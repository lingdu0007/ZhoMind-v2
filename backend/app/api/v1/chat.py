import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schemas import ChatRequest
from app.common.deps import get_current_user
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.operations.chat_capacity import get_chat_admission_gate
from app.service.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger(__name__)


def _ok(data: dict) -> dict:
    payload = ok_response(data=data, request_id=get_request_id())
    payload.update(data)
    return payload


def _chunk_text(content: str, size: int = 28) -> list[str]:
    text = content or ""
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sse_event(event: str, data: dict | str) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _record_operational_context(request: Request, result: dict) -> None:
    trace = result.get("message", {}).get("rag_trace", {})
    gate = trace.get("gate", {}) if isinstance(trace, dict) else {}
    runtime = trace.get("runtime", {}) if isinstance(trace, dict) else {}
    evidence = trace.get("evidence", []) if isinstance(trace, dict) else []
    attempts = runtime.get("provider_attempts", []) if isinstance(runtime, dict) else []
    failed_attempt = next(
        (
            item
            for item in reversed(attempts)
            if isinstance(item, dict) and isinstance(item.get("error_code"), str)
        ),
        {},
    )
    request.state.operational_event = {
        "gate_outcome": "passed" if gate.get("passed") is True else "rejected" if gate.get("passed") is False else "unavailable",
        "provider_identity": (
            runtime.get("final_provider") if isinstance(runtime, dict) else None
        ) or failed_attempt.get("provider"),
        "normalized_error": failed_attempt.get("error_code"),
        "candidate_count": len(evidence) if isinstance(evidence, list) else None,
    }


async def _run_admitted_chat(
    service: ChatService,
    *,
    user_id: str,
    question: str,
    session_id: str | None,
    progress: Callable[[str, str], Awaitable[None]] | None = None,
) -> dict:
    gate = get_chat_admission_gate()
    if not gate.try_admit():
        raise AppError(
            status_code=429,
            code="CHAT_CONCURRENCY_LIMIT_REACHED",
            message="the first-release concurrent chat limit has been reached",
        )
    try:
        return await service.run_chat(
            user_id=user_id,
            question=question,
            session_id=session_id,
            progress=progress,
        )
    finally:
        gate.release()


@router.post("")
async def chat(
    payload: ChatRequest,
    request: Request,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    service = ChatService(session)
    result = await _run_admitted_chat(
        service,
        user_id=current_user.username,
        question=payload.message,
        session_id=payload.session_id,
    )
    _record_operational_context(request, result)
    return _ok(service.project_chat_result(result, current_user.role))


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    service = ChatService(session)
    progress_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()

    async def progress(stage: str, message: str) -> None:
        await progress_queue.put((stage, message))

    async def run_chat_with_progress() -> dict:
        try:
            return await _run_admitted_chat(
                service,
                user_id=current_user.username,
                question=payload.message,
                session_id=payload.session_id,
                progress=progress,
            )
        finally:
            await progress_queue.put(None)

    chat_task = asyncio.create_task(run_chat_with_progress())
    # Gate/provider context is produced while the body streams; ask the
    # operational middleware to read request.state after the body completes.
    request.state.defer_operational_event = True

    def consume_task_result(task: asyncio.Task) -> None:
        if task.done() and not task.cancelled():
            task.exception()

    async def event_generator():
        try:
            while True:
                item = await progress_queue.get()
                if item is None:
                    break
                stage, message = item
                yield _sse_event("stage", {"stage": stage, "message": message})
            result = await chat_task
        except AppError as exc:
            consume_task_result(chat_task)
            yield _sse_event("error", {"code": exc.code, "message": exc.message})
            yield _sse_event("done", "[DONE]")
            return
        except Exception:
            logger.exception("chat stream failed")
            consume_task_result(chat_task)
            yield _sse_event("error", {"code": "CHAT_STREAM_FAILED", "message": "聊天流式处理失败，请稍后重试。"})
            yield _sse_event("done", "[DONE]")
            return
        except BaseException:
            # Client disconnect / shutdown: stop the background chat.
            if not chat_task.done():
                chat_task.cancel()
            chat_task.add_done_callback(consume_task_result)
            raise
        consume_task_result(chat_task)
        _record_operational_context(request, result)
        projected_message = service.project_message(result["message"], current_user.role)

        yield _sse_event("answer_identity", {"answer_id": projected_message["id"]})
        if projected_message.get("outcome") is not None:
            yield _sse_event("outcome", {"outcome": projected_message["outcome"]})
        for chunk in _chunk_text(projected_message["content"]):
            yield _sse_event("content", {"content": chunk})

        if projected_message.get("evidence_summary") is not None:
            yield _sse_event("evidence_summary", {"evidence_summary": projected_message["evidence_summary"]})
        if projected_message.get("retrieval_diagnostics") is not None:
            yield _sse_event(
                "retrieval_diagnostics",
                {"retrieval_diagnostics": projected_message["retrieval_diagnostics"]},
            )
        yield _sse_event("done", "[DONE]")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
