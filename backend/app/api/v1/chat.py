import json

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


async def _run_admitted_chat(service: ChatService, *, user_id: str, question: str, session_id: str | None) -> dict:
    gate = get_chat_admission_gate()
    if not gate.try_admit():
        raise AppError(
            status_code=429,
            code="CHAT_CONCURRENCY_LIMIT_REACHED",
            message="the first-release concurrent chat limit has been reached",
        )
    try:
        return await service.run_chat(user_id=user_id, question=question, session_id=session_id)
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
    result = await _run_admitted_chat(
        service,
        user_id=current_user.username,
        question=payload.message,
        session_id=payload.session_id,
    )
    _record_operational_context(request, result)
    projected_message = service.project_message(result["message"], current_user.role)

    async def event_generator():
        content = projected_message["content"]
        for chunk in _chunk_text(content):
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
