import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from time import perf_counter
from typing import TypeVar

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask
from starlette.responses import ContentStream, StreamingResponse
from starlette.types import Message, Send

from app.chat.schemas import ChatRequest
from app.common.deps import get_current_user
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.common.stream_delivery import attach_stream_delivery_observer
from app.infra.db import get_db_session
from app.operations.chat_capacity import get_chat_admission_gate
from app.service.answer_execution_store import AnswerExecutionHandle
from app.service.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

logger = logging.getLogger(__name__)
TaskResult = TypeVar("TaskResult")


class _DeliveryAwareStreamingResponse(StreamingResponse):
    """Observe ASGI send failures that occur outside the async body iterator."""

    def __init__(
        self,
        content: ContentStream,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: BackgroundTask | None = None,
        *,
        on_interruption: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(
            content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )
        self._on_interruption = on_interruption

    async def stream_response(self, send: Send) -> None:
        try:
            await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
            async for chunk in self.body_iterator:
                if not isinstance(chunk, bytes | memoryview):
                    chunk = chunk.encode(self.charset)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except BaseException:
            try:
                await self._on_interruption()
            except Exception:
                logger.exception("chat stream transport interruption cleanup failed")
            raise


class _TerminalDeliveryObserver:
    """Marks completion only after the terminal SSE response fully closes."""

    def __init__(
        self,
        *,
        terminal_body: bytes,
        on_response_completion: Callable[[], Awaitable[None]],
        on_interruption: Callable[[], Awaitable[None]],
    ) -> None:
        self._terminal_body = terminal_body
        self._on_response_completion = on_response_completion
        self._on_interruption = on_interruption
        self._terminal_body_observed = False

    async def observe_network_message(self, message: Message) -> None:
        if message.get("type") != "http.response.body":
            return
        body = message.get("body", b"")
        if isinstance(body, memoryview):
            body = body.tobytes()
        if isinstance(body, bytes) and body == self._terminal_body:
            self._terminal_body_observed = True

    async def finish_network_response(self) -> None:
        if self._terminal_body_observed:
            await self._on_response_completion()
            return
        await self._on_interruption()

    async def interrupt_network_response(self) -> None:
        await self._on_interruption()


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


async def _await_cleanup_task(task: asyncio.Task[TaskResult]) -> TaskResult:
    """Let durable stream cleanup finish despite repeated iterator cancellation."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is None:
                raise
            while current_task.cancelling():
                current_task.uncancel()
    return await task


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
    execution = result.get("message", {}).get("answer_execution", {})
    context = getattr(request.state, "operational_event", {})
    dimensions = context.get("dimensions", {})
    dimensions.update({
        "execution_state": execution.get("state"),
        "outcome": execution.get("outcome"),
        "evidence_count": len(execution.get("item_identities") or []),
    })
    request.state.operational_event = {
        **context, "dimensions": dimensions,
        "generation_route": runtime,
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
    query_conditions: list[dict] | None = None,
    inherit_conditions: bool = False,
    progress: Callable[[str, str], Awaitable[None]] | None = None,
    on_admitted: Callable[[AnswerExecutionHandle], Awaitable[None]] | None = None,
    operational: dict | None = None,
) -> dict:
    gate = get_chat_admission_gate()
    started = perf_counter()
    stage_started = started
    stage_name = "queue"
    timings: dict[str, int] = dict.fromkeys(("queue", "application", "retrieval", "provider", "persistence"), 0)
    if operational is not None:
        operational["dimensions"] = {
            "configuration_identity": gate.configuration()["identity"], "stage_durations_ms": timings,
        }
    reservation = gate.reserve(member_id=user_id)
    if reservation.state == "throttled":
        if operational is not None:
            operational["normalized_error"] = reservation.reason
            operational["dimensions"]["execution_state"] = "throttled"
        raise AppError(
            status_code=429,
            code=reservation.reason or "CHAT_QUEUE_FULL",
            message="request capacity is unavailable; retry later",
            detail={"state": "throttled", "retryable": True},
        )

    def close_stage() -> None:
        nonlocal stage_started
        now = perf_counter()
        timings[stage_name] = timings.get(stage_name, 0) + round((now - stage_started) * 1000)
        stage_started = now

    async def report(stage: str, message: str) -> None:
        nonlocal stage_name
        next_stage = {
            "queued": "queue", "running": "application", "retrieval": "retrieval",
            "generating": "provider", "persistence": "persistence",
        }.get(stage)
        if next_stage is not None and next_stage != stage_name:
            close_stage()
            stage_name = next_stage
        if progress is not None:
            await progress(stage, message)

    async def await_capacity() -> None:
        await gate.wait_for_member(reservation, session=service.session, progress=report)

    try:
        if progress is not None and reservation.state == "queued":
            await progress("queued", f"Waiting in queue: {gate.observe(reservation)['position']}")
        return await service.run_chat(
            user_id=user_id,
            question=question,
            session_id=session_id,
            query_conditions=query_conditions,
            inherit_conditions=inherit_conditions,
            progress=report,
            on_admitted=on_admitted,
            await_capacity=await_capacity,
        )
    except AppError as exc:
        if operational is not None:
            operational["normalized_error"] = exc.code
            operational["dimensions"]["execution_state"] = "failed"
        raise
    except Exception:
        if operational is not None:
            operational["normalized_error"] = {
                "retrieval": "RETRIEVAL_FAILED", "persistence": "ANSWER_EXECUTION_PERSISTENCE_FAILED",
            }.get(stage_name, operational.get("normalized_error") or "APPLICATION_FAILED")
            operational["dimensions"]["execution_state"] = "failed"
        raise
    finally:
        close_stage()
        gate.finish(reservation)


@router.post("")
async def chat(
    payload: ChatRequest,
    request: Request,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    service = ChatService(session)
    user_id, role = current_user.username, current_user.role
    result = await _run_admitted_chat(
        service,
        user_id=user_id,
        question=payload.message,
        session_id=payload.session_id,
        query_conditions=(
            [condition.model_dump() for condition in payload.query_conditions]
            if payload.query_conditions is not None
            else None
        ),
        inherit_conditions=payload.inherit_conditions,
        operational=getattr(request.state, "operational_event", None),
    )
    _record_operational_context(request, result)
    return _ok(await service.project_current_chat_result(
        result, user_id=user_id, role=role,
    ))


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    service = ChatService(session)
    user_id, role = current_user.username, current_user.role
    progress_queue: asyncio.Queue[tuple[str, str, str] | None] = asyncio.Queue()
    admitted_execution: AnswerExecutionHandle | None = None
    completed_stream_result: dict | None = None
    answer_identity_announced = False

    async def progress(stage: str, message: str) -> None:
        await progress_queue.put(("stage", stage, message))

    async def remember_admitted(execution: AnswerExecutionHandle) -> None:
        nonlocal admitted_execution
        admitted_execution = execution
        await service.record_stream_delivery_pending(
            execution_id=execution.execution_id,
            user_id=user_id,
            session_id=execution.session_id,
        )
        if not isinstance(execution.assistant_message_id, str) or not execution.assistant_message_id:
            raise ValueError("admitted answer execution has no reserved assistant message identity")
        await progress_queue.put(("answer_identity", execution.assistant_message_id, ""))

    async def run_chat_with_progress() -> dict:
        try:
            return await _run_admitted_chat(
                service,
                user_id=user_id,
                question=payload.message,
                session_id=payload.session_id,
                query_conditions=(
                    [condition.model_dump() for condition in payload.query_conditions]
                    if payload.query_conditions is not None
                    else None
                ),
                inherit_conditions=payload.inherit_conditions,
                progress=progress,
                on_admitted=remember_admitted,
                operational=getattr(request.state, "operational_event", None),
            )
        finally:
            await progress_queue.put(None)

    chat_task = asyncio.create_task(run_chat_with_progress())
    cleanup_task: asyncio.Task[dict | None] | None = None
    def consume_task_result(task: asyncio.Task) -> None:
        if task.done() and not task.cancelled():
            task.exception()

    async def _await_cancelled_chat_task() -> dict | None:
        if not chat_task.done():
            chat_task.cancel()
        try:
            return await chat_task
        except asyncio.CancelledError:
            return None
        except Exception:
            logger.exception("chat stream interruption cleanup failed")
            return None

    async def cancel_chat_task() -> dict | None:
        nonlocal cleanup_task
        if cleanup_task is None:
            # Keep the terminal-state write independent from this generator's
            # cancellation. ASGI can cancel the body iterator more than once
            # while a client disconnect is unwinding.
            cleanup_task = asyncio.create_task(_await_cancelled_chat_task())
        return await _await_cleanup_task(cleanup_task)

    delivery_interruption_task: asyncio.Task[None] | None = None
    delivery_interruption_lock = asyncio.Lock()
    delivery_finished = False

    async def _record_incomplete_delivery(result: dict | None) -> None:
        context = getattr(request.state, "operational_event", None)
        if isinstance(context, dict):
            context["normalized_error"] = "CHAT_STREAM_INTERRUPTED"
        message = result.get("message") if isinstance(result, dict) else None
        session_id = result.get("session_id") if isinstance(result, dict) else None
        execution_id = message.get("answer_execution_id") if isinstance(message, dict) else None
        if not isinstance(execution_id, str) or not execution_id or not isinstance(session_id, str) or not session_id:
            if admitted_execution is None:
                return
            execution_id = admitted_execution.execution_id
            session_id = admitted_execution.session_id
        await service.record_stream_delivery_interruption(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )

    async def _persist_incomplete_delivery(result: dict | None) -> None:
        closed_result = result if result is not None else await cancel_chat_task()
        await _record_incomplete_delivery(closed_result)

    async def _record_completed_delivery(result: dict | None) -> None:
        message = result.get("message") if isinstance(result, dict) else None
        session_id = result.get("session_id") if isinstance(result, dict) else None
        execution_id = message.get("answer_execution_id") if isinstance(message, dict) else None
        if not isinstance(execution_id, str) or not execution_id or not isinstance(session_id, str) or not session_id:
            if admitted_execution is None:
                return
            execution_id = admitted_execution.execution_id
            session_id = admitted_execution.session_id
        await service.record_stream_delivery_completion(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )

    async def _admitted_execution_projection() -> dict | None:
        if admitted_execution is None:
            return None
        return await service.get_answer_execution_projection(
            execution_id=admitted_execution.execution_id,
            user_id=user_id,
            session_id=admitted_execution.session_id,
        )

    def _terminal_events_for_admitted_execution(execution: dict) -> list[str]:
        nonlocal answer_identity_announced
        state = execution.get("state")
        if state not in {"stopped", "failed", "throttled", "rejected"}:
            raise ValueError("stream error did not retain a terminal answer execution")
        events: list[str] = []
        assistant_message_id = execution.get("assistant_message_id")
        if assistant_message_id is not None:
            if not isinstance(assistant_message_id, str) or not assistant_message_id:
                raise ValueError("terminal answer execution has a malformed assistant binding")
            if not answer_identity_announced:
                events.append(_sse_event("answer_identity", {"answer_id": assistant_message_id}))
                answer_identity_announced = True
        events.append(_sse_event("answer_execution", {"answer_execution": execution}))
        return events

    async def _admitted_terminal_events() -> list[str]:
        execution = await _admitted_execution_projection()
        if execution is None:
            return []
        return _terminal_events_for_admitted_execution(execution)

    async def _error_terminal_events(result: dict | None) -> list[str]:
        # A completed execution has already been persisted. If projection then
        # fails, its immutable result must become non-projectable before the
        # stream can emit an error terminal.
        if result is not None:
            await _await_incomplete_delivery_record(result)
            return []
        try:
            execution = await _admitted_execution_projection()
        except ValueError:
            # A post-completion task failure sees the delivery-pending gate
            # before it can read the completed projection. Record the
            # interruption directly from the admitted identity instead.
            await _await_incomplete_delivery_record(None)
            return []
        if execution is None:
            return []
        if execution.get("state") == "completed":
            await _await_incomplete_delivery_record(None)
            return []
        return _terminal_events_for_admitted_execution(execution)

    async def _await_incomplete_delivery_record(result: dict | None) -> None:
        nonlocal delivery_interruption_task
        if delivery_finished:
            return
        async with delivery_interruption_lock:
            if delivery_finished:
                return
            for attempt in range(2):
                if delivery_interruption_task is None:
                    delivery_interruption_task = asyncio.create_task(_persist_incomplete_delivery(result))
                try:
                    await _await_cleanup_task(delivery_interruption_task)
                    return
                except BaseException:
                    delivery_interruption_task = None
                    if attempt == 1:
                        raise

    async def mark_response_delivery() -> None:
        nonlocal delivery_finished
        await _record_completed_delivery(completed_stream_result)
        delivery_finished = True

    terminal_body = _sse_event("done", "[DONE]").encode("utf-8")
    attach_stream_delivery_observer(
        request.scope,
        _TerminalDeliveryObserver(
            terminal_body=terminal_body,
            on_response_completion=mark_response_delivery,
            on_interruption=lambda: _await_incomplete_delivery_record(None),
        ),
    )

    async def event_generator():
        nonlocal completed_stream_result, answer_identity_announced
        result: dict | None = None
        terminal_done_yielded = False
        try:
            try:
                while True:
                    item = await progress_queue.get()
                    if item is None:
                        break
                    event_type, value, message = item
                    if event_type == "answer_identity":
                        if answer_identity_announced:
                            raise ValueError("stream answer identity was repeated")
                        answer_identity_announced = True
                        yield _sse_event("answer_identity", {"answer_id": value})
                        continue
                    if event_type != "stage":
                        raise ValueError("stream emitted an unknown progress event")
                    yield _sse_event("stage", {"stage": value, "message": message})
                result = await chat_task
                completed_stream_result = result
                consume_task_result(chat_task)
                _record_operational_context(request, result)
                projected_message = service.project_message(result["message"], role)

                if admitted_execution is None:
                    raise ValueError("completed stream has no admitted answer execution")
                if (
                    projected_message.get("id") != admitted_execution.assistant_message_id
                ):
                    raise ValueError("completed stream contradicted its admitted answer identity")
                if not answer_identity_announced:
                    answer_identity_announced = True
                    yield _sse_event(
                        "answer_identity",
                        {"answer_id": admitted_execution.assistant_message_id},
                    )
                projected_message = await service.refresh_pending_stream_message(
                    result["message"],
                    execution_id=admitted_execution.execution_id,
                    user_id=user_id,
                    session_id=admitted_execution.session_id,
                    role=role,
                )
                if projected_message.get("answer_execution") is not None:
                    yield _sse_event("answer_execution", {"answer_execution": projected_message["answer_execution"]})
                if projected_message.get("outcome") is not None:
                    yield _sse_event("outcome", {"outcome": projected_message["outcome"]})
                if projected_message.get("insufficient_evidence_reply") is not None:
                    yield _sse_event(
                        "insufficient_evidence_reply",
                        {"insufficient_evidence_reply": projected_message["insufficient_evidence_reply"]},
                    )
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
                terminal_done_yielded = True
            except AppError as exc:
                consume_task_result(chat_task)
                for event in await _error_terminal_events(result):
                    yield event
                yield _sse_event("error", {"code": exc.code, "message": exc.message})
                yield _sse_event("done", "[DONE]")
                terminal_done_yielded = True
            except Exception:
                logger.exception("chat stream failed")
                consume_task_result(chat_task)
                try:
                    for event in await _error_terminal_events(result):
                        yield event
                except Exception:
                    logger.exception("chat stream terminal projection failed")
                    if result is not None:
                        raise
                yield _sse_event(
                    "error",
                    {"code": "CHAT_STREAM_FAILED", "message": "聊天流式处理失败，请稍后重试。"},
                )
                yield _sse_event("done", "[DONE]")
                terminal_done_yielded = True
            except asyncio.CancelledError:
                # A client disconnect is a user stop only after the private
                # execution has durably reached its stopped terminal state.
                await cancel_chat_task()
                await _await_incomplete_delivery_record(None)
                raise
            except BaseException:
                await cancel_chat_task()
                await _await_incomplete_delivery_record(None)
                raise
        except Exception:
            logger.exception("chat stream failed")
            consume_task_result(chat_task)
            try:
                for event in await _error_terminal_events(result):
                    yield event
            except Exception:
                logger.exception("chat stream terminal projection failed")
                if result is not None:
                    raise
            yield _sse_event("error", {"code": "CHAT_STREAM_FAILED", "message": "聊天流式处理失败，请稍后重试。"})
            yield _sse_event("done", "[DONE]")
            terminal_done_yielded = True
        finally:
            if not terminal_done_yielded:
                try:
                    await _await_incomplete_delivery_record(result)
                except Exception:
                    logger.exception("chat stream delivery interruption persistence failed")

    return _DeliveryAwareStreamingResponse(
        event_generator(),
        on_interruption=lambda: _await_incomplete_delivery_record(None),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
