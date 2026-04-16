import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schemas import ChatRequest
from app.common.deps import get_current_user
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
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


@router.post("")
async def chat(
    payload: ChatRequest,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await ChatService(session).run_chat(
        user_id=current_user.username,
        question=payload.message,
        session_id=payload.session_id,
    )
    message = result["message"]
    return _ok(
        {
            "session_id": result["session_id"],
            "answer": message["content"],
            "message": message,
            "rag_steps": result["rag_steps"],
            "rag_trace": message["rag_trace"],
        }
    )


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    result = await ChatService(session).run_chat(
        user_id=current_user.username,
        question=payload.message,
        session_id=payload.session_id,
    )

    async def event_generator():
        for step in result["rag_steps"]:
            yield _sse_event("rag_step", {"step": step})

        content = result["message"]["content"]
        for chunk in _chunk_text(content):
            yield _sse_event("content", {"content": chunk})

        yield _sse_event("trace", {"trace": result["message"]["rag_trace"]})
        yield _sse_event("done", "[DONE]")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
