from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import get_current_user
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.service.chat_service import ChatService

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _ok(data: dict) -> dict:
    payload = ok_response(data=data, request_id=get_request_id())
    payload.update(data)
    return payload


@router.get("")
async def list_sessions(
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    items = await ChatService(session).list_sessions(user_id=current_user.username)
    return _ok({"sessions": items})


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    items = await ChatService(session).get_session_messages(session_id=session_id, user_id=current_user.username)
    return _ok({"session_id": session_id, "messages": items})


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    deleted = await ChatService(session).delete_session(session_id=session_id, user_id=current_user.username)
    return _ok({"session_id": session_id, "deleted": deleted})
