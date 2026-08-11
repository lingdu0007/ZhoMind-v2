from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import get_current_user
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.knowledge_map.service import KnowledgeMapService

router = APIRouter(prefix="/knowledge-map", tags=["knowledge-map"])


def _ok(data: dict) -> dict:
    payload = ok_response(data=data, request_id=get_request_id())
    payload.update(data)
    return payload


@router.get("")
async def list_knowledge_map(
    _current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return _ok(await KnowledgeMapService(session).list())


@router.get("/{entry_id}")
async def get_knowledge_map_entry(
    entry_id: str,
    _current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return _ok(await KnowledgeMapService(session).get(entry_id))
