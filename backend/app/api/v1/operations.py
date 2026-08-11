from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.operations.service import OperationsService

router = APIRouter(prefix="/operations", tags=["operations"])


@router.get("")
async def read_operations(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await OperationsService(session).read()
    return ok_response(data=data, request_id=get_request_id())
