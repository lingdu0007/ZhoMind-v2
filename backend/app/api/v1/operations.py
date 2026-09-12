from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.operations.pilot_snapshot import measurement_snapshot
from app.operations.service import OperationsService

router = APIRouter(prefix="/operations", tags=["operations"])


@router.get("")
async def read_operations(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await OperationsService(session).read()
    return ok_response(data=data, request_id=get_request_id())


@router.get("/measurement-snapshot")
async def read_measurement_snapshot(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return ok_response(data=await measurement_snapshot(session), request_id=get_request_id())


@router.get("/requests/{request_id}")
async def read_request_measurement(
    request_id: UUID,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await OperationsService(session).read_request(request_id)
    return ok_response(data=data, request_id=get_request_id())
