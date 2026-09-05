from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.delivery_acceptance.schemas import (
    CreateDeliveryAcceptanceRecordRequest,
    UpdateDeliveryAcceptanceStatusRequest,
)
from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.infra.db import get_db_session

router = APIRouter(prefix="/acceptance", tags=["acceptance"])


@router.post("/records")
async def create_delivery_acceptance_record(
    payload: CreateDeliveryAcceptanceRecordRequest,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    record = await DeliveryAcceptanceService(session).create(payload, administrator)
    return ok_response(data=record, request_id=get_request_id())


@router.get("/records/{record_identity}")
async def get_delivery_acceptance_record(
    record_identity: str,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    del administrator
    record = await DeliveryAcceptanceService(session).get_projection(record_identity)
    return ok_response(data=record, request_id=get_request_id())


@router.post("/records/{record_identity}/status")
async def update_delivery_acceptance_status(
    record_identity: str,
    payload: UpdateDeliveryAcceptanceStatusRequest,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    record = await DeliveryAcceptanceService(session).update_status(record_identity, payload, administrator)
    return ok_response(data=record, request_id=get_request_id())
