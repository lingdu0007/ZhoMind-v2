from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.retention.cleanup import read_retention_status, run_retention_sweep
from app.retention.policy import PolicyChange, change_policy

router = APIRouter(prefix="/retention", tags=["retention"])


@router.get("")
async def read_retention(
    _administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return ok_response(data=await read_retention_status(session), request_id=get_request_id())


@router.put("/policy")
async def update_retention_policy(
    payload: PolicyChange,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return ok_response(data=await change_policy(session, payload, administrator), request_id=get_request_id())


@router.post("/cleanup")
async def retry_retention_cleanup(
    request: Request,
    _administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    await session.rollback()
    attempted = await run_retention_sweep(request.app.state.settings_session_factory)
    status = await read_retention_status(session)
    for data_class, result in attempted.items():
        if result.get("status") != "verified":
            status["cleanup"][data_class] = result
            status["privacy_blocked"] = True
    return ok_response(data=status, request_id=get_request_id())
