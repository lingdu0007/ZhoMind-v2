from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.common.deps import require_admin
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.settings.service import SystemSettingsDraftService

router = APIRouter(prefix="/settings", tags=["settings"])


def _require_settings_draft_rollout() -> None:
    if not get_settings().system_settings_draft_enabled:
        raise AppError(status_code=404, code="SETTINGS_DRAFT_DISABLED", message="settings draft rollout is disabled")


@router.get("/draft")
async def get_settings_draft(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    _require_settings_draft_rollout()
    data = await SystemSettingsDraftService(session).read()
    return ok_response(data=data, request_id=get_request_id())


@router.put("/draft")
async def save_settings_draft(
    payload: object = Body(...),
    current_admin=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    _require_settings_draft_rollout()
    data = await SystemSettingsDraftService(session).save(actor=current_admin.username, payload=payload)
    return ok_response(data=data, request_id=get_request_id())
