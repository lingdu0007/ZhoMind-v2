from fastapi import APIRouter, BackgroundTasks, Body, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.common.deps import require_admin
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import SessionLocal, get_db_session
from app.settings.generation_routes import GenerationRouteService
from app.settings.service import SystemSettingsDraftService

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/generation-route")
async def get_generation_route(
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await GenerationRouteService(session).read()
    return ok_response(data=data, request_id=get_request_id())


@router.put("/generation-route")
async def save_generation_route(
    payload: object = Body(...),
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await GenerationRouteService(session).save(actor=f"member:{administrator.id}", payload=payload)
    return ok_response(data=data, request_id=get_request_id())


@router.post("/generation-route/activate")
async def activate_generation_route(
    payload: object = Body(...),
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    data = await GenerationRouteService(session).activate(actor=f"member:{administrator.id}", payload=payload)
    return ok_response(data=data, request_id=get_request_id())


def _require_settings_draft_rollout() -> None:
    if not get_settings().system_settings_draft_enabled:
        raise AppError(status_code=404, code="SETTINGS_DRAFT_DISABLED", message="settings draft rollout is disabled")


def _require_settings_application_rollout() -> None:
    if not get_settings().system_settings_application_enabled:
        raise AppError(
            status_code=404,
            code="SETTINGS_APPLICATION_DISABLED",
            message="settings application rollout is disabled",
        )


async def _complete_application(session_factory, *, actor: str, version: int) -> None:
    async with session_factory() as session:
        await SystemSettingsDraftService(session).complete_application(actor=actor, version=version)


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


@router.post("/apply")
async def apply_settings_version(
    background_tasks: BackgroundTasks,
    request: Request,
    payload: object = Body(...),
    current_admin=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    _require_settings_draft_rollout()
    _require_settings_application_rollout()
    if not isinstance(payload, dict):
        raise AppError(status_code=400, code="SETTINGS_VERSION_INVALID", message="saved settings version is invalid")
    data = await SystemSettingsDraftService(session).begin_application(actor=current_admin.username, version=payload.get("version"))
    session_factory = getattr(request.app.state, "settings_session_factory", SessionLocal)
    background_tasks.add_task(
        _complete_application,
        session_factory,
        actor=current_admin.username,
        version=data["application"]["version"],
    )
    return ok_response(data=data, request_id=get_request_id())
