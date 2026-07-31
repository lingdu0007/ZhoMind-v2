from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.schemas import LoginRequest, MeData, RegisterRequest, WorkspaceCapabilities
from app.common.config import get_settings
from app.common.deps import get_current_user
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.service.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _workspace_capabilities(role: str) -> WorkspaceCapabilities:
    settings = get_settings()
    return WorkspaceCapabilities(
        system_settings=(
            role == "admin"
            and settings.system_settings_draft_enabled
            and settings.system_settings_application_enabled
        )
    )


@router.post("/register")
async def register(
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    data = await AuthService(session, redis=redis).register(
        username=payload.username,
        password=payload.password,
        role=payload.role,
        admin_code=payload.admin_code,
    )
    return ok_response(data=data, request_id=get_request_id())


@router.post("/login")
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    data = await AuthService(session, redis=redis).login(username=payload.username, password=payload.password)
    return ok_response(data=data, request_id=get_request_id())


@router.get("/me")
async def me(current_user=Depends(get_current_user)) -> dict:
    return ok_response(
        data=MeData(
            username=current_user.username,
            role=current_user.role,
            capabilities=_workspace_capabilities(current_user.role),
        ).model_dump(),
        request_id=get_request_id(),
    )
