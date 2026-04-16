from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.schemas import LoginRequest, RegisterRequest
from app.common.deps import get_current_user
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.service.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
async def register(payload: RegisterRequest, session: AsyncSession = Depends(get_db_session)) -> dict:
    data = await AuthService(session).register(
        username=payload.username,
        password=payload.password,
        role=payload.role,
        admin_code=payload.admin_code,
    )
    return ok_response(data=data, request_id=get_request_id())


@router.post("/login")
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_db_session)) -> dict:
    data = await AuthService(session).login(username=payload.username, password=payload.password)
    return ok_response(data=data, request_id=get_request_id())


@router.get("/me")
async def me(current_user=Depends(get_current_user)) -> dict:
    return ok_response(
        data={"username": current_user.username, "role": current_user.role},
        request_id=get_request_id(),
    )
