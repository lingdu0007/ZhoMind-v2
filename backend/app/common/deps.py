from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.common.security import decode_access_token
from app.infra.db import get_db_session
from app.repository.user_repository import UserRepository

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_db_session),
):
    if credentials is None:
        raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="missing bearer token")
    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub", "")
    user = await UserRepository(session).get_by_username(username)
    if user is None:
        raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="user not found")
    return user


async def require_admin(user=Depends(get_current_user)):
    if user.role != "admin":
        raise AppError(status_code=403, code="AUTH_FORBIDDEN", message="admin required")
    return user
