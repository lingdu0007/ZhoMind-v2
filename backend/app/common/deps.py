from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.common.security import build_auth_session_key, decode_access_token
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.repository.user_repository import UserRepository

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
):
    if credentials is None:
        raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="missing bearer token")

    payload = decode_access_token(credentials.credentials)
    username = str(payload.get("sub") or "")
    jti = str(payload.get("jti") or "")
    if not username or not jti:
        raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="invalid token payload")

    key = build_auth_session_key(subject=username, jti=jti)
    if not await redis.exists(key):
        raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="session not found")

    user = await UserRepository(session).get_by_username(username)
    if user is None:
        raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="user not found")
    return user


async def require_admin(user=Depends(get_current_user)):
    if user.role != "admin":
        raise AppError(status_code=403, code="AUTH_FORBIDDEN", message="admin required")
    return user
