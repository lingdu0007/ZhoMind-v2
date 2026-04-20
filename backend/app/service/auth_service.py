from datetime import datetime, timezone

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.common.exceptions import AppError
from app.common.security import build_auth_session_key, create_access_token, decode_access_token, hash_password, verify_password
from app.repository.user_repository import UserRepository


class AuthService:
    def __init__(self, session: AsyncSession, redis: Redis) -> None:
        self.repo = UserRepository(session)
        self.session = session
        self.redis = redis

    async def _persist_auth_session(self, token: str, username: str, role: str) -> None:
        payload = decode_access_token(token)
        exp = int(payload.get("exp") or 0)
        jti = str(payload.get("jti") or "")
        if exp <= 0 or not jti:
            raise AppError(status_code=500, code="AUTH_TOKEN_INVALID_PAYLOAD", message="token payload missing exp or jti")

        ttl = max(exp - int(datetime.now(timezone.utc).timestamp()), 1)
        key = build_auth_session_key(subject=username, jti=jti)
        await self.redis.hset(
            key,
            mapping={
                "username": username,
                "role": role,
                "issued_at": str(payload.get("iat") or ""),
            },
        )
        await self.redis.expire(key, ttl)

    async def register(self, username: str, password: str, role: str, admin_code: str | None) -> dict[str, str]:
        normalized_username = username.strip()
        normalized_password = password.strip()
        if not normalized_username or not normalized_password:
            raise AppError(status_code=400, code="VALIDATION_ERROR", message="username and password are required")

        normalized_role = "admin" if role == "admin" else "user"
        settings = get_settings()
        if normalized_role == "admin" and settings.admin_invite_code != (admin_code or ""):
            raise AppError(status_code=403, code="AUTH_FORBIDDEN", message="invalid admin invite code")

        existing = await self.repo.get_by_username(normalized_username)
        if existing:
            raise AppError(status_code=409, code="RESOURCE_CONFLICT", message="username already exists")

        user = await self.repo.create_user(
            username=normalized_username,
            password_hash=hash_password(normalized_password),
            role=normalized_role,
        )
        await self.session.commit()
        token = create_access_token(subject=user.username, role=user.role)
        await self._persist_auth_session(token=token, username=user.username, role=user.role)
        return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}

    async def login(self, username: str, password: str) -> dict[str, str]:
        user = await self.repo.get_by_username(username.strip())
        if not user or not verify_password(password.strip(), user.password_hash):
            raise AppError(status_code=401, code="AUTH_INVALID_CREDENTIALS", message="invalid username or password")
        token = create_access_token(subject=user.username, role=user.role)
        await self._persist_auth_session(token=token, username=user.username, role=user.role)
        return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}
