from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import cast

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.common.security import (
    build_auth_session_key,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.operations.limits import MAX_ACTIVE_MEMBERS
from app.repository.user_repository import UserRepository
from app.service.identity_audit_service import IdentityAuditService
from app.service.member_admission_service import MemberAdmissionService


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

        ttl = max(exp - int(datetime.now(UTC).timestamp()), 1)
        key = build_auth_session_key(subject=username, jti=jti)
        # redis.asyncio types hset as Union[Awaitable[int], int]; the asyncio
        # client always returns an awaitable, so cast narrows the union.
        await cast(Awaitable[int], self.redis.hset(
            key,
            mapping={
                "username": username,
                "role": role,
                "issued_at": str(payload.get("iat") or ""),
            },
        ))
        await self.redis.expire(key, ttl)

    async def register(self, username: str, password: str, invitation_code: str) -> dict[str, str]:
        normalized_username = username.strip()
        normalized_password = password.strip()
        if not normalized_username or not normalized_password:
            raise AppError(status_code=400, code="VALIDATION_ERROR", message="username and password are required")

        try:
            admission = MemberAdmissionService(self.session, self.redis)
            invitation = await admission.reserve_registration_invitation(invitation_code)
            existing = await self.repo.get_by_username(normalized_username)
            if existing:
                raise AppError(status_code=409, code="RESOURCE_CONFLICT", message="username already exists")
            await self.repo.lock_active_members()
            if await self.repo.count_active_members() >= MAX_ACTIVE_MEMBERS:
                raise AppError(
                    status_code=409,
                    code="ACTIVE_MEMBER_LIMIT_REACHED",
                    message="the first-release active member limit has been reached",
                )

            user = await self.repo.create_user(
                username=normalized_username,
                password_hash=hash_password(normalized_password),
                role="user",
            )
            await admission.consume_registration_invitation(invitation, user)
            await IdentityAuditService(self.session).record_registration_admitted(user, invitation)
            await self.session.commit()
        except AppError as exc:
            if exc.code == "INVITATION_CLAIM_RACE":
                await self.session.rollback()
                denial = await MemberAdmissionService(self.session, self.redis).registration_denial_error(invitation_code)
                await IdentityAuditService(self.session).record_registration_denied(
                    invitation_code,
                    reason=denial.code.removeprefix("INVITATION_").lower(),
                )
                await self.session.commit()
                raise denial from exc
            if exc.code.startswith("INVITATION_"):
                await self.session.rollback()
                await IdentityAuditService(self.session).record_registration_denied(
                    invitation_code,
                    reason=exc.code.removeprefix("INVITATION_").lower(),
                )
                await self.session.commit()
            raise
        token = create_access_token(subject=user.username, role=user.role)
        await self._persist_auth_session(token=token, username=user.username, role=user.role)
        return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}

    async def login(self, username: str, password: str) -> dict[str, str]:
        user = await self.repo.get_by_username(username.strip())
        if not user or not verify_password(password.strip(), user.password_hash):
            raise AppError(status_code=401, code="AUTH_INVALID_CREDENTIALS", message="invalid username or password")
        if not user.is_active:
            raise AppError(status_code=403, code="AUTH_INACTIVE", message="member is deactivated")
        token = create_access_token(subject=user.username, role=user.role)
        await self._persist_auth_session(token=token, username=user.username, role=user.role)
        return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}

    async def logout(self, token: str, current_user) -> None:
        payload = decode_access_token(token)
        username = str(payload.get("sub") or "")
        jti = str(payload.get("jti") or "")
        if not username or not jti:
            raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="invalid token payload")
        if current_user.username != username:
            raise AppError(status_code=401, code="AUTH_INVALID_TOKEN", message="invalid token subject")

        audit = IdentityAuditService(self.session)
        await audit.record_logout(
            current_user,
            jti=jti,
            outcome="pending",
            reason="session_revocation_pending",
        )
        await self.session.commit()
        try:
            await self.redis.delete(build_auth_session_key(subject=username, jti=jti))
        except Exception:
            await audit.record_logout(
                current_user,
                jti=jti,
                outcome="failed",
                reason="session_store_unavailable",
            )
            await self.session.commit()
            raise
        await audit.record_logout(current_user, jti=jti)
        await self.session.commit()
