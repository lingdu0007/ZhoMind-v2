from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import uuid

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.common.security import build_auth_session_key, hash_password
from app.model.team_invitation import TeamInvitation
from app.model.user import User
from app.repository.team_invitation_repository import TeamInvitationRepository
from app.repository.user_repository import UserRepository

DEFAULT_TEAM_INVITATION_LIFETIME = timedelta(days=7)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _invitation_code_hash(invitation_code: str) -> str:
    return hashlib.sha256(invitation_code.encode("utf-8")).hexdigest()


class MemberAdmissionService:
    def __init__(self, session: AsyncSession, redis: Redis | None) -> None:
        self.session = session
        self.redis = redis
        self.users = UserRepository(session)
        self.invitations = TeamInvitationRepository(session)

    async def create_bootstrap_administrator(self, username: str, password: str) -> None:
        normalized_username = username.strip()
        normalized_password = password.strip()
        if not normalized_username and not normalized_password:
            return
        if not normalized_username or not normalized_password:
            raise RuntimeError("bootstrap administrator credentials must include both username and password")
        if await self.users.has_bootstrap_administrator():
            return
        if await self.users.get_by_username(normalized_username):
            raise RuntimeError("bootstrap administrator username is already assigned")

        user = await self.users.create_user(
            username=normalized_username,
            password_hash=hash_password(normalized_password),
            role="admin",
        )
        user.is_bootstrap_administrator = True
        await self.session.commit()

    async def issue_invitation(self, administrator: User, expires_at: datetime | None) -> tuple[TeamInvitation, str]:
        effective_expiry = expires_at or (_now() + DEFAULT_TEAM_INVITATION_LIFETIME)
        if effective_expiry.tzinfo is None:
            raise AppError(status_code=400, code="VALIDATION_ERROR", message="invitation expiry must include a timezone")
        effective_expiry = effective_expiry.astimezone(timezone.utc)
        if effective_expiry <= _now():
            raise AppError(status_code=400, code="VALIDATION_ERROR", message="invitation expiry must be in the future")

        invitation_code = secrets.token_urlsafe(32)
        invitation = await self.invitations.create(
            code_hash=_invitation_code_hash(invitation_code),
            created_by_user_id=administrator.id,
            expires_at=effective_expiry,
        )
        await self.session.commit()
        return invitation, invitation_code

    async def list_invitations(self) -> list[TeamInvitation]:
        return await self.invitations.list_all()

    async def revoke_invitation(self, invitation_id: uuid.UUID) -> TeamInvitation:
        invitation = await self.invitations.get_by_id(invitation_id)
        if invitation is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="team invitation not found")
        if invitation.revoked_at is None:
            invitation.revoked_at = _now()
            await self.session.commit()
        return invitation

    async def verify_registration_invitation(self, invitation_code: str) -> None:
        normalized_code = invitation_code.strip()
        invitation = await self.invitations.get_by_code_hash(_invitation_code_hash(normalized_code)) if normalized_code else None
        if invitation is None or invitation.revoked_at is not None or _as_utc(invitation.expires_at) <= _now():
            raise AppError(status_code=403, code="INVITATION_INVALID", message="a valid team invitation is required")

    async def list_members(self) -> list[User]:
        return await self.users.list_members()

    async def promote_member(self, username: str) -> User:
        user = await self.users.get_by_username(username)
        if user is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="member not found")
        if not user.is_active:
            raise AppError(status_code=409, code="MEMBER_INACTIVE", message="deactivated member cannot be promoted")
        if user.role == "admin":
            raise AppError(status_code=409, code="RESOURCE_CONFLICT", message="member is already an administrator")
        await self.users.promote(user)
        await self.session.commit()
        return user

    async def deactivate_member(self, username: str) -> User:
        user = await self.users.get_by_username(username)
        if user is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="member not found")
        if user.is_active:
            await self.users.deactivate(user)
            await self.session.commit()
        await self._revoke_sessions(username=user.username)
        return user

    async def _revoke_sessions(self, username: str) -> None:
        if self.redis is None:
            raise RuntimeError("session store is required to revoke member access")
        keys = [key async for key in self.redis.scan_iter(match=f"auth:session:{username}:*")]
        if keys:
            await self.redis.delete(*keys)
