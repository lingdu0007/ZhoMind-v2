import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.common.security import hash_password
from app.model.team_invitation import TeamInvitation
from app.model.user import User
from app.repository.team_invitation_repository import TeamInvitationRepository
from app.repository.user_repository import UserRepository
from app.service.identity_audit_service import IdentityAuditService

DEFAULT_TEAM_INVITATION_LIFETIME = timedelta(days=7)


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


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
        existing_bootstrap_administrator = await self.users.get_bootstrap_administrator()
        if existing_bootstrap_administrator is not None:
            await IdentityAuditService(self.session).record_bootstrap_replayed(existing_bootstrap_administrator)
            await self.session.commit()
            return
        if await self.users.get_by_username(normalized_username):
            raise RuntimeError("bootstrap administrator username is already assigned")

        user = await self.users.create_user(
            username=normalized_username,
            password_hash=hash_password(normalized_password),
            role="admin",
        )
        user.is_bootstrap_administrator = True
        await IdentityAuditService(self.session).record_bootstrap_created(user)
        try:
            await self.session.commit()
        except IntegrityError:
            await self.session.rollback()
            existing_bootstrap_administrator = await self.users.get_bootstrap_administrator()
            if existing_bootstrap_administrator is None:
                raise
            await IdentityAuditService(self.session).record_bootstrap_replayed(existing_bootstrap_administrator)
            await self.session.commit()

    async def issue_invitation(self, administrator: User, expires_at: datetime | None) -> tuple[TeamInvitation, str]:
        effective_expiry = expires_at or (_now() + DEFAULT_TEAM_INVITATION_LIFETIME)
        if effective_expiry.tzinfo is None:
            raise AppError(status_code=400, code="VALIDATION_ERROR", message="invitation expiry must include a timezone")
        effective_expiry = effective_expiry.astimezone(UTC)
        if effective_expiry <= _now():
            raise AppError(status_code=400, code="VALIDATION_ERROR", message="invitation expiry must be in the future")

        invitation_code = secrets.token_urlsafe(32)
        invitation = await self.invitations.create(
            code_hash=_invitation_code_hash(invitation_code),
            created_by_user_id=administrator.id,
            expires_at=effective_expiry,
        )
        await IdentityAuditService(self.session).record_invitation_issued(invitation, administrator)
        await self.session.commit()
        return invitation, invitation_code

    async def list_invitations(self) -> list[TeamInvitation]:
        return await self.invitations.list_all()

    async def revoke_invitation(self, administrator: User, invitation_id: uuid.UUID) -> TeamInvitation:
        invitation = await self.invitations.get_by_id(invitation_id)
        if invitation is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="team invitation not found")
        if invitation.revoked_at is None:
            invitation.revoked_at = _now()
            await IdentityAuditService(self.session).record_invitation_revoked(invitation, administrator)
            await self.session.commit()
        return invitation

    async def reserve_registration_invitation(self, invitation_code: str) -> TeamInvitation:
        normalized_code = invitation_code.strip()
        invitation = (
            await self.invitations.get_by_code_hash_for_update(_invitation_code_hash(normalized_code))
            if normalized_code
            else None
        )
        denial = self._registration_denial_error(invitation)
        if denial is not None:
            raise denial
        assert invitation is not None
        return invitation

    async def registration_denial_error(self, invitation_code: str) -> AppError:
        normalized_code = invitation_code.strip()
        invitation = (
            await self.invitations.get_by_code_hash(_invitation_code_hash(normalized_code))
            if normalized_code
            else None
        )
        denial = self._registration_denial_error(invitation)
        if denial is None:
            return AppError(
                status_code=403,
                code="INVITATION_REPLAYED",
                message="team invitation claim did not complete",
            )
        return denial

    @staticmethod
    def _registration_denial_error(invitation: TeamInvitation | None) -> AppError | None:
        if invitation is None:
            return AppError(status_code=403, code="INVITATION_INVALID", message="a valid team invitation is required")
        if invitation.consumed_at is not None:
            return AppError(status_code=403, code="INVITATION_REPLAYED", message="team invitation has already been used")
        if invitation.revoked_at is not None:
            return AppError(status_code=403, code="INVITATION_REVOKED", message="team invitation is revoked")
        if _as_utc(invitation.expires_at) <= _now():
            return AppError(status_code=403, code="INVITATION_EXPIRED", message="team invitation has expired")
        return None

    async def consume_registration_invitation(self, invitation: TeamInvitation, user: User) -> None:
        consumed_at = _now()
        claimed = await self.invitations.consume_once(
            invitation_id=invitation.id,
            user_id=user.id,
            consumed_at=consumed_at,
        )
        if not claimed:
            raise AppError(
                status_code=403,
                code="INVITATION_CLAIM_RACE",
                message="team invitation claim did not complete",
            )

    async def list_members(self) -> list[User]:
        return await self.users.list_members()

    async def promote_member(self, administrator: User, username: str) -> User:
        user = await self.users.get_by_username(username)
        if user is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="member not found")
        if not user.is_active:
            raise AppError(status_code=409, code="MEMBER_INACTIVE", message="deactivated member cannot be promoted")
        if user.role == "admin":
            raise AppError(status_code=409, code="RESOURCE_CONFLICT", message="member is already an administrator")
        await self.users.promote(user)
        await IdentityAuditService(self.session).record_promotion(administrator, user)
        await self.session.commit()
        return user

    async def deactivate_member(self, administrator: User, username: str) -> User:
        user = await self.users.get_by_username(username)
        if user is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="member not found")
        if user.is_active:
            await self.users.deactivate(user)
            audit = IdentityAuditService(self.session)
            await audit.record_deactivation(
                administrator,
                user,
                outcome="pending",
                reason="session_revocation_pending",
            )
            await self.session.commit()
            await self._complete_deactivation_session_revocation(administrator, user, audit)
        else:
            await self._complete_deactivation_session_revocation(
                administrator,
                user,
                IdentityAuditService(self.session),
            )
        return user

    async def _complete_deactivation_session_revocation(
        self,
        administrator: User,
        user: User,
        audit: IdentityAuditService,
    ) -> None:
        try:
            await self._revoke_sessions(username=user.username)
        except Exception:
            await audit.record_deactivation(
                administrator,
                user,
                outcome="failed",
                reason="session_store_unavailable",
            )
            await self.session.commit()
            raise
        await audit.record_deactivation(administrator, user)
        await self.session.commit()

    async def _revoke_sessions(self, username: str) -> None:
        if self.redis is None:
            raise RuntimeError("session store is required to revoke member access")
        prefix = "auth:session:"
        keys: list[str] = []
        async for raw_key in self.redis.scan_iter(match=f"{prefix}*"):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            subject, separator, _jti = key.removeprefix(prefix).rpartition(":")
            if separator and subject == username:
                keys.append(key)
        if keys:
            await self.redis.delete(*keys)
