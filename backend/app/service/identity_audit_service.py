from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.canonical import CanonicalEventType, CanonicalRecordClass, StableIdentity, StableIdentityKind
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.team_invitation import TeamInvitation
from app.model.user import User


class IdentityAuditService:
    """Persist minimal, content-free identity lifecycle records and events."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def member_identity(user: User) -> str:
        return StableIdentity(StableIdentityKind.MEMBER, user.id.hex).stable_id

    @staticmethod
    def invitation_identity(invitation: TeamInvitation) -> str:
        return StableIdentity(StableIdentityKind.TEAM_INVITATION, invitation.id.hex).stable_id

    @staticmethod
    def invitation_reference(code_hash: str) -> str:
        return f"sha256:{code_hash}"

    @staticmethod
    def admission_attempt_identity(code_hash: str) -> str:
        return StableIdentity(StableIdentityKind.ADMISSION_ATTEMPT, code_hash).stable_id

    @staticmethod
    def session_reference(jti: str) -> str:
        return f"sha256:{hashlib.sha256(jti.encode('utf-8')).hexdigest()}"

    async def ensure_member_record(self, user: User, *, admission_path: str) -> str:
        identity = self.member_identity(user)
        if await self.session.get(CanonicalRecordModel, identity) is None:
            self.session.add(
                CanonicalRecordModel(
                    stable_id=identity,
                    identity_kind=StableIdentityKind.MEMBER.value,
                    identity_value=user.id.hex,
                    state="active" if user.is_active else "deactivated",
                    record_class=CanonicalRecordClass.AUTHORITATIVE.value,
                    payload={
                        "schema": "identity_record/v1",
                        "admission_path": admission_path,
                        "role": self._role_name(user.role),
                        "bootstrap_administrator": user.is_bootstrap_administrator,
                    },
                    legacy_type="users",
                    legacy_id=str(user.id),
                )
            )
            await self.session.flush()
        return identity

    async def ensure_invitation_record(self, invitation: TeamInvitation, *, issued_by_identity: str) -> str:
        identity = self.invitation_identity(invitation)
        if await self.session.get(CanonicalRecordModel, identity) is None:
            self.session.add(
                CanonicalRecordModel(
                    stable_id=identity,
                    identity_kind=StableIdentityKind.TEAM_INVITATION.value,
                    identity_value=invitation.id.hex,
                    state="active",
                    record_class=CanonicalRecordClass.AUTHORITATIVE.value,
                    payload={
                        "schema": "identity_record/v1",
                        "issued_by_identity": issued_by_identity,
                        "invitation_reference": self.invitation_reference(invitation.code_hash),
                    },
                    legacy_type="team_invitations",
                    legacy_id=str(invitation.id),
                )
            )
            await self.session.flush()
        return identity

    async def ensure_admission_attempt_record(self, code_hash: str) -> str:
        identity = self.admission_attempt_identity(code_hash)
        if await self.session.get(CanonicalRecordModel, identity) is None:
            self.session.add(
                CanonicalRecordModel(
                    stable_id=identity,
                    identity_kind=StableIdentityKind.ADMISSION_ATTEMPT.value,
                    identity_value=code_hash,
                    state="denied",
                    record_class=CanonicalRecordClass.IMMUTABLE.value,
                    payload={
                        "schema": "identity_record/v1",
                        "invitation_reference": self.invitation_reference(code_hash),
                    },
                )
            )
            await self.session.flush()
        return identity

    async def record_bootstrap_created(self, user: User) -> None:
        target = await self.ensure_member_record(user, admission_path="bootstrap")
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.MEMBER,
            event_type=CanonicalEventType.CREATED,
            from_state=None,
            to_state="active",
            action="bootstrap_administrator",
            outcome="created",
            reason="server_configuration",
            actor_identity="system:bootstrap",
            target_identity=target,
            reference_identity="bootstrap_configuration",
        )

    async def record_bootstrap_replayed(self, user: User) -> None:
        target = await self.ensure_member_record(user, admission_path="bootstrap")
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.MEMBER,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state="active",
            to_state="active",
            action="bootstrap_administrator",
            outcome="replayed",
            reason="bootstrap_administrator_exists",
            actor_identity="system:bootstrap",
            target_identity=target,
            reference_identity="bootstrap_configuration",
        )

    async def record_invitation_issued(self, invitation: TeamInvitation, administrator: User) -> None:
        actor = await self.ensure_member_record(administrator, admission_path="bootstrap_or_promotion")
        target = await self.ensure_invitation_record(invitation, issued_by_identity=actor)
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.TEAM_INVITATION,
            event_type=CanonicalEventType.CREATED,
            from_state=None,
            to_state="active",
            action="issue_invitation",
            outcome="issued",
            reason="administrator_authorized",
            actor_identity=actor,
            target_identity=target,
            reference_identity=self.invitation_reference(invitation.code_hash),
        )

    async def record_invitation_revoked(self, invitation: TeamInvitation, administrator: User) -> None:
        actor = await self.ensure_member_record(administrator, admission_path="bootstrap_or_promotion")
        target = await self.ensure_invitation_record(
            invitation,
            issued_by_identity=self._member_identity_from_uuid(invitation.created_by_user_id),
        )
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.TEAM_INVITATION,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state="active",
            to_state="revoked",
            action="revoke_invitation",
            outcome="revoked",
            reason="administrator_authorized",
            actor_identity=actor,
            target_identity=target,
            reference_identity=self.invitation_reference(invitation.code_hash),
        )

    async def record_registration_admitted(self, user: User, invitation: TeamInvitation) -> None:
        target = await self.ensure_member_record(user, admission_path="team_invitation")
        invitation_identity = await self.ensure_invitation_record(
            invitation,
            issued_by_identity=self._member_identity_from_uuid(invitation.created_by_user_id),
        )
        await self._append(
            aggregate_identity=invitation_identity,
            aggregate_kind=StableIdentityKind.TEAM_INVITATION,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state="active",
            to_state="consumed",
            action="registration_invitation",
            outcome="admitted",
            reason="valid",
            actor_identity="anonymous:registration",
            target_identity=target,
            reference_identity=self.invitation_reference(invitation.code_hash),
        )

    async def record_registration_denied(self, invitation_code: str, *, reason: str) -> None:
        code_hash = hashlib.sha256(invitation_code.strip().encode("utf-8")).hexdigest()
        reference = self.invitation_reference(code_hash)
        invitation = (
            await self.session.execute(select(TeamInvitation).where(TeamInvitation.code_hash == code_hash))
        ).scalar_one_or_none()
        if invitation is None:
            aggregate = await self.ensure_admission_attempt_record(code_hash)
            aggregate_kind = StableIdentityKind.ADMISSION_ATTEMPT
        else:
            aggregate = await self.ensure_invitation_record(
                invitation,
                issued_by_identity=self._member_identity_from_uuid(invitation.created_by_user_id),
            )
            aggregate_kind = StableIdentityKind.TEAM_INVITATION
        await self._append(
            aggregate_identity=aggregate,
            aggregate_kind=aggregate_kind,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=None,
            to_state="denied",
            action="registration_invitation",
            outcome="denied",
            reason=reason,
            actor_identity="anonymous:registration",
            target_identity=aggregate,
            reference_identity=reference,
        )

    async def record_promotion(self, administrator: User, member: User) -> None:
        actor = await self.ensure_member_record(administrator, admission_path="bootstrap_or_promotion")
        target = await self.ensure_member_record(member, admission_path="team_invitation_or_legacy")
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.MEMBER,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state="knowledge_user",
            to_state="system_administrator",
            action="promotion",
            outcome="promoted",
            reason="administrator_authorized",
            actor_identity=actor,
            target_identity=target,
            reference_identity=target,
        )

    async def record_logout(
        self,
        user: User,
        *,
        jti: str,
        outcome: str = "revoked",
        reason: str = "current_bearer",
    ) -> None:
        target = await self.ensure_member_record(user, admission_path="bootstrap_or_promotion")
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.MEMBER,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state="active",
            to_state="active",
            action="logout",
            outcome=outcome,
            reason=reason,
            actor_identity=target,
            target_identity=target,
            reference_identity=self.session_reference(jti),
        )

    async def record_deactivation(
        self,
        administrator: User,
        member: User,
        *,
        outcome: str = "deactivated",
        reason: str = "administrator_authorized",
    ) -> None:
        actor = await self.ensure_member_record(administrator, admission_path="bootstrap_or_promotion")
        target = await self.ensure_member_record(member, admission_path="team_invitation_or_legacy")
        await self._append(
            aggregate_identity=target,
            aggregate_kind=StableIdentityKind.MEMBER,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state="active",
            to_state="deactivated",
            action="deactivation",
            outcome=outcome,
            reason=reason,
            actor_identity=actor,
            target_identity=target,
            reference_identity=target,
        )

    async def list_events(self) -> list[dict[str, Any]]:
        result = await self.session.execute(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_kind.in_(
                    [StableIdentityKind.MEMBER.value, StableIdentityKind.TEAM_INVITATION.value]
                    + [StableIdentityKind.ADMISSION_ATTEMPT.value]
                )
            )
            .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
        )
        return [self._event_projection(event) for event in result.scalars()]

    async def _append(
        self,
        *,
        aggregate_identity: str,
        aggregate_kind: StableIdentityKind,
        event_type: CanonicalEventType,
        from_state: str | None,
        to_state: str,
        action: str,
        outcome: str,
        reason: str,
        actor_identity: str,
        target_identity: str,
        reference_identity: str,
    ) -> None:
        self.session.add(
            CanonicalEventModel(
                aggregate_id=aggregate_identity,
                aggregate_kind=aggregate_kind.value,
                event_type=event_type.value,
                from_state=from_state,
                to_state=to_state,
                payload={
                    "schema": "identity_audit/v1",
                    "action": action,
                    "outcome": outcome,
                    "reason": reason,
                    "actor_identity": actor_identity,
                    "target_identity": target_identity,
                    "reference_identity": reference_identity,
                },
                occurred_at=datetime.now(UTC),
                recorded_by=actor_identity if actor_identity.startswith("member:") else None,
            )
        )
        await self.session.flush()

    @staticmethod
    def _event_projection(event: CanonicalEventModel) -> dict[str, Any]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        return {
            "id": event.id,
            "action": payload.get("action", "unknown"),
            "outcome": payload.get("outcome", "unknown"),
            "reason": payload.get("reason", "unknown"),
            "actor_identity": payload.get("actor_identity", "unknown"),
            "target_identity": payload.get("target_identity", "unknown"),
            "reference_identity": payload.get("reference_identity", "unknown"),
            "occurred_at": event.occurred_at,
        }

    @staticmethod
    def _member_identity_from_uuid(user_id: object) -> str:
        return StableIdentity(StableIdentityKind.MEMBER, str(user_id).replace("-", "")).stable_id

    @staticmethod
    def _role_name(role: str) -> str:
        return "system_administrator" if role == "admin" else "knowledge_user"
