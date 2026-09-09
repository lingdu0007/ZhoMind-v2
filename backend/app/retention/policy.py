from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.service.identity_audit_service import IdentityAuditService

DataClass = Literal["conversations", "operational_events", "feedback_signals"]
REGISTRY_ID = "configuration:pilot-retention"


class RetentionDays(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    conversations: int = Field(default=30, ge=1, le=3650)
    operational_events: int = Field(default=30, ge=1, le=3650)
    feedback_signals: int = Field(default=180, ge=1, le=3650)


class PolicyChange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_identity: str = Field(pattern=r"^configuration:[0-9a-f]{64}$")
    days: RetentionDays


def policy_projection(days: RetentionDays, version: int = 1) -> dict:
    value = {
        "schema": "retention_policy/v1",
        "version": version,
        "days": days.model_dump(),
        "content_telemetry": False,
        "administrator_transcript_access": False,
    }
    return {"identity": f"configuration:{canonical_json_sha256(value)}", **value}


async def read_policy(session: AsyncSession) -> dict:
    current = policy_projection(RetentionDays())
    events = (await session.scalars(select(CanonicalEventModel).where(
        CanonicalEventModel.aggregate_id == REGISTRY_ID,
        CanonicalEventModel.event_type == "retention_policy_changed",
    ).order_by(CanonicalEventModel.occurred_at, CanonicalEventModel.id))).all()
    for event in events:
        payload = event.payload
        try:
            proposed = policy_projection(RetentionDays.model_validate(payload["days"]), current["version"] + 1)
            record = await session.get(CanonicalRecordModel, proposed["identity"])
            if (
                payload != {"previous_identity": current["identity"], **proposed}
                or event.aggregate_kind != "configuration"
                or event.from_state != str(current["version"])
                or event.to_state != str(proposed["version"])
                or not event.recorded_by
                or not event.recorded_by.startswith("member:")
                or record is None or record.payload != proposed
                or record.record_class != "immutable"
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(status_code=409, code="RETENTION_POLICY_INVALID", message="retention policy cannot be verified") from exc
        current = proposed
    return current


async def lock_registry(session: AsyncSession) -> None:
    from app.repository.chat_repository import ChatRepository

    await ChatRepository(session).acquire_private_conversation_write_fence()
    record = await session.get(CanonicalRecordModel, REGISTRY_ID)
    if record is None:
        try:
            async with session.begin_nested():
                session.add(CanonicalRecordModel(
                    stable_id=REGISTRY_ID, identity_kind="configuration",
                    identity_value="pilot-retention", state="registered", record_class="immutable",
                    payload={"schema": "retention_registry/v1"},
                ))
                await session.flush()
        except IntegrityError:
            pass
    await session.scalar(select(CanonicalRecordModel).where(
        CanonicalRecordModel.stable_id == REGISTRY_ID,
    ).with_for_update().execution_options(populate_existing=True))


async def change_policy(session: AsyncSession, change: PolicyChange, administrator: User) -> dict:
    from app.delivery_acceptance.service import DeliveryAcceptanceService

    await lock_registry(session)
    current = await read_policy(session)
    if current["identity"] != change.expected_identity:
        raise AppError(status_code=409, code="RETENTION_POLICY_STALE", message="retention policy changed; reload before retry")
    if current["days"] == change.days.model_dump():
        await session.commit()
        return {"policy": current, "reacceptance_required": False}
    proposed = policy_projection(change.days, current["version"] + 1)
    actor = await IdentityAuditService(session).ensure_member_record(administrator, admission_path="retention_administrator")
    session.add(CanonicalRecordModel(
        stable_id=proposed["identity"], identity_kind="configuration",
        identity_value=proposed["identity"].partition(":")[2], state="configured",
        record_class="immutable", payload=proposed,
    ))
    session.add(CanonicalEventModel(
        aggregate_id=REGISTRY_ID, aggregate_kind="configuration", event_type="retention_policy_changed",
        from_state=str(current["version"]), to_state=str(proposed["version"]),
        recorded_by=actor, payload={"previous_identity": current["identity"], **proposed},
    ))
    await DeliveryAcceptanceService(session).suspend_retention_policy(current["identity"], actor)
    await session.commit()
    return {"policy": proposed, "reacceptance_required": True}
