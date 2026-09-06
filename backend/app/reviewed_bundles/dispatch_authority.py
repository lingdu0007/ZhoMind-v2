from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.contracts.canonical import (
    BuildJobTerminalStatus,
    CanonicalEventType,
    CanonicalRecordClass,
    StableIdentity,
    StableIdentityKind,
)
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.reviewed_bundles.inputs import FrozenCandidateBuildInput, load_frozen_candidate_build_input
from app.reviewed_bundles.models import CandidateBuildJob

_DISPATCH_ACTIONS = frozenset({"dispatched", "retry_dispatched"})
_RETRY_SOURCE_STATUSES = frozenset(
    {
        BuildJobTerminalStatus.FAILED.value,
        BuildJobTerminalStatus.CANCELED.value,
        BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value,
    }
)


async def has_current_dispatch_authorization(session: AsyncSession, job: CandidateBuildJob) -> bool:
    if job.status != "queued" or job.dispatched_at is None:
        return False
    try:
        frozen_input = await load_frozen_candidate_build_input(session, job.id)
    except AppError:
        return False
    if not frozen_input.matches_job(job):
        return False
    result = await session.execute(
        select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == f"build_generation:{job.id}",
            CanonicalEventModel.aggregate_kind == StableIdentityKind.BUILD_GENERATION.value,
            CanonicalEventModel.recorded_by.is_not(None),
        )
    )
    for event in result.scalars().all():
        if not _matches_current_dispatch_event(event, job, frozen_input):
            continue
        if await _is_current_administrator(session, event.recorded_by):
            return True
    return False


def _matches_current_dispatch_event(
    event: CanonicalEventModel,
    job: CandidateBuildJob,
    frozen_input: FrozenCandidateBuildInput,
) -> bool:
    if event.aggregate_id != f"build_generation:{job.id}":
        return False
    if event.aggregate_kind != StableIdentityKind.BUILD_GENERATION.value:
        return False
    payload = event.payload
    if not isinstance(payload, dict):
        return False
    action = payload.get("action")
    if action not in _DISPATCH_ACTIONS:
        return False
    if (
        payload.get("schema") != "candidate_build_job_event/v1"
        or payload.get("stage") != "queued"
        or payload.get("status") != "queued"
        or payload.get("progress") != 0
        or payload.get("attempt") != job.attempt
        or payload.get("editorial_source_revision") != frozen_input.editorial_source_revision
        or payload.get("input_sha256") != frozen_input.input_sha256
        or not _matches_frozen_input_hash(payload, frozen_input)
        or payload.get("failure_reason") is not None
        or payload.get("allowed_next_action") != "cancel_or_await_candidate_build"
        or event.to_state != "queued"
    ):
        return False
    if action == "dispatched":
        return (
            event.event_type == CanonicalEventType.STATUS_CHANGED.value
            and event.from_state == "queued"
        )
    return (
        event.event_type == CanonicalEventType.STATE_CHANGED.value
        and event.from_state in _RETRY_SOURCE_STATUSES
    )


def _matches_frozen_input_hash(payload: dict, frozen_input: FrozenCandidateBuildInput) -> bool:
    if payload.get("frozen_input_sha256") == frozen_input.frozen_input_sha256:
        return True
    return frozen_input.is_legacy_hash_backfill and "frozen_input_sha256" not in payload


async def _is_current_administrator(session: AsyncSession, recorded_by: str | None) -> bool:
    if not isinstance(recorded_by, str):
        return False
    try:
        identity = StableIdentity.from_stable_id(recorded_by)
        user_id = uuid.UUID(identity.value)
    except (ValueError, AttributeError):
        return False
    if identity.kind is not StableIdentityKind.MEMBER:
        return False
    member_record = await session.get(CanonicalRecordModel, identity.stable_id)
    if (
        member_record is None
        or member_record.identity_kind != StableIdentityKind.MEMBER.value
        or member_record.identity_value != identity.value
        or member_record.state != "active"
        or member_record.record_class != CanonicalRecordClass.AUTHORITATIVE.value
        or not isinstance(member_record.payload, dict)
        or member_record.payload.get("schema") != "identity_record/v1"
    ):
        return False
    administrator = await session.scalar(
        select(User).where(
            User.id == user_id,
            User.role == "admin",
            User.is_active.is_(True),
        )
    )
    return administrator is not None
