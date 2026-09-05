from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.canonical import (
    BundleIntakeState,
    CanonicalEventType,
    StableIdentityKind,
    validate_transition,
)
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.reviewed_bundles.models import CandidateBuildJob

_COMPLETING_JOB_STATUSES = {"candidate_ready", "superseded"}


async def bundle_intake_state(session: AsyncSession, bundle_id: str) -> str:
    event = await session.scalar(
        select(CanonicalEventModel)
        .where(
            CanonicalEventModel.aggregate_id == bundle_id,
            CanonicalEventModel.aggregate_kind == StableIdentityKind.BUNDLE.value,
        )
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
        .limit(1)
    )
    return event.to_state if event is not None else BundleIntakeState.RECEIVED.value


async def complete_bundle_when_candidate_work_is_finished(session: AsyncSession, bundle_id: str) -> bool:
    bundle = await session.scalar(
        select(CanonicalRecordModel)
        .where(
            CanonicalRecordModel.stable_id == bundle_id,
            CanonicalRecordModel.identity_kind == StableIdentityKind.BUNDLE.value,
        )
        .with_for_update()
    )
    if bundle is None:
        return False
    state = await bundle_intake_state(session, bundle_id)
    if state != BundleIntakeState.PROCESSING.value:
        return False

    jobs = (
        await session.execute(
            select(CandidateBuildJob.status).where(CandidateBuildJob.bundle_id == bundle_id)
        )
    ).scalars().all()
    if not jobs or any(status not in _COMPLETING_JOB_STATUSES for status in jobs):
        return False

    completion_state = (
        BundleIntakeState.COMPLETED_WITH_REJECTIONS
        if bundle.payload.get("has_rejected_items") is True
        else BundleIntakeState.COMPLETED
    )
    session.add(
        CanonicalEventModel(
            aggregate_id=bundle_id,
            aggregate_kind=StableIdentityKind.BUNDLE.value,
            event_type=CanonicalEventType.STATE_CHANGED.value,
            from_state=BundleIntakeState.PROCESSING.value,
            to_state=validate_transition(
                BundleIntakeState,
                BundleIntakeState.PROCESSING,
                completion_state,
            ).value,
            payload={
                "schema": "reviewed_release_bundle_event/v1",
                "action": (
                    "candidate_work_completed_with_rejections"
                    if completion_state is BundleIntakeState.COMPLETED_WITH_REJECTIONS
                    else "candidate_work_completed"
                ),
            },
        )
    )
    return True
