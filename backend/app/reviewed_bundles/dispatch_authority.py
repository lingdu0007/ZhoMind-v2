from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.canonical import StableIdentityKind
from app.model.canonical import CanonicalEventModel
from app.reviewed_bundles.models import CandidateBuildJob

_DISPATCH_ACTIONS = frozenset({"dispatched", "retry_dispatched"})


async def has_current_dispatch_authorization(session: AsyncSession, job: CandidateBuildJob) -> bool:
    if job.status != "queued" or job.dispatched_at is None:
        return False
    result = await session.execute(
        select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == f"build_generation:{job.id}",
            CanonicalEventModel.aggregate_kind == StableIdentityKind.BUILD_GENERATION.value,
            CanonicalEventModel.recorded_by.is_not(None),
        )
    )
    return any(
        isinstance(event.recorded_by, str)
        and event.recorded_by.startswith("member:")
        and event.payload.get("action") in _DISPATCH_ACTIONS
        and event.payload.get("attempt") == job.attempt
        for event in result.scalars().all()
    )
