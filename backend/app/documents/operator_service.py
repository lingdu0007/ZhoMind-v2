from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.document import DocumentJob

DRAIN_MODE_KEY = "documents:drain_mode"
DRAIN_STARTED_AT_KEY = "documents:drain_started_at"


class DocumentsOperatorService:
    def __init__(self, *, redis) -> None:
        self.redis = redis

    async def enable_drain(self, *, now: datetime | None = None) -> None:
        current_started_at = await self.redis.get(DRAIN_STARTED_AT_KEY)
        timestamp = (now or datetime.now(timezone.utc)).isoformat()
        await self.redis.set(DRAIN_MODE_KEY, "1")
        if current_started_at is None:
            await self.redis.set(DRAIN_STARTED_AT_KEY, timestamp)

    async def clear_drain(self) -> None:
        await self.redis.delete(DRAIN_MODE_KEY, DRAIN_STARTED_AT_KEY)

    async def read_drain_state(self) -> tuple[bool, str | None]:
        mode = await self.redis.get(DRAIN_MODE_KEY)
        started_at = await self.redis.get(DRAIN_STARTED_AT_KEY)
        return mode == "1", started_at

    async def ensure_drain_enabled(self) -> None:
        drain_enabled, _ = await self.read_drain_state()
        if not drain_enabled:
            raise AppError(
                status_code=409,
                code="DOC_MIGRATION_DRAIN_INACTIVE",
                message="migration drain is not active",
            )

    async def collect_status(self, *, session: AsyncSession, active_dispatcher_tasks: int) -> dict:
        drain_enabled, drain_started_at = await self.read_drain_state()
        queued_jobs = int(
            await session.scalar(
                select(func.count()).select_from(DocumentJob).where(DocumentJob.status == "queued")
            )
            or 0
        )
        running_jobs = int(
            await session.scalar(
                select(func.count()).select_from(DocumentJob).where(DocumentJob.status == "running")
            )
            or 0
        )
        return {
            "drain_enabled": drain_enabled,
            "drain_started_at": drain_started_at,
            "queued_jobs": queued_jobs,
            "running_jobs": running_jobs,
            "active_dispatcher_tasks": active_dispatcher_tasks,
            "ready_for_migration": drain_enabled and running_jobs == 0 and active_dispatcher_tasks == 0,
        }
