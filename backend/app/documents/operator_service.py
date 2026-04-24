from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.extensions.registry import get_task_backend
from app.model.document import Document, DocumentJob

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

    async def reconcile_queued_jobs(
        self,
        *,
        session: AsyncSession,
        cancel_dispatcher_job,
    ) -> list[str]:
        await self.ensure_drain_enabled()
        result = await session.execute(
            select(DocumentJob).where(DocumentJob.status == "queued").order_by(DocumentJob.updated_at.asc())
        )
        reconciled_job_ids: list[str] = []
        for job in result.scalars().all():
            document = await session.get(Document, job.document_id)
            job.status = "canceled"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = "canceled during migration drain"
            if document is not None:
                self._release_claim_if_owned(document=document, job=job)
            if (
                document is not None
                and job.build_generation is not None
                and document.latest_requested_generation == job.build_generation
            ):
                if document.deleted_at is None:
                    document.latest_requested_generation = document.published_generation
                    document.status = "ready" if document.published_generation > 0 else "failed"
            with suppress(Exception):
                await get_task_backend("inmemory").cancel(job.id)
            with suppress(Exception):
                await cancel_dispatcher_job(job.id)
            reconciled_job_ids.append(job.id)
        await session.commit()
        return reconciled_job_ids

    @staticmethod
    def _release_claim_if_owned(*, document: Document, job: DocumentJob) -> None:
        if job.build_generation is None:
            return
        if document.active_build_generation == job.build_generation and document.active_build_job_id == job.id:
            document.active_build_generation = None
            document.active_build_job_id = None
            document.active_build_heartbeat_at = None
