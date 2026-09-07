from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.document import Document, DocumentJob
from app.model.operational_event import OperationalEvent
from app.operations.limits import first_release_limits
from app.settings.service import SystemSettingsDraftService


class OperationsService:
    """Projects operational state without ever reading Private Conversation Records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def read(self) -> dict:
        document_counts = await self._document_counts()
        job_counts = await self._job_counts()
        settings = await SystemSettingsDraftService(self.session).read()
        route_events = await self.session.scalars(select(OperationalEvent).where(
            OperationalEvent.generation_route.is_not(None),
        ).order_by(OperationalEvent.created_at.desc()).limit(20))
        failures, retry_actions = await self._failure_projection()
        self._append_generation_settings_failure(
            failures=failures,
            retry_actions=retry_actions,
            settings=settings,
        )
        return {
            "documents": {
                "total": document_counts["total"],
                "published_sources": document_counts["published_sources"],
                "candidate_builds": document_counts["candidate_builds"],
                "queued_builds": job_counts["queued"],
                "running_builds": job_counts["running"],
            },
            "generation": {
                "route_executions": [
                    {**event.generation_route, "request_id": event.request_id}
                    for event in route_events if event.generation_route
                ],
                "application_state": settings["application_state"],
                "active": settings["active"],
            },
            "failures": failures,
            "retry_actions": retry_actions,
            "limits": first_release_limits(),
        }

    async def _document_counts(self) -> dict[str, int]:
        live_documents = Document.deleted_at.is_(None)
        total = await self.session.scalar(select(func.count()).select_from(Document).where(live_documents))
        published_sources = await self.session.scalar(
            select(func.count()).select_from(Document).where(live_documents, Document.published_generation > 0)
        )
        candidate_builds = await self.session.scalar(
            select(func.count()).select_from(Document).where(live_documents, Document.candidate_generation.is_not(None))
        )
        return {
            "total": int(total or 0),
            "published_sources": int(published_sources or 0),
            "candidate_builds": int(candidate_builds or 0),
        }

    async def _job_counts(self) -> dict[str, int]:
        async def _count(status: str) -> int:
            value = await self.session.scalar(
                select(func.count()).select_from(DocumentJob).where(DocumentJob.status == status)
            )
            return int(value or 0)

        return {"queued": await _count("queued"), "running": await _count("running")}

    async def _failure_projection(self) -> tuple[list[dict], list[dict]]:
        result = await self.session.execute(
            select(DocumentJob)
            .where(DocumentJob.status == "failed")
            .order_by(DocumentJob.updated_at.desc())
            .limit(20)
        )
        failures: list[dict] = []
        retry_actions: list[dict] = []
        for job in result.scalars().all():
            failures.append(
                {
                    "kind": "document_build",
                    "code": "DOCUMENT_BUILD_FAILED",
                    "document_id": job.document_id,
                    "job_id": job.id,
                }
            )
            retry_actions.append(
                {
                    "action": "retry_document_build",
                    "document_id": job.document_id,
                    "method": "POST",
                    "path": f"/api/v1/documents/{job.document_id}/build",
                }
            )
        event_result = await self.session.execute(
            select(OperationalEvent)
            .where(OperationalEvent.normalized_error.is_not(None))
            .order_by(OperationalEvent.created_at.desc())
            .limit(20)
        )
        failures.extend(
            {
                "kind": "generation_provider",
                "code": event.normalized_error,
                "request_id": event.request_id,
            }
            for event in event_result.scalars().all()
        )
        return failures, retry_actions

    @staticmethod
    def _append_generation_settings_failure(*, failures: list[dict], retry_actions: list[dict], settings: dict) -> None:
        if settings.get("application_state") != "failed":
            return
        application = settings.get("application")
        version = application.get("version") if isinstance(application, dict) else None
        failures.append({"kind": "generation_settings", "code": "GENERATION_SETTINGS_APPLY_FAILED"})
        if isinstance(version, int) and version > 0:
            retry_actions.append(
                {
                    "action": "retry_generation_settings_apply",
                    "method": "POST",
                    "path": "/api/v1/settings/apply",
                    "body": {"version": version},
                }
            )
