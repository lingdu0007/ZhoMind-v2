from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.document import Document, DocumentJob
from app.model.operational_event import OperationalEvent
from app.operations.chat_capacity import get_chat_admission_gate
from app.operations.events import OperationalEventService
from app.operations.limits import first_release_limits
from app.retention.policy import read_policy
from app.reviewed_bundles.models import CandidateBuildJob
from app.settings.service import SystemSettingsDraftService


class OperationsService:
    """Projects operational state without ever reading Private Conversation Records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def read_request(self, request_id: UUID) -> dict:
        days = (await read_policy(self.session))["days"]["operational_events"]
        events = list(await self.session.scalars(
            select(OperationalEvent).where(
                OperationalEvent.request_id == str(request_id),
                OperationalEvent.created_at > datetime.now(UTC) - timedelta(days=days),
                OperationalEvent.route_outcome.in_([
                    f"POST /api/v1/chat{suffix}:{status}"
                    for suffix in ("", "/stream") for status in ("success", "client_error", "server_error")
                ]),
            ).limit(2),
        ))
        if len(events) != 1:
            raise AppError(status_code=404, code="MEASUREMENT_UNAVAILABLE", message="unique retained measurement unavailable")
        event = events[0]
        return {
            "request_id": OperationalEventService.request_identity(event.request_id),
            "created_at": event.created_at.replace(tzinfo=UTC).isoformat(),
            "duration_ms": OperationalEventService.nonnegative_integer(event.duration_ms),
            "route_class": OperationalEventService.route_class(event.route_outcome),
            "dimensions": OperationalEventService.dimensions(event.dimensions),
            "generation_route": OperationalEventService.retained_route(event.generation_route),
            "normalized_error": OperationalEventService.error_code(event.normalized_error),
        }

    async def read(self) -> dict:
        document_counts = await self._document_counts()
        job_counts = await self._job_counts()
        settings = await SystemSettingsDraftService(self.session).read()
        route_events = await self.session.scalars(select(OperationalEvent).where(
            OperationalEvent.generation_route.is_not(None),
        ).order_by(OperationalEvent.created_at.desc()).limit(20))
        failures, retry_actions = await self._failure_projection()
        recent_events = list(await self.session.scalars(
            select(OperationalEvent).order_by(OperationalEvent.created_at.desc()).limit(20),
        ))
        self._append_generation_settings_failure(
            failures=failures,
            retry_actions=retry_actions,
            settings=settings,
        )
        return {
            "admission": get_chat_admission_gate().snapshot(),
            "events": [{
                "request_id": OperationalEventService.request_identity(event.request_id),
                "route_class": OperationalEventService.route_class(event.route_outcome),
                "duration_ms": OperationalEventService.nonnegative_integer(event.duration_ms),
                "dimensions": OperationalEventService.dimensions(event.dimensions),
            } for event in recent_events],
            "documents": {
                "total": document_counts["total"],
                "published_sources": document_counts["published_sources"],
                "candidate_builds": document_counts["candidate_builds"],
                "queued_builds": job_counts["queued"],
                "running_builds": job_counts["running"],
            },
            "generation": {
                "route_executions": [
                    {**projection, "request_id": OperationalEventService.request_identity(event.request_id)}
                    for event in route_events
                    if (projection := OperationalEventService.retained_route(event.generation_route)) is not None
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
            total = 0
            for model in (DocumentJob, CandidateBuildJob):
                value = await self.session.scalar(
                    select(func.count()).select_from(model).where(model.status == status)
                )
                total += int(value or 0)
            return total

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
        candidate_jobs = await self.session.scalars(
            select(CandidateBuildJob)
            .where(CandidateBuildJob.status.in_(("failed", "canceled", "interrupted_retryable")))
            .order_by(CandidateBuildJob.updated_at.desc()).limit(20)
        )
        for job in candidate_jobs:
            failures.append({
                "kind": "candidate_build", "code": {
                    "failed": "CANDIDATE_BUILD_FAILED", "canceled": "CANDIDATE_BUILD_CANCELED",
                    "interrupted_retryable": "CANDIDATE_BUILD_INTERRUPTED",
                }[job.status], "job_id": job.id,
            })
            if job.allowed_next_action in {"retry_fixed_inputs", "reconcile_derived_data_then_retry"}:
                retry_actions.append({
                    "action": "retry_candidate_build", "job_id": job.id, "method": "POST",
                    "path": f"/api/v1/reviewed-release-bundles/jobs/{job.id}/retry",
                })
        event_result = await self.session.execute(
            select(OperationalEvent)
            .where(OperationalEvent.normalized_error.is_not(None))
            .order_by(OperationalEvent.created_at.desc())
            .limit(20)
        )
        failures.extend(
            {
                "kind": OperationalEventService.failure_category(event.normalized_error),
                "code": OperationalEventService.error_code(event.normalized_error),
                "request_id": OperationalEventService.request_identity(event.request_id),
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
