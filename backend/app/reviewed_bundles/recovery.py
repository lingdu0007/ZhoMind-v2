from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.contracts.canonical import BuildJobTerminalStatus, CanonicalEventType, StableIdentityKind
from app.model.canonical import CanonicalEventModel
from app.reviewed_bundles.dispatch_authority import has_current_dispatch_authorization
from app.reviewed_bundles.events import candidate_job_event_payload
from app.reviewed_bundles.inputs import FrozenCandidateBuildInput, load_frozen_candidate_build_input
from app.reviewed_bundles.models import CandidateBuildChunk, CandidateBuildJob


class DenseRecoveryIndexer(Protocol):
    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None: ...


class CandidateBuildRecoveryService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        dense_index_service: DenseRecoveryIndexer,
    ) -> None:
        self.session = session
        self._dense_index_service = dense_index_service

    async def recover(
        self,
        *,
        enqueue: Callable[[str], Awaitable[None]],
        now: datetime | None = None,
        recovery_owner: str | None = None,
    ) -> dict[str, list[str]]:
        current_time = now or datetime.now(UTC)
        queued_result = await self.session.execute(
            select(CandidateBuildJob)
            .where(
                CandidateBuildJob.status == "queued",
                CandidateBuildJob.dispatched_at.is_not(None),
            )
            .order_by(CandidateBuildJob.created_at.asc(), CandidateBuildJob.id.asc())
        )
        queued_job_ids: list[str] = []
        invalid_queued_job_ids: list[str] = []
        for job in queued_result.scalars().all():
            if not await self._is_current_dispatched_queued_job(job):
                continue
            try:
                await self._load_matching_input(job)
            except AppError:
                if not await self._is_current_dispatched_queued_job(job):
                    continue
                invalid_queued_job_ids.append(job.id)
                self._mark_input_integrity_failure(job, now=current_time, action="queued_input_integrity_failed")
            else:
                if await self._is_current_dispatched_queued_job(job):
                    queued_job_ids.append(job.id)

        abandoned_lease_conditions = [
            CandidateBuildJob.lease_expires_at.is_(None),
            CandidateBuildJob.lease_expires_at <= current_time,
        ]
        if recovery_owner is not None:
            abandoned_lease_conditions.extend(
                [
                    CandidateBuildJob.lease_owner.is_(None),
                    CandidateBuildJob.lease_owner != recovery_owner,
                ]
            )
        running_result = await self.session.execute(
            select(CandidateBuildJob)
            .where(
                CandidateBuildJob.status == "running",
                or_(*abandoned_lease_conditions),
            )
            .order_by(CandidateBuildJob.created_at.asc(), CandidateBuildJob.id.asc())
        )
        interrupted_job_ids: list[str] = []
        invalid_running_job_ids: list[str] = []
        for job in running_result.scalars().all():
            if not await self._is_current_abandoned_running_job(
                job,
                now=current_time,
                recovery_owner=recovery_owner,
            ):
                continue
            try:
                frozen_input = await self._load_matching_input(job)
            except AppError:
                if not await self._is_current_abandoned_running_job(
                    job,
                    now=current_time,
                    recovery_owner=recovery_owner,
                ):
                    continue
                invalid_running_job_ids.append(job.id)
                self._mark_input_integrity_failure(job, now=current_time, action="running_input_integrity_failed")
            else:
                if await self._is_current_abandoned_running_job(
                    job,
                    now=current_time,
                    recovery_owner=recovery_owner,
                ):
                    await self._interrupt_expired_lease(job, frozen_input=frozen_input, now=current_time)
                    interrupted_job_ids.append(job.id)

        # Persist the stale-worker fence before deleting any derived asset.
        # A prior worker can no longer report a terminal Candidate outcome once
        # this transaction commits, even if an external index call finishes late.
        await self.session.commit()

        excluded_from_automatic_cleanup = set(invalid_queued_job_ids + invalid_running_job_ids)
        cleanup_result = await self.session.execute(
            select(CandidateBuildJob)
            .where(
                CandidateBuildJob.derived_cleanup_pending.is_(True),
                CandidateBuildJob.status.in_(
                    [
                        BuildJobTerminalStatus.FAILED.value,
                        BuildJobTerminalStatus.CANCELED.value,
                        BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value,
                        BuildJobTerminalStatus.SUPERSEDED.value,
                    ]
                ),
            )
            .order_by(CandidateBuildJob.completed_at.asc(), CandidateBuildJob.id.asc())
        )
        reconciled_cleanup_job_ids: list[str] = []
        cleanup_pending_job_ids: list[str] = []
        for job in cleanup_result.scalars().all():
            if job.id in excluded_from_automatic_cleanup:
                continue
            if not await self._is_current_cleanup_pending_job(job):
                continue
            if await self._reconcile_pending_cleanup(job):
                reconciled_cleanup_job_ids.append(job.id)
            else:
                cleanup_pending_job_ids.append(job.id)

        await self.session.commit()
        requeued_job_ids: list[str] = []
        enqueue_failed_job_ids: list[str] = []
        for job_id in queued_job_ids:
            job = await self._current_requeueable_job(job_id)
            if job is None:
                continue
            try:
                await self._load_matching_input(job)
            except AppError:
                self._mark_input_integrity_failure(
                    job,
                    now=current_time,
                    action="requeue_input_integrity_failed",
                )
                await self.session.commit()
                invalid_queued_job_ids.append(job_id)
                continue
            try:
                await enqueue(job_id)
            except Exception:
                job = await self._current_requeueable_job(job_id)
                if job is not None:
                    self._mark_enqueue_failed(job, now=current_time)
                    await self.session.commit()
                    enqueue_failed_job_ids.append(job_id)
            else:
                job = await self._current_job(job_id)
                if job is not None and job.status in {"queued", "running"}:
                    self._append_event(
                        job,
                        event_type=CanonicalEventType.STATUS_CHANGED,
                        from_state="queued",
                        to_state=job.status,
                        action="requeued_on_startup",
                    )
                    await self.session.commit()
                    requeued_job_ids.append(job_id)
        return {
            "requeued_job_ids": requeued_job_ids,
            "interrupted_job_ids": interrupted_job_ids,
            "invalid_queued_job_ids": invalid_queued_job_ids,
            "invalid_running_job_ids": invalid_running_job_ids,
            "reconciled_cleanup_job_ids": reconciled_cleanup_job_ids,
            "cleanup_pending_job_ids": cleanup_pending_job_ids,
            "enqueue_failed_job_ids": enqueue_failed_job_ids,
        }

    async def _load_matching_input(self, job: CandidateBuildJob) -> FrozenCandidateBuildInput:
        await self.session.refresh(job, with_for_update=True)
        frozen_input = await load_frozen_candidate_build_input(self.session, job.id)
        if not frozen_input.matches_job(job):
            raise AppError(
                status_code=409,
                code="CANDIDATE_INPUT_INTEGRITY_FAILED",
                message="candidate build immutable inputs no longer verify",
                detail={"job_id": job.id},
            )
        return frozen_input

    async def _current_job(self, job_id: str) -> CandidateBuildJob | None:
        job = await self.session.get(CandidateBuildJob, job_id)
        if job is None:
            return None
        await self.session.refresh(job, with_for_update=True)
        return job

    async def _current_requeueable_job(self, job_id: str) -> CandidateBuildJob | None:
        job = await self._current_job(job_id)
        if job is None or not await has_current_dispatch_authorization(self.session, job):
            return None
        return job

    async def _is_current_dispatched_queued_job(self, job: CandidateBuildJob) -> bool:
        await self.session.refresh(job, with_for_update=True)
        return await has_current_dispatch_authorization(self.session, job)

    async def _is_current_abandoned_running_job(
        self,
        job: CandidateBuildJob,
        *,
        now: datetime,
        recovery_owner: str | None,
    ) -> bool:
        await self.session.refresh(job, with_for_update=True)
        if job.status != "running":
            return False
        lease_expired = job.lease_expires_at is None or self._lease_is_expired(job.lease_expires_at, now=now)
        owned_by_other_runtime = recovery_owner is not None and (
            job.lease_owner is None or job.lease_owner != recovery_owner
        )
        return lease_expired or owned_by_other_runtime

    @staticmethod
    def _lease_is_expired(lease_expires_at: datetime, *, now: datetime) -> bool:
        normalized_expiry = (
            lease_expires_at.replace(tzinfo=UTC)
            if lease_expires_at.tzinfo is None
            else lease_expires_at.astimezone(UTC)
        )
        normalized_now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
        return normalized_expiry <= normalized_now

    async def _is_current_cleanup_pending_job(self, job: CandidateBuildJob) -> bool:
        await self.session.refresh(job, with_for_update=True)
        return job.derived_cleanup_pending and job.status in {
            BuildJobTerminalStatus.FAILED.value,
            BuildJobTerminalStatus.CANCELED.value,
            BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value,
            BuildJobTerminalStatus.SUPERSEDED.value,
        }

    async def _interrupt_expired_lease(
        self,
        job: CandidateBuildJob,
        *,
        frozen_input: FrozenCandidateBuildInput,
        now: datetime,
    ) -> None:
        previous_stage = job.stage
        job.status = BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value
        job.terminal_state = BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value
        job.progress = min(job.progress, 99)
        job.failure_reason = {
            "code": "CANDIDATE_WORKER_LEASE_INTERRUPTED",
            "stage": previous_stage,
            "message": "startup found an expired Candidate Build worker lease",
        }
        job.completed_at = now
        job.heartbeat_at = now
        job.lease_expires_at = None
        job.lease_owner = None
        job.derived_cleanup_pending = True
        job.allowed_next_action = "reconcile_derived_data_then_retry"
        self._append_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=previous_stage,
            to_state=BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value,
            action="lease_interrupted",
        )

    def _mark_input_integrity_failure(self, job: CandidateBuildJob, *, now: datetime, action: str) -> None:
        previous_stage = job.stage
        job.status = BuildJobTerminalStatus.FAILED.value
        job.progress = min(job.progress, 99)
        job.terminal_state = BuildJobTerminalStatus.FAILED.value
        job.failure_reason = {
            "code": "CANDIDATE_INPUT_INTEGRITY_FAILED",
            "stage": previous_stage,
            "message": "Candidate Build immutable inputs no longer verify",
        }
        job.derived_cleanup_pending = True
        job.completed_at = now
        job.heartbeat_at = now
        job.lease_expires_at = None
        job.lease_owner = None
        job.allowed_next_action = "import_new_bundle"
        self._append_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=previous_stage,
            to_state=BuildJobTerminalStatus.FAILED.value,
            action=action,
        )

    async def _cleanup_derived_assets(
        self,
        job: CandidateBuildJob,
        *,
        expected: FrozenCandidateBuildInput,
    ) -> str | None:
        try:
            frozen_input = await self._load_matching_input(job)
        except AppError:
            job.derived_cleanup_pending = True
            return "immutable Candidate inputs require manual reconciliation"
        if frozen_input != expected:
            job.derived_cleanup_pending = True
            return "immutable Candidate inputs require manual reconciliation"

        await self.session.execute(delete(CandidateBuildChunk).where(CandidateBuildChunk.job_id == job.id))
        try:
            await self._dense_index_service.delete_candidate_generation(
                document_id=frozen_input.document_identity,
                generation=frozen_input.requested_generation,
                embedding_fingerprint=frozen_input.embedding_fingerprint,
            )
        except Exception:
            job.derived_cleanup_pending = True
            return "derived vector cleanup requires reconciliation before retry"
        job.derived_cleanup_pending = False
        return None

    async def _reconcile_pending_cleanup(self, job: CandidateBuildJob) -> bool:
        try:
            frozen_input = await self._load_matching_input(job)
        except AppError:
            self._mark_cleanup_pending(
                job,
                message="immutable Candidate inputs require manual reconciliation",
            )
            return False

        cleanup_note = await self._cleanup_derived_assets(job, expected=frozen_input)
        if cleanup_note is not None:
            self._mark_cleanup_pending(job, message=cleanup_note)
            return False

        failure_code = job.failure_reason.get("code") if isinstance(job.failure_reason, dict) else None
        if job.status == BuildJobTerminalStatus.SUPERSEDED.value:
            job.allowed_next_action = "none"
        elif failure_code == "CANDIDATE_INPUT_INTEGRITY_FAILED":
            job.allowed_next_action = "import_new_bundle"
        else:
            job.allowed_next_action = "retry_fixed_inputs"
        self._append_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=job.status,
            to_state=job.status,
            action="derived_cleanup_reconciled",
        )
        return True

    def _mark_cleanup_pending(self, job: CandidateBuildJob, *, message: str) -> None:
        failure_reason = dict(job.failure_reason) if isinstance(job.failure_reason, dict) else {}
        failure_reason.setdefault("code", "CANDIDATE_DERIVED_CLEANUP_REQUIRED")
        failure_reason.setdefault("stage", job.stage)
        failure_reason.setdefault("message", "Candidate Build derived data requires reconciliation")
        failure_reason["cleanup"] = message
        job.failure_reason = failure_reason
        job.derived_cleanup_pending = True
        job.allowed_next_action = (
            "reconcile_derived_data"
            if job.status == BuildJobTerminalStatus.SUPERSEDED.value
            else "reconcile_derived_data_then_retry"
        )
        self._append_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=job.status,
            to_state=job.status,
            action="derived_cleanup_pending",
        )

    def _mark_enqueue_failed(self, job: CandidateBuildJob, *, now: datetime) -> None:
        previous_stage = job.stage
        job.status = BuildJobTerminalStatus.FAILED.value
        job.progress = min(job.progress, 99)
        job.terminal_state = BuildJobTerminalStatus.FAILED.value
        job.failure_reason = {
            "code": "CANDIDATE_ENQUEUE_FAILED",
            "stage": previous_stage,
            "message": "Candidate Build could not be returned to the queue",
        }
        job.completed_at = now
        job.heartbeat_at = now
        job.lease_expires_at = None
        job.lease_owner = None
        job.allowed_next_action = "retry_fixed_inputs"
        self._append_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=previous_stage,
            to_state=BuildJobTerminalStatus.FAILED.value,
            action="enqueue_failed",
        )

    def _append_event(
        self,
        job: CandidateBuildJob,
        *,
        event_type: CanonicalEventType,
        from_state: str | None,
        to_state: str,
        action: str,
    ) -> None:
        self.session.add(
            CanonicalEventModel(
                aggregate_id=f"build_generation:{job.id}",
                aggregate_kind=StableIdentityKind.BUILD_GENERATION.value,
                event_type=event_type.value,
                from_state=from_state,
                to_state=to_state,
                payload=candidate_job_event_payload(job, action=action),
            )
        )
