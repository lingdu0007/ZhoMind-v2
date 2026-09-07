from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, TypedDict

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.common.config import Settings, get_settings
from app.common.exceptions import AppError
from app.contracts.canonical import (
    BuildJobStage,
    BuildJobTerminalStatus,
    CanonicalEventType,
    CanonicalRecordClass,
    StableIdentity,
    StableIdentityKind,
    validate_transition,
)
from app.documents.dense_index_service import DenseIndexService
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.document import DocumentChunk
from app.reviewed_bundles.dispatch_authority import has_current_dispatch_authorization
from app.reviewed_bundles.events import candidate_job_event_payload
from app.reviewed_bundles.inputs import FrozenCandidateBuildInput, load_frozen_candidate_build_input
from app.reviewed_bundles.lifecycle import complete_bundle_when_candidate_work_is_finished
from app.reviewed_bundles.models import CandidateBuildChunk, CandidateBuildJob
from app.reviewed_bundles.service import EditorialExportVerifier, candidate_embedding_configuration

_LEASE_DURATION = timedelta(seconds=30)
_LEASE_HEARTBEAT_INTERVAL_SECONDS = _LEASE_DURATION.total_seconds() / 3
_TERMINAL_STATUSES = {
    BuildJobTerminalStatus.CANDIDATE_READY.value,
    BuildJobTerminalStatus.FAILED.value,
    BuildJobTerminalStatus.CANCELED.value,
    BuildJobTerminalStatus.INTERRUPTED_RETRYABLE.value,
    BuildJobTerminalStatus.SUPERSEDED.value,
}


class CandidateChunkSpec(TypedDict):
    chunk_index: int
    content: str
    content_sha256: str
    metadata: dict[str, object]


class ParsedCandidateArtifact(TypedDict):
    entry: dict[str, object]
    body: dict[str, object]
    source_relationships_by_section: dict[str, list[dict[str, object]]]
    assurance_level: object
    applicability_conditions: list[object]
    non_applicability_conditions: list[object]
    freshness_triggers: list[object]
    editorial_revision_identity: object


class DenseCandidateIndexer(Protocol):
    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks: list[DocumentChunk],
        embedding_fingerprint: str | None = None,
    ) -> Any: ...

    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None: ...


class CandidateBuildService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        dense_index_service: DenseCandidateIndexer | None = None,
        editorial_export_verifier: EditorialExportVerifier,
        settings: Settings | None = None,
        lease_owner: str | None = None,
    ) -> None:
        self.session = session
        self._settings = settings or get_settings()
        self._dense_index_service = dense_index_service or DenseIndexService(settings=self._settings)
        self._editorial_export_verifier = editorial_export_verifier
        self._lease_owner = lease_owner or f"candidate-build-worker:{uuid.uuid4().hex}"
        self._owned_attempts: dict[str, int] = {}

    async def process_job(self, job_id: str) -> dict[str, object]:
        job = await self._get_job(job_id)
        if job.status == "candidate_ready":
            return await self._candidate_projection(job)
        if job.status in _TERMINAL_STATUSES:
            return self._job_projection(job)
        if job.status != "queued":
            raise AppError(
                status_code=409,
                code="CANDIDATE_BUILD_NOT_QUEUED",
                message="candidate build job is not queued",
                detail={"job_id": job.id, "status": job.status},
            )

        try:
            if not await self._start(job):
                return self._job_projection(job)
            frozen_input = await self._load_frozen_input(job)
            artifact = frozen_input.artifact
            await self._revalidate_editorial_authority(job, frozen_input)
            self._assert_embedding_configuration(frozen_input)
            parsed_artifact = self._parse_artifact(artifact)
            if not await self._advance(
                job,
                frozen_input=frozen_input,
                stage=BuildJobStage.CHUNKING,
                progress=45,
                action="chunking",
            ):
                return self._job_projection(job)
            chunk_specs = self._chunk_specs(parsed_artifact, frozen_input)
            if not chunk_specs:
                raise AppError(
                    status_code=422,
                    code="CANDIDATE_CHUNKING_EMPTY",
                    message="accepted editorial export did not produce any candidate chunks",
                    detail={"job_id": job.id},
                )

            await self._revalidate_editorial_authority(job, frozen_input)
            if not await self._advance(
                job,
                frozen_input=frozen_input,
                stage=BuildJobStage.INDEXING,
                progress=75,
                action="indexing",
            ):
                return self._job_projection(job)
            candidate_identity = self._candidate_identity(job)
            if not await self._write_chunks(
                job=job,
                frozen_input=frozen_input,
                candidate_id=candidate_identity.stable_id,
                chunk_specs=chunk_specs,
            ):
                return self._job_projection(job)
            dense_chunks = [
                DocumentChunk(
                    document_id=frozen_input.document_identity,
                    generation=frozen_input.requested_generation,
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    content_sha256=chunk["content_sha256"],
                    keywords=[],
                    generated_questions=[],
                    chunk_metadata=chunk["metadata"],
                )
                for chunk in chunk_specs
            ]
            await self._current_frozen_input(job, expected=frozen_input)
            if not await self._owns_running_job(job):
                return self._job_projection(job)
            await self.session.commit()
            dense_result = await self._index_candidate_generation_with_lease_heartbeat(
                job=job,
                document_id=frozen_input.document_identity,
                generation=frozen_input.requested_generation,
                chunks=dense_chunks,
                embedding_fingerprint=frozen_input.embedding_fingerprint,
            )
            await self._revalidate_editorial_authority(job, frozen_input)
            if not await self._complete(
                job=job,
                frozen_input=frozen_input,
                candidate_identity=candidate_identity,
                chunk_specs=chunk_specs,
                dense_result=dense_result,
            ):
                return self._job_projection(job)
            return await self._candidate_projection(job)
        except asyncio.CancelledError:
            if await self._owns_running_job(job):
                await self._cancel(job, message="Candidate Build worker was canceled")
            raise
        except AppError as exc:
            if not await self._owns_running_job(job):
                return self._job_projection(job)
            await self._fail(job, code=exc.code or "CANDIDATE_BUILD_FAILED", message=exc.message or "candidate build failed")
            return self._job_projection(job)
        except Exception:
            if not await self._owns_running_job(job):
                return self._job_projection(job)
            await self._fail(
                job,
                code="CANDIDATE_BUILD_UNEXPECTED_FAILURE",
                message="candidate build failed unexpectedly",
            )
            return self._job_projection(job)

    async def cancel_job(self, job_id: str) -> dict[str, object]:
        job = await self._get_job(job_id)
        await self.session.refresh(job, with_for_update=True)
        if job.status == "canceled":
            return self._job_projection(job)
        if job.status in _TERMINAL_STATUSES:
            raise AppError(
                status_code=409,
                code="CANDIDATE_CANCEL_NOT_ALLOWED",
                message="Candidate Build is already terminal and cannot be canceled",
                detail={"job_id": job.id, "status": job.status},
            )
        if job.status not in {"queued", "running"}:
            raise AppError(
                status_code=409,
                code="CANDIDATE_CANCEL_NOT_ALLOWED",
                message="Candidate Build cannot be canceled in its current state",
                detail={"job_id": job.id, "status": job.status},
            )
        await self._cancel(job, message="Candidate Build was canceled by an administrator")
        return self._job_projection(job)

    async def mark_enqueue_failed(self, job_id: str) -> dict[str, object]:
        job = await self._get_job(job_id)
        if job.status == "queued":
            await self._fail(
                job,
                code="CANDIDATE_ENQUEUE_FAILED",
                message="Candidate Build could not be returned to the queue",
            )
        return self._job_projection(job)

    async def mark_cancellation_request_failed(self, job_id: str) -> dict[str, object]:
        job = await self._get_job(job_id)
        await self.session.refresh(job, with_for_update=True)
        if job.status in _TERMINAL_STATUSES:
            return self._job_projection(job)

        from_stage = job.stage
        now = datetime.now(UTC)
        job.status = BuildJobTerminalStatus.FAILED.value
        job.progress = min(job.progress, 99)
        job.terminal_state = BuildJobTerminalStatus.FAILED.value
        job.failure_reason = {
            "code": "CANDIDATE_CANCELLATION_REQUEST_FAILED",
            "stage": from_stage,
            "message": "Candidate Build cancellation request could not reach the worker",
        }
        job.derived_cleanup_pending = True
        job.completed_at = now
        job.heartbeat_at = now
        job.lease_expires_at = None
        job.lease_owner = None
        job.allowed_next_action = "reconcile_derived_data_then_retry"
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=from_stage,
            to_state=BuildJobTerminalStatus.FAILED.value,
            action="cancellation_request_failed",
        )
        await self.session.commit()
        self._owned_attempts.pop(job.id, None)
        return self._job_projection(job)

    async def dispatch_job(self, job_id: str, *, actor_identity: str) -> bool:
        job = await self._get_job(job_id)
        await self.session.refresh(job, with_for_update=True)
        if job.status != "queued":
            raise AppError(
                status_code=409,
                code="CANDIDATE_DISPATCH_NOT_ALLOWED",
                message="only a queued Candidate Build can be dispatched",
                detail={"job_id": job.id, "status": job.status},
            )
        if await has_current_dispatch_authorization(self.session, job):
            return False

        await self._load_frozen_input(job)
        job.dispatched_at = datetime.now(UTC)
        job.allowed_next_action = "cancel_or_await_candidate_build"
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=BuildJobStage.QUEUED.value,
            to_state=BuildJobStage.QUEUED.value,
            action="dispatched",
            recorded_by=actor_identity,
        )
        await self.session.commit()
        return True

    async def retry_job(self, job_id: str, *, actor_identity: str) -> dict[str, object]:
        job = await self._get_job(job_id)
        await self.session.refresh(job, with_for_update=True)
        if job.status not in {"failed", "canceled", "interrupted_retryable"}:
            raise AppError(
                status_code=409,
                code="CANDIDATE_RETRY_NOT_ALLOWED",
                message="only a failed, canceled, or interrupted Candidate Build can be retried",
                detail={"job_id": job.id, "status": job.status},
            )
        try:
            frozen_input = await self._load_frozen_input(job)
        except AppError as exc:
            job.allowed_next_action = "import_new_bundle"
            self._record_retry_blocker(
                job,
                code="CANDIDATE_RETRY_REQUIRES_NEW_BUNDLE",
                message="immutable retry inputs no longer verify; import a new bundle",
            )
            self._append_job_event(
                job,
                event_type=CanonicalEventType.STATUS_CHANGED,
                from_state=job.status,
                to_state=job.status,
                action="retry_requires_new_bundle",
                recorded_by=actor_identity,
            )
            await self.session.commit()
            raise AppError(
                status_code=409,
                code="CANDIDATE_RETRY_REQUIRES_NEW_BUNDLE",
                message="immutable retry inputs no longer verify; import a new bundle",
                detail={"job_id": job.id, "reason": exc.code},
            ) from exc

        cleanup_message = await self._cleanup_derived_assets(job, frozen_input=frozen_input)
        if cleanup_message is not None:
            job.allowed_next_action = "reconcile_derived_data_then_retry"
            self._record_retry_blocker(
                job,
                code="CANDIDATE_DERIVED_CLEANUP_REQUIRED",
                message="derived Candidate assets must be reconciled before retry",
                cleanup=cleanup_message,
            )
            self._append_job_event(
                job,
                event_type=CanonicalEventType.STATUS_CHANGED,
                from_state=job.status,
                to_state=job.status,
                action="retry_cleanup_required",
                recorded_by=actor_identity,
            )
            await self.session.commit()
            raise AppError(
                status_code=409,
                code="CANDIDATE_DERIVED_CLEANUP_REQUIRED",
                message="derived Candidate assets must be reconciled before retry",
                detail={"job_id": job.id},
            )

        from_state = job.terminal_state or job.status
        job.status = "queued"
        job.stage = "queued"
        job.progress = 0
        job.attempt += 1
        job.terminal_state = None
        job.failure_reason = None
        job.candidate_id = None
        job.lease_owner = None
        job.started_at = None
        job.heartbeat_at = None
        job.lease_expires_at = None
        job.completed_at = None
        job.dispatched_at = datetime.now(UTC)
        job.allowed_next_action = "cancel_or_await_candidate_build"
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state=from_state,
            to_state="queued",
            action="retry_dispatched",
            recorded_by=actor_identity,
        )
        await self.session.commit()
        return self._job_projection(job)

    async def _start(self, job: CandidateBuildJob) -> bool:
        await self.session.refresh(job, with_for_update=True)
        if job.status != "queued":
            return False
        if job.dispatched_at is not None:
            try:
                await self._load_frozen_input(job)
            except AppError as exc:
                # A dispatched job whose immutable input no longer binds must
                # become a durable failure; an undispatched job still waits
                # for an administrator's current dispatch proof below.
                await self._fail(
                    job,
                    code=exc.code or "CANDIDATE_INPUT_INTEGRITY_FAILED",
                    message=exc.message or "candidate build immutable inputs no longer verify",
                )
                return False
        if not await has_current_dispatch_authorization(self.session, job):
            return False
        now = datetime.now(UTC)
        job.status = "running"
        job.stage = BuildJobStage.PARSING.value
        job.progress = 10
        job.started_at = now
        job.heartbeat_at = now
        job.lease_expires_at = now + _LEASE_DURATION
        job.lease_owner = self._lease_owner
        job.derived_cleanup_pending = False
        job.terminal_state = None
        job.failure_reason = None
        job.allowed_next_action = "cancel_or_await_candidate_build"
        self._owned_attempts[job.id] = job.attempt
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state=BuildJobStage.QUEUED.value,
            to_state=BuildJobStage.PARSING.value,
            action="started",
        )
        await self.session.commit()
        return True

    async def _advance(
        self,
        job: CandidateBuildJob,
        *,
        frozen_input: FrozenCandidateBuildInput,
        stage: BuildJobStage,
        progress: int,
        action: str,
    ) -> bool:
        if not await self._owns_running_job(job):
            return False
        await self._current_frozen_input(job, expected=frozen_input)
        if not await self._owns_running_job(job):
            return False
        from_stage = job.stage
        now = datetime.now(UTC)
        job.stage = validate_transition(BuildJobStage, from_stage, stage).value
        job.progress = progress
        job.heartbeat_at = now
        job.lease_expires_at = now + _LEASE_DURATION
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATE_CHANGED,
            from_state=from_stage,
            to_state=job.stage,
            action=action,
        )
        await self.session.commit()
        return True

    async def _index_candidate_generation_with_lease_heartbeat(
        self,
        *,
        job: CandidateBuildJob,
        document_id: str,
        generation: int,
        chunks: list[DocumentChunk],
        embedding_fingerprint: str | None,
    ) -> Any:
        heartbeat_task = asyncio.create_task(self._maintain_lease(job_id=job.id, attempt=job.attempt))
        index_task = asyncio.create_task(
            self._dense_index_service.index_candidate_generation(
                document_id=document_id,
                generation=generation,
                chunks=chunks,
                embedding_fingerprint=embedding_fingerprint,
            )
        )
        try:
            done, _ = await asyncio.wait(
                {index_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                await heartbeat_task
            return await index_task
        finally:
            for task in (index_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            for task in (index_task, heartbeat_task):
                with suppress(asyncio.CancelledError, Exception):
                    await task

    async def _maintain_lease(self, *, job_id: str, attempt: int) -> None:
        while True:
            await asyncio.sleep(_LEASE_HEARTBEAT_INTERVAL_SECONDS)
            if not await self._renew_lease(job_id=job_id, attempt=attempt):
                raise AppError(
                    status_code=409,
                    code="CANDIDATE_WORKER_LEASE_LOST",
                    message="Candidate Build worker lease could not be renewed while indexing",
                    detail={"job_id": job_id, "attempt": attempt},
                )

    async def _renew_lease(self, *, job_id: str, attempt: int) -> bool:
        bind = self.session.bind
        if bind is None:
            return False
        now = datetime.now(UTC)
        session_factory = async_sessionmaker(bind=bind, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as heartbeat_session:
            result = await heartbeat_session.execute(
                update(CandidateBuildJob)
                .where(
                    CandidateBuildJob.id == job_id,
                    CandidateBuildJob.status == "running",
                    CandidateBuildJob.lease_owner == self._lease_owner,
                    CandidateBuildJob.attempt == attempt,
                    CandidateBuildJob.lease_expires_at.is_not(None),
                    CandidateBuildJob.lease_expires_at > now,
                )
                .values(
                    heartbeat_at=now,
                    lease_expires_at=now + _LEASE_DURATION,
                )
            )
            await heartbeat_session.commit()
        return bool(result.rowcount)

    async def _write_chunks(
        self,
        *,
        job: CandidateBuildJob,
        frozen_input: FrozenCandidateBuildInput,
        candidate_id: str,
        chunk_specs: list[CandidateChunkSpec],
    ) -> bool:
        if not await self._owns_running_job(job):
            return False
        await self._current_frozen_input(job, expected=frozen_input)
        if not await self._owns_running_job(job):
            return False
        await self.session.execute(delete(CandidateBuildChunk).where(CandidateBuildChunk.job_id == job.id))
        self.session.add_all(
            [
                CandidateBuildChunk(
                    job_id=job.id,
                    candidate_id=candidate_id,
                    document_identity=frozen_input.document_identity,
                    generation=frozen_input.requested_generation,
                    attempt=job.attempt,
                    chunk_index=int(chunk["chunk_index"]),
                    content=str(chunk["content"]),
                    content_sha256=str(chunk["content_sha256"]),
                    chunk_metadata=dict(chunk["metadata"]),
                )
                for chunk in chunk_specs
            ]
        )
        await self.session.commit()
        return True

    async def _complete(
        self,
        *,
        job: CandidateBuildJob,
        frozen_input: FrozenCandidateBuildInput,
        candidate_identity: StableIdentity,
        chunk_specs: list[CandidateChunkSpec],
        dense_result: Any,
    ) -> bool:
        if not await self._owns_running_job(job):
            return False
        frozen_input = await self._current_frozen_input(job, expected=frozen_input)
        async with self._candidate_finalization_authority_guard(job, frozen_input):
            if not await self._owns_running_job(job):
                return False
            from_stage = job.stage
            candidate_record = CanonicalRecordModel(
                stable_id=candidate_identity.stable_id,
                identity_kind=StableIdentityKind.CANDIDATE.value,
                identity_value=candidate_identity.value,
                state="candidate_ready",
                record_class=CanonicalRecordClass.IMMUTABLE.value,
                payload={
                    "schema": "candidate_build_candidate/v1",
                    "bundle_id": frozen_input.bundle_id,
                    "bundle_sha256": frozen_input.bundle_sha256,
                    "bundle_item_id": frozen_input.bundle_item_id,
                    "bundle_item_sha256": frozen_input.bundle_item_sha256,
                    "build_generation_id": f"build_generation:{job.id}",
                    "entry_identity": frozen_input.entry_identity,
                    "document_identity": frozen_input.document_identity,
                    "requested_generation": frozen_input.requested_generation,
                    "editorial_source_revision": frozen_input.editorial_source_revision,
                    "input_sha256": frozen_input.input_sha256,
                    "frozen_input_sha256": frozen_input.frozen_input_sha256,
                    "chunk_strategy": frozen_input.chunk_strategy,
                    "embedding_configuration": frozen_input.embedding_configuration,
                    "embedding_active": bool(getattr(dense_result, "active", False)),
                    "embedding_fingerprint": getattr(dense_result, "fingerprint", None),
                    "attempt": job.attempt,
                    "chunk_count": len(chunk_specs),
                    "chunk_sha256s": [str(chunk["content_sha256"]) for chunk in chunk_specs],
                },
            )
            job.status = BuildJobTerminalStatus.CANDIDATE_READY.value
            job.progress = 100
            job.terminal_state = BuildJobTerminalStatus.CANDIDATE_READY.value
            job.candidate_id = candidate_identity.stable_id
            job.completed_at = datetime.now(UTC)
            job.heartbeat_at = job.completed_at
            job.lease_expires_at = None
            job.lease_owner = None
            job.derived_cleanup_pending = False
            job.allowed_next_action = "await_candidate_inspection"
            self.session.add(candidate_record)
            self._append_job_event(
                job,
                event_type=CanonicalEventType.STATE_CHANGED,
                from_state=from_stage,
                to_state=BuildJobTerminalStatus.CANDIDATE_READY.value,
                action="completed",
            )
            await complete_bundle_when_candidate_work_is_finished(self.session, job.bundle_id)
            await self.session.commit()
        self._owned_attempts.pop(job.id, None)
        return True

    async def _fail(self, job: CandidateBuildJob, *, code: str, message: str) -> None:
        cleanup_message = await self._cleanup_derived_assets(job)

        from_stage = job.stage
        job.status = BuildJobTerminalStatus.FAILED.value
        job.progress = min(job.progress, 99)
        job.terminal_state = BuildJobTerminalStatus.FAILED.value
        job.failure_reason = {
            "code": code,
            "stage": from_stage,
            "message": message,
            **({"cleanup": cleanup_message} if cleanup_message else {}),
        }
        job.completed_at = datetime.now(UTC)
        job.heartbeat_at = job.completed_at
        job.lease_expires_at = None
        job.lease_owner = None
        job.allowed_next_action = (
            "import_new_bundle"
            if code == "CANDIDATE_INPUT_INTEGRITY_FAILED"
            else "reconcile_derived_data_then_retry"
            if cleanup_message
            else "retry_fixed_inputs"
        )
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=from_stage,
            to_state=BuildJobTerminalStatus.FAILED.value,
            action="failed",
        )
        await self.session.commit()
        self._owned_attempts.pop(job.id, None)

    async def _cancel(self, job: CandidateBuildJob, *, message: str) -> None:
        cleanup_message = await self._cleanup_derived_assets(job)
        from_stage = job.stage
        job.status = BuildJobTerminalStatus.CANCELED.value
        job.progress = min(job.progress, 99)
        job.terminal_state = BuildJobTerminalStatus.CANCELED.value
        job.failure_reason = {
            "code": "CANDIDATE_BUILD_CANCELED",
            "stage": from_stage,
            "message": message,
            **({"cleanup": cleanup_message} if cleanup_message else {}),
        }
        job.completed_at = datetime.now(UTC)
        job.heartbeat_at = job.completed_at
        job.lease_expires_at = None
        job.lease_owner = None
        job.allowed_next_action = "reconcile_derived_data_then_retry" if cleanup_message else "retry_fixed_inputs"
        self._append_job_event(
            job,
            event_type=CanonicalEventType.STATUS_CHANGED,
            from_state=from_stage,
            to_state=BuildJobTerminalStatus.CANCELED.value,
            action="canceled",
        )
        await self.session.commit()
        self._owned_attempts.pop(job.id, None)

    async def _cleanup_derived_assets(
        self,
        job: CandidateBuildJob,
        *,
        frozen_input: FrozenCandidateBuildInput | None = None,
    ) -> str | None:
        try:
            cleanup_input = await self._current_frozen_input(job, expected=frozen_input)
        except AppError:
            job.derived_cleanup_pending = True
            return "immutable Candidate inputs require manual reconciliation"

        await self.session.execute(delete(CandidateBuildChunk).where(CandidateBuildChunk.job_id == job.id))
        try:
            await self._dense_index_service.delete_candidate_generation(
                document_id=cleanup_input.document_identity,
                generation=cleanup_input.requested_generation,
                embedding_fingerprint=cleanup_input.embedding_fingerprint,
            )
        except Exception:
            job.derived_cleanup_pending = True
            return "derived vector cleanup requires reconciliation before retry"
        job.derived_cleanup_pending = False
        return None

    @staticmethod
    def _record_retry_blocker(
        job: CandidateBuildJob,
        *,
        code: str,
        message: str,
        cleanup: str | None = None,
    ) -> None:
        prior_failure_reason = dict(job.failure_reason) if isinstance(job.failure_reason, dict) else None
        job.failure_reason = {
            "code": code,
            "stage": job.stage,
            "message": message,
            **({"cleanup": cleanup} if cleanup is not None else {}),
            **({"prior_failure_reason": prior_failure_reason} if prior_failure_reason is not None else {}),
        }

    async def _owns_running_job(self, job: CandidateBuildJob) -> bool:
        await self.session.refresh(job, with_for_update=True)
        expected_attempt = self._owned_attempts.get(job.id)
        return bool(
            job.status == "running"
            and expected_attempt is not None
            and job.lease_owner == self._lease_owner
            and job.attempt == expected_attempt
            and job.lease_expires_at is not None
            and self._lease_is_current(job.lease_expires_at)
        )

    @staticmethod
    def _lease_is_current(lease_expires_at: datetime) -> bool:
        normalized_expiry = (
            lease_expires_at.replace(tzinfo=UTC)
            if lease_expires_at.tzinfo is None
            else lease_expires_at.astimezone(UTC)
        )
        return normalized_expiry > datetime.now(UTC)

    async def _load_frozen_input(self, job: CandidateBuildJob) -> FrozenCandidateBuildInput:
        await self.session.refresh(job)
        frozen_input = await load_frozen_candidate_build_input(self.session, job.id)
        if not frozen_input.matches_job(job):
            raise AppError(
                status_code=409,
                code="CANDIDATE_INPUT_INTEGRITY_FAILED",
                message="candidate build immutable inputs no longer verify",
                detail={"job_id": job.id},
            )
        return frozen_input

    async def _current_frozen_input(
        self,
        job: CandidateBuildJob,
        *,
        expected: FrozenCandidateBuildInput | None = None,
    ) -> FrozenCandidateBuildInput:
        frozen_input = await self._load_frozen_input(job)
        if expected is not None and frozen_input != expected:
            raise AppError(
                status_code=409,
                code="CANDIDATE_INPUT_INTEGRITY_FAILED",
                message="candidate build immutable inputs changed during execution",
                detail={"job_id": job.id},
            )
        return frozen_input

    async def _revalidate_editorial_authority(self, job: CandidateBuildJob, frozen_input: FrozenCandidateBuildInput) -> None:
        frozen_input = await self._current_frozen_input(job, expected=frozen_input)
        verified_artifact = await self._editorial_export_verifier.verify(frozen_input.artifact, frozen_input.input_sha256)
        if verified_artifact != frozen_input.artifact:
            raise AppError(
                status_code=409,
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                message="Candidate Build inputs no longer match the approved editorial export",
                detail={"job_id": job.id},
            )

    @asynccontextmanager
    async def _candidate_finalization_authority_guard(
        self,
        job: CandidateBuildJob,
        frozen_input: FrozenCandidateBuildInput,
    ) -> AsyncIterator[None]:
        try:
            authority_context = self._editorial_export_verifier.verify_for_candidate_finalization(
                frozen_input.artifact,
                frozen_input.input_sha256,
            )
        except AttributeError as exc:
            raise AppError(
                status_code=409,
                code="EDITORIAL_AUTHORITY_FINALIZATION_FENCE_REQUIRED",
                message="Candidate finalization requires an editorial authority fence",
                detail={"job_id": job.id},
            ) from exc
        async with authority_context as verified_artifact:
            if verified_artifact != frozen_input.artifact:
                raise AppError(
                    status_code=409,
                    code="EDITORIAL_EXPORT_NOT_APPROVED",
                    message="Candidate Build inputs no longer match the approved editorial export",
                    detail={"job_id": job.id},
                )
            yield

    def _assert_embedding_configuration(self, frozen_input: FrozenCandidateBuildInput) -> None:
        if candidate_embedding_configuration(self._settings) != frozen_input.embedding_configuration:
            raise AppError(
                status_code=409,
                code="CANDIDATE_EMBEDDING_CONFIGURATION_CHANGED",
                message="Candidate Build embedding configuration changed after bundle intake",
                detail={"entry_identity": frozen_input.entry_identity},
            )

    def _parse_artifact(self, artifact: dict[str, object]) -> ParsedCandidateArtifact:
        entry = artifact.get("entry")
        if not isinstance(entry, dict):
            raise AppError(
                status_code=422,
                code="CANDIDATE_PARSER_FAILED",
                message="editorial export entry metadata is missing",
            )
        body = entry.get("body")
        if not isinstance(body, dict):
            raise AppError(
                status_code=422,
                code="CANDIDATE_PARSER_FAILED",
                message="editorial export body is missing",
            )
        entry_sources: dict[str, dict[str, object]] = {}
        raw_entry_sources = entry.get("sources")
        if isinstance(raw_entry_sources, list):
            for item in raw_entry_sources:
                if isinstance(item, dict) and isinstance(item.get("source_id"), str):
                    entry_sources[item["source_id"]] = item

        source_relationships_by_identity: dict[str, dict[str, object]] = {}
        source_snapshots = artifact.get("sources")
        if isinstance(source_snapshots, list):
            for item in source_snapshots:
                if isinstance(item, dict):
                    source_identity = item.get("source_identity")
                    if isinstance(source_identity, str):
                        source_definition = item.get("source")
                        if not isinstance(source_definition, dict):
                            source_definition = entry_sources.get(source_identity.removeprefix("source:"), {})
                        source_relationships_by_identity[source_identity] = {
                            "source_identity": source_identity,
                            "availability": item.get(
                                "availability",
                                source_definition.get("availability"),
                            ),
                            "access_scope": source_definition.get("access_scope"),
                        }

        raw_section_relationships = entry.get("section_source_relationships")
        if not isinstance(raw_section_relationships, list):
            raise AppError(
                status_code=422,
                code="CANDIDATE_PARSER_FAILED",
                message="editorial export section-source relationships are missing",
            )
        source_relationships_by_section: dict[str, list[dict[str, object]]] = {}
        for relationship in raw_section_relationships:
            if not isinstance(relationship, dict):
                raise AppError(
                    status_code=422,
                    code="CANDIDATE_PARSER_FAILED",
                    message="editorial export section-source relationship is invalid",
                )
            section_id = relationship.get("section_id")
            source_ids = relationship.get("source_ids")
            if (
                not isinstance(section_id, str)
                or not section_id
                or section_id in source_relationships_by_section
                or not isinstance(source_ids, list)
                or not source_ids
            ):
                raise AppError(
                    status_code=422,
                    code="CANDIDATE_PARSER_FAILED",
                    message="editorial export section-source relationship is invalid",
                )
            relationships: list[dict[str, object]] = []
            for source_id in source_ids:
                source_identity = f"source:{source_id}" if isinstance(source_id, str) else None
                relationship_snapshot = (
                    source_relationships_by_identity.get(source_identity)
                    if source_identity is not None
                    else None
                )
                if relationship_snapshot is None:
                    raise AppError(
                        status_code=422,
                        code="CANDIDATE_PARSER_FAILED",
                        message="editorial export section source is missing from the verified source snapshots",
                    )
                relationships.append(dict(relationship_snapshot))
            source_relationships_by_section[section_id] = sorted(
                relationships,
                key=lambda item: str(item["source_identity"]),
            )
        return {
            "entry": entry,
            "body": body,
            "source_relationships_by_section": source_relationships_by_section,
            "assurance_level": entry.get("assurance_level"),
            "applicability_conditions": list(entry.get("applicability_conditions") or []),
            "non_applicability_conditions": list(entry.get("non_applicability_conditions") or []),
            "freshness_triggers": list(entry.get("freshness_triggers") or []),
            "editorial_revision_identity": artifact.get("editorial_revision_identity"),
        }

    def _chunk_specs(
        self,
        parsed_artifact: ParsedCandidateArtifact,
        frozen_input: FrozenCandidateBuildInput,
    ) -> list[CandidateChunkSpec]:
        entry = parsed_artifact["entry"]
        body = parsed_artifact["body"]
        maximum = frozen_input.chunk_strategy.get("max_characters")
        overlap = frozen_input.chunk_strategy.get("overlap_characters")
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not isinstance(overlap, int)
            or isinstance(overlap, bool)
            or maximum < 100
            or overlap < 0
            or overlap >= maximum
        ):
            raise AppError(
                status_code=422,
                code="CANDIDATE_CHUNK_STRATEGY_INVALID",
                message="candidate build chunk strategy is invalid",
            )
        chunks: list[CandidateChunkSpec] = []
        for section_id, value in body.items():
            if not isinstance(section_id, str) or not isinstance(value, str) or not value.strip():
                continue
            source_relationships = parsed_artifact["source_relationships_by_section"].get(section_id)
            if not source_relationships:
                raise AppError(
                    status_code=422,
                    code="CANDIDATE_PARSER_FAILED",
                    message="editorial export body section has no verified source relationship",
                )
            heading = section_id.replace("_", " ").strip().title()
            section_text = f"## {heading}\n\n{value.strip()}"
            for piece in self._split_section(section_text, maximum=maximum, overlap=overlap):
                content_sha256 = self._sha256(piece)
                chunks.append(
                    {
                        "chunk_index": len(chunks),
                        "content": piece,
                        "content_sha256": content_sha256,
                        "metadata": {
                            "entry_id": entry.get("entry_id"),
                            "domain": entry.get("coverage_position"),
                            "entry_identity": frozen_input.entry_identity,
                            "editorial_revision_identity": parsed_artifact["editorial_revision_identity"],
                            "section_id": section_id,
                            "section_title": heading,
                            "chunk_strategy_id": frozen_input.chunk_strategy.get("strategy_id"),
                            "source_identities": [
                                relationship["source_identity"] for relationship in source_relationships
                            ],
                            "source_relationships": source_relationships,
                            "assurance_level": parsed_artifact["assurance_level"],
                            "applicability_conditions": parsed_artifact["applicability_conditions"],
                            "non_applicability_conditions": parsed_artifact["non_applicability_conditions"],
                            "freshness_triggers": parsed_artifact["freshness_triggers"],
                            "lifecycle_state": "candidate_build",
                            "candidate_build": True,
                        },
                    }
                )
        return chunks

    @staticmethod
    def _split_section(value: str, *, maximum: int, overlap: int) -> list[str]:
        if len(value) <= maximum:
            return [value]
        step = maximum - overlap
        return [value[start : start + maximum] for start in range(0, len(value), step) if value[start : start + maximum]]

    @staticmethod
    def _candidate_identity(job: CandidateBuildJob) -> StableIdentity:
        return StableIdentity(StableIdentityKind.CANDIDATE, f"{job.id}-attempt-{job.attempt}")

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    async def _get_job(self, job_id: str) -> CandidateBuildJob:
        job = await self.session.get(CandidateBuildJob, job_id)
        if job is None:
            raise AppError(
                status_code=404,
                code="RESOURCE_NOT_FOUND",
                message="candidate build job not found",
                detail={"job_id": job_id},
            )
        return job

    @staticmethod
    def _job_projection(job: CandidateBuildJob) -> dict[str, object]:
        return {
            "job_id": job.id,
            "status": job.status,
            "stage": job.stage,
            "progress": job.progress,
            "attempt": job.attempt,
            "terminal_state": job.terminal_state,
            "failure_reason": job.failure_reason,
            "allowed_next_action": job.allowed_next_action,
        }

    async def _candidate_projection(self, job: CandidateBuildJob) -> dict[str, object]:
        assert job.candidate_id is not None
        candidate = await self.session.get(CanonicalRecordModel, job.candidate_id)
        assert candidate is not None
        payload = candidate.payload
        return {
            "candidate_id": candidate.stable_id,
            "entry_identity": payload["entry_identity"],
            "input_sha256": payload["input_sha256"],
            "frozen_input_sha256": payload["frozen_input_sha256"],
            "chunk_count": payload["chunk_count"],
            "status": candidate.state,
            "allowed_next_action": job.allowed_next_action,
        }

    def _append_job_event(
        self,
        job: CandidateBuildJob,
        *,
        event_type: CanonicalEventType,
        from_state: str | None,
        to_state: str,
        action: str,
        recorded_by: str | None = None,
    ) -> None:
        self.session.add(
            CanonicalEventModel(
                aggregate_id=f"build_generation:{job.id}",
                aggregate_kind=StableIdentityKind.BUILD_GENERATION.value,
                event_type=event_type.value,
                from_state=from_state,
                to_state=to_state,
                payload=candidate_job_event_payload(job, action=action),
                recorded_by=recorded_by,
            )
        )
