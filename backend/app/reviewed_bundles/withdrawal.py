from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.contracts.canonical import CanonicalEventType, EntryLifecycleState, StableIdentityKind, validate_transition
from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.documents.dense_index_service import DenseIndexService
from app.documents.runtime_dense_obligations import complete_runtime_dense_target, pending_runtime_dense_targets
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.document import Document, DocumentChunk
from app.model.user import User
from app.repository.chat_repository import ChatRepository
from app.reviewed_bundles.events import candidate_job_event_payload
from app.reviewed_bundles.inputs import load_frozen_candidate_build_input, runtime_document_identity
from app.reviewed_bundles.lifecycle import complete_bundle_when_candidate_work_is_finished
from app.reviewed_bundles.models import CandidateBuildChunk, CandidateBuildJob, PublishedKnowledgePointer, PublishedKnowledgeVersion
from app.reviewed_bundles.publication import CandidatePublicationService
from app.reviewed_bundles.runtime import candidate_build_runtime
from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier
from app.reviewed_bundles.withdrawal_facts import (
    WithdrawalReason,
    WithdrawalTrigger,
    read_publication_withdrawals,
    validate_withdrawal_reconciliation,
)
from app.reviewed_bundles.writer_exit import require_candidate_writers_settled
from app.service.answer_execution_store import AnswerExecutionStore
from app.service.identity_audit_service import IdentityAuditService


class WithdrawalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reason_code: WithdrawalReason
    trigger: WithdrawalTrigger


class PublicationWithdrawalService:
    """Contain an exact current publication while retaining its immutable lineage."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, publication_identity: str) -> dict:
        event = await self._withdrawal_event(publication_identity)
        if event is None:
            raise AppError(status_code=404, code="WITHDRAWAL_NOT_FOUND", message="publication has no withdrawal record")
        return dict(event.payload)

    async def withdraw(self, publication_identity: str, request: WithdrawalRequest, administrator: User) -> dict:
        await ChatRepository(self.session).acquire_private_conversation_write_fence()
        current_actor = await self.session.scalar(
            select(User).where(User.id == administrator.id).with_for_update().execution_options(populate_existing=True)
        )
        if current_actor is None or not current_actor.is_active or current_actor.role != "admin":
            raise AppError(status_code=403, code="ADMIN_REQUIRED", message="current administrator authority is required")
        actor = await IdentityAuditService(self.session).ensure_member_record(current_actor, admission_path="publication_withdrawal")
        version = await self.session.get(PublishedKnowledgeVersion, publication_identity)
        if version is None:
            raise AppError(status_code=404, code="PUBLICATION_NOT_FOUND", message="publication was not found")
        # Match publication's canonical lock order, including Release-Assured
        # acceptance records, before taking pointer, document and job locks.
        authority_records = list((await self.session.scalars(select(CanonicalRecordModel).where(or_(
            CanonicalRecordModel.identity_kind == StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value,
            CanonicalRecordModel.stable_id == version.entry_identity,
        )).order_by(CanonicalRecordModel.stable_id).with_for_update())).all())
        entry = next((record for record in authority_records if record.stable_id == version.entry_identity), None)
        previous = await self._withdrawal_event(publication_identity)
        if previous is not None:
            if previous.payload["reason_code"] != request.reason_code or previous.payload["trigger"] != request.trigger:
                raise AppError(status_code=409, code="WITHDRAWAL_CONFLICT", message="original withdrawal reason cannot be replaced")
            return dict(previous.payload)
        pointer = await self.session.scalar(
            select(PublishedKnowledgePointer)
            .where(PublishedKnowledgePointer.entry_identity == version.entry_identity)
            .with_for_update().execution_options(populate_existing=True)
        )
        canonical = await self.session.get(CanonicalRecordModel, publication_identity)
        document = await self.session.scalar(
            select(Document).where(Document.id == version.document_identity)
            .with_for_update().execution_options(populate_existing=True)
        )
        try:
            await CandidatePublicationService(
                self.session, editorial_export_verifier=CanonicalEditorialExportVerifier(self.session),
            ).get_publication(version.candidate_id)
        except AppError as exc:
            raise AppError(
                status_code=409, code="WITHDRAWAL_BINDING_INVALID",
                message="immutable publication binding cannot be proven",
            ) from exc
        if (
            entry is None or canonical is None or canonical.payload.get("schema") != "published_knowledge_version/v1"
            or pointer is None or document is None or pointer.current_version_id != publication_identity
            or pointer.document_identity != version.document_identity or pointer.generation != version.generation
            or document.published_generation != version.generation
            or any(
                type(canonical.payload.get(field)) is not type(getattr(version, field))
                or canonical.payload.get(field) != getattr(version, field) for field in (
                "candidate_id", "entry_identity", "document_identity", "generation", "configuration_identity",
                "bundle_sha256", "frozen_input_sha256", "inspection_record_identity", "acceptance_record_identity",
                "supersedes_version_id",
            ))
        ):
            raise AppError(status_code=409, code="WITHDRAWAL_BINDING_INVALID", message="current publication binding cannot be proven")
        events = list((await self.session.scalars(
            select(CanonicalEventModel).where(
                CanonicalEventModel.aggregate_id == version.entry_identity,
                CanonicalEventModel.aggregate_kind == "entry",
            ).order_by(CanonicalEventModel.occurred_at, CanonicalEventModel.id)
        )).all())
        if not events:
            raise AppError(status_code=409, code="WITHDRAWAL_BINDING_INVALID", message="entry lifecycle cannot be proven")
        latest = max(events, key=lambda event: event.payload.get("sequence", 0))
        if latest.to_state not in {"published", "needs_re_review", "editorial_review", "candidate_build"}:
            raise AppError(status_code=409, code="WITHDRAWAL_STATE_INVALID", message="entry is not in a withdrawable state")
        validate_transition(EntryLifecycleState, latest.to_state, EntryLifecycleState.WITHDRAWN)
        now = datetime.now(UTC)
        event_id = uuid4().hex
        payload = {
            "schema": "publication_withdrawal/v1",
            "event_identity": f"event:{event_id}",
            "entry_identity": version.entry_identity,
            "publication_identity": publication_identity,
            "document_identity": version.document_identity,
            "generation": version.generation,
            "configuration_identity": version.configuration_identity,
            "supersedes_version_id": version.supersedes_version_id,
            "state": "withdrawn",
            "reason_code": request.reason_code,
            "trigger": request.trigger,
            "actor_identity": actor,
            "occurred_at": now.isoformat(),
            "affected_scope": {"scope": "entry_version", "identity": publication_identity},
        }
        self.session.add(CanonicalEventModel(
            id=event_id, aggregate_id=publication_identity, aggregate_kind="published_knowledge_version",
            event_type=CanonicalEventType.WITHDRAWN.value, from_state="published", to_state="withdrawn",
            payload=payload, occurred_at=now, recorded_by=actor,
        ))
        self.session.add(CanonicalEventModel(
            aggregate_id=version.entry_identity, aggregate_kind="entry",
            event_type=CanonicalEventType.WITHDRAWN.value, from_state=latest.to_state, to_state="withdrawn",
            payload={
                "schema": "editorial_authority_event/v1",
                "sequence": latest.payload["sequence"] + 1,
                "action": "publication_withdrawn",
                "revision_identity": latest.payload["revision_identity"],
                "withdrawal_event_identity": payload["event_identity"],
                "published_knowledge_version_identity": publication_identity,
            },
            occurred_at=now, recorded_by=actor,
        ))
        document.deleted_at = now
        jobs = await self._unpublished_jobs(version.entry_identity, for_update=True)
        for job in jobs:
            previous_state = job.status
            if previous_state not in {
                "queued", "running", "interrupted_retryable", "failed", "candidate_ready", "canceled", "superseded",
            }:
                raise AppError(
                    status_code=409, code="WITHDRAWAL_BINDING_INVALID",
                    message="Candidate lifecycle cannot be verified for withdrawal",
                )
            job.status = "superseded"
            job.terminal_state = "superseded"
            job.allowed_next_action = "none"
            job.derived_cleanup_pending = True
            job.lease_owner = None
            job.lease_expires_at = None
            job.completed_at = now
            job.failure_reason = {"code": "PUBLICATION_WITHDRAWN", "stage": job.stage, "message": "publication withdrawn"}
            self.session.add(CanonicalEventModel(
                aggregate_id=f"build_generation:{job.id}", aggregate_kind="build_generation",
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state=previous_state, to_state="superseded", recorded_by=actor, occurred_at=now,
                payload=candidate_job_event_payload(job, action="invalidated_by_withdrawal", extra={
                    "withdrawal_event_identity": payload["event_identity"],
                }),
            ))
        for bundle_id in sorted({job.bundle_id for job in jobs}):
            await complete_bundle_when_candidate_work_is_finished(self.session, bundle_id)
        self._record_reconciliation(payload, state="pending", reason_code="derived_cleanup_pending")
        await DeliveryAcceptanceService(self.session).suspend_withdrawn_publication(
            publication_identity=publication_identity, entry_identity=version.entry_identity, actor_identity=actor,
            locked_records=[
                record for record in authority_records
                if record.identity_kind == StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value
            ],
        )
        await self.session.commit()
        failure_reason = "derived_cleanup_failed"
        try:
            for job in jobs:
                reached = await candidate_build_runtime.cancel(job.id)
                if not reached:
                    frozen_job = await load_frozen_candidate_build_input(self.session, job.id)
                    await require_candidate_writers_settled(self.session, job, frozen_job)
            failure_reason = "history_redaction_failed"
            await AnswerExecutionStore(self.session, ChatRepository(self.session)).redact_document_evidence(
                document_id=payload["document_identity"], publication_identity=publication_identity, withdrawal=payload,
            )
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            self._record_reconciliation(payload, state="suspended", reason_code=failure_reason)
            await self.session.commit()
        return payload

    async def reconciliation(self, publication_identity: str) -> dict:
        withdrawal = await self.get(publication_identity)
        events = list((await self.session.scalars(select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == publication_identity,
            CanonicalEventModel.event_type == CanonicalEventType.STATUS_CHANGED.value,
        ).order_by(CanonicalEventModel.occurred_at, CanonicalEventModel.id))).all())
        if not events:
            return {
                "state": "suspended", "affected_scope": withdrawal["affected_scope"],
                "allowed_next_action": "retry_reconciliation",
            }
        for event in events:
            validate_withdrawal_reconciliation(event, withdrawal)
        state = dict(events[-1].payload)
        if state["state"] == "completed":
            chunks = await self.session.scalar(select(DocumentChunk.id).where(
                DocumentChunk.document_id == withdrawal["document_identity"],
                DocumentChunk.generation == withdrawal["generation"],
            ).limit(1))
            jobs = await self._unpublished_jobs(withdrawal["entry_identity"])
            document = await self.session.get(Document, withdrawal["document_identity"])
            try:
                for job in jobs:
                    frozen = await load_frozen_candidate_build_input(self.session, job.id)
                    if job.status != "superseded" or not frozen.matches_job(job):
                        raise ValueError("withdrawal successor binding differs")
                    await require_candidate_writers_settled(self.session, job, frozen)
                    remaining_chunk = await self.session.scalar(select(CandidateBuildChunk.id).where(
                        CandidateBuildChunk.job_id == job.id,
                    ).limit(1))
                    if remaining_chunk is not None:
                        raise ValueError("withdrawal successor chunks remain")
            except (AppError, ValueError) as exc:
                raise AppError(
                    status_code=409, code="WITHDRAWAL_RECONCILIATION_INVALID",
                    message="withdrawal successor completion cannot be verified",
                ) from exc
            if (
                chunks is not None or any(job.derived_cleanup_pending for job in jobs)
                or await pending_runtime_dense_targets(self.session, withdrawal)
                or document is not None and document.dense_ready_generation == withdrawal["generation"]
                and document.dense_ready_fingerprint is not None
            ):
                raise AppError(
                    status_code=409, code="WITHDRAWAL_RECONCILIATION_INVALID",
                    message="withdrawal reconciliation still has pending derived data",
                )
        return state

    async def reconcile(self, publication_identity: str) -> dict:
        await ChatRepository(self.session).acquire_private_conversation_write_fence()
        withdrawal = await self.get(publication_identity)
        await self.session.scalar(
            select(CanonicalRecordModel).where(
                CanonicalRecordModel.stable_id == withdrawal["entry_identity"],
            ).with_for_update()
        )
        state = await self.reconciliation(publication_identity)
        if state["state"] == "completed":
            return state
        try:
            version = await self.session.get(PublishedKnowledgeVersion, publication_identity)
            job = await self.session.scalar(select(CandidateBuildJob).where(
                CandidateBuildJob.candidate_id == version.candidate_id,
            )) if version is not None else None
            if version is None or job is None:
                raise ValueError("withdrawal input cannot be reconstructed")
            frozen = await load_frozen_candidate_build_input(self.session, job.id)
            if (
                not frozen.matches_job(job)
                or frozen.frozen_input_sha256 != version.frozen_input_sha256
                or frozen.requested_generation != withdrawal["generation"]
                or runtime_document_identity(frozen.document_identity) != withdrawal["document_identity"]
            ):
                raise ValueError("withdrawal input binding differs")
            # Answer completion locks evidence documents before execution rows.
            document = await self.session.scalar(select(Document).where(
                Document.id == withdrawal["document_identity"],
            ).with_for_update().execution_options(populate_existing=True))
            if document is None or document.published_generation != withdrawal["generation"]:
                raise ValueError("withdrawal runtime document binding differs")
            await AnswerExecutionStore(self.session, ChatRepository(self.session)).redact_document_evidence(
                document_id=withdrawal["document_identity"],
                publication_identity=publication_identity, withdrawal=withdrawal,
            )
            await DenseIndexService().delete_candidate_generation(
                document_id=frozen.document_identity, generation=frozen.requested_generation,
                embedding_fingerprint=frozen.embedding_fingerprint,
            )
            for target in await pending_runtime_dense_targets(self.session, withdrawal):
                if target.state != "settled":
                    raise ValueError("runtime dense writer has not proven termination")
                await DenseIndexService().delete_candidate_generation(
                    document_id=target.document_identity, generation=target.generation,
                    embedding_fingerprint=target.embedding_fingerprint,
                )
                complete_runtime_dense_target(self.session, target, withdrawal)
            if document.dense_ready_generation == withdrawal["generation"]:
                await DenseIndexService().delete_candidate_generation(
                    document_id=document.id, generation=document.dense_ready_generation,
                    embedding_fingerprint=document.dense_ready_fingerprint,
                )
                document.dense_ready_generation = 0
                document.dense_ready_fingerprint = None
            await self.session.execute(delete(DocumentChunk).where(
                DocumentChunk.document_id == withdrawal["document_identity"],
                DocumentChunk.generation == withdrawal["generation"],
            ))
            pending_jobs = await self._unpublished_jobs(withdrawal["entry_identity"], for_update=True)
            for pending in pending_jobs:
                pending_input = await load_frozen_candidate_build_input(self.session, pending.id)
                if pending.status != "superseded" or not pending_input.matches_job(pending):
                    raise ValueError("withdrawal successor input binding differs")
                await require_candidate_writers_settled(self.session, pending, pending_input)
                await DenseIndexService().delete_candidate_generation(
                    document_id=pending_input.document_identity,
                    generation=pending_input.requested_generation,
                    embedding_fingerprint=pending_input.embedding_fingerprint,
                )
                await self.session.execute(delete(CandidateBuildChunk).where(
                    CandidateBuildChunk.job_id == pending.id,
                    CandidateBuildChunk.generation == pending_input.requested_generation,
                ))
                pending.derived_cleanup_pending = False
                self.session.add(CanonicalEventModel(
                    aggregate_id=f"build_generation:{pending.id}", aggregate_kind="build_generation",
                    event_type=CanonicalEventType.STATUS_CHANGED.value,
                    from_state="superseded", to_state="superseded",
                    recorded_by=withdrawal["actor_identity"],
                    payload=candidate_job_event_payload(pending, action="withdrawal_cleanup_completed"),
                ))
            self._record_reconciliation(withdrawal, state="completed", reason_code="derived_cleanup_completed")
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            self._record_reconciliation(withdrawal, state="suspended", reason_code="derived_cleanup_failed")
            await self.session.commit()
        return await self.reconciliation(publication_identity)

    async def _unpublished_jobs(self, entry_identity: str, *, for_update: bool = False) -> list[CandidateBuildJob]:
        published_candidates = select(CanonicalRecordModel.payload["candidate_id"].as_string()).where(
            CanonicalRecordModel.identity_kind == "published_knowledge_version",
            CanonicalRecordModel.payload["candidate_id"].as_string().is_not(None),
        )
        published_builds = select(CanonicalRecordModel.payload["build_generation_id"].as_string()).where(
            CanonicalRecordModel.identity_kind == "candidate",
            CanonicalRecordModel.payload["build_generation_id"].as_string().is_not(None),
            or_(
                CanonicalRecordModel.stable_id.in_(published_candidates),
                CanonicalRecordModel.stable_id.in_(select(PublishedKnowledgeVersion.candidate_id)),
            ),
        )
        statement = select(CandidateBuildJob).where(
            CandidateBuildJob.entry_identity == entry_identity,
            ("build_generation:" + CandidateBuildJob.id).not_in(published_builds),
            or_(
                CandidateBuildJob.candidate_id.is_(None),
                CandidateBuildJob.candidate_id.not_in(select(PublishedKnowledgeVersion.candidate_id))
                & CandidateBuildJob.candidate_id.not_in(published_candidates),
            ),
        ).order_by(CandidateBuildJob.id)
        if for_update:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        return list((await self.session.scalars(statement)).all())

    def _record_reconciliation(self, withdrawal: dict, *, state: str, reason_code: str) -> None:
        self.session.add(CanonicalEventModel(
            aggregate_id=withdrawal["publication_identity"], aggregate_kind="published_knowledge_version",
            event_type=CanonicalEventType.STATUS_CHANGED.value, from_state="withdrawn", to_state=state,
            recorded_by=withdrawal["actor_identity"],
            payload={
                "schema": "withdrawal_reconciliation/v1",
                "withdrawal_event_identity": withdrawal["event_identity"],
                "state": state, "reason_code": reason_code,
                "affected_scope": withdrawal["affected_scope"],
                "allowed_next_action": "none" if state == "completed" else "retry_reconciliation",
            },
        ))

    async def _withdrawal_event(self, publication_identity: str) -> CanonicalEventModel | None:
        events = await read_publication_withdrawals(self.session, [publication_identity])
        return events[0] if events else None
