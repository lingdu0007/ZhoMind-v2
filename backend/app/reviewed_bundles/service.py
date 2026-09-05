from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import NoReturn, Protocol

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import Settings, get_settings
from app.common.exceptions import AppError
from app.contracts.canonical import (
    BundleIntakeState,
    CanonicalEventType,
    CanonicalRecordClass,
    StableIdentity,
    StableIdentityKind,
    validate_transition,
)
from app.editorial_authority.schemas import editorial_export_safety_findings
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.rag.dense_contract import DenseEmbeddingContract
from app.reviewed_bundles.events import candidate_job_event_payload
from app.reviewed_bundles.lifecycle import bundle_intake_state, complete_bundle_when_candidate_work_is_finished
from app.reviewed_bundles.models import CandidateBuildJob

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_SCHEMA = "reviewed_release_bundle/v1"
_EXPORT_SCHEMA = "editorial_export/v1"
_ITEM_OPERATIONS = {"create", "replace", "no_op", "proposed_withdrawal"}
_ASSURANCE_LEVELS = {"source_grounded", "claim_linked", "release_assured"}
_ACCESS_SCOPES = {"public", "controlled_internal"}
_APPROVED_EXPORT_STATUSES = {"approved", "lightweight_accepted"}
_AUDIT_REASON_CODES = {"unsafe", "invalid", "unsupported", "mismatch", "required", "duplicate", "conflict"}
_AUDIT_REASON_FIELDS = {
    "schema",
    "schema_version",
    "bundle_id",
    "editorial_source_revision",
    "exported_at",
    "bundle_sha256",
    "items",
}
_MAX_IMPORT_AUDIT_REASONS = 16
_MAX_GENERATION_ALLOCATION_RETRIES = 3


class EditorialExportVerifier(Protocol):
    async def verify(self, artifact: dict, artifact_sha256: str) -> dict: ...


class ReviewedReleaseBundleService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        editorial_export_verifier: EditorialExportVerifier,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self._editorial_export_verifier = editorial_export_verifier
        self._settings = settings or get_settings()

    async def import_bundle(
        self,
        manifest: object,
        *,
        actor_identity: str,
        _generation_retry: int = 0,
    ) -> dict:
        try:
            manifest = self._validate_bundle_integrity(manifest)
        except AppError as exc:
            if exc.code == "BUNDLE_INTEGRITY_REJECTED":
                await self._record_rejected_import(actor_identity=actor_identity, error=exc)
            raise
        bundle_id = manifest["bundle_id"]
        bundle_identity = StableIdentity(StableIdentityKind.BUNDLE, bundle_id)
        existing = await self.session.get(CanonicalRecordModel, bundle_identity.stable_id)
        if existing is not None:
            if existing.payload.get("bundle_sha256") != manifest["bundle_sha256"]:
                error = AppError(
                    status_code=409,
                    code="BUNDLE_ID_CONFLICT",
                    message="bundle_id is already bound to a different immutable bundle",
                    detail={
                        "reasons": [
                            {
                                "field": "bundle_id",
                                "code": "conflict",
                                "message": "immutable bundle identity is already bound",
                            }
                        ]
                    },
                )
                await self._record_rejected_import(actor_identity=actor_identity, error=error, action="identity_conflict")
                raise error
            return await self._projection(bundle_identity.stable_id)

        try:
            await self._assert_item_identities_available(manifest["items"])
        except AppError as exc:
            await self._record_rejected_import(actor_identity=actor_identity, error=exc, action="identity_conflict")
            raise

        bundle_record = CanonicalRecordModel(
            stable_id=bundle_identity.stable_id,
            identity_kind=StableIdentityKind.BUNDLE.value,
            identity_value=bundle_id,
            state="received",
            record_class=CanonicalRecordClass.IMMUTABLE.value,
            payload={
                "schema": _BUNDLE_SCHEMA,
                "schema_version": manifest["schema_version"],
                "editorial_source_revision": manifest["editorial_source_revision"],
                "exported_at": manifest["exported_at"],
                "bundle_sha256": manifest["bundle_sha256"],
                "manifest": manifest,
            },
        )
        records: list[CanonicalRecordModel] = [bundle_record]
        jobs: list[CandidateBuildJob] = []
        supersession_events: list[CanonicalEventModel] = []
        job_events: list[CanonicalEventModel] = []
        has_rejected_items = False
        for position, item in enumerate(manifest["items"]):
            item_identity = StableIdentity(StableIdentityKind.BUNDLE_ITEM, item["bundle_item_id"])
            artifact = item["artifact"]
            artifact_sha256 = item["artifact_sha256"]
            verified_artifact, failure_reason = await self._validate_item(
                artifact=artifact,
                artifact_sha256=artifact_sha256,
            )
            entry_identity = artifact["entry_identity"]
            state = "rejected" if failure_reason is not None else "admitted"
            allowed_next_action = "correct_item_in_new_bundle" if failure_reason is not None else "dispatch_candidate_build"
            if failure_reason is not None:
                has_rejected_items = True
            job: CandidateBuildJob | None = None

            if failure_reason is None and item["operation"] == "no_op":
                state = "no_op"
                allowed_next_action = "review_explicit_no_op"
            elif failure_reason is None and item["operation"] == "proposed_withdrawal":
                state = "proposed_withdrawal"
                allowed_next_action = "requires_t04_publication_workflow"
            elif failure_reason is None:
                assert verified_artifact is not None
                requested_generation = await self._next_generation(entry_identity)
                job = CandidateBuildJob(
                    id=uuid.uuid4().hex,
                    bundle_id=bundle_identity.stable_id,
                    bundle_item_id=item_identity.stable_id,
                    entry_identity=entry_identity,
                    document_identity=self._document_identity(entry_identity),
                    requested_generation=requested_generation,
                    editorial_source_revision=manifest["editorial_source_revision"],
                    input_sha256=artifact_sha256,
                    chunk_strategy=dict(verified_artifact["entry"]["chunk_strategy"]),
                    embedding_configuration=self._effective_embedding_configuration(),
                    status="queued",
                    stage="queued",
                    progress=0,
                    attempt=1,
                    allowed_next_action="dispatch_candidate_build",
                )
                jobs.append(job)
                supersession_events.extend(
                    await self._supersede_older_generations(
                        entry_identity=entry_identity,
                        requested_generation=requested_generation,
                        superseding_bundle_id=bundle_identity.stable_id,
                    )
                )

            item_payload: dict[str, object] = {
                "schema": "reviewed_release_bundle_item/v1",
                "bundle_id": bundle_identity.stable_id,
                "position": position,
                "operation": item["operation"],
                "entry_identity": entry_identity,
                "artifact_sha256": artifact_sha256,
                "bundle_item_sha256": item["bundle_item_sha256"],
                "artifact": verified_artifact if verified_artifact is not None else artifact,
                "allowed_next_action": allowed_next_action,
            }
            if failure_reason is not None:
                item_payload["failure_reason"] = failure_reason
            records.append(
                CanonicalRecordModel(
                    stable_id=item_identity.stable_id,
                    identity_kind=StableIdentityKind.BUNDLE_ITEM.value,
                    identity_value=item_identity.value,
                    state=state,
                    record_class=CanonicalRecordClass.IMMUTABLE.value,
                    payload=item_payload,
                )
            )
            if job is not None:
                input_identity = StableIdentity(StableIdentityKind.BUILD_GENERATION, job.id)
                records.append(
                    CanonicalRecordModel(
                        stable_id=input_identity.stable_id,
                        identity_kind=StableIdentityKind.BUILD_GENERATION.value,
                        identity_value=input_identity.value,
                        state="queued",
                        record_class=CanonicalRecordClass.IMMUTABLE.value,
                        payload={
                            "schema": "candidate_build_input/v1",
                            "bundle_id": bundle_identity.stable_id,
                            "bundle_sha256": manifest["bundle_sha256"],
                            "bundle_item_id": item_identity.stable_id,
                            "bundle_item_sha256": item["bundle_item_sha256"],
                            "entry_identity": entry_identity,
                            "document_identity": job.document_identity,
                            "requested_generation": job.requested_generation,
                            "editorial_source_revision": job.editorial_source_revision,
                            "input_sha256": artifact_sha256,
                            "chunk_strategy": job.chunk_strategy,
                            "embedding_configuration": job.embedding_configuration,
                        },
                    )
                )
                job_events.append(
                    CanonicalEventModel(
                        aggregate_id=input_identity.stable_id,
                        aggregate_kind=StableIdentityKind.BUILD_GENERATION.value,
                        event_type=CanonicalEventType.CREATED.value,
                        from_state=None,
                        to_state="queued",
                        payload=candidate_job_event_payload(job, action="queued"),
                        recorded_by=actor_identity,
                    )
                )

        bundle_record.payload["has_rejected_items"] = has_rejected_items
        events = [
            *self._bundle_events(
                bundle_identity.stable_id,
                actor_identity,
                has_candidate_work=bool(jobs),
                has_rejected_items=has_rejected_items,
            ),
            *supersession_events,
            *job_events,
        ]
        self.session.add_all(records)
        self.session.add_all(jobs)
        self.session.add_all(events)
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            winning_bundle = await self.session.get(CanonicalRecordModel, bundle_identity.stable_id)
            if winning_bundle is not None and winning_bundle.payload.get("bundle_sha256") == manifest["bundle_sha256"]:
                return await self._projection(bundle_identity.stable_id)
            if winning_bundle is None and _generation_retry < _MAX_GENERATION_ALLOCATION_RETRIES:
                return await self.import_bundle(
                    manifest,
                    actor_identity=actor_identity,
                    _generation_retry=_generation_retry + 1,
                )
            if winning_bundle is None:
                raise AppError(
                    status_code=503,
                    code="CANDIDATE_GENERATION_ALLOCATION_RETRY_REQUIRED",
                    message="Candidate generation allocation remained contended; retry the immutable bundle",
                    detail={"bundle_id": bundle_id},
                ) from exc
            error = AppError(
                status_code=409,
                code="BUNDLE_ID_CONFLICT",
                message="immutable bundle identities conflict with existing intake records",
                detail={
                    "reasons": [
                        {
                            "field": "manifest",
                            "code": "conflict",
                            "message": "immutable bundle identity conflict",
                        }
                    ]
                },
            )
            await self._record_rejected_import(actor_identity=actor_identity, error=error, action="identity_conflict")
            raise error from exc
        return await self._projection(bundle_identity.stable_id)

    async def _validate_item(self, *, artifact: dict, artifact_sha256: str) -> tuple[dict | None, dict | None]:
        try:
            verified_artifact = await self._editorial_export_verifier.verify(artifact, artifact_sha256)
        except AppError as exc:
            return None, self._verifier_failure_reason(exc)
        except Exception:
            return None, self._failure_reason(
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                field="editorial_export",
                message="item is not a verified approved editorial export",
            )

        if not isinstance(verified_artifact, dict) or verified_artifact != artifact:
            return None, self._failure_reason(
                code="EDITORIAL_EXPORT_NOT_APPROVED",
                field="editorial_export",
                message="item does not match a retained approved editorial export",
            )

        reasons = self._editorial_item_reasons(verified_artifact)
        return verified_artifact, reasons[0] if reasons else None

    async def _record_rejected_import(
        self,
        *,
        actor_identity: str,
        error: AppError,
        action: str = "integrity_rejected",
    ) -> None:
        detail = error.detail if isinstance(error.detail, dict) else {}
        raw_reasons = detail.get("reasons")
        reasons = self._safe_import_audit_reasons(raw_reasons)
        attempt_identity = StableIdentity.new(StableIdentityKind.ADMISSION_ATTEMPT)
        self.session.add(
            CanonicalRecordModel(
                stable_id=attempt_identity.stable_id,
                identity_kind=attempt_identity.kind.value,
                identity_value=attempt_identity.value,
                state=BundleIntakeState.REJECTED.value,
                record_class=CanonicalRecordClass.IMMUTABLE.value,
                payload={
                    "schema": "reviewed_release_bundle_import_attempt/v1",
                    "outcome": BundleIntakeState.REJECTED.value,
                },
            )
        )
        self.session.add(
            CanonicalEventModel(
                aggregate_id=attempt_identity.stable_id,
                aggregate_kind=attempt_identity.kind.value,
                event_type=CanonicalEventType.CREATED.value,
                from_state=None,
                to_state=BundleIntakeState.REJECTED.value,
                payload={
                    "schema": "reviewed_release_bundle_import_audit/v1",
                    "action": action,
                    "reasons": reasons,
                },
                recorded_by=actor_identity,
            )
        )
        await self.session.commit()

    @staticmethod
    def _safe_import_audit_reasons(raw_reasons: object) -> list[dict[str, str]]:
        if not isinstance(raw_reasons, list):
            return [{"field": "manifest", "code": "invalid", "message": "bundle integrity validation failed"}]

        reasons: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for raw_reason in raw_reasons:
            if not isinstance(raw_reason, dict):
                continue
            raw_field = raw_reason.get("field")
            raw_code = raw_reason.get("code")
            field = (
                "items"
                if isinstance(raw_field, str) and raw_field.startswith("items")
                else raw_field
                if isinstance(raw_field, str) and raw_field in _AUDIT_REASON_FIELDS
                else "manifest"
            )
            code = raw_code if isinstance(raw_code, str) and raw_code in _AUDIT_REASON_CODES else "invalid"
            key = (field, code)
            if key in seen:
                continue
            seen.add(key)
            reasons.append(
                {
                    "field": field,
                    "code": code,
                    "message": "bundle integrity validation failed",
                }
            )
            if len(reasons) >= _MAX_IMPORT_AUDIT_REASONS:
                break
        return reasons or [{"field": "manifest", "code": "invalid", "message": "bundle integrity validation failed"}]

    async def _assert_item_identities_available(self, items: list[dict]) -> None:
        identities = [
            StableIdentity(StableIdentityKind.BUNDLE_ITEM, item["bundle_item_id"]).stable_id
            for item in items
        ]
        existing = (
            await self.session.execute(
                select(CanonicalRecordModel.stable_id).where(CanonicalRecordModel.stable_id.in_(identities))
            )
        ).scalars().all()
        if existing:
            raise AppError(
                status_code=409,
                code="BUNDLE_ITEM_ID_CONFLICT",
                message="bundle item identity is already bound to an immutable intake record",
                detail={
                    "reasons": [
                        {
                            "field": "items",
                            "code": "conflict",
                            "message": "bundle item identity is already bound",
                        }
                    ]
                },
            )

    @classmethod
    def _editorial_item_reasons(cls, artifact: dict) -> list[dict]:
        reasons: list[dict] = []
        entry = artifact.get("entry")
        approval = artifact.get("approval")
        roles = artifact.get("roles")
        if artifact.get("schema") != _EXPORT_SCHEMA:
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_EXPORT_SCHEMA_INVALID",
                    field="schema",
                    message=f"item must use {_EXPORT_SCHEMA}",
                )
            )
        if not isinstance(entry, dict):
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_METADATA_INVALID",
                    field="entry",
                    message="item must contain editorial entry metadata",
                )
            )
            return reasons
        if not isinstance(approval, dict) or approval.get("status") not in _APPROVED_EXPORT_STATUSES:
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_APPROVAL_REQUIRED",
                    field="approval.status",
                    message="item must contain an approved editorial export",
                )
            )
        if not isinstance(roles, dict) or not all(
            isinstance(roles.get(field), str) and roles[field].startswith("member:")
            for field in (
                "author_identity",
                "approving_reviewer_identity",
                "accountable_maintainer_identity",
            )
        ):
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_ROLE_SNAPSHOT_INVALID",
                    field="roles",
                    message="item must retain the approved editorial role snapshot",
                )
            )

        entry_identity = artifact.get("entry_identity")
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or entry_identity != f"entry:{entry_id}":
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_METADATA_INVALID",
                    field="entry.entry_id",
                    message="entry metadata must match the retained entry identity",
                )
            )
        if not isinstance(entry.get("title"), str) or not entry["title"].strip():
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_METADATA_INVALID",
                    field="entry.title",
                    message="entry title is required",
                )
            )
        if entry.get("assurance_level") not in _ASSURANCE_LEVELS:
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_ASSURANCE_INVALID",
                    field="entry.assurance_level",
                    message="entry assurance level is not recognized",
                )
            )
        if entry.get("coverage_position") is None:
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_METADATA_INVALID",
                    field="entry.coverage_position",
                    message="entry coverage position is required",
                )
            )
        if not isinstance(entry.get("body"), dict) or not entry["body"]:
            reasons.append(
                cls._failure_reason(
                    code="EDITORIAL_METADATA_INVALID",
                    field="entry.body",
                    message="entry body is required",
                )
            )

        reasons.extend(cls._chunk_strategy_reasons(entry.get("chunk_strategy")))
        reasons.extend(cls._acceptance_reasons(entry.get("acceptance_material")))
        reasons.extend(cls._source_reasons(entry.get("sources"), artifact.get("sources")))
        reasons.extend(cls._release_assurance_reasons(entry.get("assurance_level"), artifact.get("release_assurance_snapshot")))
        return reasons

    @classmethod
    def _chunk_strategy_reasons(cls, value: object) -> list[dict]:
        if not isinstance(value, dict):
            return [
                cls._failure_reason(
                    code="EDITORIAL_CHUNK_STRATEGY_INVALID",
                    field="entry.chunk_strategy",
                    message="entry must retain an accepted chunk strategy",
                )
            ]
        maximum = value.get("max_characters")
        overlap = value.get("overlap_characters")
        invalid = (
            not isinstance(value.get("strategy_id"), str)
            or not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or not isinstance(overlap, int)
            or isinstance(overlap, bool)
            or overlap < 0
            or overlap >= maximum
            or value.get("preserve_section_boundaries") is not True
        )
        if invalid:
            return [
                cls._failure_reason(
                    code="EDITORIAL_CHUNK_STRATEGY_INVALID",
                    field="entry.chunk_strategy",
                    message="entry chunk strategy is incomplete or unsafe",
                )
            ]
        return []

    @classmethod
    def _acceptance_reasons(cls, value: object) -> list[dict]:
        if not isinstance(value, dict):
            return [
                cls._failure_reason(
                    code="EDITORIAL_ACCEPTANCE_INVALID",
                    field="entry.acceptance_material",
                    message="entry must retain supported and Boundary acceptance material",
                )
            ]
        if not isinstance(value.get("supported_queries"), list) or not value["supported_queries"]:
            return [
                cls._failure_reason(
                    code="EDITORIAL_ACCEPTANCE_INVALID",
                    field="entry.acceptance_material.supported_queries",
                    message="entry must retain supported acceptance queries",
                )
            ]
        if not isinstance(value.get("boundary_queries"), list) or not value["boundary_queries"]:
            return [
                cls._failure_reason(
                    code="EDITORIAL_ACCEPTANCE_INVALID",
                    field="entry.acceptance_material.boundary_queries",
                    message="entry must retain Boundary acceptance queries",
                )
            ]
        return []

    @classmethod
    def _source_reasons(cls, definitions: object, snapshots: object) -> list[dict]:
        if not isinstance(definitions, list) or not definitions:
            return [
                cls._failure_reason(
                    code="EDITORIAL_SOURCE_INVALID",
                    field="entry.sources",
                    message="entry must retain source definitions",
                )
            ]
        if not isinstance(snapshots, list) or not snapshots:
            return [
                cls._failure_reason(
                    code="EDITORIAL_SOURCE_INVALID",
                    field="sources",
                    message="item must retain verified source authority snapshots",
                )
            ]
        snapshot_ids = {
            item.get("source_identity")
            for item in snapshots
            if isinstance(item, dict) and item.get("availability") == "verified_usable"
        }
        reasons: list[dict] = []
        for source in definitions:
            if not isinstance(source, dict):
                reasons.append(
                    cls._failure_reason(
                        code="EDITORIAL_SOURCE_INVALID",
                        field="entry.sources",
                        message="source definition is invalid",
                    )
                )
                continue
            source_id = source.get("source_id")
            source_identity = f"source:{source_id}" if isinstance(source_id, str) else ""
            if (
                source.get("access_scope") not in _ACCESS_SCOPES
                or source_identity not in snapshot_ids
            ):
                reasons.append(
                    cls._failure_reason(
                        code="EDITORIAL_SOURCE_UNAVAILABLE",
                        field="entry.sources",
                        message="source authority or access scope is not accepted",
                    )
                )
        return reasons

    @classmethod
    def _release_assurance_reasons(cls, assurance_level: object, snapshot: object) -> list[dict]:
        if assurance_level != "release_assured":
            return []
        if not isinstance(snapshot, dict) or snapshot.get("schema") != "editorial_release_assurance_snapshot/v1":
            return [
                cls._failure_reason(
                    code="EDITORIAL_ASSURANCE_INVALID",
                    field="release_assurance_snapshot",
                    message="Release-Assured items must retain a verified release assurance snapshot",
                )
            ]
        return []

    @staticmethod
    def _failure_reason(*, code: str, field: str, message: str) -> dict:
        return {"code": code, "field": field, "message": message}

    @classmethod
    def _verifier_failure_reason(cls, error: AppError) -> dict:
        detail = error.detail if isinstance(error.detail, dict) else {}
        reasons = detail.get("reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                if not isinstance(reason, dict):
                    continue
                field = reason.get("field")
                if not isinstance(field, str) or not field:
                    continue
                message = reason.get("message")
                return cls._failure_reason(
                    code=error.code or "EDITORIAL_EXPORT_NOT_APPROVED",
                    field=field,
                    message=message
                    if isinstance(message, str) and message
                    else error.message or "editorial export could not be verified",
                )
        return cls._failure_reason(
            code=error.code or "EDITORIAL_EXPORT_NOT_APPROVED",
            field="editorial_export",
            message=error.message or "editorial export could not be verified",
        )

    async def _next_generation(self, entry_identity: str) -> int:
        current = await self.session.scalar(
            select(func.max(CandidateBuildJob.requested_generation)).where(
                CandidateBuildJob.entry_identity == entry_identity
            )
        )
        return int(current or 0) + 1

    async def _supersede_older_generations(
        self,
        *,
        entry_identity: str,
        requested_generation: int,
        superseding_bundle_id: str,
    ) -> list[CanonicalEventModel]:
        result = await self.session.execute(
            select(CandidateBuildJob).where(
                CandidateBuildJob.entry_identity == entry_identity,
                CandidateBuildJob.requested_generation < requested_generation,
                CandidateBuildJob.status.in_({"queued", "running", "interrupted_retryable", "candidate_ready"}),
            ).with_for_update()
        )
        events: list[CanonicalEventModel] = []
        affected_bundle_ids: set[str] = set()
        for job in result.scalars().all():
            previous_stage = job.stage
            previous_status = job.status
            affected_bundle_ids.add(job.bundle_id)
            job.status = "superseded"
            job.terminal_state = "superseded"
            job.derived_cleanup_pending = previous_status in {"running", "interrupted_retryable"}
            job.failure_reason = {
                "code": "CANDIDATE_SUPERSEDED",
                "stage": previous_stage,
                "message": "a newer immutable bundle generation was admitted",
                "superseding_bundle_id": superseding_bundle_id,
            }
            job.completed_at = job.completed_at or datetime.now(UTC)
            job.heartbeat_at = None
            job.lease_expires_at = None
            job.lease_owner = None
            job.allowed_next_action = "reconcile_derived_data" if job.derived_cleanup_pending else "none"
            events.append(
                CanonicalEventModel(
                    aggregate_id=f"build_generation:{job.id}",
                    aggregate_kind=StableIdentityKind.BUILD_GENERATION.value,
                    event_type=CanonicalEventType.SUPERSEDED.value,
                    from_state=previous_stage,
                    to_state="superseded",
                    payload=candidate_job_event_payload(
                        job,
                        action="superseded",
                        extra={
                            "previous_status": previous_status,
                            "superseding_bundle_id": superseding_bundle_id,
                        },
                    ),
                )
            )
            if job.candidate_id is not None:
                events.append(
                    CanonicalEventModel(
                        aggregate_id=job.candidate_id,
                        aggregate_kind=StableIdentityKind.CANDIDATE.value,
                        event_type=CanonicalEventType.SUPERSEDED.value,
                        from_state="candidate_ready",
                        to_state="superseded",
                        payload={
                            "schema": "candidate_build_candidate_event/v1",
                            "action": "superseded",
                            "build_generation_id": f"build_generation:{job.id}",
                            "superseding_bundle_id": superseding_bundle_id,
                        },
                    )
                )
        for bundle_id in affected_bundle_ids:
            await complete_bundle_when_candidate_work_is_finished(self.session, bundle_id)
        return events

    @staticmethod
    def _document_identity(entry_identity: str) -> str:
        entry = StableIdentity.from_stable_id(entry_identity)
        return f"runtime-document:{entry.value}"

    @staticmethod
    def _bundle_events(
        bundle_id: str,
        actor_identity: str,
        *,
        has_candidate_work: bool,
        has_rejected_items: bool,
    ) -> list[CanonicalEventModel]:
        events = [
            CanonicalEventModel(
                aggregate_id=bundle_id,
                aggregate_kind=StableIdentityKind.BUNDLE.value,
                event_type=CanonicalEventType.CREATED.value,
                from_state=None,
                to_state=BundleIntakeState.RECEIVED.value,
                payload={"schema": "reviewed_release_bundle_event/v1", "action": "received"},
                recorded_by=actor_identity,
            ),
            CanonicalEventModel(
                aggregate_id=bundle_id,
                aggregate_kind=StableIdentityKind.BUNDLE.value,
                event_type=CanonicalEventType.STATE_CHANGED.value,
                from_state=BundleIntakeState.RECEIVED.value,
                to_state=validate_transition(
                    BundleIntakeState,
                    BundleIntakeState.RECEIVED,
                    BundleIntakeState.VALIDATING,
                ).value,
                payload={"schema": "reviewed_release_bundle_event/v1", "action": "validating"},
                recorded_by=actor_identity,
            ),
            CanonicalEventModel(
                aggregate_id=bundle_id,
                aggregate_kind=StableIdentityKind.BUNDLE.value,
                event_type=CanonicalEventType.STATE_CHANGED.value,
                from_state=BundleIntakeState.VALIDATING.value,
                to_state=validate_transition(
                    BundleIntakeState,
                    BundleIntakeState.VALIDATING,
                    BundleIntakeState.VALIDATED,
                ).value,
                payload={"schema": "reviewed_release_bundle_event/v1", "action": "validated"},
                recorded_by=actor_identity,
            ),
        ]
        events.append(
            CanonicalEventModel(
                aggregate_id=bundle_id,
                aggregate_kind=StableIdentityKind.BUNDLE.value,
                event_type=CanonicalEventType.STATE_CHANGED.value,
                from_state=BundleIntakeState.VALIDATED.value,
                to_state=validate_transition(
                    BundleIntakeState,
                    BundleIntakeState.VALIDATED,
                    BundleIntakeState.PROCESSING,
                ).value,
                payload={"schema": "reviewed_release_bundle_event/v1", "action": "candidate_work_started"},
                recorded_by=actor_identity,
            )
        )
        if not has_candidate_work:
            completion_state = (
                BundleIntakeState.COMPLETED_WITH_REJECTIONS
                if has_rejected_items
                else BundleIntakeState.COMPLETED
            )
            events.append(
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
                            "item_plan_completed_with_rejections"
                            if completion_state is BundleIntakeState.COMPLETED_WITH_REJECTIONS
                            else "item_plan_completed"
                        ),
                    },
                    recorded_by=actor_identity,
                )
            )
        return events

    @classmethod
    def _validate_bundle_integrity(cls, manifest: object) -> dict:
        reasons: list[dict[str, str]] = []
        if not isinstance(manifest, dict):
            cls._raise_integrity_rejected(
                [{"field": "manifest", "code": "invalid", "message": "manifest must be an object"}]
            )

        if any(not isinstance(key, str) for key in manifest):
            reasons.append(
                {
                    "field": "manifest",
                    "code": "invalid",
                    "message": "manifest field names must be strings",
                }
            )

        findings = editorial_export_safety_findings(manifest)
        if findings:
            reasons.extend(
                {
                    "field": finding,
                    "code": "unsafe",
                    "message": "bundle cannot contain credentials or automatic-publication instructions",
                }
                for finding in findings
            )

        if manifest.get("schema") != _BUNDLE_SCHEMA:
            reasons.append(
                {
                    "field": "schema",
                    "code": "unsupported",
                    "message": f"schema must be {_BUNDLE_SCHEMA}",
                }
            )
        if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
            reasons.append(
                {
                    "field": "schema_version",
                    "code": "unsupported",
                    "message": "schema_version must be 1",
                }
            )

        bundle_id = manifest.get("bundle_id")
        if not isinstance(bundle_id, str):
            reasons.append(
                {
                    "field": "bundle_id",
                    "code": "invalid",
                    "message": "bundle_id must be a stable identity value",
                }
            )
        else:
            try:
                bundle_identity = StableIdentity(StableIdentityKind.BUNDLE, bundle_id)
            except ValueError:
                reasons.append(
                    {
                        "field": "bundle_id",
                        "code": "invalid",
                        "message": "bundle_id must be a stable identity value",
                    }
                )
            else:
                if bundle_identity.value != bundle_id:
                    reasons.append(
                        {
                            "field": "bundle_id",
                            "code": "invalid",
                            "message": "bundle_id must be a stable identity value",
                        }
                    )

        source_revision = manifest.get("editorial_source_revision")
        if not isinstance(source_revision, str) or _SHA256.fullmatch(source_revision) is None:
            reasons.append(
                {
                    "field": "editorial_source_revision",
                    "code": "invalid",
                    "message": "editorial_source_revision must be a lowercase SHA-256 revision hash",
                }
            )

        exported_at = manifest.get("exported_at")
        try:
            exported_datetime = (
                datetime.fromisoformat(exported_at.replace("Z", "+00:00"))
                if isinstance(exported_at, str) and exported_at.endswith("Z")
                else None
            )
        except ValueError:
            exported_datetime = None
        if (
            exported_datetime is None
            or exported_datetime.tzinfo is None
            or exported_datetime.utcoffset() != timedelta(0)
        ):
            reasons.append(
                {
                    "field": "exported_at",
                    "code": "invalid",
                    "message": "exported_at must be an ISO-8601 UTC timestamp",
                }
            )

        bundle_sha256 = manifest.get("bundle_sha256")
        unsigned_manifest = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
        if not isinstance(bundle_sha256, str) or _SHA256.fullmatch(bundle_sha256) is None:
            reasons.append(
                {
                    "field": "bundle_sha256",
                    "code": "invalid",
                    "message": "bundle_sha256 must be a SHA-256 hash",
                }
            )
        else:
            try:
                bundle_matches = bundle_sha256 == cls._sha256(unsigned_manifest)
            except (TypeError, ValueError):
                bundle_matches = False
            if not bundle_matches:
                reasons.append(
                    {
                        "field": "bundle_sha256",
                        "code": "mismatch",
                        "message": "bundle_sha256 does not match the immutable manifest",
                    }
                )

        items = manifest.get("items")
        if not isinstance(items, list) or not items:
            reasons.append(
                {
                    "field": "items",
                    "code": "required",
                    "message": "items must be a non-empty list",
                }
            )
        else:
            seen_item_ids: set[str] = set()
            seen_entry_identities: set[str] = set()
            for index, item in enumerate(items):
                prefix = f"items[{index}]"
                if not isinstance(item, dict):
                    reasons.append(
                        {
                            "field": prefix,
                            "code": "invalid",
                            "message": "bundle item must be an object",
                        }
                    )
                    continue
                item_id = item.get("bundle_item_id")
                if not isinstance(item_id, str):
                    reasons.append(
                        {
                            "field": f"{prefix}.bundle_item_id",
                            "code": "invalid",
                            "message": "bundle_item_id must be a stable identity value",
                        }
                    )
                    item_identity = None
                else:
                    try:
                        item_identity = StableIdentity(StableIdentityKind.BUNDLE_ITEM, item_id)
                    except ValueError:
                        reasons.append(
                            {
                                "field": f"{prefix}.bundle_item_id",
                                "code": "invalid",
                                "message": "bundle_item_id must be a stable identity value",
                            }
                        )
                        item_identity = None
                    else:
                        if item_identity.value != item_id:
                            reasons.append(
                                {
                                    "field": f"{prefix}.bundle_item_id",
                                    "code": "invalid",
                                    "message": "bundle_item_id must be a stable identity value",
                                }
                            )
                            item_identity = None
                if item_identity is not None:
                    if item_identity.value in seen_item_ids:
                        reasons.append(
                            {
                                "field": f"{prefix}.bundle_item_id",
                                "code": "duplicate",
                                "message": "bundle item identity must be unique",
                            }
                        )
                    seen_item_ids.add(item_identity.value)

                operation = item.get("operation")
                if not isinstance(operation, str) or operation not in _ITEM_OPERATIONS:
                    reasons.append(
                        {
                            "field": f"{prefix}.operation",
                            "code": "unsupported",
                            "message": "operation must be create, replace, no_op, or proposed_withdrawal",
                        }
                    )

                artifact = item.get("artifact")
                artifact_sha256 = item.get("artifact_sha256")
                bundle_item_sha256 = item.get("bundle_item_sha256")
                if not isinstance(artifact, dict):
                    reasons.append(
                        {
                            "field": f"{prefix}.artifact",
                            "code": "invalid",
                            "message": "artifact must be an object",
                        }
                    )
                    continue
                if not isinstance(artifact_sha256, str) or _SHA256.fullmatch(artifact_sha256) is None:
                    reasons.append(
                        {
                            "field": f"{prefix}.artifact_sha256",
                            "code": "invalid",
                            "message": "artifact_sha256 must be a SHA-256 hash",
                        }
                    )
                else:
                    try:
                        artifact_matches = artifact_sha256 == cls._sha256(artifact)
                    except (TypeError, ValueError):
                        artifact_matches = False
                    if not artifact_matches:
                        reasons.append(
                            {
                                "field": f"{prefix}.artifact_sha256",
                                "code": "mismatch",
                                "message": "artifact_sha256 does not match the supplied artifact",
                            }
                        )

                expected_item_payload = {
                    "bundle_item_id": item_id,
                    "operation": operation,
                    "artifact_sha256": artifact_sha256,
                    "artifact": artifact,
                }
                if not isinstance(bundle_item_sha256, str) or _SHA256.fullmatch(bundle_item_sha256) is None:
                    reasons.append(
                        {
                            "field": f"{prefix}.bundle_item_sha256",
                            "code": "invalid",
                            "message": "bundle_item_sha256 must be a SHA-256 hash",
                        }
                    )
                else:
                    try:
                        item_matches = bundle_item_sha256 == cls._sha256(expected_item_payload)
                    except (TypeError, ValueError):
                        item_matches = False
                    if not item_matches:
                        reasons.append(
                            {
                                "field": f"{prefix}.bundle_item_sha256",
                                "code": "mismatch",
                                "message": "bundle_item_sha256 does not match the immutable bundle item",
                            }
                        )

                if artifact.get("revision_sha256") != source_revision:
                    reasons.append(
                        {
                            "field": f"{prefix}.artifact.revision_sha256",
                            "code": "mismatch",
                            "message": "every approved export must match the bundle editorial_source_revision",
                        }
                    )

                entry_identity = artifact.get("entry_identity")
                if not isinstance(entry_identity, str):
                    entry = None
                else:
                    try:
                        entry = StableIdentity.from_stable_id(entry_identity)
                    except ValueError:
                        entry = None
                if entry is None or entry.kind is not StableIdentityKind.ENTRY or entry.stable_id != entry_identity:
                    reasons.append(
                        {
                            "field": f"{prefix}.artifact.entry_identity",
                            "code": "invalid",
                            "message": "artifact must name an entry identity",
                        }
                    )
                elif entry.stable_id in seen_entry_identities:
                    reasons.append(
                        {
                            "field": f"{prefix}.artifact.entry_identity",
                            "code": "conflict",
                            "message": "an entry may appear in only one bundle operation",
                        }
                    )
                else:
                    seen_entry_identities.add(entry.stable_id)

        if reasons:
            cls._raise_integrity_rejected(reasons)
        return manifest

    @staticmethod
    def _raise_integrity_rejected(reasons: list[dict[str, str]]) -> NoReturn:
        raise AppError(
            status_code=422,
            code="BUNDLE_INTEGRITY_REJECTED",
            message="reviewed release bundle failed integrity validation",
            detail={"reasons": reasons},
        )

    async def _projection(self, bundle_id: str) -> dict:
        bundle = await self.session.get(CanonicalRecordModel, bundle_id)
        assert bundle is not None
        items_result = await self.session.execute(
            select(CanonicalRecordModel)
            .where(CanonicalRecordModel.identity_kind == StableIdentityKind.BUNDLE_ITEM.value)
            .order_by(CanonicalRecordModel.stable_id.asc())
        )
        items = sorted(
            (
                item
                for item in items_result.scalars()
                if isinstance(item.payload, dict) and item.payload.get("bundle_id") == bundle_id
            ),
            key=lambda item: int(item.payload.get("position", 0)),
        )
        result_items: list[dict[str, object]] = []
        for item in items:
            payload = item.payload
            result: dict[str, object] = {
                "bundle_item_id": item.identity_value,
                "entry_identity": payload["entry_identity"],
                "operation": payload["operation"],
                "state": item.state,
                "artifact_sha256": payload["artifact_sha256"],
                "bundle_item_sha256": payload["bundle_item_sha256"],
                "allowed_next_action": payload["allowed_next_action"],
            }
            failure_reason = payload.get("failure_reason")
            if isinstance(failure_reason, dict):
                result["failure_reason"] = failure_reason
            job_result = await self.session.execute(
                select(CandidateBuildJob).where(CandidateBuildJob.bundle_item_id == item.stable_id)
            )
            job = job_result.scalar_one_or_none()
            if job is not None:
                result["job_id"] = job.id
            result_items.append(result)
        return {
            "bundle_id": bundle.identity_value,
            "state": await bundle_intake_state(self.session, bundle_id),
            "schema_version": bundle.payload["schema_version"],
            "editorial_source_revision": bundle.payload["editorial_source_revision"],
            "exported_at": bundle.payload["exported_at"],
            "bundle_sha256": bundle.payload["bundle_sha256"],
            "items": result_items,
        }

    async def get_bundle(self, bundle_id: str) -> dict:
        try:
            identity = StableIdentity(StableIdentityKind.BUNDLE, bundle_id)
        except ValueError as exc:
            raise AppError(
                status_code=404,
                code="RESOURCE_NOT_FOUND",
                message="reviewed release bundle not found",
                detail={"bundle_id": bundle_id},
            ) from exc
        bundle = await self.session.get(CanonicalRecordModel, identity.stable_id)
        if bundle is None or bundle.identity_kind != StableIdentityKind.BUNDLE.value:
            raise AppError(
                status_code=404,
                code="RESOURCE_NOT_FOUND",
                message="reviewed release bundle not found",
                detail={"bundle_id": bundle_id},
            )
        return await self._projection(identity.stable_id)

    async def list_bundles(self) -> list[dict]:
        result = await self.session.execute(
            select(CanonicalRecordModel)
            .where(CanonicalRecordModel.identity_kind == StableIdentityKind.BUNDLE.value)
            .order_by(CanonicalRecordModel.created_at.desc(), CanonicalRecordModel.stable_id.desc())
        )
        return [await self._projection(bundle.stable_id) for bundle in result.scalars().all()]

    def _effective_embedding_configuration(self) -> dict[str, object]:
        return candidate_embedding_configuration(self._settings)

    @staticmethod
    def _sha256(value: object) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def candidate_embedding_configuration(settings: Settings) -> dict[str, object]:
    contract = DenseEmbeddingContract.from_settings(settings)
    configuration = {
        "schema": "candidate_embedding_configuration/v1",
        "active": contract.active,
        "embedding_model": contract.model,
        "dense_embedding_dim": contract.dimension,
    }
    encoded = json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**configuration, "fingerprint": hashlib.sha256(encoded).hexdigest()}
