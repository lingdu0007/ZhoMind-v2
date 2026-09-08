from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.config import Settings, get_settings
from app.common.exceptions import AppError
from app.contracts.canonical import (
    CanonicalEventType,
    CanonicalRecordClass,
    StableIdentity,
    StableIdentityKind,
)
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.document import Document, DocumentChunk
from app.operations.limits import MAX_PUBLISHED_SOURCES
from app.rag.evidence_sufficiency import AnswerEvidenceSet, QueryConditionSet, decide_answer_evidence
from app.reviewed_bundles.candidate_metadata import (
    candidate_frozen_chunk_metadata,
    candidate_frozen_published_chunk_metadata,
)
from app.reviewed_bundles.inputs import (
    FrozenCandidateBuildInput,
    load_frozen_candidate_build_input,
    runtime_document_identity,
)
from app.reviewed_bundles.models import (
    CandidateBuildChunk,
    CandidateBuildJob,
    CandidatePublicationConfirmation,
    PublishedKnowledgePointer,
    PublishedKnowledgeVersion,
)
from app.reviewed_bundles.service import EditorialExportVerifier, candidate_embedding_configuration

_APPROVED_EDITORIAL_STATUSES = frozenset({"approved", "lightweight_accepted"})
_GOVERNING_SECTION = "recommendation_or_reviewed_branches"
_CONFIRMATION_LEASE_SECONDS = 300
_COVERAGE_TOKEN = re.compile(r"[\w-]+", re.UNICODE)
_COVERAGE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "applies",
        "are",
        "can",
        "does",
        "for",
        "how",
        "is",
        "it",
        "of",
        "or",
        "the",
        "to",
        "what",
        "which",
    }
)


@dataclass(frozen=True)
class CandidatePublicationBinding:
    candidate: CanonicalRecordModel
    job: CandidateBuildJob
    frozen_input: FrozenCandidateBuildInput
    chunks: tuple[CandidateBuildChunk, ...]

    @property
    def configuration_identity(self) -> str:
        configuration_hash = canonical_json_sha256(self.frozen_input.embedding_configuration)
        return StableIdentity(
            StableIdentityKind.CONFIGURATION,
            f"candidate-build-{configuration_hash}",
        ).stable_id


@dataclass(frozen=True)
class CandidatePublicationSnapshot:
    """Immutable inspection and acceptance facts selected for one publication."""

    inspection_record_identity: str | None
    acceptance_record_identity: str | None
    replaces_published_knowledge_version_identity: str | None

    @property
    def is_complete(self) -> bool:
        return self.inspection_record_identity is not None and self.acceptance_record_identity is not None


def _candidate_build_chunk_metadata(
    binding: CandidatePublicationBinding,
    *,
    section_id: str,
) -> dict[str, Any]:
    return candidate_frozen_chunk_metadata(
        artifact=binding.frozen_input.artifact,
        entry_identity=binding.frozen_input.entry_identity,
        chunk_strategy=binding.frozen_input.chunk_strategy,
        section_id=section_id,
    )


def _candidate_published_chunk_metadata(
    binding: CandidatePublicationBinding,
    *,
    section_id: str,
    publication_identity: str,
    source_identity: str | None = None,
    include_source_evidence_projections: bool = True,
) -> dict[str, Any]:
    return candidate_frozen_published_chunk_metadata(
        artifact=binding.frozen_input.artifact,
        entry_identity=binding.frozen_input.entry_identity,
        chunk_strategy=binding.frozen_input.chunk_strategy,
        section_id=section_id,
        publication_identity=publication_identity,
        source_identity=source_identity,
        include_source_evidence_projections=include_source_evidence_projections,
    )


def _candidate_decision_query(entry: dict[str, Any]) -> str:
    body = entry.get("body")
    decision_query = body.get("decision_query") if isinstance(body, dict) else None
    if not isinstance(decision_query, str) or not decision_query.strip():
        raise ValueError("Candidate artifact decision query is invalid")
    return decision_query.strip()


class CandidateAcceptanceExecutionAdapter:
    """Run the closed evidence contract in an admin-only Candidate acceptance context."""

    def __init__(self, binding: CandidatePublicationBinding) -> None:
        self._binding = binding

    def run(self) -> dict[str, Any]:
        material = self._acceptance_material()
        supported_query = self._query(material.get("supported_queries"), expected_outcome="supported")
        boundary_query = self._query(
            material.get("boundary_queries"),
            expected_outcome="insufficient_evidence",
        )
        supported = self._supported_result(
            supported_query["query"],
            self._query_conditions(supported_query),
        )
        boundary = self._boundary_result(
            boundary_query["query"],
            self._query_conditions(boundary_query),
        )
        return {
            "schema": "candidate_acceptance_result/v1",
            "candidate_id": self._binding.candidate.stable_id,
            "configuration_identity": self._binding.configuration_identity,
            "supported": supported,
            "boundary": boundary,
        }

    def _acceptance_material(self) -> dict[str, Any]:
        entry = self._binding.frozen_input.artifact.get("entry")
        material = entry.get("acceptance_material") if isinstance(entry, dict) else None
        if not isinstance(material, dict):
            raise self._acceptance_error("Candidate has no approved acceptance material")
        return material

    @staticmethod
    def _query(raw_queries: object, *, expected_outcome: str) -> dict[str, Any]:
        if not isinstance(raw_queries, list):
            raise CandidateAcceptanceExecutionAdapter._acceptance_error(
                "Candidate acceptance material is incomplete"
            )
        for raw_query in raw_queries:
            if (
                isinstance(raw_query, dict)
                and raw_query.get("expected_outcome") == expected_outcome
                and isinstance(raw_query.get("query"), str)
                and raw_query["query"].strip()
            ):
                return {**raw_query, "query": raw_query["query"].strip()}
        raise CandidateAcceptanceExecutionAdapter._acceptance_error(
            "Candidate acceptance material does not include the required query"
        )

    @staticmethod
    def _query_conditions(query: dict[str, Any]) -> QueryConditionSet:
        raw_query = query.get("query")
        assert isinstance(raw_query, str)
        normalized_query = raw_query.strip()
        raw_conditions = query.get("query_conditions")
        if raw_conditions is None:
            return QueryConditionSet.from_question(normalized_query)
        if not isinstance(raw_conditions, list):
            raise CandidateAcceptanceExecutionAdapter._acceptance_error(
                "Candidate acceptance query conditions must be a list"
            )
        try:
            return QueryConditionSet.from_records(
                normalized_question=normalized_query,
                records=raw_conditions,
            )
        except ValueError as exc:
            raise CandidateAcceptanceExecutionAdapter._acceptance_error(
                "Candidate acceptance query conditions are invalid"
            ) from exc

    def _supported_result(self, query: str, conditions: QueryConditionSet) -> dict[str, Any]:
        decision = self._decision(query, conditions)
        if not decision.is_sufficient or decision.evidence_set is None:
            raise self._acceptance_error(
                f"Supported Query cannot satisfy the Candidate-bound closed evidence contract: {decision.reason}"
            )
        evidence_set = self._candidate_bound_evidence_set(decision.evidence_set, conditions)
        return {
            "outcome": "evidence_gated_answer",
            "query": query,
            "query_condition_set_identity": conditions.identity,
            "expected_governing_entry_identity": self._binding.frozen_input.entry_identity,
            "expected_governing_section_id": _GOVERNING_SECTION,
            "answer_evidence_set": evidence_set.to_record(),
            "citation_markers": [citation.marker for citation in evidence_set.citations],
            "provider_call_count": 0,
        }

    def _boundary_result(self, query: str, conditions: QueryConditionSet) -> dict[str, Any]:
        decision = self._decision(query, conditions)
        if decision.is_sufficient:
            raise self._acceptance_error("Boundary Query is covered by the Candidate closed evidence contract")
        assert decision.insufficient_reply is not None
        return {
            **decision.insufficient_reply.to_record(),
            "citation_markers": [],
            "provider_call_count": 0,
        }

    def _decision(self, query: str, conditions: QueryConditionSet):
        try:
            return decide_answer_evidence(
                normalized_question=query,
                query_conditions=conditions,
                candidates=tuple(
                    candidate
                    for chunk in self._binding.chunks
                    for candidate in self._acceptance_candidates(chunk)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise self._acceptance_error("Candidate closed evidence input is invalid") from exc

    def _acceptance_candidates(self, chunk: CandidateBuildChunk) -> tuple[dict[str, Any], ...]:
        metadata = dict(chunk.chunk_metadata) if isinstance(chunk.chunk_metadata, dict) else None
        section_id = metadata.get("section_id") if metadata is not None else None
        if not isinstance(section_id, str) or not section_id:
            raise self._acceptance_error("Candidate chunk metadata is invalid")
        try:
            publication_identity = CandidatePublicationService._published_version_identity(self._binding).stable_id
            expected_metadata = _candidate_build_chunk_metadata(self._binding, section_id=section_id)
        except ValueError as exc:
            raise self._acceptance_error("Candidate frozen citation binding is invalid") from exc
        relationships = expected_metadata.get("source_relationships")
        if not isinstance(relationships, list) or not relationships:
            raise self._acceptance_error("Candidate source relationships are invalid")
        candidates: list[dict[str, Any]] = []
        for relationship in relationships:
            source_identity = relationship.get("source_identity") if isinstance(relationship, dict) else None
            if not isinstance(source_identity, str) or not source_identity:
                raise self._acceptance_error("Candidate source relationships are invalid")
            acceptance_document_identity = (
                f"{self._binding.frozen_input.document_identity}@candidate-acceptance:{source_identity}"
            )
            try:
                projected_metadata = _candidate_published_chunk_metadata(
                    self._binding,
                    section_id=section_id,
                    publication_identity=publication_identity,
                    source_identity=source_identity,
                    include_source_evidence_projections=False,
                )
            except ValueError as exc:
                raise self._acceptance_error("Candidate frozen citation binding is invalid") from exc
            candidates.append(
                {
                    "chunk_id": f"{chunk.id}@{source_identity}",
                    "document_id": acceptance_document_identity,
                    "generation": chunk.generation,
                    "chunk_index": chunk.chunk_index,
                    "content_sha256": chunk.content_sha256,
                    "content_preview": chunk.content,
                    "content_length": len(chunk.content),
                    "metadata": projected_metadata,
                    "retrieval_source": "candidate_acceptance",
                    "entry_id": projected_metadata["entry_id"],
                    "entry_identity": projected_metadata["entry_identity"],
                    "editorial_revision_identity": projected_metadata["editorial_revision_identity"],
                    "publication_identity": publication_identity,
                    "publication_version": publication_identity,
                    "section_id": section_id,
                    "section_identity": f"{self._binding.frozen_input.entry_identity}#{section_id}",
                    "decision_query": projected_metadata["decision_query"],
                    "source_relationships": projected_metadata["source_relationships"],
                    "assurance_level": projected_metadata["assurance_level"],
                    "applicability_conditions": projected_metadata["applicability_conditions"],
                    "non_applicability_conditions": projected_metadata["non_applicability_conditions"],
                    "freshness_triggers": projected_metadata["freshness_triggers"],
                    "access_scope": projected_metadata["source_access_scope"],
                    "candidate_acceptance_source_identity": source_identity,
                    "chunk_identity": {
                        "document_id": acceptance_document_identity,
                        "generation": chunk.generation,
                        "chunk_index": chunk.chunk_index,
                        "content_sha256": chunk.content_sha256,
                    },
                    "answer_evidence_eligible": True,
                }
            )
        return tuple(candidates)

    def _candidate_bound_evidence_set(
        self,
        evidence_set: AnswerEvidenceSet,
        conditions: QueryConditionSet,
    ) -> AnswerEvidenceSet:
        chunks_by_acceptance_id = {
            candidate["chunk_id"]: (chunk, candidate["candidate_acceptance_source_identity"])
            for chunk in self._binding.chunks
            for candidate in self._acceptance_candidates(chunk)
        }
        bindings: list[dict[str, Any]] = []
        for evidence in evidence_set.items:
            chunk_binding = chunks_by_acceptance_id.get(evidence.source_id)
            if chunk_binding is None:
                raise self._acceptance_error("Candidate evidence refers to an unknown chunk")
            chunk, source_identity = chunk_binding
            bindings.append(
                {
                    "candidate_identity": self._binding.candidate.stable_id,
                    "build_generation_identity": f"build_generation:{self._binding.job.id}",
                    "entry_identity": self._binding.frozen_input.entry_identity,
                    "editorial_revision_identity": self._binding.frozen_input.artifact["editorial_revision_identity"],
                    "section_id": dict(evidence.metadata_items).get("section_id"),
                    "chunk_identity": {
                        "candidate_id": self._binding.candidate.stable_id,
                        "generation": chunk.generation,
                        "attempt": chunk.attempt,
                        "chunk_index": chunk.chunk_index,
                        "content_sha256": chunk.content_sha256,
                        "source_identity": source_identity,
                    },
                    "snapshot_id": evidence.snapshot_id,
                }
            )
        governing_snapshot_id = evidence_set.governing_citation.snapshot_id
        governing = next((item for item in evidence_set.items if item.snapshot_id == governing_snapshot_id), None)
        if governing is None:
            raise self._acceptance_error("Candidate governing evidence binding is invalid")
        return AnswerEvidenceSet.freeze(
            query_conditions=conditions,
            items=evidence_set.items,
            governing_item=governing,
            item_identity_bindings=tuple(bindings),
            item_identities=tuple(canonical_json_sha256(binding) for binding in bindings),
        )

    @staticmethod
    def _acceptance_error(message: str) -> AppError:
        return AppError(
            status_code=409,
            code="CANDIDATE_ACCEPTANCE_MATERIAL_INVALID",
            message=message,
        )


class CandidatePublicationService:
    """Inspect exact Candidate Build bindings before later acceptance and publication."""

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

    async def inspect(self, candidate_id: str, *, actor_identity: str) -> dict[str, Any]:
        binding = await self._load_binding(candidate_id)
        self._assert_current_embedding_configuration(binding)
        await self._assert_current_editorial_approval(binding)
        await self._ensure_configuration_record(binding)
        current = await self._current_version(
            await self._current_pointer(binding.frozen_input.entry_identity)
        )

        inspection_identity = StableIdentity.new(StableIdentityKind.EVENT)
        inspection_payload = self._inspection_payload(
            binding,
            inspection_identity=inspection_identity.stable_id,
            actor_identity=actor_identity,
            replaces_published_knowledge_version_identity=(current.id if current is not None else None),
        )
        self.session.add(
            CanonicalRecordModel(
                stable_id=inspection_identity.stable_id,
                identity_kind=inspection_identity.kind.value,
                identity_value=inspection_identity.value,
                state="recorded",
                record_class=CanonicalRecordClass.IMMUTABLE.value,
                payload=inspection_payload,
            )
        )
        self.session.add(
            CanonicalEventModel(
                aggregate_id=binding.candidate.stable_id,
                aggregate_kind=StableIdentityKind.CANDIDATE.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state=binding.candidate.state,
                to_state=binding.candidate.state,
                payload={
                    "schema": "candidate_inspection_event/v1",
                    "inspection_record_identity": inspection_identity.stable_id,
                    "candidate_id": binding.candidate.stable_id,
                    "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
                    "configuration_identity": binding.configuration_identity,
                    "replaces_published_knowledge_version_identity": (
                        current.id if current is not None else None
                    ),
                },
                recorded_by=actor_identity,
            )
        )
        binding.job.allowed_next_action = "await_candidate_acceptance"
        await self.session.commit()
        return await self._projection(binding, inspection_payload)

    async def get_inspection(self, candidate_id: str) -> dict[str, Any]:
        binding = await self._load_binding(candidate_id, require_latest_generation=False)
        inspection = await self._latest_valid_inspection(binding)
        return await self._projection(binding, inspection)

    async def accept(self, candidate_id: str, *, actor_identity: str) -> dict[str, Any]:
        binding = await self._load_binding(candidate_id)
        self._assert_current_embedding_configuration(binding)
        await self._assert_current_editorial_approval(binding)
        inspection = await self._latest_valid_inspection(binding)
        if inspection is None:
            raise AppError(
                status_code=409,
                code="CANDIDATE_INSPECTION_REQUIRED",
                message="Candidate requires an exact durable inspection before acceptance",
                detail={"candidate_id": candidate_id},
            )
        acceptance_result = CandidateAcceptanceExecutionAdapter(binding).run()
        acceptance_identity = StableIdentity.new(StableIdentityKind.EVENT)
        acceptance_payload = {
            "schema": "candidate_acceptance/v1",
            "acceptance_record_identity": acceptance_identity.stable_id,
            "candidate_id": binding.candidate.stable_id,
            "inspection_record_identity": inspection["inspection_record_identity"],
            "build_generation_id": f"build_generation:{binding.job.id}",
            "entry_identity": binding.frozen_input.entry_identity,
            "bundle_sha256": binding.frozen_input.bundle_sha256,
            "bundle_item_sha256": binding.frozen_input.bundle_item_sha256,
            "input_sha256": binding.frozen_input.input_sha256,
            "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
            "configuration_identity": binding.configuration_identity,
            "configuration": binding.frozen_input.embedding_configuration,
            "replaces_published_knowledge_version_identity": inspection[
                "replaces_published_knowledge_version_identity"
            ],
            "result": acceptance_result,
            "accepted_by": actor_identity,
        }
        self.session.add(
            CanonicalRecordModel(
                stable_id=acceptance_identity.stable_id,
                identity_kind=acceptance_identity.kind.value,
                identity_value=acceptance_identity.value,
                state="passed",
                record_class=CanonicalRecordClass.IMMUTABLE.value,
                payload=acceptance_payload,
            )
        )
        self.session.add(
            CanonicalEventModel(
                aggregate_id=binding.candidate.stable_id,
                aggregate_kind=StableIdentityKind.CANDIDATE.value,
                event_type=CanonicalEventType.STATUS_CHANGED.value,
                from_state=binding.candidate.state,
                to_state=binding.candidate.state,
                payload={
                    "schema": "candidate_acceptance_event/v1",
                    "acceptance_record_identity": acceptance_identity.stable_id,
                    "candidate_id": binding.candidate.stable_id,
                    "inspection_record_identity": inspection["inspection_record_identity"],
                    "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
                    "configuration_identity": binding.configuration_identity,
                    "replaces_published_knowledge_version_identity": inspection[
                        "replaces_published_knowledge_version_identity"
                    ],
                },
                recorded_by=actor_identity,
            )
        )
        binding.job.allowed_next_action = "await_explicit_publication"
        await self.session.commit()
        return {
            "record_identity": acceptance_identity.stable_id,
            "candidate_id": binding.candidate.stable_id,
            "replaces_published_knowledge_version_identity": inspection[
                "replaces_published_knowledge_version_identity"
            ],
            **acceptance_result,
        }

    async def publication_eligibility(self, candidate_id: str) -> dict[str, Any]:
        binding = await self._load_binding(candidate_id)
        snapshot = await self._current_publication_snapshot(binding)
        reasons = await self._eligibility_reasons(binding, snapshot=snapshot)
        pointer = await self._current_pointer(binding.frozen_input.entry_identity)
        current_version = await self._current_version(pointer)
        effect = "replace" if current_version is not None else "create"
        return {
            "candidate_id": binding.candidate.stable_id,
            "eligible": not reasons,
            "reasons": reasons,
            "effect": effect,
            "current_published_knowledge_version": (
                self._version_projection(current_version) if current_version is not None else None
            ),
            "configuration_identity": binding.configuration_identity,
            "generation": binding.frozen_input.requested_generation,
            "inspection_record_identity": snapshot.inspection_record_identity,
            "acceptance_record_identity": snapshot.acceptance_record_identity,
        }

    async def get_publication(self, candidate_id: str) -> dict[str, Any]:
        """Return the immutable publication record for one Candidate without exposing a write path."""

        binding = await self._load_binding(candidate_id, require_latest_generation=False)
        version = await self._published_version_for_candidate(binding.candidate.stable_id)
        pointer = await self._current_pointer(binding.frozen_input.entry_identity)
        current = await self._current_version(pointer)
        return {
            "candidate_id": binding.candidate.stable_id,
            "published_knowledge_version": (
                self._version_projection(version) if version is not None else None
            ),
            "current_published_knowledge_version": (
                self._version_projection(current) if current is not None else None
            ),
            "is_current_for_entry": version is not None and current is not None and current.id == version.id,
        }

    async def confirm_publication_batch(
        self,
        payload: object,
        *,
        actor_identity: str,
    ) -> dict[str, Any]:
        confirmation_id, selected_items = self._normalize_confirmation(payload)
        selection_sha256 = canonical_json_sha256(selected_items)
        existing = await self._current_confirmation(confirmation_id)
        if existing is not None:
            if existing.selection_sha256 != selection_sha256 or existing.actor_identity != actor_identity:
                raise AppError(
                    status_code=409,
                    code="PUBLICATION_CONFIRMATION_CONFLICT",
                    message="confirmation identity is already bound to a different selected batch",
                )
            if existing.state != "processing":
                return dict(existing.results)
            return await self._claim_and_resume_publication_confirmation(existing)

        await self._validate_confirmed_selection(selected_items)
        confirmation = CandidatePublicationConfirmation(
            id=confirmation_id,
            selection_sha256=selection_sha256,
            selected_items=selected_items,
            actor_identity=actor_identity,
            state="processing",
            results={"batch_complete": False, "published": [], "failed": [], "skipped": []},
        )
        self.session.add(confirmation)
        try:
            await self.session.commit()
        except IntegrityError as exc:
            await self.session.rollback()
            existing = await self._current_confirmation(confirmation_id)
            if existing is None or existing.selection_sha256 != selection_sha256 or existing.actor_identity != actor_identity:
                raise AppError(
                    status_code=409,
                    code="PUBLICATION_CONFIRMATION_CONFLICT",
                    message="confirmation identity is already bound to a different selected batch",
                ) from exc
            if existing.state != "processing":
                return dict(existing.results)
            return await self._claim_and_resume_publication_confirmation(existing)
        return await self._claim_and_resume_publication_confirmation(confirmation)

    async def _current_confirmation(
        self,
        confirmation_id: str,
        *,
        for_update: bool = False,
    ) -> CandidatePublicationConfirmation | None:
        statement = (
            select(CandidatePublicationConfirmation)
            .where(CandidatePublicationConfirmation.id == confirmation_id)
            .execution_options(populate_existing=True)
        )
        if for_update:
            statement = statement.with_for_update()
        return await self.session.scalar(statement)

    async def _claim_and_resume_publication_confirmation(
        self,
        confirmation: CandidatePublicationConfirmation,
    ) -> dict[str, Any]:
        lease_owner = uuid4().hex
        now = datetime.now(UTC)
        claimed = await self.session.execute(
            update(CandidatePublicationConfirmation)
            .where(
                CandidatePublicationConfirmation.id == confirmation.id,
                CandidatePublicationConfirmation.state == "processing",
                or_(
                    CandidatePublicationConfirmation.lease_expires_at.is_(None),
                    CandidatePublicationConfirmation.lease_expires_at <= now,
                ),
            )
            .values(
                lease_owner=lease_owner,
                lease_expires_at=now + timedelta(seconds=_CONFIRMATION_LEASE_SECONDS),
            )
            .execution_options(synchronize_session=False)
        )
        await self.session.commit()
        current = await self._current_confirmation(confirmation.id)
        if current is None:
            raise RuntimeError("Candidate publication confirmation disappeared")
        if claimed.rowcount != 1:
            if current.state != "processing":
                return dict(current.results)
            raise AppError(
                status_code=409,
                code="PUBLICATION_CONFIRMATION_IN_PROGRESS",
                message="the selected publication confirmation is already processing",
                detail={"confirmation_id": confirmation.id},
            )
        if current.lease_owner != lease_owner:
            raise RuntimeError("Candidate publication confirmation lease was not retained")
        return await self._resume_publication_confirmation(current, lease_owner=lease_owner)

    async def _resume_publication_confirmation(
        self,
        confirmation: CandidatePublicationConfirmation,
        *,
        lease_owner: str,
    ) -> dict[str, Any]:
        confirmation_id = confirmation.id
        actor_identity = confirmation.actor_identity
        selected_items = list(confirmation.selected_items)
        results = self._confirmation_results(confirmation)
        completed_candidate_ids = self._completed_confirmation_candidate_ids(results)
        for selected in selected_items:
            candidate_id = selected.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise RuntimeError("Candidate publication confirmation has an invalid selected item")
            if candidate_id in completed_candidate_ids:
                continue
            result = await self._publish_selected_item(
                selected,
                actor_identity=actor_identity,
                confirmation_id=confirmation_id,
                lease_owner=lease_owner,
            )
            category = str(result.pop("category"))
            persisted_confirmation = await self._owned_processing_confirmation(
                confirmation_id,
                lease_owner=lease_owner,
            )
            persisted_results = self._confirmation_results(persisted_confirmation)
            persisted_completed = self._completed_confirmation_candidate_ids(persisted_results)
            if candidate_id in persisted_completed:
                results = persisted_results
                completed_candidate_ids = persisted_completed
                continue
            results = {
                **persisted_results,
                category: [*list(persisted_results.get(category, [])), result],
                "batch_complete": False,
            }
            persisted_confirmation.results = results
            persisted_confirmation.lease_expires_at = datetime.now(UTC) + timedelta(
                seconds=_CONFIRMATION_LEASE_SECONDS
            )
            await self.session.commit()
            completed_candidate_ids.add(candidate_id)

        persisted_confirmation = await self._owned_processing_confirmation(
            confirmation_id,
            lease_owner=lease_owner,
        )
        final_results = self._confirmation_results(persisted_confirmation)
        final_results["batch_complete"] = not final_results["failed"] and not final_results["skipped"]
        persisted_confirmation.results = final_results
        persisted_confirmation.state = (
            "completed" if final_results["batch_complete"] else "completed_with_item_results"
        )
        persisted_confirmation.completed_at = datetime.now(UTC)
        persisted_confirmation.lease_owner = None
        persisted_confirmation.lease_expires_at = None
        await self.session.commit()
        return dict(final_results)

    async def _owned_processing_confirmation(
        self,
        confirmation_id: str,
        *,
        lease_owner: str,
    ) -> CandidatePublicationConfirmation:
        confirmation = await self._current_confirmation(confirmation_id, for_update=True)
        if confirmation is None:
            raise RuntimeError("Candidate publication confirmation disappeared")
        if (
            confirmation.state != "processing"
            or confirmation.lease_owner != lease_owner
            or confirmation.lease_expires_at is None
            or self._utc_timestamp(confirmation.lease_expires_at) <= datetime.now(UTC)
        ):
            raise AppError(
                status_code=409,
                code="PUBLICATION_CONFIRMATION_LEASE_LOST",
                message="publication confirmation execution lease was not retained",
                detail={"confirmation_id": confirmation_id},
            )
        return confirmation

    @staticmethod
    def _utc_timestamp(value: datetime) -> datetime:
        """Normalize SQLite's naive temporal values to the UTC model contract."""
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _confirmation_results(confirmation: CandidatePublicationConfirmation) -> dict[str, Any]:
        results = dict(confirmation.results) if isinstance(confirmation.results, dict) else {}
        return {
            "batch_complete": False,
            "published": list(results.get("published", [])),
            "failed": list(results.get("failed", [])),
            "skipped": list(results.get("skipped", [])),
        }

    @staticmethod
    def _completed_confirmation_candidate_ids(results: dict[str, Any]) -> set[str]:
        completed: set[str] = set()
        for category in ("published", "failed", "skipped"):
            items = results.get(category)
            if not isinstance(items, list):
                raise RuntimeError("Candidate publication confirmation results are invalid")
            for item in items:
                candidate_id = item.get("candidate_id") if isinstance(item, dict) else None
                if not isinstance(candidate_id, str) or candidate_id in completed:
                    raise RuntimeError("Candidate publication confirmation results are invalid")
                completed.add(candidate_id)
        return completed

    async def _load_binding(
        self,
        candidate_id: str,
        *,
        for_update: bool = False,
        require_latest_generation: bool = True,
    ) -> CandidatePublicationBinding:
        try:
            candidate_identity = StableIdentity.from_stable_id(candidate_id)
        except ValueError as exc:
            raise AppError(
                status_code=404,
                code="CANDIDATE_NOT_READY_FOR_PUBLICATION",
                message="Candidate is unavailable for inspection or publication",
                detail={"candidate_id": candidate_id},
            ) from exc
        if candidate_identity.kind is not StableIdentityKind.CANDIDATE:
            raise AppError(
                status_code=404,
                code="CANDIDATE_NOT_READY_FOR_PUBLICATION",
                message="Candidate is unavailable for inspection or publication",
                detail={"candidate_id": candidate_id},
            )

        candidate = await self.session.get(CanonicalRecordModel, candidate_id)
        if (
            candidate is None
            or candidate.identity_kind != StableIdentityKind.CANDIDATE.value
            or candidate.record_class != CanonicalRecordClass.IMMUTABLE.value
            or candidate.state != "candidate_ready"
            or not isinstance(candidate.payload, dict)
            or candidate.payload.get("schema") != "candidate_build_candidate/v1"
        ):
            raise self._candidate_integrity_error(candidate_id)

        job_statement = select(CandidateBuildJob).where(CandidateBuildJob.candidate_id == candidate_id)
        if require_latest_generation:
            job_statement = job_statement.where(CandidateBuildJob.status == "candidate_ready")
        job_statement = job_statement.order_by(CandidateBuildJob.id.asc())
        if for_update:
            job_statement = job_statement.execution_options(populate_existing=True).with_for_update()
        jobs = (await self.session.execute(job_statement)).scalars().all()
        if len(jobs) != 1:
            raise self._candidate_integrity_error(candidate_id)
        job = jobs[0]
        if job.candidate_id != candidate_id or job.attempt < 1:
            raise self._candidate_integrity_error(candidate_id)

        frozen_input = await self._load_matching_input(job, candidate_id=candidate_id)
        self._assert_candidate_payload(candidate, job=job, frozen_input=frozen_input)
        if require_latest_generation:
            await self._assert_latest_generation(job, candidate_id=candidate_id)

        chunks = tuple(
            (
                await self.session.execute(
                    select(CandidateBuildChunk)
                    .where(
                        CandidateBuildChunk.job_id == job.id,
                        CandidateBuildChunk.candidate_id == candidate_id,
                        CandidateBuildChunk.attempt == job.attempt,
                    )
                    .order_by(CandidateBuildChunk.chunk_index.asc(), CandidateBuildChunk.id.asc())
                )
            ).scalars()
        )
        self._assert_chunk_binding(candidate, job=job, frozen_input=frozen_input, chunks=chunks)
        return CandidatePublicationBinding(
            candidate=candidate,
            job=job,
            frozen_input=frozen_input,
            chunks=chunks,
        )

    async def _eligibility_reasons(
        self,
        binding: CandidatePublicationBinding,
        *,
        snapshot: CandidatePublicationSnapshot,
    ) -> list[str]:
        reasons: list[str] = []
        if await self._published_version_for_candidate(binding.candidate.stable_id) is not None:
            reasons.append("CANDIDATE_ALREADY_PUBLISHED")
        if candidate_embedding_configuration(self._settings) != binding.frozen_input.embedding_configuration:
            reasons.append("CANDIDATE_EMBEDDING_CONFIGURATION_CHANGED")
        try:
            await self._assert_current_editorial_approval(binding)
        except AppError as exc:
            reasons.append(exc.code)
        if snapshot.inspection_record_identity is None:
            reasons.append("CANDIDATE_INSPECTION_REQUIRED")
        if snapshot.acceptance_record_identity is None:
            reasons.append("CANDIDATE_ACCEPTANCE_REQUIRED")
        if snapshot.inspection_record_identity is not None:
            current = await self._current_version(
                await self._current_pointer(binding.frozen_input.entry_identity)
            )
            current_identity = current.id if current is not None else None
            if current_identity != snapshot.replaces_published_knowledge_version_identity:
                reasons.append("CANDIDATE_REPLACEMENT_POINTER_STALE")
        return reasons

    async def _current_publication_snapshot(
        self,
        binding: CandidatePublicationBinding,
    ) -> CandidatePublicationSnapshot:
        acceptance = await self._latest_valid_acceptance(binding)
        if acceptance is not None:
            inspection_identity = acceptance.get("inspection_record_identity")
            acceptance_identity = acceptance.get("acceptance_record_identity")
            if (
                isinstance(inspection_identity, str)
                and isinstance(acceptance_identity, str)
                and (inspection := await self._exact_inspection_payload(binding, inspection_identity)) is not None
                and acceptance.get("replaces_published_knowledge_version_identity")
                == inspection.get("replaces_published_knowledge_version_identity")
            ):
                return CandidatePublicationSnapshot(
                    inspection_record_identity=inspection_identity,
                    acceptance_record_identity=acceptance_identity,
                    replaces_published_knowledge_version_identity=inspection[
                        "replaces_published_knowledge_version_identity"
                    ],
                )
        inspection = await self._latest_valid_inspection(binding)
        return CandidatePublicationSnapshot(
            inspection_record_identity=(
                str(inspection["inspection_record_identity"]) if inspection is not None else None
            ),
            acceptance_record_identity=None,
            replaces_published_knowledge_version_identity=(
                inspection["replaces_published_knowledge_version_identity"]
                if inspection is not None
                else None
            ),
        )

    async def _latest_valid_acceptance(self, binding: CandidatePublicationBinding) -> dict[str, Any] | None:
        records = (
            await self.session.execute(
                select(CanonicalRecordModel)
                .where(CanonicalRecordModel.identity_kind == StableIdentityKind.EVENT.value)
                .order_by(CanonicalRecordModel.created_at.desc(), CanonicalRecordModel.stable_id.desc())
            )
        ).scalars()
        for record in records:
            if not isinstance(record.payload, dict) or record.payload.get("schema") != "candidate_acceptance/v1":
                continue
            if (
                record.record_class == CanonicalRecordClass.IMMUTABLE.value
                and record.state == "passed"
                and record.payload.get("acceptance_record_identity") == record.stable_id
                and self._acceptance_matches(record.payload, binding)
            ):
                return dict(record.payload)
        return None

    async def _exact_inspection_payload(
        self,
        binding: CandidatePublicationBinding,
        inspection_identity: str,
    ) -> dict[str, Any] | None:
        try:
            identity = StableIdentity.from_stable_id(inspection_identity)
        except ValueError:
            return None
        if identity.kind is not StableIdentityKind.EVENT:
            return None
        record = await self.session.get(CanonicalRecordModel, inspection_identity)
        if (
            record is None
            or record.identity_kind != StableIdentityKind.EVENT.value
            or record.record_class != CanonicalRecordClass.IMMUTABLE.value
            or record.state != "recorded"
            or not isinstance(record.payload, dict)
            or record.payload.get("schema") != "candidate_inspection/v1"
            or record.payload.get("inspection_record_identity") != record.stable_id
            or not self._inspection_matches(record.payload, binding)
        ):
            return None
        return dict(record.payload)

    async def _exact_acceptance_payload(
        self,
        binding: CandidatePublicationBinding,
        *,
        inspection_identity: str,
        acceptance_identity: str,
    ) -> dict[str, Any] | None:
        try:
            identity = StableIdentity.from_stable_id(acceptance_identity)
        except ValueError:
            return None
        if identity.kind is not StableIdentityKind.EVENT:
            return None
        record = await self.session.get(CanonicalRecordModel, acceptance_identity)
        if (
            record is None
            or record.identity_kind != StableIdentityKind.EVENT.value
            or record.record_class != CanonicalRecordClass.IMMUTABLE.value
            or record.state != "passed"
            or not isinstance(record.payload, dict)
            or record.payload.get("schema") != "candidate_acceptance/v1"
            or record.payload.get("acceptance_record_identity") != record.stable_id
            or record.payload.get("inspection_record_identity") != inspection_identity
            or not self._acceptance_matches(record.payload, binding)
        ):
            return None
        return dict(record.payload)

    def _acceptance_matches(self, payload: dict[str, Any], binding: CandidatePublicationBinding) -> bool:
        result = payload.get("result")
        boundary = result.get("boundary") if isinstance(result, dict) else None
        binding_matches = (
            payload.get("candidate_id") == binding.candidate.stable_id
            and payload.get("build_generation_id") == f"build_generation:{binding.job.id}"
            and payload.get("entry_identity") == binding.frozen_input.entry_identity
            and payload.get("bundle_sha256") == binding.frozen_input.bundle_sha256
            and payload.get("bundle_item_sha256") == binding.frozen_input.bundle_item_sha256
            and payload.get("input_sha256") == binding.frozen_input.input_sha256
            and payload.get("frozen_input_sha256") == binding.frozen_input.frozen_input_sha256
            and payload.get("configuration_identity") == binding.configuration_identity
            and payload.get("configuration") == binding.frozen_input.embedding_configuration
            and "replaces_published_knowledge_version_identity" in payload
            and (
                payload.get("replaces_published_knowledge_version_identity") is None
                or isinstance(payload.get("replaces_published_knowledge_version_identity"), str)
            )
            and isinstance(boundary, dict)
            and boundary.get("reason") in {"decision_not_covered", "decisive_condition_missing"}
        )
        if not binding_matches:
            return False
        try:
            expected = CandidateAcceptanceExecutionAdapter(binding).run()
            # Compare the whole deterministic result, including JSON scalar types.
            return canonical_json_sha256(result) == canonical_json_sha256(expected)
        except (AppError, TypeError, ValueError):
            return False

    async def _published_version_for_candidate(
        self,
        candidate_id: str,
    ) -> PublishedKnowledgeVersion | None:
        version = await self.session.scalar(
            select(PublishedKnowledgeVersion).where(PublishedKnowledgeVersion.candidate_id == candidate_id)
        )
        if version is not None:
            await self._assert_version_integrity(version)
        return version

    async def _current_pointer(
        self,
        entry_identity: str,
        *,
        for_update: bool = False,
    ) -> PublishedKnowledgePointer | None:
        if for_update:
            return await self.session.scalar(
                select(PublishedKnowledgePointer)
                .where(PublishedKnowledgePointer.entry_identity == entry_identity)
                .with_for_update()
            )
        return await self.session.get(PublishedKnowledgePointer, entry_identity)

    async def _current_version(
        self,
        pointer: PublishedKnowledgePointer | None,
    ) -> PublishedKnowledgeVersion | None:
        if pointer is None:
            return None
        version = await self.session.get(PublishedKnowledgeVersion, pointer.current_version_id)
        if (
            version is None
            or version.entry_identity != pointer.entry_identity
            or version.document_identity != pointer.document_identity
            or version.generation != pointer.generation
        ):
            raise AppError(
                status_code=409,
                code="PUBLISHED_KNOWLEDGE_POINTER_INTEGRITY_FAILED",
                message="published Knowledge Version pointer does not resolve",
                detail={"entry_identity": pointer.entry_identity},
            )
        await self._assert_version_integrity(version)
        return version

    async def _assert_version_integrity(self, version: PublishedKnowledgeVersion) -> None:
        failure = AppError(
            status_code=409,
            code="PUBLISHED_KNOWLEDGE_POINTER_INTEGRITY_FAILED",
            message="published Knowledge Version does not match its immutable authority",
            detail={"entry_identity": version.entry_identity},
        )
        record = await self.session.get(CanonicalRecordModel, version.id)
        if (
            record is None
            or record.identity_kind != StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION.value
            or record.record_class != CanonicalRecordClass.IMMUTABLE.value
            or record.state != "published"
            or not isinstance(record.payload, dict)
            or record.payload.get("schema") != "published_knowledge_version/v1"
        ):
            raise failure
        try:
            binding = await self._load_binding(version.candidate_id, require_latest_generation=False)
        except AppError as exc:
            raise failure from exc
        frozen = binding.frozen_input
        expected_projection = {
            "candidate_id": binding.candidate.stable_id,
            "entry_identity": frozen.entry_identity,
            "document_identity": runtime_document_identity(frozen.document_identity),
            "generation": frozen.requested_generation,
            "bundle_sha256": frozen.bundle_sha256,
            "frozen_input_sha256": frozen.frozen_input_sha256,
            "configuration_identity": binding.configuration_identity,
            "inspection_record_identity": version.inspection_record_identity,
            "acceptance_record_identity": version.acceptance_record_identity,
            "supersedes_version_id": version.supersedes_version_id,
        }
        expected_record = {
            **expected_projection,
            "schema": "published_knowledge_version/v1",
            "bundle_id": frozen.bundle_id,
            "bundle_item_id": frozen.bundle_item_id,
            "bundle_item_sha256": frozen.bundle_item_sha256,
            "input_sha256": frozen.input_sha256,
            "editorial_revision_identity": frozen.artifact["editorial_revision_identity"],
            "replaces_published_knowledge_version_identity": version.supersedes_version_id,
        }
        if (
            version.id != self._published_version_identity(binding).stable_id
            or any(getattr(version, key) != value for key, value in expected_projection.items())
            or canonical_json_sha256(record.payload) != canonical_json_sha256(expected_record)
        ):
            raise failure

    @staticmethod
    def _version_projection(version: PublishedKnowledgeVersion) -> dict[str, Any]:
        return {
            "identity": version.id,
            "candidate_id": version.candidate_id,
            "entry_identity": version.entry_identity,
            "generation": version.generation,
            "bundle_sha256": version.bundle_sha256,
            "frozen_input_sha256": version.frozen_input_sha256,
            "configuration_identity": version.configuration_identity,
            "inspection_record_identity": version.inspection_record_identity,
            "acceptance_record_identity": version.acceptance_record_identity,
            "supersedes_published_knowledge_version_identity": version.supersedes_version_id,
        }

    def _normalize_confirmation(self, payload: object) -> tuple[str, list[dict[str, str | None]]]:
        if not isinstance(payload, dict) or set(payload) != {"confirmation_id", "selected_items"}:
            raise AppError(
                status_code=422,
                code="PUBLICATION_CONFIRMATION_INVALID",
                message="publication confirmation must name an exact selected batch",
            )
        confirmation_id = payload.get("confirmation_id")
        selected = payload.get("selected_items")
        if (
            not isinstance(confirmation_id, str)
            or not confirmation_id
            or len(confirmation_id) > 160
            or not isinstance(selected, list)
            or not selected
        ):
            raise AppError(
                status_code=422,
                code="PUBLICATION_CONFIRMATION_INVALID",
                message="publication confirmation must name an exact selected batch",
            )
        normalized: list[dict[str, str | None]] = []
        candidate_ids: set[str] = set()
        for item in selected:
            if not isinstance(item, dict) or set(item) != {
                "candidate_id",
                "effect",
                "current_published_knowledge_version",
                "inspection_record_identity",
                "acceptance_record_identity",
            }:
                raise AppError(
                    status_code=422,
                    code="PUBLICATION_CONFIRMATION_INVALID",
                    message="publication confirmation contains an invalid selected item",
                )
            candidate_id = item.get("candidate_id")
            effect = item.get("effect")
            current_version = item.get("current_published_knowledge_version")
            inspection_identity = item.get("inspection_record_identity")
            acceptance_identity = item.get("acceptance_record_identity")
            if (
                not isinstance(candidate_id, str)
                or not isinstance(effect, str)
                or effect not in {"create", "replace"}
                or (current_version is not None and not isinstance(current_version, str))
                or not isinstance(inspection_identity, str)
                or not isinstance(acceptance_identity, str)
                or candidate_id in candidate_ids
            ):
                raise AppError(
                    status_code=422,
                    code="PUBLICATION_CONFIRMATION_INVALID",
                    message="publication confirmation contains an invalid selected item",
                )
            candidate_ids.add(candidate_id)
            normalized.append(
                {
                    "candidate_id": candidate_id,
                    "effect": effect,
                    "current_published_knowledge_version": current_version,
                    "inspection_record_identity": inspection_identity,
                    "acceptance_record_identity": acceptance_identity,
                }
            )
        return confirmation_id, normalized

    async def _validate_confirmed_selection(self, selected_items: list[dict[str, str | None]]) -> None:
        for selected in selected_items:
            candidate_id = str(selected["candidate_id"])
            eligibility = await self.publication_eligibility(candidate_id)
            if not eligibility["eligible"]:
                code = (
                    "CANDIDATE_ALREADY_PUBLISHED"
                    if "CANDIDATE_ALREADY_PUBLISHED" in eligibility["reasons"]
                    else "CANDIDATE_NOT_PUBLICATION_ELIGIBLE"
                )
                raise AppError(
                    status_code=409,
                    code=code,
                    message="Candidate is not eligible for explicit publication",
                    detail={"candidate_id": candidate_id, "reasons": eligibility["reasons"]},
                )
            if selected["effect"] != eligibility["effect"]:
                raise AppError(
                    status_code=409,
                    code="PUBLICATION_CONFIRMATION_EFFECT_MISMATCH",
                    message="confirmation effect does not match the current Candidate publication effect",
                    detail={"candidate_id": candidate_id, "expected_effect": eligibility["effect"]},
                )
            current = eligibility["current_published_knowledge_version"]
            current_identity = current["identity"] if isinstance(current, dict) else None
            if selected["current_published_knowledge_version"] != current_identity:
                raise AppError(
                    status_code=409,
                    code="PUBLICATION_CONFIRMATION_POINTER_MISMATCH",
                    message="confirmation does not bind the current replacement pointer",
                    detail={"candidate_id": candidate_id, "current_published_knowledge_version": current_identity},
                )
            if (
                selected["inspection_record_identity"] != eligibility["inspection_record_identity"]
                or selected["acceptance_record_identity"] != eligibility["acceptance_record_identity"]
            ):
                raise AppError(
                    status_code=409,
                    code="PUBLICATION_CONFIRMATION_ACCEPTANCE_MISMATCH",
                    message="confirmation must bind the currently eligible inspection and acceptance records",
                    detail={
                        "candidate_id": candidate_id,
                        "inspection_record_identity": eligibility["inspection_record_identity"],
                        "acceptance_record_identity": eligibility["acceptance_record_identity"],
                    },
                )

    async def _selected_publication_snapshot(
        self,
        binding: CandidatePublicationBinding,
        selected: dict[str, str | None],
    ) -> CandidatePublicationSnapshot:
        inspection_identity = selected["inspection_record_identity"]
        acceptance_identity = selected["acceptance_record_identity"]
        if not isinstance(inspection_identity, str) or not isinstance(acceptance_identity, str):
            raise AppError(
                status_code=409,
                code="PUBLICATION_CONFIRMATION_ACCEPTANCE_MISMATCH",
                message="publication confirmation does not bind exact acceptance records",
                detail={"candidate_id": binding.candidate.stable_id},
            )
        inspection = await self._exact_inspection_payload(binding, inspection_identity)
        acceptance = await self._exact_acceptance_payload(
            binding,
            inspection_identity=inspection_identity,
            acceptance_identity=acceptance_identity,
        )
        if inspection is None or acceptance is None:
            raise AppError(
                status_code=409,
                code="PUBLICATION_CONFIRMATION_ACCEPTANCE_MISMATCH",
                message="publication confirmation acceptance records are not valid for this Candidate",
                detail={"candidate_id": binding.candidate.stable_id},
            )
        replacement_identity = inspection["replaces_published_knowledge_version_identity"]
        if (
            replacement_identity != acceptance.get("replaces_published_knowledge_version_identity")
            or selected["current_published_knowledge_version"] != replacement_identity
        ):
            raise AppError(
                status_code=409,
                code="PUBLICATION_CONFIRMATION_POINTER_MISMATCH",
                message="publication confirmation does not bind the inspected replacement version",
                detail={"candidate_id": binding.candidate.stable_id},
            )
        return CandidatePublicationSnapshot(
            inspection_record_identity=inspection_identity,
            acceptance_record_identity=acceptance_identity,
            replaces_published_knowledge_version_identity=replacement_identity,
        )

    async def _publish_selected_item(
        self,
        selected: dict[str, str | None],
        *,
        actor_identity: str,
        confirmation_id: str,
        lease_owner: str,
    ) -> dict[str, Any]:
        candidate_id = str(selected["candidate_id"])
        try:
            existing_version = await self._published_version_for_candidate(candidate_id)
            if existing_version is not None:
                if (
                    existing_version.inspection_record_identity == selected["inspection_record_identity"]
                    and existing_version.acceptance_record_identity == selected["acceptance_record_identity"]
                ):
                    return {
                        "category": "published",
                        "candidate_id": candidate_id,
                        "effect": selected["effect"],
                        "publication_identity": existing_version.id,
                    }
                return {
                    "category": "skipped",
                    "candidate_id": candidate_id,
                    "effect": selected["effect"],
                    "reason": "CANDIDATE_ALREADY_PUBLISHED",
                }
            binding = await self._load_binding(candidate_id)
            try:
                authority_context = self._editorial_export_verifier.verify_for_candidate_finalization(
                    binding.frozen_input.artifact,
                    binding.frozen_input.input_sha256,
                )
            except AttributeError as exc:
                raise AppError(
                    status_code=409,
                    code="EDITORIAL_PUBLICATION_FENCE_REQUIRED",
                    message="Candidate publication requires a final editorial authority fence",
                    detail={"candidate_id": candidate_id},
                ) from exc
            async with authority_context as verified_artifact:
                if verified_artifact != binding.frozen_input.artifact:
                    raise AppError(
                        status_code=409,
                        code="CANDIDATE_EDITORIAL_APPROVAL_INVALID",
                        message="Candidate no longer proves a current approved editorial export",
                        detail={"candidate_id": candidate_id},
                    )
                binding = await self._load_binding(candidate_id, for_update=True)
                existing_version = await self._published_version_for_candidate(candidate_id)
                if existing_version is not None:
                    if (
                        existing_version.inspection_record_identity == selected["inspection_record_identity"]
                        and existing_version.acceptance_record_identity == selected["acceptance_record_identity"]
                    ):
                        return {
                            "category": "published",
                            "candidate_id": candidate_id,
                            "effect": selected["effect"],
                            "publication_identity": existing_version.id,
                        }
                    return {
                        "category": "skipped",
                        "candidate_id": candidate_id,
                        "effect": selected["effect"],
                        "reason": "CANDIDATE_ALREADY_PUBLISHED",
                    }
                eligibility = await self.publication_eligibility(candidate_id)
                if not eligibility["eligible"]:
                    return {
                        "category": "skipped",
                        "candidate_id": candidate_id,
                        "effect": selected["effect"],
                        "reason": "CANDIDATE_NOT_PUBLICATION_ELIGIBLE",
                    }
                if eligibility["effect"] != selected["effect"]:
                    return {
                        "category": "skipped",
                        "candidate_id": candidate_id,
                        "effect": selected["effect"],
                        "reason": "PUBLICATION_CONFIRMATION_STALE",
                    }
                current = eligibility["current_published_knowledge_version"]
                current_identity = current["identity"] if isinstance(current, dict) else None
                if current_identity != selected["current_published_knowledge_version"]:
                    return {
                        "category": "skipped",
                        "candidate_id": candidate_id,
                        "effect": selected["effect"],
                        "reason": "PUBLICATION_CONFIRMATION_STALE",
                    }
                await self._owned_processing_confirmation(
                    confirmation_id,
                    lease_owner=lease_owner,
                )
                snapshot = await self._selected_publication_snapshot(binding, selected)
                version = await self._publish_one(
                    binding,
                    previous_version_identity=current_identity,
                    actor_identity=actor_identity,
                    snapshot=snapshot,
                )
                await self._editorial_export_verifier.record_candidate_publication(
                    binding.frozen_input.artifact,
                    candidate_identity=binding.candidate.stable_id,
                    published_knowledge_version_identity=version.id,
                    actor_identity=actor_identity,
                )
                await self.session.commit()
            return {
                "category": "published",
                "candidate_id": candidate_id,
                "effect": selected["effect"],
                "publication_identity": version.id,
            }
        except AppError as exc:
            await self.session.rollback()
            return {
                "category": "failed",
                "candidate_id": candidate_id,
                "effect": selected["effect"],
                "reason": exc.code,
            }
        except (SQLAlchemyError, OSError, RuntimeError):
            await self.session.rollback()
            return {
                "category": "failed",
                "candidate_id": candidate_id,
                "effect": selected["effect"],
                "reason": "PUBLICATION_ITEM_RETRYABLE_FAILURE",
            }

    async def _publish_one(
        self,
        binding: CandidatePublicationBinding,
        *,
        previous_version_identity: str | None,
        actor_identity: str,
        snapshot: CandidatePublicationSnapshot,
    ) -> PublishedKnowledgeVersion:
        if not snapshot.is_complete:
            raise AppError(
                status_code=409,
                code="PUBLICATION_CONFIRMATION_ACCEPTANCE_MISMATCH",
                message="publication requires an exact inspection and acceptance snapshot",
                detail={"candidate_id": binding.candidate.stable_id},
            )
        assert snapshot.inspection_record_identity is not None
        assert snapshot.acceptance_record_identity is not None
        self._assert_current_embedding_configuration(binding)
        document_identity = runtime_document_identity(binding.frozen_input.document_identity)
        version_identity = self._published_version_identity(binding)
        existing = await self.session.get(PublishedKnowledgeVersion, version_identity.stable_id)
        if existing is not None:
            if existing.candidate_id != binding.candidate.stable_id:
                raise self._candidate_integrity_error(binding.candidate.stable_id)
            return existing

        pointer = await self._current_pointer(
            binding.frozen_input.entry_identity,
            for_update=True,
        )
        current = await self._current_version(pointer)
        current_identity = current.id if current is not None else None
        if current_identity != snapshot.replaces_published_knowledge_version_identity:
            raise AppError(
                status_code=409,
                code="PUBLICATION_POINTER_CHANGED",
                message="published Knowledge Version pointer differs from the inspected replacement version",
                detail={"entry_identity": binding.frozen_input.entry_identity},
            )
        if current_identity != previous_version_identity:
            raise AppError(
                status_code=409,
                code="PUBLICATION_POINTER_CHANGED",
                message="published Knowledge Version pointer changed before publication",
                detail={"entry_identity": binding.frozen_input.entry_identity},
            )

        document = await self.session.get(Document, document_identity)
        await self._ensure_published_source_capacity(document)
        if document is None:
            document = Document(
                id=document_identity,
                filename=f"{binding.frozen_input.entry_identity.removeprefix('entry:')}.reviewed",
                file_type="reviewed_release_bundle",
                file_size=sum(len(chunk.content.encode("utf-8")) for chunk in binding.chunks),
                status="pending",
                chunk_strategy=str(binding.frozen_input.chunk_strategy.get("strategy_id", "reviewed")),
                next_generation=binding.frozen_input.requested_generation + 1,
                latest_requested_generation=binding.frozen_input.requested_generation,
            )
            self.session.add(document)
            await self.session.flush()

        runtime_chunk_count = await self._write_runtime_projection(
            binding,
            document_identity=document_identity,
            publication_identity=version_identity.stable_id,
        )
        await self._before_pointer_switch(binding)

        version = PublishedKnowledgeVersion(
            id=version_identity.stable_id,
            candidate_id=binding.candidate.stable_id,
            entry_identity=binding.frozen_input.entry_identity,
            document_identity=document_identity,
            generation=binding.frozen_input.requested_generation,
            bundle_sha256=binding.frozen_input.bundle_sha256,
            frozen_input_sha256=binding.frozen_input.frozen_input_sha256,
            configuration_identity=binding.configuration_identity,
            inspection_record_identity=snapshot.inspection_record_identity,
            acceptance_record_identity=snapshot.acceptance_record_identity,
            supersedes_version_id=current_identity,
            published_by=actor_identity,
        )
        self.session.add(version)
        self.session.add(
            CanonicalRecordModel(
                stable_id=version_identity.stable_id,
                identity_kind=StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION.value,
                identity_value=version_identity.value,
                state="published",
                record_class=CanonicalRecordClass.IMMUTABLE.value,
                payload={
                    "schema": "published_knowledge_version/v1",
                    "candidate_id": binding.candidate.stable_id,
                    "entry_identity": binding.frozen_input.entry_identity,
                    "document_identity": document_identity,
                    "generation": binding.frozen_input.requested_generation,
                    "bundle_id": binding.frozen_input.bundle_id,
                    "bundle_sha256": binding.frozen_input.bundle_sha256,
                    "bundle_item_id": binding.frozen_input.bundle_item_id,
                    "bundle_item_sha256": binding.frozen_input.bundle_item_sha256,
                    "input_sha256": binding.frozen_input.input_sha256,
                    "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
                    "configuration_identity": binding.configuration_identity,
                    "editorial_revision_identity": binding.frozen_input.artifact["editorial_revision_identity"],
                    "inspection_record_identity": snapshot.inspection_record_identity,
                    "acceptance_record_identity": snapshot.acceptance_record_identity,
                    "replaces_published_knowledge_version_identity": (
                        snapshot.replaces_published_knowledge_version_identity
                    ),
                    "supersedes_version_id": current_identity,
                },
            )
        )
        self.session.add(
            CanonicalEventModel(
                aggregate_id=version_identity.stable_id,
                aggregate_kind=StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION.value,
                event_type=CanonicalEventType.PUBLISHED.value,
                from_state=None,
                to_state="published",
                payload={
                    "schema": "candidate_publication_event/v1",
                    "candidate_id": binding.candidate.stable_id,
                    "entry_identity": binding.frozen_input.entry_identity,
                    "previous_published_knowledge_version": current_identity,
                    "replaces_published_knowledge_version_identity": (
                        snapshot.replaces_published_knowledge_version_identity
                    ),
                    "bundle_sha256": binding.frozen_input.bundle_sha256,
                    "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
                    "configuration_identity": binding.configuration_identity,
                    "inspection_record_identity": snapshot.inspection_record_identity,
                    "acceptance_record_identity": snapshot.acceptance_record_identity,
                },
                recorded_by=actor_identity,
            )
        )
        if pointer is None:
            self.session.add(
                PublishedKnowledgePointer(
                    entry_identity=binding.frozen_input.entry_identity,
                    current_version_id=version_identity.stable_id,
                    document_identity=document_identity,
                    generation=binding.frozen_input.requested_generation,
                )
            )
        else:
            pointer.current_version_id = version_identity.stable_id
            pointer.document_identity = document_identity
            pointer.generation = binding.frozen_input.requested_generation
        document.published_generation = binding.frozen_input.requested_generation
        document.next_generation = max(
            document.next_generation,
            binding.frozen_input.requested_generation + 1,
        )
        document.latest_requested_generation = max(
            document.latest_requested_generation,
            binding.frozen_input.requested_generation,
        )
        document.candidate_generation = None
        document.candidate_chunk_count = 0
        document.candidate_chunk_strategy = None
        document.status = "ready"
        document.chunk_count = runtime_chunk_count
        document.chunk_strategy = str(binding.frozen_input.chunk_strategy.get("strategy_id", "reviewed"))
        binding.job.allowed_next_action = "published"
        return version

    async def _write_runtime_projection(
        self,
        binding: CandidatePublicationBinding,
        *,
        document_identity: str,
        publication_identity: str,
    ) -> int:
        existing = (
            await self.session.execute(
                select(DocumentChunk.id).where(
                    DocumentChunk.document_id == document_identity,
                    DocumentChunk.generation == binding.frozen_input.requested_generation,
                )
            )
        ).scalars().all()
        if existing:
            raise AppError(
                status_code=409,
                code="PUBLICATION_RUNTIME_GENERATION_CONFLICT",
                message="runtime projection already has the Candidate generation",
                detail={"candidate_id": binding.candidate.stable_id},
            )
        projected_chunks: list[DocumentChunk] = []
        for candidate_chunk in binding.chunks:
            raw_metadata = candidate_chunk.chunk_metadata
            section_id = raw_metadata.get("section_id") if isinstance(raw_metadata, dict) else None
            if not isinstance(section_id, str) or not section_id:
                raise self._candidate_integrity_error(binding.candidate.stable_id)
            try:
                metadata = _candidate_published_chunk_metadata(
                    binding,
                    section_id=section_id,
                    publication_identity=publication_identity,
                )
            except ValueError as exc:
                raise self._candidate_integrity_error(binding.candidate.stable_id) from exc
            projected_chunks.append(
                DocumentChunk(
                    document_id=document_identity,
                    generation=binding.frozen_input.requested_generation,
                    chunk_index=candidate_chunk.chunk_index,
                    content=candidate_chunk.content,
                    chunk_metadata=metadata,
                )
            )
        self.session.add_all(projected_chunks)
        return len(projected_chunks)

    async def _ensure_published_source_capacity(self, document: Document | None) -> None:
        if document is not None and document.published_generation > 0:
            return
        await self.session.execute(
            select(Document.id)
            .where(Document.deleted_at.is_(None), Document.published_generation > 0)
            .with_for_update()
        )
        published_sources = await self.session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.deleted_at.is_(None), Document.published_generation > 0)
        )
        if int(published_sources or 0) >= MAX_PUBLISHED_SOURCES:
            raise AppError(
                status_code=409,
                code="PUBLISHED_SOURCE_LIMIT_REACHED",
                message="the first-release published source limit has been reached",
            )

    async def _before_pointer_switch(self, binding: CandidatePublicationBinding) -> None:
        del binding

    @staticmethod
    def _published_version_identity(binding: CandidatePublicationBinding) -> StableIdentity:
        return StableIdentity(
            StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION,
            canonical_json_sha256(
                {
                    "candidate_id": binding.candidate.stable_id,
                    "entry_identity": binding.frozen_input.entry_identity,
                    "generation": binding.frozen_input.requested_generation,
                    "bundle_sha256": binding.frozen_input.bundle_sha256,
                    "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
                    "configuration_identity": binding.configuration_identity,
                }
            ),
        )

    async def _load_matching_input(
        self,
        job: CandidateBuildJob,
        *,
        candidate_id: str,
    ) -> FrozenCandidateBuildInput:
        try:
            frozen_input = await load_frozen_candidate_build_input(self.session, job.id)
        except AppError as exc:
            raise self._candidate_integrity_error(candidate_id) from exc
        if not frozen_input.matches_job(job):
            raise self._candidate_integrity_error(candidate_id)
        return frozen_input

    async def _assert_latest_generation(self, job: CandidateBuildJob, *, candidate_id: str) -> None:
        latest_generation = (
            await self.session.execute(
                select(func.max(CandidateBuildJob.requested_generation)).where(
                    CandidateBuildJob.entry_identity == job.entry_identity
                )
            )
        ).scalar_one()
        if latest_generation != job.requested_generation:
            raise AppError(
                status_code=409,
                code="CANDIDATE_GENERATION_STALE",
                message="Candidate is not the latest immutable generation for its entry",
                detail={"candidate_id": candidate_id, "entry_identity": job.entry_identity},
            )

    def _assert_candidate_payload(
        self,
        candidate: CanonicalRecordModel,
        *,
        job: CandidateBuildJob,
        frozen_input: FrozenCandidateBuildInput,
    ) -> None:
        payload = candidate.payload
        expected = {
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
            "attempt": job.attempt,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise self._candidate_integrity_error(candidate.stable_id)

    def _assert_chunk_binding(
        self,
        candidate: CanonicalRecordModel,
        *,
        job: CandidateBuildJob,
        frozen_input: FrozenCandidateBuildInput,
        chunks: tuple[CandidateBuildChunk, ...],
    ) -> None:
        payload = candidate.payload
        expected_hashes = payload.get("chunk_sha256s")
        binding = CandidatePublicationBinding(
            candidate=candidate,
            job=job,
            frozen_input=frozen_input,
            chunks=chunks,
        )
        metadata_matches = True
        for chunk in chunks:
            section_id = chunk.chunk_metadata.get("section_id") if isinstance(chunk.chunk_metadata, dict) else None
            try:
                expected_metadata = (
                    _candidate_build_chunk_metadata(binding, section_id=section_id)
                    if isinstance(section_id, str)
                    else None
                )
            except ValueError:
                metadata_matches = False
                break
            if chunk.chunk_metadata != expected_metadata:
                metadata_matches = False
                break
        if (
            not chunks
            or payload.get("chunk_count") != len(chunks)
            or not isinstance(expected_hashes, list)
            or expected_hashes != [chunk.content_sha256 for chunk in chunks]
            or [chunk.chunk_index for chunk in chunks] != list(range(len(chunks)))
            or any(
                chunk.document_identity != frozen_input.document_identity
                or chunk.generation != frozen_input.requested_generation
                or chunk.attempt != job.attempt
                or chunk.candidate_id != candidate.stable_id
                or hashlib.sha256(chunk.content.encode("utf-8")).hexdigest() != chunk.content_sha256
                for chunk in chunks
            )
            or not metadata_matches
        ):
            raise self._candidate_integrity_error(candidate.stable_id)

    async def _assert_current_editorial_approval(self, binding: CandidatePublicationBinding) -> None:
        verified_artifact = await self._editorial_export_verifier.verify(
            binding.frozen_input.artifact,
            binding.frozen_input.input_sha256,
        )
        approval = binding.frozen_input.artifact.get("approval")
        if (
            verified_artifact != binding.frozen_input.artifact
            or not isinstance(approval, dict)
            or approval.get("status") not in _APPROVED_EDITORIAL_STATUSES
        ):
            raise AppError(
                status_code=409,
                code="CANDIDATE_EDITORIAL_APPROVAL_INVALID",
                message="Candidate no longer proves a current approved editorial export",
                detail={"candidate_id": binding.candidate.stable_id},
            )

    def _assert_current_embedding_configuration(self, binding: CandidatePublicationBinding) -> None:
        if candidate_embedding_configuration(self._settings) != binding.frozen_input.embedding_configuration:
            raise AppError(
                status_code=409,
                code="CANDIDATE_EMBEDDING_CONFIGURATION_CHANGED",
                message="Candidate embedding configuration changed after Candidate Build",
                detail={"candidate_id": binding.candidate.stable_id},
            )

    async def _ensure_configuration_record(self, binding: CandidatePublicationBinding) -> None:
        configuration_identity = StableIdentity.from_stable_id(binding.configuration_identity)
        payload = {
            "schema": "candidate_publication_configuration/v1",
            "configuration": binding.frozen_input.embedding_configuration,
            "configuration_sha256": canonical_json_sha256(binding.frozen_input.embedding_configuration),
        }
        existing = await self.session.get(CanonicalRecordModel, configuration_identity.stable_id)
        if existing is None:
            self.session.add(
                CanonicalRecordModel(
                    stable_id=configuration_identity.stable_id,
                    identity_kind=configuration_identity.kind.value,
                    identity_value=configuration_identity.value,
                    state="frozen",
                    record_class=CanonicalRecordClass.IMMUTABLE.value,
                    payload=payload,
                )
            )
            return
        if (
            existing.identity_kind != StableIdentityKind.CONFIGURATION.value
            or existing.record_class != CanonicalRecordClass.IMMUTABLE.value
            or existing.payload != payload
        ):
            raise self._candidate_integrity_error(binding.candidate.stable_id)

    async def _latest_valid_inspection(self, binding: CandidatePublicationBinding) -> dict[str, Any] | None:
        records = (
            await self.session.execute(
                select(CanonicalRecordModel)
                .where(CanonicalRecordModel.identity_kind == StableIdentityKind.EVENT.value)
                .order_by(CanonicalRecordModel.created_at.desc(), CanonicalRecordModel.stable_id.desc())
            )
        ).scalars()
        for record in records:
            if not isinstance(record.payload, dict) or record.payload.get("schema") != "candidate_inspection/v1":
                continue
            if (
                record.record_class == CanonicalRecordClass.IMMUTABLE.value
                and record.state == "recorded"
                and record.payload.get("inspection_record_identity") == record.stable_id
                and self._inspection_matches(record.payload, binding)
            ):
                return dict(record.payload)
        return None

    @staticmethod
    def _inspection_matches(
        payload: dict[str, Any],
        binding: CandidatePublicationBinding,
    ) -> bool:
        return (
            payload.get("candidate_id") == binding.candidate.stable_id
            and payload.get("build_generation_id") == f"build_generation:{binding.job.id}"
            and payload.get("entry_identity") == binding.frozen_input.entry_identity
            and payload.get("document_identity") == binding.frozen_input.document_identity
            and payload.get("bundle_id") == binding.frozen_input.bundle_id
            and payload.get("bundle_item_id") == binding.frozen_input.bundle_item_id
            and payload.get("editorial_source_revision") == binding.frozen_input.editorial_source_revision
            and payload.get("bundle_sha256") == binding.frozen_input.bundle_sha256
            and payload.get("bundle_item_sha256") == binding.frozen_input.bundle_item_sha256
            and payload.get("input_sha256") == binding.frozen_input.input_sha256
            and payload.get("frozen_input_sha256") == binding.frozen_input.frozen_input_sha256
            and payload.get("configuration_identity") == binding.configuration_identity
            and payload.get("configuration") == binding.frozen_input.embedding_configuration
            and payload.get("requested_generation") == binding.frozen_input.requested_generation
            and payload.get("chunk_sha256s") == [chunk.content_sha256 for chunk in binding.chunks]
            and "replaces_published_knowledge_version_identity" in payload
            and (
                payload.get("replaces_published_knowledge_version_identity") is None
                or isinstance(payload.get("replaces_published_knowledge_version_identity"), str)
            )
        )

    def _inspection_payload(
        self,
        binding: CandidatePublicationBinding,
        *,
        inspection_identity: str,
        actor_identity: str,
        replaces_published_knowledge_version_identity: str | None,
    ) -> dict[str, Any]:
        return {
            "schema": "candidate_inspection/v1",
            "inspection_record_identity": inspection_identity,
            "candidate_id": binding.candidate.stable_id,
            "build_generation_id": f"build_generation:{binding.job.id}",
            "entry_identity": binding.frozen_input.entry_identity,
            "document_identity": binding.frozen_input.document_identity,
            "bundle_id": binding.frozen_input.bundle_id,
            "bundle_sha256": binding.frozen_input.bundle_sha256,
            "bundle_item_id": binding.frozen_input.bundle_item_id,
            "bundle_item_sha256": binding.frozen_input.bundle_item_sha256,
            "editorial_source_revision": binding.frozen_input.editorial_source_revision,
            "input_sha256": binding.frozen_input.input_sha256,
            "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
            "requested_generation": binding.frozen_input.requested_generation,
            "configuration_identity": binding.configuration_identity,
            "configuration": binding.frozen_input.embedding_configuration,
            "chunk_sha256s": [chunk.content_sha256 for chunk in binding.chunks],
            "replaces_published_knowledge_version_identity": replaces_published_knowledge_version_identity,
            "inspected_by": actor_identity,
        }

    async def _projection(
        self,
        binding: CandidatePublicationBinding,
        inspection: dict[str, Any] | None,
    ) -> dict[str, Any]:
        acceptance = await self._latest_valid_acceptance(binding)
        if acceptance is not None and await self._exact_inspection_payload(
            binding, acceptance["inspection_record_identity"],
        ) is None:
            acceptance = None
        return {
            "candidate": {
                "candidate_id": binding.candidate.stable_id,
                "entry_identity": binding.frozen_input.entry_identity,
                "document_identity": binding.frozen_input.document_identity,
                "bundle_id": binding.frozen_input.bundle_id,
                "bundle_item_id": binding.frozen_input.bundle_item_id,
                "bundle_sha256": binding.frozen_input.bundle_sha256,
                "bundle_item_sha256": binding.frozen_input.bundle_item_sha256,
                "editorial_source_revision": binding.frozen_input.editorial_source_revision,
                "input_sha256": binding.frozen_input.input_sha256,
                "frozen_input_sha256": binding.frozen_input.frozen_input_sha256,
                "generation": binding.frozen_input.requested_generation,
                "configuration_identity": binding.configuration_identity,
                "configuration": binding.frozen_input.embedding_configuration,
                "metadata": {
                    key: value
                    for key, value in binding.frozen_input.artifact["entry"].items()
                    if key not in {"body", "acceptance_material"}
                },
                "chunks": [
                    {
                        "chunk_id": chunk.id,
                        "chunk_index": chunk.chunk_index,
                        "content_sha256": chunk.content_sha256,
                        "content": chunk.content,
                        "metadata": chunk.chunk_metadata,
                    }
                    for chunk in binding.chunks
                ],
            },
            "inspection": (
                {
                    "record_identity": inspection["inspection_record_identity"],
                    "candidate_id": inspection["candidate_id"],
                    "generation": inspection["requested_generation"],
                    "frozen_input_sha256": inspection["frozen_input_sha256"],
                    "configuration_identity": inspection["configuration_identity"],
                    "inspected_by": inspection["inspected_by"],
                    "replaces_published_knowledge_version_identity": inspection[
                        "replaces_published_knowledge_version_identity"
                    ],
                }
                if inspection is not None
                else None
            ),
            "acceptance": (
                {
                    "record_identity": acceptance["acceptance_record_identity"],
                    "inspection_record_identity": acceptance["inspection_record_identity"],
                    "accepted_by": acceptance["accepted_by"],
                    "frozen_input_sha256": acceptance["frozen_input_sha256"],
                    "configuration_identity": acceptance["configuration_identity"],
                    **acceptance["result"],
                }
                if acceptance is not None
                else None
            ),
            "replacement": await self._replacement_projection(
                binding,
                replacement_version_identity=(
                    inspection["replaces_published_knowledge_version_identity"]
                    if inspection is not None
                    else None
                ),
                inspect_snapshot=inspection is not None,
            ),
        }

    async def _replacement_projection(
        self,
        binding: CandidatePublicationBinding,
        *,
        replacement_version_identity: str | None,
        inspect_snapshot: bool,
    ) -> dict[str, Any]:
        if inspect_snapshot:
            current = (
                await self.session.get(PublishedKnowledgeVersion, replacement_version_identity)
                if replacement_version_identity is not None
                else None
            )
            if replacement_version_identity is not None:
                if current is None or current.entry_identity != binding.frozen_input.entry_identity:
                    raise self._candidate_integrity_error(binding.candidate.stable_id)
                await self._assert_version_integrity(current)
        else:
            pointer = await self._current_pointer(binding.frozen_input.entry_identity)
            current = await self._current_version(pointer)
        if current is None:
            return {
                "effect": "create",
                "current_published_knowledge_version": None,
                "diff": {
                    "schema": "candidate_replacement_diff/v1",
                    "added": [
                        {
                            "chunk_index": chunk.chunk_index,
                            "candidate_content_sha256": chunk.content_sha256,
                        }
                        for chunk in binding.chunks
                    ],
                    "removed": [],
                    "changed": [],
                },
            }
        current_chunks = (
            await self.session.execute(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == current.document_identity,
                    DocumentChunk.generation == current.generation,
                )
                .order_by(DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
            )
        ).scalars().all()
        candidate_by_index = {chunk.chunk_index: chunk for chunk in binding.chunks}
        published_by_index = {chunk.chunk_index: chunk for chunk in current_chunks}
        added: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        changed: list[dict[str, Any]] = []
        for chunk_index in sorted(set(candidate_by_index) | set(published_by_index)):
            candidate_chunk = candidate_by_index.get(chunk_index)
            published_chunk = published_by_index.get(chunk_index)
            if candidate_chunk is None and published_chunk is not None:
                removed.append(
                    {
                        "chunk_index": chunk_index,
                        "published_content_sha256": published_chunk.content_sha256,
                        "published_content": published_chunk.content,
                    }
                )
            elif published_chunk is None and candidate_chunk is not None:
                added.append(
                    {
                        "chunk_index": chunk_index,
                        "candidate_content_sha256": candidate_chunk.content_sha256,
                        "candidate_content": candidate_chunk.content,
                    }
                )
            elif (
                candidate_chunk is not None
                and published_chunk is not None
                and candidate_chunk.content_sha256 != published_chunk.content_sha256
            ):
                changed.append(
                    {
                        "chunk_index": chunk_index,
                        "candidate_content_sha256": candidate_chunk.content_sha256,
                        "published_content_sha256": published_chunk.content_sha256,
                        "candidate_content": candidate_chunk.content,
                        "published_content": published_chunk.content,
                    }
                )
        return {
            "effect": "replace",
            "current_published_knowledge_version": self._version_projection(current),
            "diff": {
                "schema": "candidate_replacement_diff/v1",
                "added": added,
                "removed": removed,
                "changed": changed,
            },
        }

    @staticmethod
    def _candidate_integrity_error(candidate_id: str) -> AppError:
        return AppError(
            status_code=409,
            code="CANDIDATE_PUBLICATION_INTEGRITY_FAILED",
            message="Candidate immutable publication bindings do not verify",
            detail={"candidate_id": candidate_id},
        )
