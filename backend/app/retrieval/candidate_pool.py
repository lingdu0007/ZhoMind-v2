from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.config import Settings
from app.common.exceptions import AppError
from app.contracts.canonical import CanonicalRecordClass, StableIdentity, StableIdentityKind
from app.editorial_authority.service import EditorialAuthorityService
from app.maintenance.containment import active_answer_blocks, publication_is_blocked
from app.model.canonical import CanonicalRecordModel
from app.model.document import Document, DocumentChunk
from app.rag.interfaces import RetrieveResult
from app.retrieval.policy import PILOT_RETRIEVAL_PROFILE_ID, get_retrieval_policy
from app.retrieval.sparse_bm25 import Bm25Chunk, LiteralPreservingTokenizer, SparseBm25Index
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
    PublishedKnowledgePointer,
    PublishedKnowledgeVersion,
)

_ALLOWED_ASSURANCE_LEVELS = frozenset({"source_grounded", "claim_linked", "release_assured"})
_KNOWN_ACCESS_SCOPES = frozenset({"public", "controlled_internal"})
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _PublishedRuntimeProjection:
    content_sha256: str
    metadata: dict[str, object]


class CurrentRetrievalAuthority(Protocol):
    async def get_retrieval_authority(self, entry_id: str, *, now: datetime | None = None) -> dict[str, Any]: ...

    async def get_retrieval_authority_for_revision(
        self,
        entry_id: str,
        revision_identity: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]: ...


class AuthorizedRetrievalCandidatePool:
    """Build the Pilot's pre-sufficiency pool from current published versions only."""

    name = "authorized-pilot-sparse-bm25-retriever"

    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        now: datetime | None = None,
        editorial_authority: CurrentRetrievalAuthority | None = None,
        allowed_access_scopes: frozenset[str] | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._now = now
        self._authority = editorial_authority or EditorialAuthorityService(session)
        self._allowed_access_scopes = (
            _KNOWN_ACCESS_SCOPES if allowed_access_scopes is None else allowed_access_scopes
        )

    async def _eligible_candidates(self) -> tuple[dict[str, dict[str, Any]], list[Bm25Chunk], list[dict[str, str]]]:
        policy = self._pilot_policy()
        maintenance_blocks = await active_answer_blocks(self._session)
        statement = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .order_by(Document.id.asc(), DocumentChunk.generation.asc(), DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
        )
        result = await self._session.execute(statement)
        candidates: dict[str, dict[str, Any]] = {}
        bm25_chunks: list[Bm25Chunk] = []
        exclusions: list[dict[str, str]] = []
        authority_cache: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        published_authority_cache: dict[tuple[str, str], tuple[dict[str, Any] | None, str | None]] = {}
        published_runtime_projection_cache: dict[
            tuple[str, int],
            _PublishedRuntimeProjection | None,
        ] = {}

        for chunk, document in result.all():
            metadata = dict(chunk.chunk_metadata) if isinstance(chunk.chunk_metadata, dict) else {}
            reason = self._published_static_exclusion_reason(chunk=chunk, document=document, metadata=metadata)
            if reason is not None:
                exclusions.append({"chunk_id": chunk.id, "reason": reason})
                continue

            metadata_identity = self._metadata_identity(metadata)
            if metadata_identity is None:
                exclusions.append({"chunk_id": chunk.id, "reason": "identity_metadata_missing"})
                continue
            entry_id, entry_identity, revision_identity, section_id = metadata_identity
            published_version_state = await self._published_knowledge_version(
                document=document,
                metadata=metadata,
            )
            if published_version_state is None:
                exclusions.append({"chunk_id": chunk.id, "reason": "published_version_identity_invalid"})
                continue
            publication_identity, published_version = published_version_state
            if publication_is_blocked(
                maintenance_blocks, entry_identity=entry_identity, publication_identity=publication_identity,
            ):
                exclusions.append({"chunk_id": chunk.id, "reason": "maintenance_containment"})
                continue
            authority, authority_error = await self._current_authority(entry_id, authority_cache)
            reason = self._authority_exclusion_reason(authority, authority_error)
            needs_published_version_authority = (
                reason == "entry_not_published"
                or (
                    reason in {None, "source_unavailable"}
                    and authority is not None
                    and authority.get("editorial_revision_identity") != revision_identity
                )
            )
            if needs_published_version_authority:
                if published_version is None:
                    exclusions.append({"chunk_id": chunk.id, "reason": reason or "editorial_revision_mismatch"})
                    continue
                preserved_authority, preserved_error = await self._published_version_authority(
                    version=published_version,
                    entry_id=entry_id,
                    entry_identity=entry_identity,
                    revision_identity=revision_identity,
                    current_authority=authority,
                    cache=published_authority_cache,
                )
                if preserved_authority is None:
                    exclusions.append(
                        {
                            "chunk_id": chunk.id,
                            "reason": preserved_error or reason or "published_version_authority_invalid",
                        }
                    )
                    continue
                authority = preserved_authority
            elif reason is not None:
                exclusions.append({"chunk_id": chunk.id, "reason": reason})
                continue
            assert authority is not None
            if authority.get("entry_identity") != entry_identity:
                exclusions.append({"chunk_id": chunk.id, "reason": "current_authority_missing"})
                continue
            if authority.get("editorial_revision_identity") != revision_identity:
                exclusions.append({"chunk_id": chunk.id, "reason": "editorial_revision_mismatch"})
                continue
            decision_query = authority.get("decision_query")
            if not isinstance(decision_query, str) or not decision_query.strip():
                exclusions.append({"chunk_id": chunk.id, "reason": "decision_query_missing"})
                continue
            source_relationships = self._section_relationships(authority, section_id)
            if source_relationships is None:
                exclusions.append({"chunk_id": chunk.id, "reason": "source_relationship_missing"})
                continue
            if any(item["access_scope"] not in self._allowed_access_scopes for item in source_relationships):
                exclusions.append({"chunk_id": chunk.id, "reason": "unauthorized_access_scope"})
                continue
            reason = self._metadata_binding_exclusion_reason(
                metadata,
                source_relationships=source_relationships,
                assurance_level=authority.get("assurance_level"),
                applicability_conditions=authority.get("applicability_conditions"),
                non_applicability_conditions=authority.get("non_applicability_conditions"),
                freshness_triggers=authority.get("freshness_triggers"),
            )
            if reason is not None:
                exclusions.append({"chunk_id": chunk.id, "reason": reason})
                continue
            reason = await self._published_runtime_projection_exclusion_reason(
                version=published_version,
                chunk=chunk,
                metadata=metadata,
                cache=published_runtime_projection_cache,
            )
            if reason is not None:
                exclusions.append({"chunk_id": chunk.id, "reason": reason})
                continue
            candidate = self._published_candidate(
                chunk=chunk,
                document=document,
                metadata=metadata,
                authority=authority,
                source_relationships=source_relationships,
                profile_identity=policy.identity,
                publication_identity=publication_identity,
            )
            candidates[chunk.id] = candidate
            bm25_chunks.append(
                Bm25Chunk(
                    chunk_id=chunk.id,
                    document=candidate["publication_identity"],
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    tie_breaker=(
                        candidate["publication_identity"],
                        chunk.chunk_index,
                        chunk.id,
                    ),
                )
            )

        return candidates, bm25_chunks, exclusions

    async def measurement_snapshot(self) -> dict:
        """Hash the same authorized corpus before ranking, without exposing text."""
        candidates, chunks, _ = await self._eligible_candidates()
        return {
            "editorial_inputs_sha256": canonical_json_sha256(sorted({
                (candidate["entry_identity"], candidate["editorial_revision_identity"])
                for candidate in candidates.values()
            })),
            "corpus": "corpus:" + canonical_json_sha256([
                {"candidate": candidate, "content_sha256": hashlib.sha256(chunk.content.encode()).hexdigest()}
                for chunk in chunks if (candidate := candidates.get(chunk.chunk_id)) is not None
            ]),
            "active_entries": len({candidate["entry_identity"] for candidate in candidates.values()}),
            "eligible_chunks": len(chunks),
        }

    async def retrieve(self, query: str, top_k: int) -> RetrieveResult:
        del top_k
        policy = self._pilot_policy()
        candidates, bm25_chunks, exclusions = await self._eligible_candidates()
        items, ranked_exclusions = self._rank_pre_sufficiency_candidates(
            candidates=candidates,
            bm25_chunks=bm25_chunks,
            query=query,
        )
        exclusions.extend(ranked_exclusions)

        return RetrieveResult(
            items=items,
            strategy=policy.strategy,
            lexical_candidate_count=len(bm25_chunks),
            merged_count=len(items),
            profile_identity=policy.identity,
            candidate_pool_scope="published_knowledge",
            candidate_exclusions=exclusions,
        )

    async def preview_candidate(self, candidate_id: str, query: str) -> RetrieveResult:
        """Return a Candidate-only diagnostic pool for an authenticated administrator."""
        policy = self._pilot_policy()
        candidate_record = await self._session.get(CanonicalRecordModel, candidate_id)
        if (
            candidate_record is None
            or candidate_record.identity_kind != "candidate"
            or candidate_record.state != "candidate_ready"
            or candidate_record.record_class != "immutable"
            or not isinstance(candidate_record.payload, dict)
            or candidate_record.payload.get("schema") != "candidate_build_candidate/v1"
        ):
            raise AppError(
                status_code=404,
                code="CANDIDATE_PREVIEW_NOT_READY",
                message="Candidate preview is unavailable",
                detail={"candidate_id": candidate_id},
            )

        result = await self._session.execute(
            select(CandidateBuildChunk, CandidateBuildJob)
            .join(CandidateBuildJob, CandidateBuildChunk.job_id == CandidateBuildJob.id)
            .where(
                CandidateBuildChunk.candidate_id == candidate_id,
                CandidateBuildJob.candidate_id == candidate_id,
                CandidateBuildJob.status == "candidate_ready",
                CandidateBuildChunk.attempt == CandidateBuildJob.attempt,
            )
            .order_by(
                CandidateBuildChunk.chunk_index.asc(),
                CandidateBuildChunk.id.asc(),
            )
        )
        rows: list[tuple[CandidateBuildChunk, CandidateBuildJob]] = [
            (row[0], row[1]) for row in result.all()
        ]
        binding = await self._candidate_preview_binding(
            candidate_record=candidate_record,
            candidate_id=candidate_id,
            rows=rows,
        )
        candidates: dict[str, dict[str, Any]] = {}
        bm25_chunks: list[Bm25Chunk] = []
        for chunk, job in rows:
            metadata = dict(chunk.chunk_metadata) if isinstance(chunk.chunk_metadata, dict) else {}
            section_id = self._candidate_preview_section_id(
                candidate_id=candidate_id,
                chunk=chunk,
                job=job,
                metadata=metadata,
                binding=binding,
            )
            source_relationships = self._section_relationships(binding, section_id)
            if source_relationships is None:
                raise self._candidate_preview_integrity_error(candidate_id)
            candidate = self._candidate_preview_item(
                chunk=chunk,
                job=job,
                metadata=metadata,
                binding=binding,
                section_id=section_id,
                source_relationships=source_relationships,
                profile_identity=policy.identity,
            )
            candidates[chunk.id] = candidate
            bm25_chunks.append(
                Bm25Chunk(
                    chunk_id=chunk.id,
                    document=candidate["candidate_version"],
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    tie_breaker=(
                        candidate["candidate_version"],
                        chunk.chunk_index,
                        chunk.id,
                    ),
                )
            )

        items, exclusions = self._rank_pre_sufficiency_candidates(
            candidates=candidates,
            bm25_chunks=bm25_chunks,
            query=query,
        )
        return RetrieveResult(
            items=items,
            strategy=policy.strategy,
            lexical_candidate_count=len(bm25_chunks),
            merged_count=len(items),
            profile_identity=policy.identity,
            candidate_pool_scope="candidate_preview",
            candidate_exclusions=exclusions,
        )

    def _pilot_policy(self):
        policy = get_retrieval_policy(self._settings)
        if policy.identity != PILOT_RETRIEVAL_PROFILE_ID:
            raise ValueError("authorized retrieval candidate pool requires retrieval-answer-policy/pilot-v1")
        return policy

    def _rank_pre_sufficiency_candidates(
        self,
        *,
        candidates: dict[str, dict[str, Any]],
        bm25_chunks: list[Bm25Chunk],
        query: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        policy = self._pilot_policy()
        index = SparseBm25Index(
            bm25_chunks,
            tokenizer=LiteralPreservingTokenizer(),
            k1=policy.bm25_k1 or 1.5,
            b=policy.bm25_b or 0.75,
        )
        ranked = index.search(query, top_k=len(bm25_chunks))
        items: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []
        exact_content_seen: set[str] = set()
        entry_section_seen: set[tuple[str, str]] = set()
        for ranked_item in ranked:
            candidate = dict(candidates[ranked_item["chunk_id"]])
            content_sha256 = str(candidate["content_sha256"])
            entry_section = (str(candidate["entry_identity"]), str(candidate["section_id"]))
            if content_sha256 in exact_content_seen:
                exclusions.append({"chunk_id": candidate["chunk_id"], "reason": "exact_duplicate"})
                continue
            if entry_section in entry_section_seen:
                exclusions.append({"chunk_id": candidate["chunk_id"], "reason": "entry_section_duplicate"})
                continue
            exact_content_seen.add(content_sha256)
            entry_section_seen.add(entry_section)
            # Raw BM25 scores are ordering observations only. They never change
            # the pre-sufficiency evidence status set by the candidate boundary.
            candidate["raw_score"] = ranked_item["score"]
            candidate["score"] = ranked_item["score"]
            items.append(candidate)
            if len(items) >= policy.candidate_depth:
                break
        return items, exclusions

    @staticmethod
    def _published_static_exclusion_reason(
        *,
        chunk: DocumentChunk,
        document: Document,
        metadata: dict[str, Any],
    ) -> str | None:
        if metadata.get("candidate_build") is True:
            return "candidate_not_in_preview"
        if document.deleted_at is not None:
            return "withdrawn"
        if document.published_generation < 1:
            return "entry_not_published"
        if chunk.generation != document.published_generation:
            return "superseded_publication"
        return None

    @staticmethod
    def _metadata_identity(metadata: dict[str, Any]) -> tuple[str, str, str, str] | None:
        entry_id = metadata.get("entry_id")
        entry_identity = metadata.get("entry_identity")
        revision_identity = metadata.get("editorial_revision_identity")
        section_id = metadata.get("section_id")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or not isinstance(revision_identity, str)
            or not isinstance(entry_identity, str)
            or not isinstance(section_id, str)
            or not section_id
        ):
            return None
        try:
            entry = StableIdentity.from_stable_id(entry_identity)
            revision = StableIdentity.from_stable_id(revision_identity)
        except ValueError:
            return None
        if (
            entry.kind is not StableIdentityKind.ENTRY
            or entry.value != entry_id
            or revision.kind is not StableIdentityKind.EDITORIAL_REVISION
        ):
            return None
        return entry_id, entry_identity, revision_identity, section_id

    async def _current_authority(
        self,
        entry_id: str,
        cache: dict[str, tuple[dict[str, Any] | None, str | None]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        cached = cache.get(entry_id)
        if cached is not None:
            return cached
        try:
            authority = await self._authority.get_retrieval_authority(entry_id, now=self._now)
        except AppError as exc:
            if exc.code == "EDITORIAL_ENTRY_NOT_FOUND":
                cached = (None, "current_authority_missing")
            elif "SOURCE" in exc.code:
                cached = (None, "source_unavailable")
            elif "ASSURANCE" in exc.code or "ACCEPTANCE" in exc.code:
                cached = (None, "assurance_ineligible")
            else:
                cached = (None, "current_authority_invalid")
        except (RuntimeError, ValueError):
            cached = (None, "current_authority_invalid")
        else:
            cached = (authority, None) if isinstance(authority, dict) else (None, "current_authority_invalid")
        cache[entry_id] = cached
        return cached

    @staticmethod
    def _authority_exclusion_reason(authority: dict[str, Any] | None, error: str | None) -> str | None:
        if error is not None:
            return error
        if authority is None:
            return "current_authority_invalid"
        if authority.get("answer_eligible") is True:
            return None
        raw_reasons = authority.get("eligibility_reasons")
        reasons = set(raw_reasons) if isinstance(raw_reasons, list) and all(isinstance(item, str) for item in raw_reasons) else set()
        if "unauthorized_access_scope" in reasons:
            return "unauthorized_access_scope"
        if "assurance_ineligible" in reasons:
            return "assurance_ineligible"
        if "known_contradiction" in reasons:
            return "known_contradiction"
        if "integrity_defect" in reasons:
            return "integrity_defect"
        if "needs_review_grace_expired" in reasons:
            return "needs_review_grace_expired"
        if {"source_unavailable", "source_availability_missing", "decisive_source_loss"} & reasons:
            return "source_unavailable"
        if "applicability_not_explicit" in reasons:
            return "applicability_missing"
        if authority.get("lifecycle_state") == "withdrawn":
            return "withdrawn"
        return "entry_not_published"

    async def _published_version_authority(
        self,
        *,
        version: PublishedKnowledgeVersion,
        entry_id: str,
        entry_identity: str,
        revision_identity: str,
        current_authority: dict[str, Any] | None,
        cache: dict[tuple[str, str], tuple[dict[str, Any] | None, str | None]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if current_authority is None:
            return None, "current_authority_missing"
        reasons = current_authority.get("eligibility_reasons")
        if current_authority.get("editorial_revision_identity") != revision_identity:
            if current_authority.get("answer_eligible") is not True and (
                not isinstance(reasons, list)
                or set(reasons) - {
                    "editorial_approval_missing", "not_published",
                    "source_unavailable", "source_availability_missing", "decisive_source_loss",
                }
            ):
                return None, "entry_not_published"
        elif not isinstance(reasons, list) or set(reasons) - {
            "editorial_approval_missing",
            "not_published",
        }:
            return None, "entry_not_published"
        cache_key = (version.id, revision_identity)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        candidate = await self._session.get(CanonicalRecordModel, version.candidate_id)
        job = await self._session.scalar(
            select(CandidateBuildJob).where(CandidateBuildJob.candidate_id == version.candidate_id)
        )
        if (
            candidate is None
            or candidate.stable_id != version.candidate_id
            or candidate.identity_kind != StableIdentityKind.CANDIDATE.value
            or candidate.record_class != CanonicalRecordClass.IMMUTABLE.value
            or candidate.state != "candidate_ready"
            or not isinstance(candidate.payload, dict)
            or candidate.payload.get("schema") != "candidate_build_candidate/v1"
            or job is None
            or job.candidate_id != version.candidate_id
            or job.entry_identity != entry_identity
        ):
            cached = (None, "published_version_binding_invalid")
            cache[cache_key] = cached
            return cached
        try:
            frozen_input = await load_frozen_candidate_build_input(self._session, job.id)
        except AppError:
            cached = (None, "published_version_binding_invalid")
            cache[cache_key] = cached
            return cached
        artifact = frozen_input.artifact
        if (
            not frozen_input.matches_job(job)
            or version.entry_identity != entry_identity
            or version.document_identity != runtime_document_identity(frozen_input.document_identity)
            or version.generation != frozen_input.requested_generation
            or version.frozen_input_sha256 != frozen_input.frozen_input_sha256
            or candidate.payload.get("entry_identity") != entry_identity
            or candidate.payload.get("frozen_input_sha256") != frozen_input.frozen_input_sha256
            or artifact.get("entry_id") != entry_id
            or artifact.get("entry_identity") != entry_identity
        ):
            cached = (None, "published_version_binding_invalid")
            cache[cache_key] = cached
            return cached
        if artifact.get("editorial_revision_identity") != revision_identity:
            cached = (None, "editorial_revision_mismatch")
            cache[cache_key] = cached
            return cached
        try:
            authority = await self._authority.get_retrieval_authority_for_revision(
                entry_id,
                revision_identity,
                now=self._now,
            )
        except AppError as exc:
            if "SOURCE" in exc.code:
                cached = (None, "source_unavailable")
            elif "ASSURANCE" in exc.code or "ACCEPTANCE" in exc.code:
                cached = (None, "assurance_ineligible")
            else:
                cached = (None, "published_version_authority_invalid")
        except (RuntimeError, ValueError):
            cached = (None, "published_version_authority_invalid")
        else:
            if (
                not isinstance(authority, dict)
                or authority.get("answer_eligible") is not True
                or authority.get("entry_id") != entry_id
                or authority.get("entry_identity") != entry_identity
                or authority.get("editorial_revision_identity") != revision_identity
            ):
                cached = (None, "published_version_authority_invalid")
            else:
                cached = (authority, None)
        cache[cache_key] = cached
        return cached

    @staticmethod
    def _section_relationships(authority: dict[str, Any], section_id: str) -> list[dict[str, str]] | None:
        relationships_by_section = authority.get("section_source_relationships")
        if not isinstance(relationships_by_section, dict):
            return None
        return AuthorizedRetrievalCandidatePool._normalize_relationships(relationships_by_section.get(section_id))

    @staticmethod
    def _normalize_relationships(value: object) -> list[dict[str, str]] | None:
        if not isinstance(value, list) or not value:
            return None
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for relationship in value:
            if not isinstance(relationship, dict) or set(relationship) != {
                "source_identity",
                "availability",
                "access_scope",
            }:
                return None
            source_identity = relationship.get("source_identity")
            availability = relationship.get("availability")
            access_scope = relationship.get("access_scope")
            if (
                not isinstance(source_identity, str)
                or not isinstance(availability, str)
                or not isinstance(access_scope, str)
                or availability != "verified_usable"
                or access_scope not in _KNOWN_ACCESS_SCOPES
                or source_identity in seen
            ):
                return None
            try:
                source = StableIdentity.from_stable_id(source_identity)
            except ValueError:
                return None
            if source.kind is not StableIdentityKind.SOURCE:
                return None
            seen.add(source_identity)
            normalized.append(
                {
                    "source_identity": source_identity,
                    "availability": availability,
                    "access_scope": access_scope,
                }
            )
        return sorted(normalized, key=lambda item: item["source_identity"])

    @staticmethod
    def _metadata_binding_exclusion_reason(
        metadata: dict[str, Any],
        *,
        source_relationships: list[dict[str, str]],
        assurance_level: object,
        applicability_conditions: object,
        non_applicability_conditions: object,
        freshness_triggers: object,
    ) -> str | None:
        metadata_relationships = AuthorizedRetrievalCandidatePool._normalize_relationships(metadata.get("source_relationships"))
        if metadata_relationships is None:
            return "source_relationship_missing"
        if metadata_relationships != source_relationships:
            return "source_relationship_mismatch"
        candidate_evidence_source_identity = metadata.get("candidate_evidence_source_identity")
        if candidate_evidence_source_identity is not None and (
            not isinstance(candidate_evidence_source_identity, str)
            or candidate_evidence_source_identity
            not in {relationship["source_identity"] for relationship in source_relationships}
        ):
            return "candidate_evidence_projection_invalid"
        if metadata.get("assurance_level") != assurance_level or assurance_level not in _ALLOWED_ASSURANCE_LEVELS:
            return "assurance_ineligible"
        if not isinstance(metadata.get("applicability_conditions"), list) or not metadata["applicability_conditions"]:
            return "applicability_missing"
        if not isinstance(applicability_conditions, list) or not applicability_conditions:
            return "applicability_missing"
        if canonical_json_sha256(metadata["applicability_conditions"]) != canonical_json_sha256(applicability_conditions):
            return "applicability_metadata_mismatch"
        if non_applicability_conditions is not None:
            if not isinstance(non_applicability_conditions, list):
                return "non_applicability_missing"
            metadata_non_applicability = metadata.get("non_applicability_conditions")
            if metadata_non_applicability is not None and (
                not isinstance(metadata_non_applicability, list)
                or canonical_json_sha256(metadata_non_applicability)
                != canonical_json_sha256(non_applicability_conditions)
            ):
                return "non_applicability_metadata_mismatch"
        if not isinstance(metadata.get("freshness_triggers"), list) or not metadata["freshness_triggers"]:
            return "freshness_metadata_missing"
        if not isinstance(freshness_triggers, list) or not freshness_triggers:
            return "freshness_metadata_missing"
        if canonical_json_sha256(metadata["freshness_triggers"]) != canonical_json_sha256(freshness_triggers):
            return "freshness_metadata_mismatch"
        return None

    async def _published_knowledge_version(
        self,
        *,
        document: Document,
        metadata: dict[str, Any],
    ) -> tuple[str, PublishedKnowledgeVersion | None] | None:
        if "published_knowledge_version_identity" not in metadata:
            if document.file_type == "reviewed_release_bundle":
                return None
            return f"published_knowledge_version:legacy:{document.id}:v{document.published_generation}", None
        recorded_identity = metadata["published_knowledge_version_identity"]
        if not isinstance(recorded_identity, str):
            return None
        try:
            identity = StableIdentity.from_stable_id(recorded_identity)
        except ValueError:
            return None
        if identity.kind is not StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION:
            return None
        version = await self._session.get(PublishedKnowledgeVersion, identity.stable_id)
        record = await self._session.get(CanonicalRecordModel, identity.stable_id)
        entry_identity = metadata.get("entry_identity")
        if (
            version is None
            or record is None
            or record.identity_kind != StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION.value
            or record.record_class != CanonicalRecordClass.IMMUTABLE.value
            or record.state != "published"
            or not isinstance(record.payload, dict)
            or record.payload.get("schema") != "published_knowledge_version/v1"
            or not isinstance(entry_identity, str)
            or version.entry_identity != entry_identity
            or version.document_identity != document.id
            or version.generation != document.published_generation
            or any(
                record.payload.get(field) != value
                for field, value in {
                    "candidate_id": version.candidate_id,
                    "entry_identity": version.entry_identity,
                    "document_identity": version.document_identity,
                    "generation": version.generation,
                    "bundle_sha256": version.bundle_sha256,
                    "frozen_input_sha256": version.frozen_input_sha256,
                    "configuration_identity": version.configuration_identity,
                    "inspection_record_identity": version.inspection_record_identity,
                    "acceptance_record_identity": version.acceptance_record_identity,
                    "supersedes_version_id": version.supersedes_version_id,
                }.items()
            )
        ):
            return None
        pointer = await self._session.get(PublishedKnowledgePointer, version.entry_identity)
        if (
            pointer is None
            or pointer.current_version_id != version.id
            or pointer.entry_identity != version.entry_identity
            or pointer.document_identity != version.document_identity
            or pointer.generation != version.generation
        ):
            return None
        return version.id, version

    async def _published_runtime_projection_exclusion_reason(
        self,
        *,
        version: PublishedKnowledgeVersion | None,
        chunk: DocumentChunk,
        metadata: dict[str, Any],
        cache: dict[tuple[str, int], _PublishedRuntimeProjection | None],
    ) -> str | None:
        if version is None:
            return None
        cache_key = (version.id, chunk.chunk_index)
        if cache_key not in cache:
            cache[cache_key] = await self._published_runtime_projection(
                version=version,
                chunk_index=chunk.chunk_index,
            )
        expected = cache[cache_key]
        if expected is None:
            return "published_runtime_projection_invalid"
        if (
            chunk.content_sha256 != expected.content_sha256
            or hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            != expected.content_sha256
            or metadata != expected.metadata
        ):
            return "published_runtime_projection_invalid"
        return None

    async def _published_runtime_projection(
        self,
        *,
        version: PublishedKnowledgeVersion,
        chunk_index: int,
    ) -> _PublishedRuntimeProjection | None:
        candidate = await self._session.get(CanonicalRecordModel, version.candidate_id)
        job = await self._session.scalar(
            select(CandidateBuildJob).where(CandidateBuildJob.candidate_id == version.candidate_id)
        )
        if (
            candidate is None
            or candidate.stable_id != version.candidate_id
            or candidate.identity_kind != StableIdentityKind.CANDIDATE.value
            or candidate.record_class != CanonicalRecordClass.IMMUTABLE.value
            or candidate.state != "candidate_ready"
            or not isinstance(candidate.payload, dict)
            or candidate.payload.get("schema") != "candidate_build_candidate/v1"
            or job is None
            or job.candidate_id != version.candidate_id
            or job.entry_identity != version.entry_identity
        ):
            return None
        try:
            frozen_input = await load_frozen_candidate_build_input(self._session, job.id)
        except AppError:
            return None
        artifact = frozen_input.artifact
        entry = artifact.get("entry")
        if (
            not frozen_input.matches_job(job)
            or version.document_identity != runtime_document_identity(frozen_input.document_identity)
            or version.generation != frozen_input.requested_generation
            or version.frozen_input_sha256 != frozen_input.frozen_input_sha256
            or candidate.payload.get("entry_identity") != version.entry_identity
            or candidate.payload.get("frozen_input_sha256") != frozen_input.frozen_input_sha256
            or artifact.get("entry_identity") != version.entry_identity
            or not isinstance(entry, dict)
            or entry.get("assurance_level") not in _ALLOWED_ASSURANCE_LEVELS
        ):
            return None
        candidate_chunks = (
            await self._session.execute(
                select(CandidateBuildChunk).where(
                    CandidateBuildChunk.candidate_id == version.candidate_id,
                    CandidateBuildChunk.job_id == job.id,
                    CandidateBuildChunk.chunk_index == chunk_index,
                )
            )
        ).scalars().all()
        if len(candidate_chunks) != 1:
            return None
        candidate_chunk = candidate_chunks[0]
        expected_hashes = candidate.payload.get("chunk_sha256s")
        if (
            not isinstance(expected_hashes, list)
            or chunk_index < 0
            or chunk_index >= len(expected_hashes)
            or not isinstance(expected_hashes[chunk_index], str)
            or candidate_chunk.document_identity != frozen_input.document_identity
            or candidate_chunk.generation != frozen_input.requested_generation
            or candidate_chunk.attempt != job.attempt
            or candidate_chunk.content_sha256 != expected_hashes[chunk_index]
            or hashlib.sha256(candidate_chunk.content.encode("utf-8")).hexdigest()
            != candidate_chunk.content_sha256
            or not isinstance(candidate_chunk.chunk_metadata, dict)
        ):
            return None
        section_id = candidate_chunk.chunk_metadata.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            return None
        try:
            expected_metadata = candidate_frozen_chunk_metadata(
                artifact=artifact,
                entry_identity=version.entry_identity,
                chunk_strategy=frozen_input.chunk_strategy,
                section_id=section_id,
            )
            runtime_metadata = candidate_frozen_published_chunk_metadata(
                artifact=artifact,
                entry_identity=version.entry_identity,
                chunk_strategy=frozen_input.chunk_strategy,
                section_id=section_id,
                publication_identity=version.id,
            )
        except ValueError:
            return None
        if candidate_chunk.chunk_metadata != expected_metadata:
            return None
        return _PublishedRuntimeProjection(
            content_sha256=candidate_chunk.content_sha256,
            metadata=runtime_metadata,
        )

    @staticmethod
    def _published_candidate(
        *,
        chunk: DocumentChunk,
        document: Document,
        metadata: dict[str, Any],
        authority: dict[str, Any],
        source_relationships: list[dict[str, str]],
        profile_identity: str,
        publication_identity: str,
    ) -> dict[str, Any]:
        entry_id = str(authority["entry_id"])
        entry_identity = str(authority["entry_identity"])
        editorial_revision_identity = str(metadata["editorial_revision_identity"])
        section_id = str(metadata["section_id"])
        access_scopes = sorted({item["access_scope"] for item in source_relationships})
        access_scope = access_scopes[0] if len(access_scopes) == 1 else "mixed_team_shared"
        recorded_version = metadata.get("publication_version")
        publication_version = (
            publication_identity
            if metadata.get("published_knowledge_version_identity") == publication_identity
            else (
                recorded_version.strip()
                if isinstance(recorded_version, str) and recorded_version.strip()
                else f"v{document.published_generation}"
            )
        )
        authoritative_metadata = dict(metadata)
        authoritative_metadata.update(
            {
                "entry_id": entry_id,
                "entry_identity": entry_identity,
                "editorial_revision_identity": editorial_revision_identity,
                "source_relationships": source_relationships,
                "assurance_level": authority["assurance_level"],
                "applicability_conditions": authority["applicability_conditions"],
                "freshness_triggers": authority["freshness_triggers"],
                "lifecycle_state": authority["lifecycle_state"],
                "decision_query": authority["decision_query"],
                "publication_version": publication_version,
            }
        )
        if isinstance(authority.get("non_applicability_conditions"), list):
            authoritative_metadata["non_applicability_conditions"] = authority["non_applicability_conditions"]
        if authority.get("release_assurance_snapshot") is not None:
            authoritative_metadata["release_assurance_snapshot"] = authority["release_assurance_snapshot"]
        projected = {
            "chunk_id": chunk.id,
            "document_id": document.id,
            "generation": chunk.generation,
            "chunk_index": chunk.chunk_index,
            "content_sha256": chunk.content_sha256,
            "content_preview": chunk.content[:1200],
            "content_length": len(chunk.content),
            "metadata": authoritative_metadata,
            "retrieval_source": "sparse_bm25",
            "entry_id": entry_id,
            "entry_identity": entry_identity,
            "editorial_revision_identity": editorial_revision_identity,
            "publication_identity": publication_identity,
            "publication_version": publication_version,
            "section_id": section_id,
            "section_identity": f"{entry_identity}#{section_id}",
            "decision_query": authority["decision_query"],
            "source_relationships": source_relationships,
            "assurance_level": authority["assurance_level"],
            "applicability_conditions": list(authority["applicability_conditions"]),
            "freshness_triggers": list(authority["freshness_triggers"]),
            "access_scope": access_scope,
            "chunk_identity": {
                "document_id": document.id,
                "generation": chunk.generation,
                "chunk_index": chunk.chunk_index,
                "content_sha256": chunk.content_sha256,
            },
            "retrieval_profile_identity": profile_identity,
            "answer_evidence_eligible": True,
        }
        if isinstance(authority.get("non_applicability_conditions"), list):
            projected["non_applicability_conditions"] = list(authority["non_applicability_conditions"])
        return projected

    async def _candidate_preview_binding(
        self,
        *,
        candidate_record: CanonicalRecordModel,
        candidate_id: str,
        rows: list[tuple[CandidateBuildChunk, CandidateBuildJob]],
    ) -> dict[str, Any]:
        if not rows or len({job.id for _, job in rows}) != 1:
            raise self._candidate_preview_integrity_error(candidate_id)
        job = rows[0][1]
        try:
            frozen_input = await load_frozen_candidate_build_input(self._session, job.id)
        except AppError as exc:
            raise self._candidate_preview_integrity_error(candidate_id) from exc
        if not frozen_input.matches_job(job):
            raise self._candidate_preview_integrity_error(candidate_id)

        expected_identity = StableIdentity(StableIdentityKind.CANDIDATE, f"{job.id}-attempt-{job.attempt}")
        if (
            candidate_id != expected_identity.stable_id
            or job.candidate_id != expected_identity.stable_id
            or candidate_record.stable_id != expected_identity.stable_id
            or candidate_record.identity_value != expected_identity.value
            or job.terminal_state != "candidate_ready"
        ):
            raise self._candidate_preview_integrity_error(candidate_id)

        chunks = [chunk for chunk, _ in rows]
        chunk_hashes: list[str] = []
        for expected_index, chunk in enumerate(chunks):
            if (
                chunk.job_id != job.id
                or chunk.candidate_id != expected_identity.stable_id
                or chunk.document_identity != frozen_input.document_identity
                or chunk.generation != frozen_input.requested_generation
                or chunk.attempt != job.attempt
                or chunk.chunk_index != expected_index
                or not self._is_sha256(chunk.content_sha256)
                or hashlib.sha256(chunk.content.encode("utf-8")).hexdigest() != chunk.content_sha256
            ):
                raise self._candidate_preview_integrity_error(candidate_id)
            chunk_hashes.append(chunk.content_sha256)

        payload = candidate_record.payload if isinstance(candidate_record.payload, dict) else None
        expected_payload = {
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
            "attempt": job.attempt,
            "chunk_count": len(chunks),
            "chunk_sha256s": chunk_hashes,
        }
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected_payload.items()):
            raise self._candidate_preview_integrity_error(candidate_id)
        return self._candidate_preview_authority(frozen_input, candidate_id=candidate_id)

    def _candidate_preview_authority(
        self,
        frozen_input: FrozenCandidateBuildInput,
        *,
        candidate_id: str,
    ) -> dict[str, Any]:
        artifact = frozen_input.artifact
        entry = artifact.get("entry")
        entry_id = artifact.get("entry_id")
        entry_identity = artifact.get("entry_identity")
        revision_identity = artifact.get("editorial_revision_identity")
        if (
            not isinstance(entry, dict)
            or not isinstance(entry_id, str)
            or not isinstance(entry_identity, str)
            or entry_identity != frozen_input.entry_identity
            or not isinstance(revision_identity, str)
            or artifact.get("revision_sha256") != frozen_input.editorial_source_revision
            or entry.get("entry_id") != entry_id
        ):
            raise self._candidate_preview_integrity_error(candidate_id)
        try:
            entry_stable_identity = StableIdentity.from_stable_id(entry_identity)
            revision_stable_identity = StableIdentity.from_stable_id(revision_identity)
        except ValueError as exc:
            raise self._candidate_preview_integrity_error(candidate_id) from exc
        if (
            entry_stable_identity.kind is not StableIdentityKind.ENTRY
            or entry_stable_identity.value != entry_id
            or revision_stable_identity.kind is not StableIdentityKind.EDITORIAL_REVISION
        ):
            raise self._candidate_preview_integrity_error(candidate_id)

        source_by_id: dict[str, dict[str, str]] = {}
        raw_sources = artifact.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise self._candidate_preview_integrity_error(candidate_id)
        for snapshot in raw_sources:
            source = snapshot.get("source") if isinstance(snapshot, dict) else None
            source_identity = snapshot.get("source_identity") if isinstance(snapshot, dict) else None
            if (
                not isinstance(source, dict)
                or not isinstance(source_identity, str)
                or snapshot.get("availability") != "verified_usable"
                or not isinstance(source.get("source_id"), str)
                or source.get("access_scope") not in _KNOWN_ACCESS_SCOPES
            ):
                raise self._candidate_preview_integrity_error(candidate_id)
            try:
                source_stable_identity = StableIdentity.from_stable_id(source_identity)
            except ValueError as exc:
                raise self._candidate_preview_integrity_error(candidate_id) from exc
            source_id = source["source_id"]
            if (
                source_stable_identity.kind is not StableIdentityKind.SOURCE
                or source_stable_identity.value != source_id
                or source_id in source_by_id
            ):
                raise self._candidate_preview_integrity_error(candidate_id)
            source_by_id[source_id] = {
                "source_identity": source_identity,
                "availability": "verified_usable",
                "access_scope": str(source["access_scope"]),
            }

        raw_relationships = entry.get("section_source_relationships")
        if not isinstance(raw_relationships, list) or not raw_relationships:
            raise self._candidate_preview_integrity_error(candidate_id)
        relationships_by_section: dict[str, list[dict[str, str]]] = {}
        for relationship in raw_relationships:
            section_id = relationship.get("section_id") if isinstance(relationship, dict) else None
            source_ids = relationship.get("source_ids") if isinstance(relationship, dict) else None
            if (
                not isinstance(section_id, str)
                or not section_id
                or section_id in relationships_by_section
                or not isinstance(source_ids, list)
                or not source_ids
            ):
                raise self._candidate_preview_integrity_error(candidate_id)
            section_sources: list[dict[str, str]] = []
            seen_source_ids: set[str] = set()
            for source_id in source_ids:
                if not isinstance(source_id, str) or source_id in seen_source_ids or source_id not in source_by_id:
                    raise self._candidate_preview_integrity_error(candidate_id)
                seen_source_ids.add(source_id)
                section_sources.append(dict(source_by_id[source_id]))
            relationships_by_section[section_id] = sorted(section_sources, key=lambda item: item["source_identity"])

        assurance_level = entry.get("assurance_level")
        applicability_conditions = entry.get("applicability_conditions")
        non_applicability_conditions = entry.get("non_applicability_conditions")
        freshness_triggers = entry.get("freshness_triggers")
        if (
            assurance_level not in _ALLOWED_ASSURANCE_LEVELS
            or not isinstance(applicability_conditions, list)
            or not applicability_conditions
            or not isinstance(freshness_triggers, list)
            or not freshness_triggers
        ):
            raise self._candidate_preview_integrity_error(candidate_id)
        return {
            "entry_id": entry_id,
            "entry_identity": entry_identity,
            "editorial_revision_identity": revision_identity,
            "section_source_relationships": relationships_by_section,
            "assurance_level": assurance_level,
            "applicability_conditions": applicability_conditions,
            "non_applicability_conditions": non_applicability_conditions,
            "freshness_triggers": freshness_triggers,
        }

    def _candidate_preview_section_id(
        self,
        *,
        candidate_id: str,
        chunk: CandidateBuildChunk,
        job: CandidateBuildJob,
        metadata: dict[str, Any],
        binding: dict[str, Any],
    ) -> str:
        metadata_identity = self._metadata_identity(metadata)
        if (
            metadata_identity is None
            or metadata_identity[0] != binding["entry_id"]
            or metadata_identity[1] != binding["entry_identity"]
            or metadata_identity[2] != binding["editorial_revision_identity"]
            or metadata.get("candidate_build") is not True
            or chunk.document_identity != job.document_identity
        ):
            raise self._candidate_preview_integrity_error(candidate_id)
        section_id = metadata_identity[3]
        source_relationships = self._section_relationships(binding, section_id)
        if source_relationships is None or (
            self._metadata_binding_exclusion_reason(
                metadata,
                source_relationships=source_relationships,
                assurance_level=binding["assurance_level"],
                applicability_conditions=binding["applicability_conditions"],
                non_applicability_conditions=binding.get("non_applicability_conditions"),
                freshness_triggers=binding["freshness_triggers"],
            )
            is not None
        ):
            raise self._candidate_preview_integrity_error(candidate_id)
        return section_id

    @staticmethod
    def _candidate_preview_item(
        *,
        chunk: CandidateBuildChunk,
        job: CandidateBuildJob,
        metadata: dict[str, Any],
        binding: dict[str, Any],
        section_id: str,
        source_relationships: list[dict[str, str]],
        profile_identity: str,
    ) -> dict[str, Any]:
        access_scopes = sorted({item["access_scope"] for item in source_relationships})
        access_scope = access_scopes[0] if len(access_scopes) == 1 else "mixed_team_shared"
        candidate_version = f"{chunk.candidate_id}:generation:{chunk.generation}:attempt:{chunk.attempt}"
        authoritative_metadata = dict(metadata)
        authoritative_metadata.update(
            {
                "entry_id": binding["entry_id"],
                "entry_identity": binding["entry_identity"],
                "editorial_revision_identity": binding["editorial_revision_identity"],
                "source_relationships": source_relationships,
                "assurance_level": binding["assurance_level"],
                "applicability_conditions": binding["applicability_conditions"],
                "non_applicability_conditions": binding["non_applicability_conditions"],
                "freshness_triggers": binding["freshness_triggers"],
            }
        )
        return {
            "chunk_id": chunk.id,
            "document_identity": chunk.document_identity,
            "generation": chunk.generation,
            "attempt": chunk.attempt,
            "chunk_index": chunk.chunk_index,
            "content_sha256": chunk.content_sha256,
            "content_preview": chunk.content[:160],
            "metadata": authoritative_metadata,
            "retrieval_source": "sparse_bm25",
            "entry_id": binding["entry_id"],
            "entry_identity": binding["entry_identity"],
            "editorial_revision_identity": binding["editorial_revision_identity"],
            "candidate_id": chunk.candidate_id,
            "candidate_identity": chunk.candidate_id,
            "candidate_version": candidate_version,
            "publication_identity": None,
            "publication_version": None,
            "build_generation_identity": f"build_generation:{job.id}",
            "section_id": section_id,
            "section_identity": f"{binding['entry_identity']}#{section_id}",
            "source_relationships": source_relationships,
            "assurance_level": binding["assurance_level"],
            "applicability_conditions": list(binding["applicability_conditions"]),
            "freshness_triggers": list(binding["freshness_triggers"]),
            "access_scope": access_scope,
            "chunk_identity": {
                "candidate_id": chunk.candidate_id,
                "document_identity": chunk.document_identity,
                "generation": chunk.generation,
                "attempt": chunk.attempt,
                "chunk_index": chunk.chunk_index,
                "content_sha256": chunk.content_sha256,
            },
            "retrieval_profile_identity": profile_identity,
            "diagnostic_only": True,
            "answer_evidence_eligible": False,
        }

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(character in _SHA256_HEX for character in value)

    @staticmethod
    def _candidate_preview_integrity_error(candidate_id: str) -> AppError:
        return AppError(
            status_code=409,
            code="CANDIDATE_PREVIEW_INTEGRITY_FAILED",
            message="Candidate preview immutable bindings do not verify",
            detail={"candidate_id": candidate_id},
        )
