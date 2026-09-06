from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import Settings
from app.common.exceptions import AppError
from app.model.base import Base
from app.model.canonical import CanonicalRecordModel
from app.model.document import Document, DocumentChunk
from app.retrieval.candidate_pool import AuthorizedRetrievalCandidatePool
from app.retrieval.policy import LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID, PILOT_RETRIEVAL_PROFILE_ID
from app.reviewed_bundles.models import CandidateBuildChunk, CandidateBuildJob
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService


def _metadata(
    *,
    entry_id: str,
    section_id: str,
    lifecycle_state: str = "published",
    source_availability: str = "verified_usable",
    access_scope: str = "controlled_internal",
    assurance_level: str = "source_grounded",
    assurance_eligible: bool = True,
    known_contradiction: bool = False,
    integrity_defect: bool = False,
    candidate_build: bool = False,
    needs_review_at: datetime | None = None,
) -> dict:
    return {
        "entry_id": entry_id,
        "entry_identity": f"entry:{entry_id}",
        "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
        "section_id": section_id,
        "source_relationships": [
            {
                "source_identity": f"source:{entry_id}-source",
                "availability": source_availability,
                "access_scope": access_scope,
            }
        ],
        "assurance_level": assurance_level,
        "assurance_eligible": assurance_eligible,
        "applicability_conditions": [{"condition_id": f"{entry_id}-applicability"}],
        "freshness_triggers": [{"trigger_id": f"{entry_id}-freshness"}],
        "lifecycle_state": lifecycle_state,
        "known_contradiction": known_contradiction,
        "integrity_defect": integrity_defect,
        "candidate_build": candidate_build,
        **(
            {"needs_review_at": needs_review_at.isoformat()}
            if needs_review_at is not None
            else {}
        ),
    }


def _document(document_id: str, *, published_generation: int = 1, deleted_at=None) -> Document:
    return Document(
        id=document_id,
        filename=f"{document_id}.md",
        file_type="md",
        file_size=100,
        status="ready",
        chunk_strategy="section-aware",
        chunk_count=10,
        published_generation=published_generation,
        next_generation=published_generation + 1,
        latest_requested_generation=published_generation,
        deleted_at=deleted_at,
    )


def _chunk(
    chunk_id: str,
    *,
    document_id: str,
    generation: int,
    chunk_index: int,
    content: str,
    metadata: dict,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        generation=generation,
        chunk_index=chunk_index,
        content=content,
        keywords=[],
        generated_questions=[],
        chunk_metadata=metadata,
    )


class _StaticAuthority:
    def __init__(self, *, ineligible: dict[str, list[str]] | None = None) -> None:
        self._ineligible = ineligible or {}

    async def get_retrieval_authority(self, entry_id: str, *, now: datetime | None = None) -> dict:
        del now
        reasons = self._ineligible.get(entry_id, [])
        source_relationships = [
            {
                "source_identity": f"source:{entry_id}-source",
                "availability": "verified_usable",
                "access_scope": "controlled_internal",
            }
        ]
        return {
            "entry_id": entry_id,
            "entry_identity": f"entry:{entry_id}",
            "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
            "lifecycle_state": "published" if not reasons else "needs_re_review",
            "answer_eligible": not reasons,
            "eligibility_reasons": reasons,
            "section_source_relationships": {
                section_id: source_relationships
                for section_id in (
                    "recommendation",
                    "different-section",
                    "failure_modes",
                    "recommendation_or_reviewed_branches",
                )
            },
            "assurance_level": "source_grounded",
            "applicability_conditions": [{"condition_id": f"{entry_id}-applicability"}],
            "freshness_triggers": [{"trigger_id": f"{entry_id}-freshness"}],
        }


def test_pilot_pool_uses_bm25_filters_ineligible_content_and_preserves_identity() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="pilot-candidate-pool-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def run() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        stale_grace = datetime.now(UTC) - timedelta(days=8)
        malformed_source_identity = _metadata(entry_id="malformed-source-001", section_id="recommendation")
        malformed_source_identity["source_relationships"] = [
            {
                "source_identity": "source:",
                "availability": "verified_usable",
                "access_scope": "controlled_internal",
            }
        ]
        malformed_relationship_shape = _metadata(entry_id="malformed-shape-001", section_id="recommendation")
        malformed_relationship_shape["source_relationships"] = [None]
        async with session_factory() as session:
            session.add_all(
                [
                    _document("published-eligible"),
                    _document("published-ineligible"),
                    _document("published-superseded", published_generation=2),
                    _document("withdrawn-document", deleted_at=datetime.now(UTC)),
                    _chunk(
                        "eligible-alpha",
                        document_id="published-eligible",
                        generation=1,
                        chunk_index=0,
                        content="Sparse BM25 keeps RUNTIME_RETRIEVAL_PROFILE and /api/v1/retrieval literal.",
                        metadata=_metadata(entry_id="decision-001", section_id="recommendation"),
                    ),
                    _chunk(
                        "eligible-exact-duplicate",
                        document_id="published-eligible",
                        generation=1,
                        chunk_index=1,
                        content="Sparse BM25 keeps RUNTIME_RETRIEVAL_PROFILE and /api/v1/retrieval literal.",
                        metadata=_metadata(entry_id="decision-002", section_id="different-section"),
                    ),
                    _chunk(
                        "eligible-same-section-duplicate",
                        document_id="published-eligible",
                        generation=1,
                        chunk_index=2,
                        content="Sparse BM25 profile for RUNTIME_RETRIEVAL_PROFILE has an alternate recommendation.",
                        metadata=_metadata(entry_id="decision-001", section_id="recommendation"),
                    ),
                    _chunk(
                        "eligible-complementary",
                        document_id="published-eligible",
                        generation=1,
                        chunk_index=3,
                        content="中文说明：Sparse BM25 保留 /api/v1/retrieval 和 RUNTIME_RETRIEVAL_PROFILE。",
                        metadata=_metadata(entry_id="decision-001", section_id="failure_modes"),
                    ),
                    _chunk(
                        "candidate-outside-preview",
                        document_id="published-eligible",
                        generation=1,
                        chunk_index=4,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE candidate preview must stay private.",
                        metadata=_metadata(
                            entry_id="candidate-001",
                            section_id="recommendation",
                            candidate_build=True,
                        ),
                    ),
                    _chunk(
                        "draft-entry",
                        document_id="published-eligible",
                        generation=1,
                        chunk_index=5,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE draft material.",
                        metadata=_metadata(
                            entry_id="draft-001",
                            section_id="recommendation",
                            lifecycle_state="draft",
                        ),
                    ),
                    _chunk(
                        "unavailable-high-score",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=0,
                        content=(
                            "Sparse BM25 Sparse BM25 Sparse BM25 "
                            "RUNTIME_RETRIEVAL_PROFILE /api/v1/retrieval"
                        ),
                        metadata=_metadata(
                            entry_id="unavailable-001",
                            section_id="recommendation",
                            source_availability="unavailable_for_new_evidence",
                        ),
                    ),
                    _chunk(
                        "contradicted-entry",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=1,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE contradicted material.",
                        metadata=_metadata(
                            entry_id="contradicted-001",
                            section_id="recommendation",
                            known_contradiction=True,
                        ),
                    ),
                    _chunk(
                        "expired-grace",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=2,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE expired grace material.",
                        metadata=_metadata(
                            entry_id="expired-001",
                            section_id="recommendation",
                            lifecycle_state="needs_re_review",
                            needs_review_at=stale_grace,
                        ),
                    ),
                    _chunk(
                        "unauthorized-access",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=3,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE restricted material.",
                        metadata=_metadata(
                            entry_id="unauthorized-001",
                            section_id="recommendation",
                            access_scope="restricted",
                        ),
                    ),
                    _chunk(
                        "assurance-ineligible",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=4,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE unassured material.",
                        metadata=_metadata(
                            entry_id="assurance-001",
                            section_id="recommendation",
                            assurance_eligible=False,
                        ),
                    ),
                    _chunk(
                        "malformed-source-identity",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=5,
                        content="Sparse BM25 malformed source identity must not become eligible.",
                        metadata=malformed_source_identity,
                    ),
                    _chunk(
                        "malformed-relationship-shape",
                        document_id="published-ineligible",
                        generation=1,
                        chunk_index=6,
                        content="Sparse BM25 malformed relationship shape must not become eligible.",
                        metadata=malformed_relationship_shape,
                    ),
                    _chunk(
                        "superseded-generation",
                        document_id="published-superseded",
                        generation=1,
                        chunk_index=0,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE superseded material.",
                        metadata=_metadata(entry_id="superseded-001", section_id="recommendation"),
                    ),
                    _chunk(
                        "withdrawn-generation",
                        document_id="withdrawn-document",
                        generation=1,
                        chunk_index=0,
                        content="Sparse BM25 RUNTIME_RETRIEVAL_PROFILE withdrawn material.",
                        metadata=_metadata(entry_id="withdrawn-001", section_id="recommendation"),
                    ),
                ]
            )
            await session.commit()

        async with session_factory() as session:
            pool = AuthorizedRetrievalCandidatePool(
                session,
                settings=Settings(),
                editorial_authority=_StaticAuthority(
                    ineligible={
                        "draft-001": ["not_published"],
                        "unavailable-001": ["source_unavailable", "decisive_source_loss"],
                        "contradicted-001": ["known_contradiction"],
                        "expired-001": ["needs_review_grace_expired"],
                        "unauthorized-001": ["unauthorized_access_scope"],
                        "assurance-001": ["assurance_ineligible"],
                    }
                ),
            )
            english = await pool.retrieve("Sparse BM25 RUNTIME_RETRIEVAL_PROFILE /api/v1/retrieval", top_k=1)
            chinese = await pool.retrieve("中文 Sparse BM25 /api/v1/retrieval", top_k=1)
            mixed = await pool.retrieve("中文 RUNTIME_RETRIEVAL_PROFILE Sparse BM25", top_k=1)

        assert english.strategy == "sparse_bm25"
        assert english.profile_identity == PILOT_RETRIEVAL_PROFILE_ID
        assert english.candidate_pool_scope == "published_knowledge"
        assert [item["chunk_id"] for item in english.items] == ["eligible-alpha", "eligible-complementary"]
        assert [item["chunk_id"] for item in chinese.items] == ["eligible-complementary", "eligible-alpha"]
        assert [item["chunk_id"] for item in mixed.items] == ["eligible-complementary", "eligible-alpha"]

        candidate = english.items[0]
        assert candidate["entry_id"] == "decision-001"
        assert candidate["entry_identity"] == "entry:decision-001"
        assert candidate["publication_identity"] == "published_knowledge_version:legacy:published-eligible:v1"
        assert candidate["publication_version"] == "v1"
        assert candidate["section_identity"] == "entry:decision-001#recommendation"
        assert candidate["source_relationships"] == [
            {
                "source_identity": "source:decision-001-source",
                "availability": "verified_usable",
                "access_scope": "controlled_internal",
            }
        ]
        assert candidate["assurance_level"] == "source_grounded"
        assert candidate["applicability_conditions"] == [{"condition_id": "decision-001-applicability"}]
        assert candidate["freshness_triggers"] == [{"trigger_id": "decision-001-freshness"}]
        assert candidate["access_scope"] == "controlled_internal"
        assert candidate["chunk_identity"] == {
            "document_id": "published-eligible",
            "generation": 1,
            "chunk_index": 0,
            "content_sha256": candidate["content_sha256"],
        }
        assert candidate["retrieval_profile_identity"] == PILOT_RETRIEVAL_PROFILE_ID
        assert candidate["answer_evidence_eligible"] is True
        assert "sufficient" not in candidate
        assert "answer_eligible" not in candidate

        assert {item["reason"] for item in english.candidate_exclusions} == {
            "candidate_not_in_preview",
            "assurance_ineligible",
            "entry_section_duplicate",
            "entry_not_published",
            "exact_duplicate",
            "known_contradiction",
            "needs_review_grace_expired",
            "source_relationship_missing",
            "source_unavailable",
            "superseded_publication",
            "unauthorized_access_scope",
            "withdrawn",
        }
        assert "unavailable-high-score" not in {item["chunk_id"] for item in english.items}

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
        os.remove(db_path)


def test_pilot_pool_uses_fixed_depth_twenty_and_versioned_deterministic_tie_breaking() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="pilot-candidate-depth-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def run() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            for index in range(25):
                document_id = f"published-{index:02d}"
                session.add(_document(document_id))
                session.add(
                    _chunk(
                        f"chunk-{index:02d}",
                        document_id=document_id,
                        generation=1,
                        chunk_index=0,
                        content=f"same Sparse BM25 literal RUNTIME_RETRIEVAL_PROFILE item{index:02d}",
                        metadata=_metadata(entry_id=f"entry-{index:02d}", section_id="recommendation"),
                    )
                )
            await session.commit()

        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=Settings(),
                editorial_authority=_StaticAuthority(),
            ).retrieve(
                "Sparse BM25 RUNTIME_RETRIEVAL_PROFILE",
                top_k=1,
            )

        assert len(result.items) == 20
        assert [item["chunk_id"] for item in result.items] == [f"chunk-{index:02d}" for index in range(20)]
        assert all(item["raw_score"] == result.items[0]["raw_score"] for item in result.items)
        assert all(item["answer_evidence_eligible"] is True for item in result.items)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
        os.remove(db_path)


def test_pilot_pool_fails_closed_when_the_caller_has_no_authorized_access_scope() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="pilot-empty-access-scope-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def run() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(_document("access-scope-document"))
            session.add(
                _chunk(
                    "access-scope-chunk",
                    document_id="access-scope-document",
                    generation=1,
                    chunk_index=0,
                    content="Sparse BM25 must fail closed for an empty access scope set.",
                    metadata=_metadata(entry_id="access-scope-entry-001", section_id="recommendation"),
                )
            )
            await session.commit()

        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=Settings(),
                editorial_authority=_StaticAuthority(),
                allowed_access_scopes=frozenset(),
            ).retrieve("Sparse BM25 access scope", top_k=1)

        assert result.items == []
        assert result.candidate_exclusions == [
            {
                "chunk_id": "access-scope-chunk",
                "reason": "unauthorized_access_scope",
            }
        ]

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
        os.remove(db_path)


def test_retrieval_service_dispatches_pilot_and_keeps_legacy_heuristic_in_explicit_migration_mode() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="pilot-profile-dispatch-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def run() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add_all(
                [
                    _document("profile-dispatch"),
                    _document("legacy-profile-dispatch"),
                    _chunk(
                        "profile-dispatch-chunk",
                        document_id="profile-dispatch",
                        generation=1,
                        chunk_index=0,
                        content="Sparse BM25 profile dispatch keeps RUNTIME_RETRIEVAL_PROFILE literal.",
                        metadata=_metadata(entry_id="profile-dispatch-001", section_id="recommendation"),
                    ),
                    _chunk(
                        "legacy-profile-dispatch-chunk",
                        document_id="legacy-profile-dispatch",
                        generation=1,
                        chunk_index=0,
                        content="Legacy heuristic diagnostic keeps RUNTIME_RETRIEVAL_PROFILE literal.",
                        metadata={"source": "legacy-migration-diagnostic"},
                    ),
                ]
            )
            await session.commit()

        async with session_factory() as session:
            pilot = await MixedModeDocumentRetrieverService(
                session,
                settings=Settings(),
                editorial_authority=_StaticAuthority(),
            ).retrieve(
                "Sparse BM25 RUNTIME_RETRIEVAL_PROFILE",
                top_k=1,
            )
            migration = await MixedModeDocumentRetrieverService(
                session,
                settings=Settings(RUNTIME_RETRIEVAL_PROFILE=LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID),
            ).retrieve(
                "Sparse BM25 RUNTIME_RETRIEVAL_PROFILE",
                top_k=1,
            )

        assert pilot.strategy == "sparse_bm25"
        assert pilot.profile_identity == PILOT_RETRIEVAL_PROFILE_ID
        assert pilot.items[0]["retrieval_source"] == "sparse_bm25"
        assert pilot.items[0]["answer_evidence_eligible"] is True
        assert migration.strategy == "lexical_heuristic_migration"
        assert migration.profile_identity == LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID
        assert migration.items[0]["chunk_id"] == "legacy-profile-dispatch-chunk"
        assert migration.items[0]["retrieval_source"] == "lexical"
        assert "bm25" not in migration.strategy

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
        os.remove(db_path)


def test_pilot_pool_rejects_metadata_that_has_no_current_editorial_authority() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="pilot-metadata-only-authority-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def run() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(_document("metadata-only"))
            session.add(
                _chunk(
                    "metadata-only-chunk",
                    document_id="metadata-only",
                    generation=1,
                    chunk_index=0,
                    content="Sparse BM25 metadata alone cannot establish authority.",
                    metadata=_metadata(entry_id="metadata-only-001", section_id="recommendation"),
                )
            )
            await session.commit()

        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(session, settings=Settings()).retrieve(
                "Sparse BM25 authority",
                top_k=1,
            )

        assert result.items == []
        assert result.candidate_exclusions == [
            {
                "chunk_id": "metadata-only-chunk",
                "reason": "current_authority_missing",
            }
        ]

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
        os.remove(db_path)


def test_candidate_preview_fails_closed_without_a_complete_immutable_binding() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="pilot-candidate-preview-binding-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    candidate_id = "candidate:preview-binding-job-001-attempt-1"

    async def run() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            session.add(
                CanonicalRecordModel(
                    stable_id=candidate_id,
                    identity_kind="candidate",
                    identity_value="preview-binding-job-001-attempt-1",
                    state="candidate_ready",
                    record_class="immutable",
                    payload={"schema": "candidate_build_candidate/v1"},
                )
            )
            session.add(
                CandidateBuildJob(
                    id="preview-binding-job-001",
                    bundle_id="bundle:preview-binding-001",
                    bundle_item_id="bundle_item:preview-binding-001",
                    entry_identity="entry:preview-binding-001",
                    document_identity="document:preview-binding-001",
                    requested_generation=1,
                    editorial_source_revision="a" * 64,
                    input_sha256="b" * 64,
                    frozen_input_sha256="c" * 64,
                    chunk_strategy={"strategy_id": "section-aware"},
                    embedding_configuration={"active": False},
                    status="candidate_ready",
                    terminal_state="candidate_ready",
                    attempt=1,
                    candidate_id=candidate_id,
                )
            )
            content = "Sparse BM25 Candidate preview content."
            session.add(
                CandidateBuildChunk(
                    id="preview-binding-chunk-001",
                    job_id="preview-binding-job-001",
                    candidate_id=candidate_id,
                    document_identity="document:preview-binding-001",
                    generation=1,
                    attempt=1,
                    chunk_index=0,
                    content=content,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    chunk_metadata={
                        "entry_id": "preview-binding-001",
                        "entry_identity": "entry:preview-binding-001",
                        "editorial_revision_identity": "editorial_revision:preview-binding-001.r1",
                        "section_id": "recommendation_or_reviewed_branches",
                    },
                )
            )
            await session.commit()

        async with session_factory() as session:
            pool = AuthorizedRetrievalCandidatePool(session, settings=Settings())
            with pytest.raises(AppError) as exc_info:
                await pool.preview_candidate(candidate_id, "Sparse BM25")

        assert exc_info.value.code == "CANDIDATE_PREVIEW_INTEGRITY_FAILED"

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
        os.remove(db_path)
