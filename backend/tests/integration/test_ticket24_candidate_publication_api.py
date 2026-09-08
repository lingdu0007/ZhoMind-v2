from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Generator
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.canonical_json import canonical_json_sha256
from app.common.config import get_settings
from app.common.exceptions import AppError
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.canonical import CanonicalRecordModel
from app.model.document import Document, DocumentChunk
from app.operations.limits import MAX_PUBLISHED_SOURCES
from app.rag.answer_evidence import evidence_summary_from_execution
from app.rag.evidence_sufficiency import AnswerEvidenceSet, QueryConditionSet, decide_answer_evidence
from app.retrieval.candidate_pool import AuthorizedRetrievalCandidatePool
from app.reviewed_bundles.assurance import candidate_assurance_metadata
from app.reviewed_bundles.inputs import frozen_candidate_build_input_sha256
from app.reviewed_bundles.models import (
    CandidateBuildChunk,
    CandidateBuildJob,
    CandidatePublicationConfirmation,
    PublishedKnowledgePointer,
    PublishedKnowledgeVersion,
)
from app.reviewed_bundles.publication import CandidatePublicationService
from app.reviewed_bundles.service import candidate_embedding_configuration
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = {str(name): str(value) for name, value in mapping.items()}
        return len(mapping)

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self._store and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self._store)

    async def get(self, key: str) -> None:
        del key
        return None


class _ApprovedExportVerifier:
    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        assert artifact_sha256 == _sha256(artifact)
        return artifact

    @asynccontextmanager
    async def verify_for_candidate_finalization(self, artifact: dict, artifact_sha256: str):
        yield await self.verify(artifact, artifact_sha256)

    async def record_candidate_publication(
        self,
        artifact: dict,
        *,
        candidate_identity: str,
        published_knowledge_version_identity: str,
        actor_identity: str,
    ) -> None:
        del artifact, candidate_identity, published_knowledge_version_identity, actor_identity


def _sha256(value: object) -> str:
    return canonical_json_sha256(value)


def _approved_export(
    *,
    entry_id: str = "ticket24-entry-001",
    supported_query: str = "Which Candidate publication contract applies?",
    boundary_query: str = "Can Candidate inspection activate an unreviewed provider route?",
    applicability_conditions: list[dict[str, str]] | None = None,
    supported_query_conditions: list[dict[str, str]] | None = None,
    boundary_query_conditions: list[dict[str, str]] | None = None,
) -> dict:
    entry_identity = f"entry:{entry_id}"
    default_applicability_conditions = [
        {
            "condition_id": "ticket24-production",
            "field": "deployment",
            "operator": "equals",
            "value": "production",
        }
    ]
    default_supported_query_conditions = [dict(default_applicability_conditions[0])]
    supported_acceptance_query = {
        "query_id": "supported-ticket24",
        "query": supported_query,
        "expected_outcome": "supported",
        "query_conditions": (
            default_supported_query_conditions
            if supported_query_conditions is None
            else supported_query_conditions
        ),
    }
    boundary_acceptance_query: dict[str, object] = {
        "query_id": "boundary-ticket24",
        "query": boundary_query,
        "expected_outcome": "insufficient_evidence",
    }
    if boundary_query_conditions is not None:
        boundary_acceptance_query["query_conditions"] = boundary_query_conditions
    return {
        "schema": "editorial_export/v1",
        "entry_identity": entry_identity,
        "entry_id": entry_id,
        "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
        "revision_number": 1,
        "revision_sha256": "a" * 64,
        "roles": {
            "author_identity": "member:ticket24-author-001",
            "approving_reviewer_identity": "member:ticket24-reviewer-001",
            "accountable_maintainer_identity": "member:ticket24-maintainer-001",
        },
        "approval": {"status": "approved", "reviewer_identity": "member:ticket24-reviewer-001"},
        "entry": {
            "schema_version": 1,
            "entry_id": entry_id,
            "title": "Ticket 24 Candidate publication contract",
            "coverage_position": "rag_source_admission_and_chunking",
            "assurance_level": "source_grounded",
            "applicability_conditions": (
                default_applicability_conditions
                if applicability_conditions is None
                else applicability_conditions
            ),
            "non_applicability_conditions": [],
            "review_date": "2026-09-08",
            "freshness_triggers": [{"trigger_id": "ticket24-freshness"}],
            "chunk_strategy": {
                "strategy_id": "section-aware-900-120",
                "max_characters": 900,
                "overlap_characters": 120,
                "preserve_section_boundaries": True,
            },
            "acceptance_material": {
                "supported_queries": [supported_acceptance_query],
                "boundary_queries": [boundary_acceptance_query],
            },
            "body": {
                "decision_query": "Which Candidate publication contract applies to reviewed bundle entries?",
                "recommendation_or_reviewed_branches": (
                    "Only a Candidate with recorded inspection, Candidate-bound acceptance, and explicit "
                    "administrator confirmation may become a Published Knowledge Version."
                ),
            },
            "section_source_relationships": [
                {"section_id": "decision_query", "source_ids": ["source-ticket24-001"]},
                {
                    "section_id": "recommendation_or_reviewed_branches",
                    "source_ids": ["source-ticket24-001"],
                },
            ],
            "sources": [
                {
                    "source_id": "source-ticket24-001",
                    "source_tier": "primary_evidence_source",
                    "title": "Ticket 24 publication authority",
                    "authority": "ZhoMind architecture group",
                    "version_or_date": "2026-09-08",
                    "access_scope": "public",
                    "public_url": "https://example.com/ticket24",
                    "availability": "verified_usable",
                }
            ],
        },
        "sources": [
            {
                "source_identity": "source:source-ticket24-001",
                "availability": "verified_usable",
                "source_definition_sha256": "b" * 64,
                "availability_event": {"event_id": "event:ticket24-source-availability-001"},
                "source": {
                    "source_id": "source-ticket24-001",
                    "source_tier": "primary_evidence_source",
                    "title": "Ticket 24 publication authority",
                    "authority": "ZhoMind architecture group",
                    "version_or_date": "2026-09-08",
                    "access_scope": "public",
                    "public_url": "https://example.com/ticket24",
                    "availability": "verified_usable",
                },
            }
        ],
        "release_assurance_snapshot": None,
        "editorial_audit": [{"sequence": 1, "action": "revision_approved"}],
    }


def _claim_linked_export() -> dict:
    artifact = _approved_export()
    artifact["entry"]["assurance_level"] = "claim_linked"
    artifact["entry"]["claims"] = [
        {
            "claim_id": "ticket24-publication-claim",
            "claim_kind": "prescriptive",
            "statement": (
                "Only a Candidate with recorded inspection, Candidate-bound acceptance, and explicit "
                "administrator confirmation may become a Published Knowledge Version."
            ),
            "section_id": "recommendation_or_reviewed_branches",
            "source_ids": ["source-ticket24-001"],
            "scope": "Candidate publication for reviewed bundle entries.",
            "material": True,
        }
    ]
    contract = {
        "schema_version": 1,
        "review_id": "ticket24-editorial-review",
        "review_revision": "2026-09-08.1",
        "conflict_state": "none",
        "unknown_state": "none",
        "resolver": {
            "resolver_id": "ticket24-resolver",
            "calibration_id": "ticket24-calibration",
            "calibration_version": "2026-09-08",
            "minimum_confidence": 0.8,
        },
        "claims": [
            {
                "claim_id": "ticket24-publication-claim",
                "scope": "Candidate publication for reviewed bundle entries.",
                "evidence": [
                    {
                        "section_id": "recommendation_or_reviewed_branches",
                        "source_id": "source-ticket24-001",
                    }
                ],
            }
        ],
    }
    contract_json = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    artifact["claim_evidence_contract"] = contract_json
    artifact["claim_evidence_contract_sha256"] = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return artifact


def _claim_linked_export_without_release_assured_contract() -> dict:
    artifact = _claim_linked_export()
    contract = {
        "schema": "candidate_claim_evidence_contract/v1",
        "entry_identity": artifact["entry_identity"],
        "editorial_revision_identity": artifact["editorial_revision_identity"],
        "claims": [
            {
                "claim_id": "ticket24-publication-claim",
                "section_id": "recommendation_or_reviewed_branches",
                "source_ids": ["source-ticket24-001"],
            }
        ],
    }
    contract_json = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    artifact["claim_evidence_contract"] = contract_json
    artifact["claim_evidence_contract_sha256"] = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    artifact["entry"].pop("claim_evidence_contract", None)
    return artifact


def _multi_source_claim_linked_export() -> dict:
    artifact = _claim_linked_export()
    second_source = {
        "source_id": "source-ticket24-002",
        "source_tier": "primary_evidence_source",
        "title": "Ticket 24 independent publication authority",
        "authority": "ZhoMind review council",
        "version_or_date": "2026-09-08",
        "access_scope": "public",
        "public_url": "https://example.com/ticket24-independent",
        "availability": "verified_usable",
    }
    artifact["entry"]["sources"].append(second_source)
    artifact["sources"].append(
        {
            "source_identity": "source:source-ticket24-002",
            "availability": "verified_usable",
            "source_definition_sha256": "c" * 64,
            "availability_event": {"event_id": "event:ticket24-source-availability-002"},
            "source": second_source,
        }
    )
    artifact["entry"]["section_source_relationships"][1]["source_ids"].append("source-ticket24-002")
    artifact["entry"]["claims"][0]["source_ids"].append("source-ticket24-002")
    contract = json.loads(artifact["claim_evidence_contract"])
    contract["claims"][0]["evidence"].append(
        {
            "section_id": "recommendation_or_reviewed_branches",
            "source_id": "source-ticket24-002",
        }
    )
    contract_json = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    artifact["claim_evidence_contract"] = contract_json
    artifact["claim_evidence_contract_sha256"] = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return artifact


def _same_citation_multi_source_claim_linked_export() -> dict:
    artifact = _multi_source_claim_linked_export()
    first_source = artifact["entry"]["sources"][0]
    second_source = artifact["entry"]["sources"][1]
    for field in ("title", "authority", "version_or_date", "access_scope", "public_url"):
        second_source[field] = first_source[field]
    return artifact


@pytest.fixture
def client(tmp_path) -> Generator[TestClient, None, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ticket24-publication-api.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()

    async def initialize_database() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(initialize_database())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.settings_session_factory = session_factory
    app.state.test_auth_session_factory = session_factory
    app.state.test_auth_redis = redis
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = SessionLocal
    asyncio.run(engine.dispose())


async def _headers(client: TestClient, *, username: str, role: str) -> dict[str, str]:
    token = await create_authenticated_test_token(
        client.app.state.test_auth_session_factory,
        client.app.state.test_auth_redis,
        username=username,
        role=role,
    )
    return {"Authorization": f"Bearer {token}"}


def _data(response) -> dict:
    return response.json()["data"]


async def _seed_ready_candidate(
    client: TestClient,
    *,
    seed: str = "001",
    entry_id: str = "ticket24-entry-001",
    generation: int = 1,
    chunk_content_suffix: str = "",
    artifact: dict | None = None,
) -> str:
    session_factory = client.app.state.test_auth_session_factory
    job_id = f"ticket24-candidate-job-{seed}"
    candidate_id = f"candidate:{job_id}-attempt-1"
    artifact = artifact or _approved_export(entry_id=entry_id)
    bundle_id = f"bundle:ticket24-bundle-{seed}"
    bundle_item_id = f"bundle_item:ticket24-item-{seed}"
    document_identity = f"runtime-document:{entry_id}"
    input_sha256 = _sha256(artifact)
    bundle_item_sha256 = _sha256(
        {
            "bundle_item_id": f"ticket24-item-{seed}",
            "operation": "create",
            "artifact_sha256": input_sha256,
            "artifact": artifact,
        }
    )
    manifest = {
        "schema": "reviewed_release_bundle/v1",
        "schema_version": 1,
        "bundle_id": f"ticket24-bundle-{seed}",
        "editorial_source_revision": artifact["revision_sha256"],
        "exported_at": "2026-09-08T12:00:00Z",
        "items": [
            {
                "bundle_item_id": f"ticket24-item-{seed}",
                "operation": "create",
                "artifact_sha256": input_sha256,
                "artifact": artifact,
                "bundle_item_sha256": bundle_item_sha256,
            }
        ],
    }
    bundle_sha256 = _sha256(manifest)
    manifest["bundle_sha256"] = bundle_sha256
    input_payload = {
        "schema": "candidate_build_input/v1",
        "bundle_id": bundle_id,
        "bundle_sha256": bundle_sha256,
        "bundle_item_id": bundle_item_id,
        "bundle_item_sha256": bundle_item_sha256,
        "entry_identity": artifact["entry_identity"],
        "document_identity": document_identity,
        "requested_generation": generation,
        "editorial_source_revision": artifact["revision_sha256"],
        "input_sha256": input_sha256,
        "chunk_strategy": artifact["entry"]["chunk_strategy"],
        "embedding_configuration": candidate_embedding_configuration(get_settings()),
    }
    frozen_input_sha256 = frozen_candidate_build_input_sha256(input_payload)
    input_payload["frozen_input_sha256"] = frozen_input_sha256
    chunk_content = (
        "## Recommendation Or Reviewed Branches\n\n"
        "Only a Candidate with recorded inspection, Candidate-bound acceptance, and explicit "
        f"administrator confirmation may become a Published Knowledge Version.{chunk_content_suffix}"
    )
    chunk_sha256 = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()
    source_definitions = {
        source["source_id"]: source
        for source in artifact["entry"]["sources"]
    }
    section_source_ids = next(
        (
            relationship["source_ids"]
            for relationship in artifact["entry"]["section_source_relationships"]
            if relationship["section_id"] == "recommendation_or_reviewed_branches"
        ),
        [],
    )
    source_relationships = [
        {
            "source_identity": f"source:{source_id}",
            "availability": source_definitions[source_id]["availability"],
            "access_scope": source_definitions[source_id]["access_scope"],
        }
        for source_id in section_source_ids
    ]
    chunk_metadata = {
        "entry_id": artifact["entry_id"],
        "domain": artifact["entry"]["coverage_position"],
        "entry_identity": artifact["entry_identity"],
        "editorial_revision_identity": artifact["editorial_revision_identity"],
        "section_id": "recommendation_or_reviewed_branches",
        "section_title": "Recommendation Or Reviewed Branches",
        "chunk_strategy_id": artifact["entry"]["chunk_strategy"]["strategy_id"],
        "source_identities": [relationship["source_identity"] for relationship in source_relationships],
        "source_relationships": source_relationships,
        "assurance_level": artifact["entry"]["assurance_level"],
        "applicability_conditions": artifact["entry"]["applicability_conditions"],
        "non_applicability_conditions": artifact["entry"]["non_applicability_conditions"],
        "freshness_triggers": artifact["entry"]["freshness_triggers"],
        "lifecycle_state": "candidate_build",
        "candidate_build": True,
    }
    if artifact["entry"]["assurance_level"] == "claim_linked":
        chunk_metadata.update(
            {
                "claim_evidence_contract": artifact["claim_evidence_contract"],
                "claim_evidence_contract_sha256": artifact["claim_evidence_contract_sha256"],
            }
        )
    if artifact["entry"]["assurance_level"] == "release_assured":
        chunk_metadata["release_assurance_snapshot"] = artifact["release_assurance_snapshot"]
    async with session_factory() as session:
        session.add_all(
            [
                CanonicalRecordModel(
                    stable_id=bundle_id,
                    identity_kind="bundle",
                    identity_value=f"ticket24-bundle-{seed}",
                    state="processing",
                    record_class="immutable",
                    payload={
                        "schema": "reviewed_release_bundle/v1",
                        "bundle_sha256": bundle_sha256,
                        "manifest": manifest,
                    },
                ),
                CanonicalRecordModel(
                    stable_id=bundle_item_id,
                    identity_kind="bundle_item",
                    identity_value=f"ticket24-item-{seed}",
                    state="admitted",
                    record_class="immutable",
                    payload={
                        "schema": "reviewed_release_bundle_item/v1",
                        "bundle_id": bundle_id,
                        "entry_identity": artifact["entry_identity"],
                        "operation": "create",
                        "artifact_sha256": input_sha256,
                        "artifact": artifact,
                        "bundle_item_sha256": bundle_item_sha256,
                    },
                ),
                CanonicalRecordModel(
                    stable_id=f"build_generation:{job_id}",
                    identity_kind="build_generation",
                    identity_value=job_id,
                    state="frozen",
                    record_class="immutable",
                    payload=input_payload,
                ),
                CanonicalRecordModel(
                    stable_id=candidate_id,
                    identity_kind="candidate",
                    identity_value=f"{job_id}-attempt-1",
                    state="candidate_ready",
                    record_class="immutable",
                    payload={
                        "schema": "candidate_build_candidate/v1",
                        "bundle_id": bundle_id,
                        "bundle_sha256": bundle_sha256,
                        "bundle_item_id": bundle_item_id,
                        "bundle_item_sha256": bundle_item_sha256,
                        "build_generation_id": f"build_generation:{job_id}",
                        "entry_identity": artifact["entry_identity"],
                        "document_identity": document_identity,
                        "requested_generation": generation,
                        "editorial_source_revision": artifact["revision_sha256"],
                        "input_sha256": input_sha256,
                        "frozen_input_sha256": frozen_input_sha256,
                        "chunk_strategy": artifact["entry"]["chunk_strategy"],
                        "embedding_configuration": input_payload["embedding_configuration"],
                        "attempt": 1,
                        "chunk_count": 1,
                        "chunk_sha256s": [chunk_sha256],
                    },
                ),
                CandidateBuildJob(
                    id=job_id,
                    bundle_id=bundle_id,
                    bundle_item_id=bundle_item_id,
                    entry_identity=artifact["entry_identity"],
                    document_identity=document_identity,
                    requested_generation=generation,
                    editorial_source_revision=artifact["revision_sha256"],
                    input_sha256=input_sha256,
                    frozen_input_sha256=frozen_input_sha256,
                    chunk_strategy=artifact["entry"]["chunk_strategy"],
                    embedding_configuration=input_payload["embedding_configuration"],
                    status="candidate_ready",
                    stage="indexing",
                    progress=100,
                    attempt=1,
                    terminal_state="candidate_ready",
                    allowed_next_action="await_candidate_inspection",
                    candidate_id=candidate_id,
                ),
                CandidateBuildChunk(
                    id=f"ticket24-candidate-chunk-{seed}",
                    job_id=job_id,
                    candidate_id=candidate_id,
                    document_identity=document_identity,
                    generation=generation,
                    attempt=1,
                    chunk_index=0,
                    content=chunk_content,
                    content_sha256=chunk_sha256,
                    chunk_metadata=chunk_metadata,
                ),
            ]
        )
        await session.commit()
    return candidate_id


def _candidate_urls(candidate_id: str) -> tuple[str, str, str]:
    base = f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}"
    return f"{base}/inspection", f"{base}/acceptance", f"{base}/publication-eligibility"


def _publication_url(candidate_id: str) -> str:
    return f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}/publication"


def _publication_state(client: TestClient, *, candidate_id: str, headers: dict[str, str]) -> dict:
    response = client.get(_publication_url(candidate_id), headers=headers)
    assert response.status_code == 200, response.json()
    return _data(response)


def _publish_selection(candidate_id: str, eligibility: dict) -> dict:
    current = eligibility["current_published_knowledge_version"]
    return {
        "candidate_id": candidate_id,
        "effect": eligibility["effect"],
        "current_published_knowledge_version": current["identity"] if current else None,
        "inspection_record_identity": eligibility["inspection_record_identity"],
        "acceptance_record_identity": eligibility["acceptance_record_identity"],
    }


def _inspect_and_accept(client: TestClient, *, candidate_id: str, headers: dict[str, str]) -> None:
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)
    assert client.post(inspection_url, headers=headers, json={}).status_code == 200
    assert client.post(acceptance_url, headers=headers, json={}).status_code == 200


class _PublishedRetrievalAuthority:
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        del now
        artifact = _approved_export(entry_id=entry_id)
        entry = artifact["entry"]
        source_relationships = [
            {
                "source_identity": "source:source-ticket24-001",
                "availability": "verified_usable",
                "access_scope": "public",
            }
        ]
        return {
            "entry_id": entry_id,
            "entry_identity": artifact["entry_identity"],
            "editorial_revision_identity": artifact["editorial_revision_identity"],
            "lifecycle_state": "published",
            "answer_eligible": True,
            "eligibility_reasons": [],
            "section_source_relationships": {
                "recommendation_or_reviewed_branches": source_relationships,
            },
            "assurance_level": "source_grounded",
            "applicability_conditions": entry["applicability_conditions"],
            "freshness_triggers": entry["freshness_triggers"],
            "decision_query": entry["body"]["decision_query"],
        }


class _ClaimLinkedMultiSourceRetrievalAuthority:
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        del now
        artifact = _multi_source_claim_linked_export()
        entry = artifact["entry"]
        return {
            "entry_id": entry_id,
            "entry_identity": artifact["entry_identity"],
            "editorial_revision_identity": artifact["editorial_revision_identity"],
            "lifecycle_state": "published",
            "answer_eligible": True,
            "eligibility_reasons": [],
            "section_source_relationships": {
                "recommendation_or_reviewed_branches": [
                    {
                        "source_identity": "source:source-ticket24-001",
                        "availability": "verified_usable",
                        "access_scope": "public",
                    },
                    {
                        "source_identity": "source:source-ticket24-002",
                        "availability": "verified_usable",
                        "access_scope": "public",
                    },
                ],
            },
            "assurance_level": "claim_linked",
            "applicability_conditions": entry["applicability_conditions"],
            "freshness_triggers": entry["freshness_triggers"],
            "decision_query": entry["body"]["decision_query"],
        }


class _ClaimLinkedWordingSuccessorRetrievalAuthority(_ClaimLinkedMultiSourceRetrievalAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        authority.update(
            {
                "editorial_revision_identity": f"editorial_revision:{entry_id}.r2",
                "entry_title": "An approved successor must not relabel an older publication.",
                "review_date": "2026-09-09",
                "decision_query": "A wording-only successor remains unbound to the published Candidate.",
            }
        )
        return authority


class _UnpublishedSuccessorRetrievalAuthority(_PublishedRetrievalAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        authority.update(
            {
                "editorial_revision_identity": f"editorial_revision:{entry_id}.r2",
                "lifecycle_state": "editorial_review",
                "answer_eligible": False,
                "eligibility_reasons": ["editorial_approval_missing", "not_published"],
                "decision_query": "A successor revision remains under editorial review.",
            }
        )
        return authority

    async def get_retrieval_authority_for_revision(self, entry_id: str, revision_identity: str, *, now=None) -> dict:
        authority = await _PublishedRetrievalAuthority().get_retrieval_authority(entry_id, now=now)
        assert authority["editorial_revision_identity"] == revision_identity
        return authority


class _WordingSuccessorRetrievalAuthority(_PublishedRetrievalAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        authority.update(
            {
                "editorial_revision_identity": f"editorial_revision:{entry_id}.r2",
                "lifecycle_state": "published",
                "answer_eligible": True,
                "eligibility_reasons": [],
                "decision_query": "A wording-only successor has been approved without replacing the published version.",
            }
        )
        return authority

    async def get_retrieval_authority_for_revision(self, entry_id: str, revision_identity: str, *, now=None) -> dict:
        authority = await _PublishedRetrievalAuthority().get_retrieval_authority(entry_id, now=now)
        assert authority["editorial_revision_identity"] == revision_identity
        return authority


class _SourceLostUnpublishedSuccessorRetrievalAuthority(_UnpublishedSuccessorRetrievalAuthority):
    async def get_retrieval_authority_for_revision(self, entry_id: str, revision_identity: str, *, now=None) -> dict:
        del entry_id, revision_identity, now
        raise AppError(
            status_code=409,
            code="EDITORIAL_SOURCE_UNAVAILABLE",
            message="the retained source is no longer verified usable",
        )


class _UnverifiedSourceSuccessorRetrievalAuthority(_UnpublishedSuccessorRetrievalAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        authority["eligibility_reasons"] = ["not_published", "source_unavailable"]
        return authority


def test_inspection_is_durable_and_exactly_bound(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    user_headers = asyncio.run(_headers(client, username="ticket24-reader", role="user"))
    admin_headers = asyncio.run(_headers(client, username="ticket24-admin", role="admin"))
    url = f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}/inspection"

    assert client.get(url).status_code == 401
    assert client.get(url, headers=user_headers).status_code == 403

    inspected = client.post(url, headers=admin_headers, json={})

    assert inspected.status_code == 200
    payload = _data(inspected)
    assert payload["candidate"]["candidate_id"] == candidate_id
    assert payload["candidate"]["generation"] == 1
    assert payload["candidate"]["bundle_sha256"] == _sha256(
        {
            "schema": "reviewed_release_bundle/v1",
            "schema_version": 1,
            "bundle_id": "ticket24-bundle-001",
            "editorial_source_revision": "a" * 64,
            "exported_at": "2026-09-08T12:00:00Z",
            "items": [
                {
                    "bundle_item_id": "ticket24-item-001",
                    "operation": "create",
                    "artifact_sha256": _sha256(_approved_export()),
                    "artifact": _approved_export(),
                    "bundle_item_sha256": _sha256(
                        {
                            "bundle_item_id": "ticket24-item-001",
                            "operation": "create",
                            "artifact_sha256": _sha256(_approved_export()),
                            "artifact": _approved_export(),
                        }
                    ),
                }
            ],
        }
    )
    assert payload["inspection"]["candidate_id"] == candidate_id
    assert payload["inspection"]["frozen_input_sha256"] == payload["candidate"]["frozen_input_sha256"]
    assert payload["inspection"]["configuration_identity"].startswith("configuration:")
    assert payload["inspection"]["record_identity"].startswith("event:")
    assert payload["replacement"]["effect"] == "create"


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/candidates/candidate:private/inspection"),
        ("POST", "/candidates/candidate:private/inspection"),
        ("POST", "/candidates/candidate:private/acceptance"),
        ("GET", "/candidates/candidate:private/publication-eligibility"),
        ("GET", "/candidates/candidate:private/publication"),
        ("POST", "/publication-batches"),
    ],
)
def test_candidate_publication_surface_rejects_anonymous_and_knowledge_users(
    client: TestClient, method: str, path: str,
) -> None:
    headers = asyncio.run(_headers(client, username="ticket24-private-reader", role="user"))
    url = f"/api/v1/reviewed-release-bundles{path}"
    assert client.request(method, url, json={}).status_code == 401
    denied = client.request(method, url, headers=headers, json={})
    assert denied.status_code == 403
    assert "candidate:private" not in denied.text


def test_inspection_rejects_chunk_content_or_index_tampering(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-integrity-admin", role="admin"))

    async def tamper_chunk(
        candidate_id: str,
        *,
        content: str | None = None,
        chunk_index: int | None = None,
    ) -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            chunk = await session.scalar(
                select(CandidateBuildChunk).where(CandidateBuildChunk.candidate_id == candidate_id)
            )
            assert chunk is not None
            if content is not None:
                chunk.content = content
            if chunk_index is not None:
                chunk.chunk_index = chunk_index
            await session.commit()

    content_candidate = asyncio.run(_seed_ready_candidate(client))
    asyncio.run(tamper_chunk(content_candidate, content="tampered Candidate content"))
    content_result = client.post(
        f"/api/v1/reviewed-release-bundles/candidates/{content_candidate}/inspection",
        headers=admin_headers,
        json={},
    )
    assert content_result.status_code == 409
    assert content_result.json()["code"] == "CANDIDATE_PUBLICATION_INTEGRITY_FAILED"

    index_candidate = asyncio.run(
        _seed_ready_candidate(client, seed="002", entry_id="ticket24-entry-002")
    )
    asyncio.run(tamper_chunk(index_candidate, chunk_index=2))
    index_result = client.post(
        f"/api/v1/reviewed-release-bundles/candidates/{index_candidate}/inspection",
        headers=admin_headers,
        json={},
    )
    assert index_result.status_code == 409
    assert index_result.json()["code"] == "CANDIDATE_PUBLICATION_INTEGRITY_FAILED"


def test_acceptance_rejects_a_supported_query_outside_the_candidate_decision(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            artifact=_approved_export(
                supported_query="Which quantum encryption algorithm applies?"
            ),
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-supported-query-admin", role="admin"))
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)

    assert client.post(inspection_url, headers=admin_headers, json={}).status_code == 200
    rejected = client.post(acceptance_url, headers=admin_headers, json={})

    assert rejected.status_code == 409
    assert rejected.json()["code"] == "CANDIDATE_ACCEPTANCE_MATERIAL_INVALID"


def test_candidate_assurance_rejects_claim_linked_inputs_without_a_frozen_claim_evidence_contract() -> None:
    artifact = _claim_linked_export()
    artifact.pop("claim_evidence_contract")
    artifact.pop("claim_evidence_contract_sha256")

    with pytest.raises(ValueError, match="Claim-Evidence contract is invalid"):
        candidate_assurance_metadata(
            artifact=artifact,
            entry=artifact["entry"],
            entry_identity=artifact["entry_identity"],
        )



def test_claim_linked_candidate_acceptance_projects_the_frozen_claim_evidence_contract(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client, artifact=_claim_linked_export()))
    admin_headers = asyncio.run(_headers(client, username="ticket24-claim-linked-admin", role="admin"))
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)

    inspected = client.post(inspection_url, headers=admin_headers, json={})
    accepted = client.post(acceptance_url, headers=admin_headers, json={})

    assert inspected.status_code == 200, inspected.json()
    assert accepted.status_code == 200, accepted.json()
    assert _data(accepted)["supported"]["outcome"] == "evidence_gated_answer"


def test_claim_linked_candidate_acceptance_uses_reviewed_links_without_a_release_assured_resolver_contract(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="claim-linked-links-only",
            artifact=_claim_linked_export_without_release_assured_contract(),
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-claim-links-only-admin", role="admin"))
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)

    inspected = client.post(inspection_url, headers=admin_headers, json={})
    accepted = client.post(acceptance_url, headers=admin_headers, json={})

    assert inspected.status_code == 200, inspected.json()
    assert accepted.status_code == 200, accepted.json()
    assert _data(accepted)["supported"]["outcome"] == "evidence_gated_answer"


def test_publication_rejects_a_candidate_after_effective_embedding_configuration_changes(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="configuration-changed",
            entry_id="ticket24-configuration-changed",
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-configuration-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)

    monkeypatch.setattr(
        reviewed_bundles_api,
        "get_runtime_settings",
        lambda: get_settings().model_copy(update={"embedding_model": "ticket24-changed-embedding-model"}),
    )
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    rejected = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-configuration-changed",
            "selected_items": [_publish_selection(candidate_id, eligibility)],
        },
    )

    assert eligibility["eligible"] is False
    assert eligibility["reasons"] == ["CANDIDATE_EMBEDDING_CONFIGURATION_CHANGED"]
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "CANDIDATE_NOT_PUBLICATION_ELIGIBLE"


def test_claim_linked_candidate_acceptance_requires_every_frozen_source_link(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="claim-linked-multi-source",
            artifact=_multi_source_claim_linked_export(),
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-claim-multi-source-admin", role="admin"))
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)

    assert client.post(inspection_url, headers=admin_headers, json={}).status_code == 200
    accepted = client.post(acceptance_url, headers=admin_headers, json={})

    assert accepted.status_code == 200, accepted.json()
    evidence = _data(accepted)["supported"]["answer_evidence_set"]
    assert len(evidence["items"]) == 2
    assert {
        item["evidence"]["metadata"]["source_id"]
        for item in evidence["items"]
    } == {"source-ticket24-001", "source-ticket24-002"}


def test_claim_linked_candidate_publication_retains_every_frozen_source_link_for_runtime_evidence(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="claim-linked-runtime-multi-source",
            artifact=_multi_source_claim_linked_export(),
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-claim-runtime-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-claim-runtime-multi-source",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        )
    )

    async def retrieve() -> list[dict]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_ClaimLinkedMultiSourceRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items

    items = asyncio.run(retrieve())
    decision = decide_answer_evidence(
        normalized_question="Which Candidate publication contract applies?",
        query_conditions=QueryConditionSet.from_records(
            normalized_question="Which Candidate publication contract applies?",
            records=[
                {
                    "condition_id": "ticket24-production",
                    "field": "deployment",
                    "operator": "equals",
                    "value": "production",
                }
            ],
        ),
        candidates=items,
    )

    assert published["batch_complete"] is True
    assert len(items) == 1
    assert items[0]["metadata"]["source_evidence_projections"] == [
        {
            "source_identity": "source:source-ticket24-001",
            "source_id": "source-ticket24-001",
            "source_tier": "primary_evidence_source",
            "source_access_scope": "public",
            "source_title": "Ticket 24 publication authority",
            "source_authority": "ZhoMind architecture group",
            "source_url": "https://example.com/ticket24",
            "source_version": "2026-09-08",
            "source_review_date": "2026-09-08",
        },
        {
            "source_identity": "source:source-ticket24-002",
            "source_id": "source-ticket24-002",
            "source_tier": "primary_evidence_source",
            "source_access_scope": "public",
            "source_title": "Ticket 24 independent publication authority",
            "source_authority": "ZhoMind review council",
            "source_url": "https://example.com/ticket24-independent",
            "source_version": "2026-09-08",
            "source_review_date": "2026-09-08",
        },
    ]
    assert decision.is_sufficient is True, items
    assert decision.evidence_set is not None
    assert {
        dict(item.metadata_items)["source_id"]
        for item in decision.evidence_set.items
    } == {"source-ticket24-001", "source-ticket24-002"}
    summary = evidence_summary_from_execution(
        {
            "outcome": "evidence_gated_answer",
            "evidence_set": decision.evidence_set.to_record(),
        }
    )
    assert summary["coverage"] == "sufficient"
    assert summary["source_count"] == 2
    assert {source["source_title"] for source in summary["sources"]} == {
        "Ticket 24 publication authority",
        "Ticket 24 independent publication authority",
    }
    source_id_mismatch = json.loads(json.dumps(items))
    source_id_mismatch[0]["metadata"]["source_evidence_projections"][0]["source_id"] = (
        "source-ticket24-002"
    )
    mismatch_decision = decide_answer_evidence(
        normalized_question="Which Candidate publication contract applies?",
        query_conditions=decision.query_conditions,
        candidates=source_id_mismatch,
    )
    assert mismatch_decision.is_sufficient is False
    assert mismatch_decision.reason == "no_eligible_published_evidence"

    frozen_record = decision.evidence_set.to_record()
    null_bound_bindings = [
        dict(item["identity_binding"])
        for item in frozen_record["items"]
    ]
    null_bound_bindings[0]["candidate_evidence_source_identity"] = None
    null_bound_ids = [
        canonical_json_sha256(binding)
        for binding in null_bound_bindings
    ]
    governing_index = next(
        index
        for index, citation in enumerate(decision.evidence_set.citations)
        if citation.item_identity == decision.evidence_set.governing_citation.item_identity
    )
    null_bound_evidence_set = AnswerEvidenceSet.freeze(
        query_conditions=decision.evidence_set.query_conditions,
        items=decision.evidence_set.items,
        governing_item=decision.evidence_set.items[governing_index],
        item_identities=null_bound_ids,
        item_identity_bindings=null_bound_bindings,
    )
    with pytest.raises(ValueError, match="frozen Answer Evidence Set cannot be projected"):
        evidence_summary_from_execution(
            {
                "outcome": "evidence_gated_answer",
                "evidence_set": null_bound_evidence_set.to_record(),
            }
        )

    async def tamper_published_source_projection() -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            version = await session.get(
                PublishedKnowledgeVersion,
                published["published"][0]["publication_identity"],
            )
            assert version is not None
            chunk = await session.scalar(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == version.document_identity,
                    DocumentChunk.generation == version.generation,
                )
            )
            assert chunk is not None
            metadata = dict(chunk.chunk_metadata)
            projections = json.loads(json.dumps(metadata["source_evidence_projections"]))
            projections[0]["source_url"] = "https://example.org/unreviewed"
            metadata["source_evidence_projections"] = projections
            chunk.chunk_metadata = metadata
            await session.commit()

    asyncio.run(tamper_published_source_projection())
    tampered_items = asyncio.run(retrieve())

    assert tampered_items == []


def test_multi_source_runtime_projection_does_not_invent_replacement_diff_changes(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-multi-source-replacement-admin", role="admin"))
    original_candidate = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="multi-source-replacement-original",
            entry_id="ticket24-multi-source-replacement",
            artifact=_multi_source_claim_linked_export(),
        )
    )
    _inspect_and_accept(client, candidate_id=original_candidate, headers=admin_headers)
    _, _, original_eligibility_url = _candidate_urls(original_candidate)
    original_eligibility = _data(client.get(original_eligibility_url, headers=admin_headers))
    published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-multi-source-replacement-original",
                "selected_items": [_publish_selection(original_candidate, original_eligibility)],
            },
        )
    )

    replacement_candidate = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="multi-source-replacement-next",
            entry_id="ticket24-multi-source-replacement",
            generation=2,
            artifact=_multi_source_claim_linked_export(),
        )
    )
    replacement_inspection_url, _, _ = _candidate_urls(replacement_candidate)
    replacement_inspection = _data(
        client.post(replacement_inspection_url, headers=admin_headers, json={})
    )

    assert published["batch_complete"] is True
    assert replacement_inspection["replacement"]["effect"] == "replace"
    assert replacement_inspection["replacement"]["diff"] == {
        "schema": "candidate_replacement_diff/v1",
        "added": [],
        "removed": [],
        "changed": [],
    }


@pytest.mark.parametrize(
    ("mutation", "authority"),
    [
        ("content", _ClaimLinkedMultiSourceRetrievalAuthority),
        ("claim_contract", _ClaimLinkedMultiSourceRetrievalAuthority),
        ("citation_metadata", _ClaimLinkedMultiSourceRetrievalAuthority),
        ("successor_revision", _ClaimLinkedWordingSuccessorRetrievalAuthority),
    ],
)
def test_ordinary_retrieval_rejects_runtime_drift_from_the_frozen_claim_linked_candidate(
    client: TestClient,
    monkeypatch,
    mutation: str,
    authority,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed=f"frozen-runtime-{mutation}",
            entry_id=f"ticket24-frozen-runtime-{mutation}",
            artifact=_multi_source_claim_linked_export(),
        )
    )
    admin_headers = asyncio.run(
        _headers(client, username=f"ticket24-frozen-runtime-{mutation}", role="admin")
    )
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": f"ticket24-frozen-runtime-{mutation}",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )

    async def tamper_and_retrieve() -> tuple[list[dict], list[dict[str, str]]]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            version = await session.scalar(
                select(PublishedKnowledgeVersion).where(
                    PublishedKnowledgeVersion.candidate_id == candidate_id
                )
            )
            assert version is not None
            chunk = await session.scalar(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == version.document_identity,
                    DocumentChunk.generation == version.generation,
                )
            )
            assert chunk is not None
            if mutation == "content":
                chunk.content = (
                    "Which Candidate publication contract applies? "
                    "A runtime projection must never become self-authorizing."
                )
            elif mutation == "claim_contract":
                metadata = dict(chunk.chunk_metadata)
                contract = json.loads(metadata["claim_evidence_contract"])
                contract["claims"][0]["evidence"] = [contract["claims"][0]["evidence"][0]]
                canonical_contract = json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                metadata["claim_evidence_contract"] = canonical_contract
                metadata["claim_evidence_contract_sha256"] = hashlib.sha256(
                    canonical_contract.encode("utf-8")
                ).hexdigest()
                chunk.chunk_metadata = metadata
            else:
                metadata = dict(chunk.chunk_metadata)
                if mutation == "citation_metadata":
                    metadata["title"] = "Unreviewed runtime citation title"
                    metadata["entry_title"] = "Unreviewed runtime entry title"
                    metadata["domain"] = "unreviewed-runtime-domain"
                    metadata["review_date"] = "2099-01-01"
                else:
                    metadata["editorial_revision_identity"] = str(
                        metadata["editorial_revision_identity"]
                    ).removesuffix(".r1") + ".r2"
                chunk.chunk_metadata = metadata
            await session.commit()
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=authority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items, result.candidate_exclusions

    items, exclusions = asyncio.run(tamper_and_retrieve())

    assert items == []
    assert {item["reason"] for item in exclusions} == {"published_runtime_projection_invalid"}


def test_claim_linked_runtime_evidence_distinguishes_source_identities_with_identical_citations(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="claim-linked-identical-citation-runtime",
            artifact=_same_citation_multi_source_claim_linked_export(),
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-identical-citation-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-identical-citation-runtime",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )

    async def retrieve() -> list[dict]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_ClaimLinkedMultiSourceRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items

    decision = decide_answer_evidence(
        normalized_question="Which Candidate publication contract applies?",
        query_conditions=QueryConditionSet.from_records(
            normalized_question="Which Candidate publication contract applies?",
            records=[
                {
                    "condition_id": "ticket24-production",
                    "field": "deployment",
                    "operator": "equals",
                    "value": "production",
                }
            ],
        ),
        candidates=asyncio.run(retrieve()),
    )

    assert decision.is_sufficient is True
    assert decision.evidence_set is not None
    frozen = decision.evidence_set.to_record()
    assert len({item["item_identity"] for item in frozen["items"]}) == 2
    assert evidence_summary_from_execution(
        {"outcome": "evidence_gated_answer", "evidence_set": frozen}
    )["source_count"] == 2


def test_candidate_acceptance_requires_exact_supported_and_boundary_records(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-acceptance-admin", role="admin"))
    inspection_url = f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}/inspection"
    acceptance_url = f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}/acceptance"

    assert client.post(acceptance_url, headers=admin_headers, json={}).status_code == 409
    assert client.post(inspection_url, headers=admin_headers, json={}).status_code == 200

    accepted = client.post(acceptance_url, headers=admin_headers, json={})

    assert accepted.status_code == 200
    payload = _data(accepted)
    assert payload["record_identity"].startswith("event:")
    assert payload["candidate_id"] == candidate_id
    assert payload["supported"]["outcome"] == "evidence_gated_answer"
    assert payload["supported"]["expected_governing_entry_identity"] == "entry:ticket24-entry-001"
    assert payload["supported"]["expected_governing_section_id"] == "recommendation_or_reviewed_branches"
    assert payload["supported"]["answer_evidence_set"]["identity"]
    assert payload["supported"]["citation_markers"] == ["S1"]
    assert payload["boundary"]["outcome"] == "insufficient_evidence_reply"
    assert payload["boundary"]["reason"] == "decisive_condition_missing"
    assert payload["boundary"]["query_condition_set_identity"]
    assert payload["boundary"]["citation_markers"] == []
    assert payload["boundary"]["provider_call_count"] == 0
    reloaded = _data(client.get(inspection_url, headers=admin_headers))
    assert reloaded["acceptance"]["record_identity"] == payload["record_identity"]
    assert reloaded["acceptance"]["supported"] == payload["supported"]
    assert reloaded["acceptance"]["boundary"] == payload["boundary"]
    assert reloaded["acceptance"]["accepted_by"] == reloaded["inspection"]["inspected_by"]
    assert reloaded["inspection"]["generation"] == 1
    assert reloaded["candidate"]["metadata"]["title"] == "Ticket 24 Candidate publication contract"
    assert "body" not in reloaded["candidate"]["metadata"]


@pytest.mark.parametrize(
    "record_kind,field_path,replacement",
    [
        ("inspection", ("entry_identity",), "entry:foreign-entry"),
        ("inspection", ("document_identity",), "runtime-document:foreign-document"),
        ("inspection", ("bundle_id",), "bundle:foreign-bundle"),
        ("inspection", ("bundle_item_id",), "bundle_item:foreign-item"),
        ("inspection", ("editorial_source_revision",), "0" * 64),
        ("inspection", ("requested_generation",), True),
        ("inspection", ("configuration", "active"), 0),
        ("acceptance", ("configuration", "active"), 0),
        ("acceptance", ("entry_identity",), "entry:foreign-entry"),
        ("acceptance", ("result", "candidate_id"), "candidate:foreign-candidate"),
        ("acceptance", ("result", "supported", "answer_evidence_set", "identity"), "forged-evidence"),
        ("acceptance", ("result", "supported", "citation_markers"), ["S99"]),
        ("acceptance", ("result", "boundary", "query_condition_set_identity"), "forged-conditions"),
        ("acceptance", ("result", "boundary", "provider_call_count"), False),
    ],
)
def test_publication_rejects_corrupted_retained_inspection_or_acceptance(
    client: TestClient, monkeypatch, record_kind: str, field_path: tuple[str, ...], replacement: object,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(reviewed_bundles_api, "CanonicalEditorialExportVerifier", lambda _session: _ApprovedExportVerifier())
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    headers = asyncio.run(_headers(client, username="ticket24-retained-record-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=headers)
    inspection_url, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=headers))
    selected = _publish_selection(candidate_id, eligibility)
    record_identity = eligibility[f"{record_kind}_record_identity"]

    async def corrupt_retained_record() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, record_identity)
            payload = deepcopy(record.payload)
            target = payload
            for field in field_path[:-1]:
                target = target[field]
            target[field_path[-1]] = replacement
            # Simulate storage corruption below the immutable ORM write boundary.
            table = CanonicalRecordModel.__table__
            await session.execute(table.update().where(table.c.stable_id == record_identity).values(payload=payload))
            await session.commit()

    asyncio.run(corrupt_retained_record())
    refreshed = _data(client.get(eligibility_url, headers=headers))
    assert refreshed["eligible"] is False
    assert f"CANDIDATE_{record_kind.upper()}_REQUIRED" in refreshed["reasons"]
    if record_kind == "inspection":
        assert _data(client.get(inspection_url, headers=headers))["inspection"] is None
    rejected = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=headers,
        json={"confirmation_id": "ticket24-corrupted-record", "selected_items": [selected]},
    )
    assert rejected.status_code == 409
    assert _publication_state(client, candidate_id=candidate_id, headers=headers)["published_knowledge_version"] is None


def test_candidate_acceptance_binds_explicit_conditions_for_supported_and_boundary_queries(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    production = {
        "condition_id": "environment-production",
        "field": "environment",
        "operator": "equals",
        "value": "production",
    }
    staging = {
        "condition_id": "environment-staging",
        "field": "environment",
        "operator": "equals",
        "value": "staging",
    }
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            artifact=_approved_export(
                applicability_conditions=[production],
                supported_query_conditions=[production],
                boundary_query="Which Candidate publication contract applies?",
                boundary_query_conditions=[staging],
            ),
        )
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-conditions-admin", role="admin"))
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)

    assert client.post(inspection_url, headers=admin_headers, json={}).status_code == 200
    accepted = client.post(acceptance_url, headers=admin_headers, json={})

    assert accepted.status_code == 200
    payload = _data(accepted)
    assert payload["supported"]["answer_evidence_set"]["query_conditions"]["conditions"] == [production]
    assert payload["boundary"]["reason"] == "decision_not_covered"
    assert payload["boundary"]["query_condition_set_identity"] != payload["supported"]["query_condition_set_identity"]


def test_publication_requires_exact_eligible_confirmation_and_creates_a_version(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-publication-admin", role="admin"))
    candidate_url = f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}"
    inspection_url = f"{candidate_url}/inspection"
    acceptance_url = f"{candidate_url}/acceptance"
    eligibility_url = f"{candidate_url}/publication-eligibility"
    batch_url = "/api/v1/reviewed-release-bundles/publication-batches"

    inspection = _data(client.post(inspection_url, headers=admin_headers, json={}))
    acceptance = _data(client.post(acceptance_url, headers=admin_headers, json={}))

    eligibility = client.get(eligibility_url, headers=admin_headers)

    assert eligibility.status_code == 200
    eligibility_data = _data(eligibility)
    assert eligibility_data["eligible"] is True
    assert eligibility_data["effect"] == "create"
    assert eligibility_data["current_published_knowledge_version"] is None
    assert eligibility_data["inspection_record_identity"] == inspection["inspection"]["record_identity"]
    assert eligibility_data["acceptance_record_identity"] == acceptance["record_identity"]

    invalid_confirmation = client.post(
        batch_url,
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-publication-batch-001",
            "selected_items": [
                {
                    "candidate_id": candidate_id,
                    "effect": "replace",
                    "current_published_knowledge_version": "published_knowledge_version:invented",
                    "inspection_record_identity": inspection["inspection"]["record_identity"],
                    "acceptance_record_identity": acceptance["record_identity"],
                }
            ],
        },
    )
    assert invalid_confirmation.status_code == 409

    selected = _publish_selection(candidate_id, eligibility_data)
    published = client.post(
        batch_url,
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-publication-batch-001",
            "selected_items": [selected],
        },
    )

    assert published.status_code == 200
    payload = _data(published)
    assert payload["batch_complete"] is True
    assert payload["published"] == [
        {
            "candidate_id": candidate_id,
            "effect": "create",
            "publication_identity": payload["published"][0]["publication_identity"],
        }
    ]
    assert payload["failed"] == []
    assert payload["skipped"] == []
    assert payload["published"][0]["publication_identity"].startswith("published_knowledge_version:")

    persisted = _publication_state(client, candidate_id=candidate_id, headers=admin_headers)
    version = persisted["published_knowledge_version"]
    assert version is not None
    assert persisted["is_current_for_entry"] is True
    assert persisted["current_published_knowledge_version"] == version
    assert version["candidate_id"] == candidate_id
    assert version["supersedes_published_knowledge_version_identity"] is None
    assert version["inspection_record_identity"] == inspection["inspection"]["record_identity"]
    assert version["acceptance_record_identity"] == acceptance["record_identity"]

    repeated = client.post(
        batch_url,
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-publication-batch-001",
            "selected_items": [selected],
        },
    )
    assert repeated.status_code == 200
    assert _data(repeated) == payload

    repeated_state = _publication_state(client, candidate_id=candidate_id, headers=admin_headers)
    assert repeated_state["published_knowledge_version"] == version
    assert repeated_state["is_current_for_entry"] is True

    republish = client.post(
        batch_url,
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-publication-batch-002",
            "selected_items": [
                {
                    **selected,
                    "effect": "replace",
                    "current_published_knowledge_version": version["identity"],
                }
            ],
        },
    )
    assert republish.status_code == 409
    assert republish.json()["code"] == "CANDIDATE_ALREADY_PUBLISHED"


def test_replacement_inspection_and_acceptance_retain_the_inspected_version_after_pointer_changes(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-replacement-audit-admin", role="admin"))
    original_candidate = asyncio.run(_seed_ready_candidate(client, seed="audit-001"))
    _inspect_and_accept(client, candidate_id=original_candidate, headers=admin_headers)
    _, _, original_eligibility_url = _candidate_urls(original_candidate)
    original_eligibility = _data(client.get(original_eligibility_url, headers=admin_headers))
    original_published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-replacement-audit-original",
                "selected_items": [_publish_selection(original_candidate, original_eligibility)],
            },
        )
    )
    original_version = original_published["published"][0]["publication_identity"]

    inspected_candidate = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="audit-002",
            generation=2,
            chunk_content_suffix=" Inspection binding retains this prior version.",
        )
    )
    inspection_url, acceptance_url, _ = _candidate_urls(inspected_candidate)
    inspected = _data(client.post(inspection_url, headers=admin_headers, json={}))
    accepted = _data(client.post(acceptance_url, headers=admin_headers, json={}))

    assert inspected["inspection"]["replaces_published_knowledge_version_identity"] == original_version
    assert accepted["replaces_published_knowledge_version_identity"] == original_version

    successor_candidate = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="audit-003",
            generation=3,
            chunk_content_suffix=" A different Candidate replaces the original version first.",
        )
    )
    _inspect_and_accept(client, candidate_id=successor_candidate, headers=admin_headers)
    _, _, successor_eligibility_url = _candidate_urls(successor_candidate)
    successor_eligibility = _data(client.get(successor_eligibility_url, headers=admin_headers))
    successor_published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-replacement-audit-successor",
                "selected_items": [_publish_selection(successor_candidate, successor_eligibility)],
            },
        )
    )
    successor_version = successor_published["published"][0]["publication_identity"]

    async def mark_inspected_candidate_superseded() -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            inspected_job = await session.get(CandidateBuildJob, "ticket24-candidate-job-audit-002")
            assert inspected_job is not None
            inspected_job.status = "superseded"
            inspected_job.terminal_state = "superseded"
            inspected_job.allowed_next_action = "import_new_bundle"
            await session.commit()

    asyncio.run(mark_inspected_candidate_superseded())
    reloaded = _data(client.get(inspection_url, headers=admin_headers))

    assert successor_version != original_version
    assert reloaded["candidate"]["candidate_id"] == inspected_candidate
    assert reloaded["replacement"]["current_published_knowledge_version"]["identity"] == original_version


def test_candidate_acceptance_fails_closed_for_non_semantic_applicability_conditions(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    artifact = _approved_export(
        applicability_conditions=[{"condition_id": "ticket24-incomplete-condition"}],
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client, artifact=artifact))
    admin_headers = asyncio.run(_headers(client, username="ticket24-conditions-admin", role="admin"))
    inspection_url, acceptance_url, _ = _candidate_urls(candidate_id)

    assert client.post(inspection_url, headers=admin_headers, json={}).status_code == 200
    rejected = client.post(acceptance_url, headers=admin_headers, json={})

    assert rejected.status_code == 409
    assert rejected.json()["code"] == "CANDIDATE_ACCEPTANCE_MATERIAL_INVALID"


def test_inspection_rejects_candidate_chunk_metadata_that_no_longer_matches_the_frozen_artifact(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-tamper-admin", role="admin"))

    async def tamper_chunk_metadata() -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            chunk = await session.get(CandidateBuildChunk, "ticket24-candidate-chunk-001")
            assert chunk is not None
            chunk.chunk_metadata = {
                **chunk.chunk_metadata,
                "source_relationships": [
                    {
                        "source_identity": "source:unrelated-ticket24-source",
                        "availability": "verified_usable",
                        "access_scope": "public",
                    }
                ],
            }
            await session.commit()

    asyncio.run(tamper_chunk_metadata())
    inspection_url, _, _ = _candidate_urls(candidate_id)
    rejected = client.post(inspection_url, headers=admin_headers, json={})

    assert rejected.status_code == 409
    assert rejected.json()["code"] == "CANDIDATE_PUBLICATION_INTEGRITY_FAILED"


def test_replacement_failure_is_isolated_and_preserves_the_existing_pointer(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api
    from app.reviewed_bundles.publication import CandidatePublicationService

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-replacement-admin", role="admin"))
    original_candidate = asyncio.run(_seed_ready_candidate(client, seed="001"))
    original_inspection_url, original_acceptance_url, original_eligibility_url = _candidate_urls(original_candidate)
    original_inspection = _data(client.post(original_inspection_url, headers=admin_headers, json={}))
    assert client.post(original_acceptance_url, headers=admin_headers, json={}).status_code == 200
    _, _, original_eligibility_url = _candidate_urls(original_candidate)
    original_eligibility = _data(client.get(original_eligibility_url, headers=admin_headers))
    original_published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-original-publication",
                "selected_items": [_publish_selection(original_candidate, original_eligibility)],
            },
        )
    )
    original_version = original_published["published"][0]["publication_identity"]

    replacement_candidate = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="002",
            entry_id="ticket24-entry-001",
            generation=2,
            chunk_content_suffix=" Replacement proposal changes this approved guidance.",
        )
    )
    independent_candidate = asyncio.run(
        _seed_ready_candidate(client, seed="003", entry_id="ticket24-entry-003", generation=1)
    )
    replacement_inspection_url, replacement_acceptance_url, _ = _candidate_urls(replacement_candidate)
    replacement_inspection = _data(
        client.post(replacement_inspection_url, headers=admin_headers, json={})
    )
    assert replacement_inspection["replacement"]["effect"] == "replace"
    assert replacement_inspection["replacement"]["current_published_knowledge_version"]["identity"] == original_version
    assert replacement_inspection["replacement"]["diff"]["schema"] == "candidate_replacement_diff/v1"
    assert replacement_inspection["replacement"]["diff"]["changed"] == [
        {
            "chunk_index": 0,
            "candidate_content_sha256": replacement_inspection["candidate"]["chunks"][0]["content_sha256"],
            "published_content_sha256": original_inspection["candidate"]["chunks"][0]["content_sha256"],
            "candidate_content": replacement_inspection["candidate"]["chunks"][0]["content"],
            "published_content": original_inspection["candidate"]["chunks"][0]["content"],
        }
    ]
    assert client.post(replacement_acceptance_url, headers=admin_headers, json={}).status_code == 200
    _inspect_and_accept(client, candidate_id=independent_candidate, headers=admin_headers)
    _, _, replacement_eligibility_url = _candidate_urls(replacement_candidate)
    _, _, independent_eligibility_url = _candidate_urls(independent_candidate)
    replacement_eligibility = _data(client.get(replacement_eligibility_url, headers=admin_headers))
    independent_eligibility = _data(client.get(independent_eligibility_url, headers=admin_headers))

    assert replacement_eligibility["effect"] == "replace"
    assert replacement_eligibility["current_published_knowledge_version"]["identity"] == original_version

    original_before_pointer_switch = CandidatePublicationService._before_pointer_switch

    async def fail_only_the_replacement(service, binding):
        if binding.candidate.stable_id == replacement_candidate:
            raise OSError("injected replacement pointer-switch failure")
        await original_before_pointer_switch(service, binding)

    monkeypatch.setattr(CandidatePublicationService, "_before_pointer_switch", fail_only_the_replacement)
    batch = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-isolated-batch",
            "selected_items": [
                _publish_selection(replacement_candidate, replacement_eligibility),
                _publish_selection(independent_candidate, independent_eligibility),
            ],
        },
    )

    assert batch.status_code == 200
    result = _data(batch)
    assert result["batch_complete"] is False
    assert result["published"] == [
        {
            "candidate_id": independent_candidate,
            "effect": "create",
            "publication_identity": result["published"][0]["publication_identity"],
        }
    ]
    assert result["failed"] == [
        {
            "candidate_id": replacement_candidate,
            "effect": "replace",
            "reason": "PUBLICATION_ITEM_RETRYABLE_FAILURE",
        }
    ]
    assert result["skipped"] == []

    replaced_entry = _publication_state(client, candidate_id=original_candidate, headers=admin_headers)
    assert replaced_entry["published_knowledge_version"]["identity"] == original_version
    assert replaced_entry["is_current_for_entry"] is True

    independent_entry = _publication_state(client, candidate_id=independent_candidate, headers=admin_headers)
    assert independent_entry["published_knowledge_version"]["identity"] == result["published"][0]["publication_identity"]
    assert independent_entry["is_current_for_entry"] is True

    repeated = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-isolated-batch",
            "selected_items": [
                _publish_selection(replacement_candidate, replacement_eligibility),
                _publish_selection(independent_candidate, independent_eligibility),
            ],
        },
    )
    assert repeated.status_code == 200
    assert _data(repeated) == result

    monkeypatch.setattr(CandidatePublicationService, "_before_pointer_switch", original_before_pointer_switch)
    retry = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-replacement-retry",
            "selected_items": [_publish_selection(replacement_candidate, replacement_eligibility)],
        },
    )
    assert retry.status_code == 200
    retried = _data(retry)
    assert retried["batch_complete"] is True
    assert retried["failed"] == []
    assert retried["skipped"] == []
    replacement_state = _publication_state(client, candidate_id=replacement_candidate, headers=admin_headers)
    assert replacement_state["is_current_for_entry"] is True
    assert replacement_state["published_knowledge_version"]["supersedes_published_knowledge_version_identity"] == original_version
    assert _publication_state(client, candidate_id=independent_candidate, headers=admin_headers) == independent_entry


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("pointer", "generation", 999),
        ("pointer", "document_identity", "runtime-document:foreign"),
        ("version", "bundle_sha256", "0" * 64),
        ("version", "frozen_input_sha256", "0" * 64),
        ("version", "configuration_identity", "configuration:foreign"),
        ("version", "inspection_record_identity", "event:foreign"),
    ],
)
def test_replacement_inspection_rejects_corrupted_publication_projection(
    client: TestClient, monkeypatch, target: str, field: str, value: object,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(reviewed_bundles_api, "CanonicalEditorialExportVerifier", lambda _session: _ApprovedExportVerifier())
    headers = asyncio.run(_headers(client, username="ticket24-pointer-integrity-admin", role="admin"))
    original = asyncio.run(_seed_ready_candidate(client))
    _inspect_and_accept(client, candidate_id=original, headers=headers)
    _, _, eligibility_url = _candidate_urls(original)
    published = _data(client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=headers,
        json={
            "confirmation_id": "ticket24-pointer-integrity",
            "selected_items": [_publish_selection(original, _data(client.get(eligibility_url, headers=headers)))],
        },
    ))
    version_id = published["published"][0]["publication_identity"]
    replacement = asyncio.run(_seed_ready_candidate(client, seed="replacement", generation=2))

    async def corrupt_projection() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            if target == "pointer":
                row = await session.get(PublishedKnowledgePointer, "entry:ticket24-entry-001")
            else:
                row = await session.get(PublishedKnowledgeVersion, version_id)
            setattr(row, field, value)
            await session.commit()

    asyncio.run(corrupt_projection())
    inspection_url, _, _ = _candidate_urls(replacement)
    for response in (client.get(inspection_url, headers=headers), client.post(inspection_url, headers=headers, json={})):
        assert response.status_code == 409
        assert response.json()["code"] == "PUBLISHED_KNOWLEDGE_POINTER_INTEGRITY_FAILED"

    async def retrieve() -> list[dict]:
        async with client.app.state.test_auth_session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session, settings=get_settings(), editorial_authority=_PublishedRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
            return result.items

    assert asyncio.run(retrieve()) == []


def test_processing_confirmation_recovers_a_persisted_item_before_its_result_is_recorded(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-recovery-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    selected_items = [_publish_selection(candidate_id, eligibility)]
    confirmation_id = "ticket24-recover-processing-confirmation"
    batch_url = "/api/v1/reviewed-release-bundles/publication-batches"

    published = _data(
        client.post(
            batch_url,
            headers=admin_headers,
            json={"confirmation_id": confirmation_id, "selected_items": selected_items},
        )
    )
    publication_identity = published["published"][0]["publication_identity"]

    async def erase_persisted_item_result() -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            confirmation = await session.get(CandidatePublicationConfirmation, confirmation_id)
            assert confirmation is not None
            confirmation.state = "processing"
            confirmation.completed_at = None
            confirmation.results = {"batch_complete": False, "published": [], "failed": [], "skipped": []}
            await session.commit()

    asyncio.run(erase_persisted_item_result())

    recovered = client.post(
        batch_url,
        headers=admin_headers,
        json={"confirmation_id": confirmation_id, "selected_items": selected_items},
    )

    assert recovered.status_code == 200
    assert _data(recovered) == {
        "batch_complete": True,
        "published": [
            {
                "candidate_id": candidate_id,
                "effect": "create",
                "publication_identity": publication_identity,
            }
        ],
        "failed": [],
        "skipped": [],
    }


def test_concurrent_confirmation_execution_is_leased_and_retains_one_persisted_outcome(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-concurrent-confirmation-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    selected_items = [_publish_selection(candidate_id, eligibility)]
    payload = {
        "confirmation_id": "ticket24-concurrent-confirmation",
        "selected_items": selected_items,
    }

    async def run() -> tuple[dict, dict, int]:
        session_factory = client.app.state.test_auth_session_factory
        first_started = asyncio.Event()
        allow_first_to_publish = asyncio.Event()

        class _BlockingPublicationService(CandidatePublicationService):
            async def _publish_selected_item(self, selected, *, actor_identity, **kwargs):
                first_started.set()
                await allow_first_to_publish.wait()
                return await super()._publish_selected_item(selected, actor_identity=actor_identity, **kwargs)

        async with session_factory() as first_session, session_factory() as second_session:
            first = _BlockingPublicationService(
                first_session,
                editorial_export_verifier=_ApprovedExportVerifier(),
            )
            second = CandidatePublicationService(
                second_session,
                editorial_export_verifier=_ApprovedExportVerifier(),
            )
            first_task = asyncio.create_task(
                first.confirm_publication_batch(payload, actor_identity="member:ticket24-concurrency-admin")
            )
            await first_started.wait()
            with pytest.raises(AppError) as in_progress:
                await second.confirm_publication_batch(payload, actor_identity="member:ticket24-concurrency-admin")
            allow_first_to_publish.set()
            completed = await first_task
            replayed = await second.confirm_publication_batch(
                payload,
                actor_identity="member:ticket24-concurrency-admin",
            )

        async with session_factory() as session:
            version_count = await session.scalar(select(func.count()).select_from(PublishedKnowledgeVersion))

        assert in_progress.value.code == "PUBLICATION_CONFIRMATION_IN_PROGRESS"
        return completed, replayed, int(version_count or 0)

    completed, replayed, version_count = asyncio.run(run())

    assert replayed == completed
    assert completed["batch_complete"] is True
    assert len(completed["published"]) == 1
    assert version_count == 1


def test_expired_confirmation_lease_cannot_publish_after_a_new_executor_takes_over(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-expired-confirmation-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    payload = {
        "confirmation_id": "ticket24-expired-confirmation",
        "selected_items": [_publish_selection(candidate_id, eligibility)],
    }

    async def run() -> tuple[dict, int]:
        session_factory = client.app.state.test_auth_session_factory
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        allow_first_to_continue = asyncio.Event()
        allow_second_to_continue = asyncio.Event()

        class _BlockingPublicationService(CandidatePublicationService):
            def __init__(self, *args, started, release, **kwargs):
                super().__init__(*args, **kwargs)
                self._started = started
                self._release = release

            async def _publish_selected_item(self, selected, *, actor_identity, **kwargs):
                self._started.set()
                await self._release.wait()
                return await super()._publish_selected_item(selected, actor_identity=actor_identity, **kwargs)

        async with session_factory() as first_session, session_factory() as second_session:
            first = _BlockingPublicationService(
                first_session,
                editorial_export_verifier=_ApprovedExportVerifier(),
                started=first_started,
                release=allow_first_to_continue,
            )
            second = _BlockingPublicationService(
                second_session,
                editorial_export_verifier=_ApprovedExportVerifier(),
                started=second_started,
                release=allow_second_to_continue,
            )
            first_task = asyncio.create_task(
                first.confirm_publication_batch(payload, actor_identity="member:ticket24-expired-first")
            )
            await first_started.wait()

            async with session_factory() as recovery_session:
                confirmation = await recovery_session.get(
                    CandidatePublicationConfirmation,
                    payload["confirmation_id"],
                )
                assert confirmation is not None
                confirmation.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
                await recovery_session.commit()

            second_task = asyncio.create_task(
                second.confirm_publication_batch(payload, actor_identity="member:ticket24-expired-first")
            )
            await second_started.wait()
            allow_first_to_continue.set()
            with pytest.raises(AppError) as stale_executor:
                await first_task

            async with session_factory() as session:
                version_count_before_new_executor = await session.scalar(
                    select(func.count()).select_from(PublishedKnowledgeVersion)
                )

            allow_second_to_continue.set()
            completed = await second_task

        assert stale_executor.value.code == "PUBLICATION_CONFIRMATION_LEASE_LOST"
        return completed, int(version_count_before_new_executor or 0)

    completed, version_count_before_new_executor = asyncio.run(run())

    assert version_count_before_new_executor == 0
    assert completed["batch_complete"] is True
    assert len(completed["published"]) == 1


def test_ordinary_retrieval_observes_only_the_canonical_published_projection(client: TestClient, monkeypatch) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-retrieval-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-retrieval-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        )
    )
    publication_identity = published["published"][0]["publication_identity"]

    async def retrieve() -> tuple[list[dict], int]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            candidate_chunks = (
                await session.execute(
                    select(CandidateBuildChunk).where(CandidateBuildChunk.candidate_id == candidate_id)
                )
            ).scalars().all()
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_PublishedRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items, len(candidate_chunks)

    items, candidate_chunk_count = asyncio.run(retrieve())

    assert candidate_chunk_count == 1
    assert [item["publication_identity"] for item in items] == [publication_identity]
    assert [item["chunk_id"] for item in items] != ["ticket24-candidate-chunk-001"]
    assert items[0]["metadata"]["candidate_build"] is False
    decision = decide_answer_evidence(
        normalized_question="Which Candidate publication contract applies?",
        query_conditions=QueryConditionSet.from_records(
            normalized_question="Which Candidate publication contract applies?",
            records=[
                {
                    "condition_id": "ticket24-production",
                    "field": "deployment",
                    "operator": "equals",
                    "value": "production",
                }
            ],
        ),
        candidates=items,
    )
    assert decision.is_sufficient is True, items
    assert decision.evidence_set is not None
    assert decision.evidence_set.items[0].publication_version == publication_identity


@pytest.mark.parametrize("mutation", ["remove", "null"])
def test_ordinary_retrieval_rejects_a_ticket24_runtime_projection_without_its_canonical_identity(
    client: TestClient,
    monkeypatch,
    mutation: str,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(
        _seed_ready_candidate(
            client,
            seed=f"canonical-identity-{mutation}",
            entry_id=f"ticket24-canonical-identity-{mutation}",
        )
    )
    admin_headers = asyncio.run(
        _headers(client, username=f"ticket24-canonical-identity-{mutation}", role="admin")
    )
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": f"ticket24-canonical-identity-{mutation}",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )

    async def tamper_and_retrieve() -> tuple[list[dict], list[dict[str, str]]]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            version = await session.scalar(
                select(PublishedKnowledgeVersion).where(
                    PublishedKnowledgeVersion.candidate_id == candidate_id
                )
            )
            assert version is not None
            chunk = await session.scalar(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == version.document_identity,
                    DocumentChunk.generation == version.generation,
                )
            )
            assert chunk is not None
            metadata = dict(chunk.chunk_metadata)
            if mutation == "remove":
                metadata.pop("published_knowledge_version_identity")
            else:
                metadata["published_knowledge_version_identity"] = None
            chunk.chunk_metadata = metadata
            await session.commit()
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_PublishedRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items, result.candidate_exclusions

    items, exclusions = asyncio.run(tamper_and_retrieve())

    assert items == []
    assert {item["reason"] for item in exclusions} == {"published_version_identity_invalid"}


def test_legacy_build_routes_cannot_mutate_a_published_candidate_runtime_projection(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-runtime-projection-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-runtime-projection-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )
    document_id = "runtime-document:ticket24-entry-001"

    single_rejected = client.post(
        f"/api/v1/documents/{document_id}/build",
        headers=admin_headers,
        json={"chunk_strategy": "general"},
    )
    batch_rejected = client.post(
        "/api/v1/documents/batch-build",
        headers=admin_headers,
        json={"document_ids": [document_id], "chunk_strategy": "general"},
    )

    async def load_projection() -> tuple[Document, list[DocumentChunk]]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            chunks = (
                await session.execute(
                    select(DocumentChunk)
                    .where(
                        DocumentChunk.document_id == document_id,
                        DocumentChunk.generation == document.published_generation,
                    )
                    .order_by(DocumentChunk.chunk_index.asc())
                )
            ).scalars().all()
        return document, chunks

    document, chunks = asyncio.run(load_projection())

    assert single_rejected.status_code == 409
    assert single_rejected.json()["code"] == "REVIEWED_BUNDLE_RUNTIME_MUTATION_REJECTED"
    assert batch_rejected.status_code == 409
    assert batch_rejected.json()["code"] == "REVIEWED_BUNDLE_RUNTIME_MUTATION_REJECTED"
    assert document.status == "ready"
    assert document.published_generation == 1
    assert document.next_generation == 2
    assert len(chunks) == 1


def test_ordinary_retrieval_excludes_a_published_chunk_with_a_forged_current_revision(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-forged-revision-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-forged-revision-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )

    async def forge_revision_and_retrieve() -> tuple[list[dict], list[dict[str, str]]]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            chunk = await session.scalar(
                select(DocumentChunk).where(DocumentChunk.document_id == "runtime-document:ticket24-entry-001")
            )
            assert chunk is not None
            metadata = dict(chunk.chunk_metadata)
            metadata["editorial_revision_identity"] = "editorial_revision:ticket24-entry-001.r999"
            chunk.chunk_metadata = metadata
            await session.commit()
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_PublishedRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items, result.candidate_exclusions

    items, exclusions = asyncio.run(forge_revision_and_retrieve())

    assert items == []
    assert {item["reason"] for item in exclusions} == {"editorial_revision_mismatch"}


@pytest.mark.parametrize(
    "authority_type", [_UnpublishedSuccessorRetrievalAuthority, _UnverifiedSourceSuccessorRetrievalAuthority],
)
def test_ordinary_retrieval_preserves_a_published_version_during_an_unpublished_successor_revision(
    client: TestClient,
    monkeypatch,
    authority_type,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-successor-retrieval-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    publication = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-successor-retrieval-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        )
    )
    publication_identity = publication["published"][0]["publication_identity"]

    async def retrieve() -> list[dict]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=authority_type(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items

    items = asyncio.run(retrieve())

    assert [item["publication_identity"] for item in items] == [publication_identity]
    assert items[0]["editorial_revision_identity"] == "editorial_revision:ticket24-entry-001.r1"
    assert items[0]["decision_query"] == (
        "Which Candidate publication contract applies to reviewed bundle entries?"
    )


def test_ordinary_retrieval_preserves_a_published_version_during_a_published_wording_successor(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-wording-successor-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    publication = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-wording-successor-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        )
    )
    publication_identity = publication["published"][0]["publication_identity"]

    async def retrieve() -> list[dict]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_WordingSuccessorRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items

    items = asyncio.run(retrieve())

    assert [item["publication_identity"] for item in items] == [publication_identity]
    assert items[0]["editorial_revision_identity"] == "editorial_revision:ticket24-entry-001.r1"
    assert items[0]["decision_query"] == (
        "Which Candidate publication contract applies to reviewed bundle entries?"
    )


def test_ordinary_retrieval_preserves_a_long_identity_published_version_during_an_unpublished_successor(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    entry_id = f"ticket24-long-{'a' * 72}"
    candidate_id = asyncio.run(_seed_ready_candidate(client, entry_id=entry_id))
    admin_headers = asyncio.run(_headers(client, username="ticket24-long-identity-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-long-identity-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        )
    )
    publication_identity = published["published"][0]["publication_identity"]

    async def retrieve() -> list[dict]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_UnpublishedSuccessorRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items

    items = asyncio.run(retrieve())

    assert [item["publication_identity"] for item in items] == [publication_identity]
    assert items[0]["editorial_revision_identity"] == f"editorial_revision:{entry_id}.r1"


def test_ordinary_retrieval_excludes_a_historical_version_when_its_source_is_no_longer_usable(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-source-loss-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-source-loss-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )

    async def retrieve() -> tuple[list[dict], list[dict[str, str]]]:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            result = await AuthorizedRetrievalCandidatePool(
                session,
                settings=get_settings(),
                editorial_authority=_SourceLostUnpublishedSuccessorRetrievalAuthority(),
            ).retrieve("Which Candidate publication contract applies?", top_k=5)
        return result.items, result.candidate_exclusions

    items, exclusions = asyncio.run(retrieve())

    assert items == []
    assert {item["reason"] for item in exclusions} == {"source_unavailable"}


def test_publication_enforces_the_shared_published_source_limit_only_for_new_entries(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    admin_headers = asyncio.run(_headers(client, username="ticket24-capacity-admin", role="admin"))
    original_candidate = asyncio.run(_seed_ready_candidate(client, seed="capacity-001"))
    _inspect_and_accept(client, candidate_id=original_candidate, headers=admin_headers)
    _, _, original_eligibility_url = _candidate_urls(original_candidate)
    original_eligibility = _data(client.get(original_eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-capacity-original",
                "selected_items": [_publish_selection(original_candidate, original_eligibility)],
            },
        ).status_code
        == 200
    )

    async def fill_published_capacity() -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id=f"ticket24-capacity-{index}",
                        filename=f"ticket24-capacity-{index}.reviewed",
                        file_type="reviewed_release_bundle",
                        file_size=1,
                        status="ready",
                        published_generation=1,
                    )
                    for index in range(MAX_PUBLISHED_SOURCES - 1)
                ]
            )
            await session.commit()

    asyncio.run(fill_published_capacity())

    replacement_candidate = asyncio.run(
        _seed_ready_candidate(
            client,
            seed="capacity-002",
            entry_id="ticket24-entry-001",
            generation=2,
            chunk_content_suffix=" Replacement remains eligible at capacity.",
        )
    )
    new_candidate = asyncio.run(_seed_ready_candidate(client, seed="capacity-003", entry_id="ticket24-entry-003"))
    _inspect_and_accept(client, candidate_id=replacement_candidate, headers=admin_headers)
    _inspect_and_accept(client, candidate_id=new_candidate, headers=admin_headers)
    _, _, replacement_eligibility_url = _candidate_urls(replacement_candidate)
    _, _, new_eligibility_url = _candidate_urls(new_candidate)
    replacement_eligibility = _data(client.get(replacement_eligibility_url, headers=admin_headers))
    new_eligibility = _data(client.get(new_eligibility_url, headers=admin_headers))
    batch = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches",
        headers=admin_headers,
        json={
            "confirmation_id": "ticket24-capacity-batch",
            "selected_items": [
                _publish_selection(replacement_candidate, replacement_eligibility),
                _publish_selection(new_candidate, new_eligibility),
            ],
        },
    )

    assert batch.status_code == 200
    result = _data(batch)
    assert [item["candidate_id"] for item in result["published"]] == [replacement_candidate]
    assert result["failed"] == [
        {
            "candidate_id": new_candidate,
            "effect": "create",
            "reason": "PUBLISHED_SOURCE_LIMIT_REACHED",
        }
    ]


def test_legacy_documents_publication_endpoint_rejects_a_canonical_runtime_projection(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-legacy-route-admin", role="admin"))
    _inspect_and_accept(client, candidate_id=candidate_id, headers=admin_headers)
    _, _, eligibility_url = _candidate_urls(candidate_id)
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    assert (
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-legacy-route-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        ).status_code
        == 200
    )

    bypass = client.post(
        "/api/v1/documents/runtime-document:ticket24-entry-001/publish",
        headers=admin_headers,
    )

    assert bypass.status_code == 410
    assert bypass.json()["code"] == "LEGACY_PUBLICATION_BYPASS_REJECTED"


def test_delivery_acceptance_persists_an_exact_candidate_publication_binding(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.api.v1 import reviewed_bundles as reviewed_bundles_api

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    candidate_id = asyncio.run(_seed_ready_candidate(client))
    admin_headers = asyncio.run(_headers(client, username="ticket24-acceptance-record-admin", role="admin"))
    inspection_url, acceptance_url, eligibility_url = _candidate_urls(candidate_id)
    inspection = _data(client.post(inspection_url, headers=admin_headers, json={}))
    acceptance = _data(client.post(acceptance_url, headers=admin_headers, json={}))
    eligibility = _data(client.get(eligibility_url, headers=admin_headers))
    published = _data(
        client.post(
            "/api/v1/reviewed-release-bundles/publication-batches",
            headers=admin_headers,
            json={
                "confirmation_id": "ticket24-delivery-acceptance-publication",
                "selected_items": [_publish_selection(candidate_id, eligibility)],
            },
        )
    )
    publication_identity = published["published"][0]["publication_identity"]
    binding = {
        "candidate_identity": candidate_id,
        "inspection_record_identity": inspection["inspection"]["record_identity"],
        "acceptance_record_identity": acceptance["record_identity"],
        "published_knowledge_version_identity": publication_identity,
        "entry_identity": "entry:ticket24-entry-001",
        "configuration_identity": inspection["candidate"]["configuration_identity"],
        "bundle_sha256": inspection["candidate"]["bundle_sha256"],
        "frozen_input_sha256": inspection["candidate"]["frozen_input_sha256"],
    }
    exact_dependencies = list(binding.values())[:6]
    payload = {
        "stage": "local_development",
        "affected_scope": {
            "entry_identities": [binding["entry_identity"], publication_identity],
            "collection_identities": [],
            "product_path_identities": [],
            "configuration_identities": [binding["configuration_identity"]],
            "protected_capability_identities": [],
            "public_claim_identities": [],
            "deployment_identity": "deployment:ticket24-candidate-publication",
            "expected_blocking_scope": "entry_version",
            "blocking_scope_identity": publication_identity,
        },
        "content_identities": [
            binding["candidate_identity"],
            binding["inspection_record_identity"],
            binding["acceptance_record_identity"],
            binding["published_knowledge_version_identity"],
            binding["entry_identity"],
        ],
        "product_identities": [binding["configuration_identity"]],
        "conditions": {"audience": "ticket24"},
        "assumptions": ["published candidate remains immutable"],
        "checks": [
            {
                "check_id": "check:impact-declaration",
                "result": "passed",
                "evidence_links": ["evidence://ticket24/impact"],
            },
            {
                "check_id": "check:bundle-secret-scan",
                "result": "passed",
                "evidence_links": ["evidence://ticket24/secret-scan"],
            },
            {
                "check_id": "check:entry-supported-query",
                "result": "passed",
                "evidence_links": ["evidence://ticket24/supported"],
                "identity_dependencies": exact_dependencies,
            },
            {
                "check_id": "check:entry-boundary-query",
                "result": "passed",
                "evidence_links": ["evidence://ticket24/boundary"],
                "identity_dependencies": exact_dependencies,
            },
            {
                "check_id": "check:evidence-citation-identity",
                "result": "passed",
                "evidence_links": ["evidence://ticket24/citation"],
                "identity_dependencies": exact_dependencies,
            },
            {
                "check_id": "check:configuration-impact",
                "result": "passed",
                "evidence_links": ["evidence://ticket24/configuration"],
                "identity_dependencies": [binding["configuration_identity"]],
            },
        ],
        "candidate_publication_binding": binding,
        "known_limits": ["withdrawal remains a separate Ticket 25 workflow"],
        "risks": ["replacement requires a fresh exact binding"],
        "evidence_links": ["evidence://ticket24"],
        "reacceptance_triggers": ["candidate generation changes"],
    }

    created = client.post("/api/v1/acceptance/records", headers=admin_headers, json=payload)

    assert created.status_code == 200, created.json()
    assert created.json()["data"]["candidate_publication_binding"] == binding

    bypass_payload = {
        **payload,
        "content_identities": [
            binding["published_knowledge_version_identity"],
            binding["entry_identity"],
        ],
        "checks": [
            (
                {
                    **check,
                    "identity_dependencies": [
                        binding["published_knowledge_version_identity"],
                        binding["entry_identity"],
                        binding["configuration_identity"],
                    ],
                }
                if check["check_id"]
                in {
                    "check:entry-supported-query",
                    "check:entry-boundary-query",
                    "check:evidence-citation-identity",
                }
                else check
            )
            for check in payload["checks"]
        ],
    }
    bypass_payload.pop("candidate_publication_binding")

    bypass = client.post("/api/v1/acceptance/records", headers=admin_headers, json=bypass_payload)

    assert bypass.status_code == 409
    assert bypass.json()["code"] == "ACCEPTANCE_CANDIDATE_PUBLICATION_BINDING_INVALID"
