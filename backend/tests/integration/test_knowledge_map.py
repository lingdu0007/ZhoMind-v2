import asyncio
import json
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.knowledge_map.service import KnowledgeMapService
from app.main import app
from app.model.base import Base
from app.model.document import Document, DocumentChunk
from tests.support.auth import create_authenticated_test_token

_ENTRY_ID = "pae-workflow-001"
_ENTRY_IDENTITY = "entry:pae-workflow-001"
_REVISION_IDENTITY = "editorial_revision:pae-workflow-001.r1"
_SOURCE_IDENTITY = "source:source-workflow-001"
_SOURCE_RELATIONSHIP = {
    "source_identity": _SOURCE_IDENTITY,
    "availability": "verified_usable",
    "access_scope": "controlled_internal",
}
_APPLICABILITY_CONDITIONS = [
    {
        "condition_id": "condition-known-path",
        "field": "execution_path",
        "operator": "equals",
        "value": "known",
    }
]
_FRESHNESS_TRIGGERS = [
    {
        "trigger_id": "freshness-workflow-source",
        "trigger_type": "source_release",
        "review_within_days": 7,
    }
]


class _InMemoryRedis:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._values[key] = mapping
        return len(mapping)

    async def expire(self, key: str, _ttl: int) -> bool:
        return key in self._values

    async def exists(self, key: str) -> int:
        return int(key in self._values)


def _metadata(
    *,
    entry_id: str,
    review_date: str = "2026-08-12",
    review_status: str = "approved",
    strategy: str = "agent",
    assurance_level: str = "claim_linked",
    sources: list[dict] | None = None,
) -> dict:
    return {
        "strategy": strategy,
        "entry_id": entry_id,
        "entry_title": "Prefer deterministic workflows",
        "domain": "workflow-vs-agent",
        "review_status": review_status,
        "review_date": review_date,
        "assurance_level": assurance_level,
        "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
        "sources": sources
        if sources is not None
        else [
            {
                "title": "Building effective agents",
                "authority": "Anthropic",
                "url": "https://www.anthropic.com/engineering/building-effective-agents",
                "version": "2024-12-19",
                "availability": "verified",
            },
            {
                "title": "Agents",
                "authority": "OpenAI Agents SDK",
                "url": "https://openai.github.io/openai-agents-python/agents/",
                "version": "verified 2026-08-12",
                "availability": "verified",
            },
        ],
    }


def _bound_metadata(metadata: dict) -> dict:
    return {
        **metadata,
        "entry_identity": _ENTRY_IDENTITY,
        "editorial_revision_identity": _REVISION_IDENTITY,
        "source_relationships": [_SOURCE_RELATIONSHIP],
        "applicability_conditions": _APPLICABILITY_CONDITIONS,
        "freshness_triggers": _FRESHNESS_TRIGGERS,
    }


class _CurrentMapAuthority:
    """Current editorial facts for a map projection test fixture."""

    def __init__(self, _session: AsyncSession) -> None:
        pass

    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        del now
        if entry_id != _ENTRY_ID:
            return {
                "entry_id": entry_id,
                "entry_identity": f"entry:{entry_id}",
                "editorial_revision_identity": f"editorial_revision:{entry_id}.r1",
                "lifecycle_state": "published",
                "answer_eligible": False,
                "eligibility_reasons": ["source_unavailable"],
            }
        return {
            "entry_id": _ENTRY_ID,
            "entry_identity": _ENTRY_IDENTITY,
            "editorial_revision_identity": _REVISION_IDENTITY,
            "lifecycle_state": "published",
            "answer_eligible": True,
            "eligibility_reasons": [],
            "entry_title": "Prefer deterministic workflows",
            "coverage_position": "orchestration_retry_human_intervention_and_side_effects",
            "review_date": "2026-08-12",
            "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
            "assurance_level": "claim_linked",
            "applicability_conditions": _APPLICABILITY_CONDITIONS,
            "freshness_triggers": _FRESHNESS_TRIGGERS,
            "section_source_relationships": {
                "decision-question": [_SOURCE_RELATIONSHIP],
                "recommendation": [_SOURCE_RELATIONSHIP],
            },
            "decision_query": "什么时候使用 deterministic workflow？",
            "source_definitions": [
                {
                    "source_identity": _SOURCE_IDENTITY,
                    "title": "Reviewed controlled workflow runbook",
                    "authority": "ZhoMind architecture group",
                    "version": "2026-09-07",
                    "access_scope": "controlled_internal",
                    "controlled_locator": "controlled://knowledge/reviewed-workflow-runbook",
                }
            ],
        }


class _SourceLostMapAuthority(_CurrentMapAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        if entry_id == _ENTRY_ID:
            authority.update(answer_eligible=False, eligibility_reasons=["source_unavailable"])
        return authority


class _OldButEligibleMapAuthority(_CurrentMapAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        if entry_id == _ENTRY_ID:
            authority["review_date"] = "2020-01-01"
        return authority


class _UnsafePublicSourceMapAuthority(_CurrentMapAuthority):
    async def get_retrieval_authority(self, entry_id: str, *, now=None) -> dict:
        authority = await super().get_retrieval_authority(entry_id, now=now)
        if entry_id == _ENTRY_ID:
            authority["source_definitions"] = [
                {
                    "source_identity": _SOURCE_IDENTITY,
                    "title": "Unsafe published source",
                    "authority": "ZhoMind architecture group",
                    "version": "2026-09-07",
                    "access_scope": "public",
                    "public_url": "https://example.com/reference?signed=temporary-access",
                }
            ]
        return authority


async def _seed(session_factory) -> None:
    async with session_factory() as session:
        documents = [
            Document(
                id="internal-published-document",
                filename="private-published-entry.md",
                file_type="md",
                file_size=10,
                status="ready",
                chunk_strategy="section-aware",
                chunk_count=2,
                published_generation=2,
                next_generation=3,
                latest_requested_generation=2,
            ),
            Document(
                id="internal-candidate-document",
                filename="private-candidate-entry.md",
                file_type="md",
                file_size=10,
                status="candidate",
                chunk_strategy="agent",
                published_generation=0,
                candidate_generation=1,
                candidate_chunk_strategy="agent",
                candidate_chunk_count=1,
                next_generation=2,
                latest_requested_generation=1,
            ),
            Document(
                id="internal-stale-document",
                filename="private-stale-entry.md",
                file_type="md",
                file_size=10,
                status="ready",
                chunk_strategy="agent",
                chunk_count=1,
                published_generation=1,
                next_generation=2,
                latest_requested_generation=1,
            ),
            Document(
                id="internal-superseded-document",
                filename="private-superseded-entry.md",
                file_type="md",
                file_size=10,
                status="ready",
                chunk_strategy="agent",
                chunk_count=1,
                published_generation=1,
                next_generation=2,
                latest_requested_generation=1,
            ),
            Document(
                id="internal-unapproved-document",
                filename="private-unapproved-entry.md",
                file_type="md",
                file_size=10,
                status="ready",
                chunk_strategy="agent",
                chunk_count=1,
                published_generation=1,
                next_generation=2,
                latest_requested_generation=1,
            ),
        ]
        session.add_all(documents)
        published_metadata = _bound_metadata(
            _metadata(
                entry_id=_ENTRY_ID,
                strategy="section-aware",
                assurance_level="claim_linked",
                sources=[
                    {
                        "title": "Reviewed controlled workflow runbook",
                        "authority": "ZhoMind architecture group",
                        "controlled_locator": "controlled://knowledge/reviewed-workflow-runbook",
                        "access_scope": "controlled_internal",
                        "version": "2026-09-07",
                        "availability": "verified",
                    }
                ],
            )
        )
        published_metadata.pop("domain")
        session.add_all(
            [
                DocumentChunk(
                    id="internal-question-chunk",
                    document_id="internal-published-document",
                    generation=2,
                    chunk_index=0,
                    content="# Decision Question\n\n什么时候使用 deterministic workflow？",
                    chunk_metadata={**published_metadata, "section_id": "decision-question"},
                ),
                DocumentChunk(
                    id="internal-recommendation-chunk",
                    document_id="internal-published-document",
                    generation=2,
                    chunk_index=1,
                    content="## Recommendation\n\n已知路径默认使用 deterministic workflow；动态路径使用 bounded Agent。",
                    chunk_metadata={**published_metadata, "section_id": "recommendation"},
                ),
                DocumentChunk(
                    id="internal-candidate-chunk",
                    document_id="internal-candidate-document",
                    generation=1,
                    chunk_index=0,
                    content="candidate private text must stay hidden",
                    chunk_metadata={**_metadata(entry_id="pae-candidate-001"), "section_id": "recommendation"},
                ),
                DocumentChunk(
                    id="internal-stale-chunk",
                    document_id="internal-stale-document",
                    generation=1,
                    chunk_index=0,
                    content="stale private text must stay hidden",
                    chunk_metadata={
                        **_metadata(entry_id="pae-stale-001", review_date="2020-01-01"),
                        "section_id": "recommendation",
                    },
                ),
                DocumentChunk(
                    id="internal-superseded-chunk",
                    document_id="internal-superseded-document",
                    generation=1,
                    chunk_index=0,
                    content="superseded private text must stay hidden",
                    chunk_metadata={
                        **published_metadata,
                        "editorial_revision_identity": "editorial_revision:pae-workflow-001.r0",
                        "section_id": "recommendation",
                    },
                ),
                DocumentChunk(
                    id="internal-unapproved-chunk",
                    document_id="internal-unapproved-document",
                    generation=1,
                    chunk_index=0,
                    content="unapproved private text must stay hidden",
                    chunk_metadata={
                        **_metadata(entry_id="pae-unapproved-001", review_status="editorial_review"),
                        "section_id": "recommendation",
                    },
                ),
            ]
        )
        await session.commit()


def test_knowledge_map_projects_current_authority_entries_without_legacy_chunk_domain(monkeypatch) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()

    async def init() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    asyncio.run(init())
    asyncio.run(_seed(session_factory))
    monkeypatch.setattr(
        "app.knowledge_map.service.EditorialAuthorityService",
        _CurrentMapAuthority,
        raising=False,
    )
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.test_auth_session_factory = session_factory
    app.state.test_auth_redis = redis

    try:
        with TestClient(app) as client:
            user_token = asyncio.run(
                create_authenticated_test_token(session_factory, redis, username="map-user", role="user")
            )
            admin_token = asyncio.run(
                create_authenticated_test_token(session_factory, redis, username="map-admin", role="admin")
            )
            assert client.get("/api/v1/knowledge-map").status_code == 401

            response = client.get(
                "/api/v1/knowledge-map",
                headers={"Authorization": f"Bearer {user_token}"},
            )
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["total_entries"] == 1
            assert data["themes"] == [
                {
                    "domain": "orchestration",
                    "label": "Agent Orchestration",
                    "entries": [
                        {
                            "entry_id": "pae-workflow-001",
                            "title": "Prefer deterministic workflows",
                            "approved_summary": "已知路径默认使用 deterministic workflow；动态路径使用 bounded Agent。",
                            "review_date": "2026-08-12",
                            "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
                            "publication_version": "v2",
                            "source_count": 1,
                            "public_source_count": 0,
                            "controlled_source_count": 1,
                            "suggested_query": "什么时候使用 deterministic workflow？",
                            "assurance_level": "claim_linked",
                            "sources": [
                                {
                                    "title": "Reviewed controlled workflow runbook",
                                    "authority": "ZhoMind architecture group",
                                    "url": "controlled://knowledge/reviewed-workflow-runbook",
                                    "version": "2026-09-07",
                                    "access_scope": "controlled_internal",
                                },
                            ],
                        }
                    ],
                }
            ]
            serialized = json.dumps(data)
            assert "internal-" not in serialized
            assert "private-" not in serialized
            assert "candidate private text" not in serialized
            assert "stale private text" not in serialized
            assert "superseded private text" not in serialized

            detail = client.get(
                "/api/v1/knowledge-map/pae-workflow-001",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert detail.status_code == 200
            assert detail.json()["data"] == data["themes"][0]["entries"][0]
            missing = client.get(
                "/api/v1/knowledge-map/pae-candidate-001",
                headers={"Authorization": f"Bearer {user_token}"},
            )
            assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())


def test_knowledge_map_fails_closed_for_current_source_loss_or_unsafe_public_locator() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def init() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def project(authority_type) -> dict:
        async with session_factory() as session:
            return await KnowledgeMapService(
                session,
                editorial_authority=authority_type(session),
            ).list()

    asyncio.run(init())
    asyncio.run(_seed(session_factory))
    try:
        for authority_type in (_SourceLostMapAuthority, _UnsafePublicSourceMapAuthority):
            assert asyncio.run(project(authority_type)) == {"total_entries": 0, "themes": []}
    finally:
        asyncio.run(engine.dispose())


def test_knowledge_map_keeps_currently_eligible_coverage_without_a_global_review_age_cutoff() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def init() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def project() -> dict:
        async with session_factory() as session:
            return await KnowledgeMapService(
                session,
                editorial_authority=_OldButEligibleMapAuthority(session),
            ).list()

    asyncio.run(init())
    asyncio.run(_seed(session_factory))
    try:
        payload = asyncio.run(project())
        assert payload["total_entries"] == 1
        assert payload["themes"][0]["entries"][0]["review_date"] == "2020-01-01"
    finally:
        asyncio.run(engine.dispose())


def test_knowledge_map_excludes_candidate_marked_chunks_even_when_their_authority_binding_matches() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def init() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def project_candidate_marked_publication() -> dict:
        async with session_factory() as session:
            chunk = await session.get(DocumentChunk, "internal-recommendation-chunk")
            assert chunk is not None
            chunk.chunk_metadata = {**chunk.chunk_metadata, "candidate_build": True}
            await session.commit()
            return await KnowledgeMapService(
                session,
                editorial_authority=_CurrentMapAuthority(session),
            ).list()

    asyncio.run(init())
    asyncio.run(_seed(session_factory))
    try:
        assert asyncio.run(project_candidate_marked_publication()) == {
            "total_entries": 0,
            "themes": [],
        }
    finally:
        asyncio.run(engine.dispose())
