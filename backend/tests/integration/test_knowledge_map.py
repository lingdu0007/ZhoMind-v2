import asyncio
import json
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.document import Document, DocumentChunk
from tests.support.auth import create_authenticated_test_token


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


def _metadata(*, entry_id: str, review_date: str = "2026-08-12", review_status: str = "approved") -> dict:
    return {
        "strategy": "agent",
        "entry_id": entry_id,
        "entry_title": "Prefer deterministic workflows",
        "domain": "workflow-vs-agent",
        "review_status": review_status,
        "review_date": review_date,
        "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
        "sources": [
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


async def _seed(session_factory) -> None:
    async with session_factory() as session:
        documents = [
            Document(
                id="internal-published-document",
                filename="private-published-entry.md",
                file_type="md",
                file_size=10,
                status="ready",
                chunk_strategy="agent",
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
        published_metadata = _metadata(entry_id="pae-workflow-001")
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


def test_knowledge_map_projects_only_fresh_published_agent_entries() -> None:
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
                    "domain": "workflow-vs-agent",
                    "label": "Workflow 与 Agent",
                    "entries": [
                        {
                            "entry_id": "pae-workflow-001",
                            "title": "Prefer deterministic workflows",
                            "approved_summary": "已知路径默认使用 deterministic workflow；动态路径使用 bounded Agent。",
                            "review_date": "2026-08-12",
                            "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
                            "publication_version": "v2",
                            "public_source_count": 2,
                            "suggested_query": "什么时候使用 deterministic workflow？",
                            "sources": [
                                {
                                    "title": "Building effective agents",
                                    "authority": "Anthropic",
                                    "url": "https://www.anthropic.com/engineering/building-effective-agents",
                                    "version": "2024-12-19",
                                },
                                {
                                    "title": "Agents",
                                    "authority": "OpenAI Agents SDK",
                                    "url": "https://openai.github.io/openai-agents-python/agents/",
                                    "version": "verified 2026-08-12",
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
