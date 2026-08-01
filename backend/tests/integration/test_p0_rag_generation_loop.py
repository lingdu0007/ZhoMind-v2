import asyncio
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.extensions.registry import get_extension_registry
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.document import Document, DocumentChunk


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0


class _RecordingProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        return "已基于发布资料生成回答。"


class _UnavailableProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        raise TimeoutError("upstream timeout")


def _stream_event_data(payload: str, event: str) -> str | None:
    for block in payload.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0] == f"event: {event}" and lines[1].startswith("data: "):
            return lines[1][6:]
    return None


def test_knowledge_user_chat_cites_only_published_document_in_normal_and_streaming_contracts(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()
    provider = _RecordingProvider()

    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="published-doc",
                        filename="发布运行手册.md",
                        file_type="md",
                        file_size=100,
                        status="ready",
                        published_generation=1,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    DocumentChunk(
                        id="published-chunk",
                        document_id="published-doc",
                        generation=1,
                        chunk_index=0,
                        content="发布循环验收锚点：仅已发布版本可以回答。",
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={},
                    ),
                    Document(
                        id="candidate-doc",
                        filename="候选草稿.md",
                        file_type="md",
                        file_size=100,
                        status="processing",
                        published_generation=0,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    DocumentChunk(
                        id="candidate-chunk",
                        document_id="candidate-doc",
                        generation=1,
                        chunk_index=0,
                        content="发布循环验收锚点：候选草稿不得作为回答证据。",
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={},
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    registry = get_extension_registry()
    registry.register_llm("ark", provider)

    try:
        with TestClient(app) as client:
            registration = client.post(
                "/api/v1/auth/register",
                json={"username": "knowledge-user", "password": "secret-123", "role": "user"},
            )
            headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}

            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "发布循环验收锚点", "session_id": "published-loop"},
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["answer"] == "已基于发布资料生成回答。"
            assert data["message"]["evidence_summary"] == {
                "coverage": "sufficient",
                "source_count": 1,
                "sources": [
                    {
                        "source_id": "published-chunk",
                        "metadata": {"title": "发布运行手册.md", "publication_version": "v1"},
                        "excerpt": "发布循环验收锚点：仅已发布版本可以回答。",
                    }
                ],
            }
            assert "发布循环验收锚点" in provider.prompts[0]
            assert "仅已发布版本可以回答" in provider.prompts[0]
            assert "候选草稿不得作为回答证据" not in provider.prompts[0]

            stream_response = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "发布循环验收锚点", "session_id": "published-loop"},
            )

            assert stream_response.status_code == 200
            assert "已基于发布资料生成回答。" in stream_response.text
            assert _stream_event_data(stream_response.text, "evidence_summary") is not None
            assert "event: done" in stream_response.text
    finally:
        app.dependency_overrides.clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_knowledge_user_with_no_published_evidence_is_not_sent_to_generation(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()
    provider = _RecordingProvider()

    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    registry = get_extension_registry()
    registry.register_llm("ark", provider)

    try:
        with TestClient(app) as client:
            registration = client.post(
                "/api/v1/auth/register",
                json={"username": "no-evidence-user", "password": "secret-123", "role": "user"},
            )
            headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}

            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "没有发布资料能回答的问题", "session_id": "no-evidence"},
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert "未检索到足够相关的知识片段" in data["answer"]
            assert data["message"]["evidence_summary"] == {
                "coverage": "insufficient",
                "source_count": 0,
                "sources": [],
            }
            assert provider.prompts == []
    finally:
        app.dependency_overrides.clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_narrow_social_reply_is_explicitly_labeled_as_non_knowledge_base(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()
    provider = _RecordingProvider()

    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    registry = get_extension_registry()
    registry.register_llm("ark", provider)

    try:
        with TestClient(app) as client:
            registration = client.post(
                "/api/v1/auth/register",
                json={"username": "social-user", "password": "secret-123", "role": "user"},
            )
            headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}

            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "你好", "session_id": "smalltalk"},
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["answer"].startswith("【非知识库回复】")
            assert data["message"]["evidence_summary"] == {
                "coverage": "unavailable",
                "source_count": 0,
                "sources": [],
            }
            assert provider.prompts == []
    finally:
        app.dependency_overrides.clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_generation_outage_fails_closed_with_published_sources_in_normal_and_streaming_contracts(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()
    primary = _UnavailableProvider()
    secondary = _RecordingProvider()

    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="outage-doc",
                        filename="故障演练手册.md",
                        file_type="md",
                        file_size=100,
                        status="ready",
                        published_generation=1,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    DocumentChunk(
                        id="outage-chunk",
                        document_id="outage-doc",
                        generation=1,
                        chunk_index=0,
                        content="故障演练锚点：provider 故障时保留已发布来源。",
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={},
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    registry = get_extension_registry()
    registry.register_llm("ark", primary)
    registry.register_llm("openai", secondary)

    try:
        with TestClient(app) as client:
            registration = client.post(
                "/api/v1/auth/register",
                json={"username": "outage-user", "password": "secret-123", "role": "user"},
            )
            headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}

            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "故障演练锚点", "session_id": "provider-outage"},
            )

            assert response.status_code == 200
            data = response.json()["data"]
            assert data["answer"].startswith("【生成不可用】")
            assert data["message"]["evidence_summary"] == {
                "coverage": "sufficient",
                "source_count": 1,
                "sources": [
                    {
                        "source_id": "outage-chunk",
                        "metadata": {"title": "故障演练手册.md", "publication_version": "v1"},
                        "excerpt": "故障演练锚点：provider 故障时保留已发布来源。",
                    }
                ],
            }
            assert "retrieval_diagnostics" not in data
            assert len(primary.prompts) == 1
            assert secondary.prompts == []

            stream_response = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "故障演练锚点", "session_id": "provider-outage"},
            )

            assert stream_response.status_code == 200
            assert "【生成不可用】" in stream_response.text
            assert _stream_event_data(stream_response.text, "evidence_summary") is not None
            assert "event: done" in stream_response.text
            assert len(primary.prompts) == 2
            assert secondary.prompts == []
    finally:
        app.dependency_overrides.clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())
