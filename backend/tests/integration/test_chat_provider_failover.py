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


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0


class _RetryableFailProvider:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise TimeoutError("upstream timeout")


class _OkProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.calls += 1
        return "fallback-ok"


class _PublishedEvidenceRetriever:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        return [
            {
                "chunk_id": "published-chunk-1",
                "document_id": "published-document-1",
                "generation": 1,
                "content_preview": "已发布资料中的可引用事实。",
                "metadata": {"title": "已发布资料", "published_generation": 1},
                "retrieval_source": "lexical",
            }
        ]


def test_chat_does_not_fallback_when_the_active_provider_fails(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "openai")
    monkeypatch.setenv("ADMIN_INVITE_CODE", "provider-test-admin-code")
    get_settings.cache_clear()

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
    prev_ark = registry.get_llm("ark")
    prev_openai = registry.get_llm("openai")
    prev_retriever = registry.get_retriever("chat-default-retriever")

    registry.register_llm("ark", _RetryableFailProvider())
    secondary = _OkProvider()
    registry.register_llm("openai", secondary)
    registry.register_retriever("chat-default-retriever", _PublishedEvidenceRetriever())

    try:
        with TestClient(app) as client:
            reg = client.post(
                "/api/v1/auth/register",
                json={
                    "username": "fallback-admin",
                    "password": "secret-123",
                    "role": "admin",
                    "admin_code": "provider-test-admin-code",
                },
            )
            token = reg.json()["data"]["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            resp = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "请给出答案", "session_id": "fallback_s1"},
            )

            assert resp.status_code == 200
            data = resp.json()["data"]
            diagnostics = data["retrieval_diagnostics"]
            assert data["answer"] == "生成服务暂不可用，请稍后重试。"
            assert data["message"]["evidence_summary"] == {
                "coverage": "sufficient",
                "source_count": 1,
                "sources": [
                    {
                        "source_id": "published-chunk-1",
                        "metadata": {"title": "已发布资料"},
                        "excerpt": "已发布资料中的可引用事实。",
                    }
                ],
            }
            assert secondary.calls == 0
            assert diagnostics["fallback"] == {"state": "not_used", "hops": 0, "final_provider": "ark"}
            assert diagnostics["provider_errors"] == [{"stage": "generate", "code": "TimeoutError", "type": None}]
            assert "rag_trace" not in data
    finally:
        if prev_ark is None:
            registry.llm_providers.pop("ark", None)
        else:
            registry.register_llm("ark", prev_ark)

        if prev_openai is None:
            registry.llm_providers.pop("openai", None)
        else:
            registry.register_llm("openai", prev_openai)

        if prev_retriever is None:
            registry.retrievers.pop("chat-default-retriever", None)
        else:
            registry.register_retriever("chat-default-retriever", prev_retriever)

        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())
