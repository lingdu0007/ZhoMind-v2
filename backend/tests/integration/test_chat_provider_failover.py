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
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return "fallback-ok"


def test_chat_fallback_to_secondary_provider(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    monkeypatch.setenv("RAG_DISABLE_GATE", "true")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "openai")
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

    registry.register_llm("ark", _RetryableFailProvider())
    registry.register_llm("openai", _OkProvider())

    try:
        with TestClient(app) as client:
            reg = client.post(
                "/api/v1/auth/register",
                json={"username": "fallback-u", "password": "secret-123", "role": "user"},
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
            runtime = data["rag_trace"]["runtime"]
            assert data["answer"] == "fallback-ok"
            assert runtime["final_provider"] == "openai"
            assert runtime["fallback_hops"] == 1
            assert runtime["provider_attempts"][0]["provider"] == "ark"
            assert runtime["provider_attempts"][-1]["provider"] == "openai"
    finally:
        if prev_ark is None:
            registry.llm_providers.pop("ark", None)
        else:
            registry.register_llm("ark", prev_ark)

        if prev_openai is None:
            registry.llm_providers.pop("openai", None)
        else:
            registry.register_llm("openai", prev_openai)

        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())
