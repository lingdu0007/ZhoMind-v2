import asyncio

from app.common.config import Settings
from app.common.responses import ok_response
from app.extensions.registry import ExtensionRegistry, get_extension_registry, get_task_backend
from app.infra.milvus import get_milvus_client, get_milvus_provider
from app.infra.minio import get_minio_client, get_minio_provider
from app.infra.redis import RedisProvider, get_redis_client, get_redis_provider
from app.tasks.interfaces import InMemoryTaskBackend, create_inmemory_task_backend


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.api_v1_prefix == "/api/v1"


def test_ok_response_shape() -> None:
    payload = ok_response(data={"status": "up"}, request_id="rid-1")
    assert payload == {
        "code": "OK",
        "message": "success",
        "data": {"status": "up"},
        "request_id": "rid-1",
    }


def test_redis_provider_is_lazy() -> None:
    provider = RedisProvider("redis://localhost:6379/0")
    assert provider._client is None


def test_settings_database_url_from_env_alias() -> None:
    settings = Settings(DATABASE_URL="postgresql+asyncpg://postgres:postgres@postgres:5432/zhomind_app")
    assert settings.database_url == "postgresql+asyncpg://postgres:postgres@postgres:5432/zhomind_app"


def test_infra_dependency_getters_delegate_to_cached_providers(monkeypatch) -> None:
    get_redis_provider.cache_clear()
    get_milvus_provider.cache_clear()
    get_minio_provider.cache_clear()

    redis_provider = get_redis_provider()
    milvus_provider = get_milvus_provider()
    minio_provider = get_minio_provider()

    redis_marker = object()
    milvus_marker = object()
    minio_marker = object()

    monkeypatch.setattr(redis_provider, "get_client", lambda: redis_marker)
    monkeypatch.setattr(milvus_provider, "get_client", lambda: milvus_marker)
    monkeypatch.setattr(minio_provider, "get_client", lambda: minio_marker)

    assert get_redis_client() is redis_marker
    assert get_milvus_client() is milvus_marker
    assert get_minio_client() is minio_marker


def test_extension_registry_register_and_get() -> None:
    registry = ExtensionRegistry()

    class DummyLlm:
        async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
            return prompt

    class DummyEmbedding:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] for _ in texts]

    class DummyReranker:
        async def rerank(self, query: str, items: list[dict]) -> list[dict]:
            return items

    class DummyRetriever:
        async def retrieve(self, query: str, top_k: int) -> list[dict]:
            return []

    class DummyJudge:
        async def judge(self, query: str, context: list[dict]) -> bool:
            return True

    backend = create_inmemory_task_backend()

    registry.register_llm("dummy", DummyLlm())
    registry.register_embedding("dummy", DummyEmbedding())
    registry.register_rerank("dummy", DummyReranker())
    registry.register_retriever("dummy", DummyRetriever())
    registry.register_judge("dummy", DummyJudge())
    registry.register_task_backend("dummy", backend)

    assert registry.get_llm("dummy") is not None
    assert registry.get_embedding("dummy") is not None
    assert registry.get_rerank("dummy") is not None
    assert registry.get_retriever("dummy") is not None
    assert registry.get_judge("dummy") is not None
    assert registry.get_task_backend("dummy") is backend


def test_extension_registry_default_task_backend() -> None:
    get_extension_registry.cache_clear()
    registry = get_extension_registry()

    default_backend = registry.get_task_backend("inmemory")
    assert default_backend is not None
    assert get_task_backend("inmemory") is default_backend

    fallback_backend = get_task_backend("unknown")
    assert isinstance(fallback_backend, InMemoryTaskBackend)


def test_inmemory_task_backend_lifecycle() -> None:
    backend = create_inmemory_task_backend()

    task_id = asyncio.run(backend.enqueue("build_document", {"document_id": "doc-1"}))
    status = asyncio.run(backend.get_status(task_id))
    assert status["task_id"] == task_id
    assert status["status"] == "queued"

    asyncio.run(backend.cancel(task_id))
    canceled = asyncio.run(backend.get_status(task_id))
    assert canceled["status"] == "canceled"
