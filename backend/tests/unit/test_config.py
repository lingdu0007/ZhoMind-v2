import asyncio

from app.common.config import Settings, get_settings
from app.common.responses import ok_response
from app.extensions.langchain_chat_providers import OpenAICompatibleChatProvider
from app.extensions.registry import ExtensionRegistry, get_extension_registry, get_task_backend
from app.infra.milvus import get_milvus_client, get_milvus_provider
from app.infra.minio import get_minio_client, get_minio_provider
from app.infra.redis import RedisProvider, get_redis_client, get_redis_provider
from app.tasks.interfaces import InMemoryTaskBackend, create_inmemory_task_backend


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.rag_graph_alias == "default_v1"
    assert settings.rag_default_llm_provider == "chat-default-llm"


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




def test_settings_rag_fields_from_env_aliases() -> None:
    settings = Settings(
        RAG_GRAPH_ALIAS="experimental_graph",
        RAG_ENABLE_TOOLS=True,
        RAG_TOOL_MAX_CALLS=5,
        RAG_TOOL_MAX_PARALLEL=4,
        RAG_TOOL_TIMEOUT_MS=12000,
        RAG_DEFAULT_LLM_PROVIDER="provider-x",
    )
    assert settings.rag_graph_alias == "experimental_graph"
    assert settings.rag_enable_tools is True
    assert settings.rag_tool_max_calls == 5
    assert settings.rag_tool_max_parallel == 4
    assert settings.rag_tool_timeout_ms == 12000
    assert settings.rag_default_llm_provider == "provider-x"


def test_document_pipeline_settings_aliases() -> None:
    settings = Settings()
    assert settings.document_allowed_extensions_raw == "txt,md,pdf"
    assert settings.doc_worker_enabled is True
    assert settings.doc_worker_max_concurrency == 1
    assert settings.document_allowed_extensions == ["txt", "md", "pdf"]


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


def test_settings_provider_fields_from_env_aliases() -> None:
    settings = Settings(
        ARK_API_KEY="ark-key",
        MODEL="ark-model",
        GRADE_MODEL="ark-grade",
        FAST_MODEL="ark-fast",
        BASE_URL="https://ark.example.com/api/v3",
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=1536,
        RERANK_MODEL="rerank-model",
        RERANK_BINDING_HOST="https://rerank.example.com/v1/rerank",
        RERANK_API_KEY="rerank-key",
        BM25_STATE_PATH="/tmp/bm25_state.json",
        RAG_DISABLE_GATE=True,
    )

    assert settings.ark_api_key == "ark-key"
    assert settings.llm_model == "ark-model"
    assert settings.grade_model == "ark-grade"
    assert settings.fast_model == "ark-fast"
    assert settings.llm_base_url == "https://ark.example.com/api/v3"
    assert settings.embedding_api_key == "emb-key"
    assert settings.embedding_base_url == "https://emb.example.com/v1"
    assert settings.embedding_model == "emb-model"
    assert settings.dense_embedding_dim == 1536
    assert settings.rerank_model == "rerank-model"
    assert settings.rerank_binding_host == "https://rerank.example.com/v1/rerank"
    assert settings.rerank_api_key == "rerank-key"
    assert settings.bm25_state_path == "/tmp/bm25_state.json"
    assert settings.rag_disable_gate is True


def test_registry_registers_ark_llm_from_env(monkeypatch) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    monkeypatch.setenv("MODEL", "ark-model")
    monkeypatch.setenv("BASE_URL", "https://ark.example.com/api/v3")

    registry = get_extension_registry()

    provider = registry.get_llm("chat-default-llm")
    assert isinstance(provider, OpenAICompatibleChatProvider)
    get_settings.cache_clear()
    get_extension_registry.cache_clear()


def test_llm_provider_routing_settings(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "openai,anthropic")

    settings = get_settings()

    assert settings.rag_primary_llm_provider == "ark"
    assert settings.rag_llm_fallback_providers == ["openai", "anthropic"]
    get_settings.cache_clear()
