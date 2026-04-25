from __future__ import annotations

import asyncio

from app.common.config import Settings, get_settings
from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
from app.extensions.langchain_embedding_providers import OpenAIEmbeddingProvider
from app.extensions.registry import get_extension_registry
from app.model.document import DocumentChunk
from app.rag.dense_contract import build_embedding_contract_fingerprint, build_milvus_collection_name


class _FakeEmbeddingProvider:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self.vectors


class _FakeDocumentIndex:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str, int]] = []
        self.upsert_calls: list[tuple[str, list[dict[str, object]]]] = []
        self.delete_calls: list[tuple[str, str, int]] = []

    async def ensure_collection(self, *, collection_name: str, dimension: int) -> None:
        self.ensure_calls.append((collection_name, dimension))

    async def upsert_generation(self, *, collection_name: str, rows: list[dict[str, object]]) -> None:
        self.upsert_calls.append((collection_name, rows))

    async def delete_generation(self, *, collection_name: str, document_id: str, generation: int) -> None:
        self.delete_calls.append((collection_name, document_id, generation))


def _dense_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "EMBEDDING_API_KEY": "emb-key",
        "EMBEDDING_BASE_URL": "https://emb.example.com/v1",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "DENSE_EMBEDDING_DIM": 3,
        "MILVUS_URI": "http://milvus.example.com:19530",
    }
    values.update(overrides)
    return Settings(**values)


def test_dense_index_service_embeds_chunks_and_upserts_generation_rows() -> None:
    async def _run() -> None:
        settings = _dense_settings()
        embedding_provider = _FakeEmbeddingProvider(vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        document_index = _FakeDocumentIndex()
        service = DenseIndexService(
            settings=settings,
            embedding_provider=embedding_provider,
            document_index=document_index,
        )
        fingerprint = build_embedding_contract_fingerprint(settings)
        expected_collection = build_milvus_collection_name(fingerprint)
        chunks = [
            DocumentChunk(document_id="doc-1", generation=2, chunk_index=0, content="alpha", chunk_metadata={}),
            DocumentChunk(document_id="doc-1", generation=2, chunk_index=1, content="beta", chunk_metadata={}),
        ]

        result = await service.index_candidate_generation(document_id="doc-1", generation=2, chunks=chunks)

        assert result == DenseIndexResult(active=True, fingerprint=fingerprint)
        assert embedding_provider.calls == [["alpha", "beta"]]
        assert document_index.ensure_calls == [(expected_collection, 3)]
        assert len(document_index.upsert_calls) == 1
        collection_name, rows = document_index.upsert_calls[0]
        assert collection_name == expected_collection
        assert rows == [
            {
                "document_id": "doc-1",
                "generation": 2,
                "chunk_index": 0,
                "content_sha256": chunks[0].content_sha256,
                "embedding_contract_fingerprint": fingerprint,
                "vector": [0.1, 0.2, 0.3],
            },
            {
                "document_id": "doc-1",
                "generation": 2,
                "chunk_index": 1,
                "content_sha256": chunks[1].content_sha256,
                "embedding_contract_fingerprint": fingerprint,
                "vector": [0.4, 0.5, 0.6],
            },
        ]

    asyncio.run(_run())


def test_dense_index_service_skips_indexing_when_contract_is_inactive() -> None:
    async def _run() -> None:
        settings = _dense_settings(EMBEDDING_API_KEY="")
        embedding_provider = _FakeEmbeddingProvider(vectors=[[0.1, 0.2, 0.3]])
        document_index = _FakeDocumentIndex()
        service = DenseIndexService(
            settings=settings,
            embedding_provider=embedding_provider,
            document_index=document_index,
        )
        chunks = [
            DocumentChunk(document_id="doc-1", generation=2, chunk_index=0, content="alpha", chunk_metadata={}),
        ]

        result = await service.index_candidate_generation(document_id="doc-1", generation=2, chunks=chunks)

        assert result == DenseIndexResult(active=False, fingerprint=None)
        assert embedding_provider.calls == []
        assert document_index.ensure_calls == []
        assert document_index.upsert_calls == []

    asyncio.run(_run())


def test_dense_index_service_deletes_candidate_vectors_for_generation() -> None:
    async def _run() -> None:
        settings = _dense_settings()
        document_index = _FakeDocumentIndex()
        service = DenseIndexService(
            settings=settings,
            embedding_provider=_FakeEmbeddingProvider(vectors=[]),
            document_index=document_index,
        )
        fingerprint = build_embedding_contract_fingerprint(settings)

        await service.delete_candidate_generation(document_id="doc-1", generation=2)

        assert document_index.delete_calls == [
            (build_milvus_collection_name(fingerprint), "doc-1", 2),
        ]

    asyncio.run(_run())


def test_openai_embedding_provider_disables_tiktoken(monkeypatch) -> None:
    constructed: dict[str, object] = {}

    class _FakeEmbeddings:
        def __init__(self, **kwargs) -> None:
            constructed.update(kwargs)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.7, 0.8] for _ in texts]

    monkeypatch.setattr("app.extensions.langchain_embedding_providers.OpenAIEmbeddings", _FakeEmbeddings)

    async def _run() -> None:
        provider = OpenAIEmbeddingProvider(
            api_key="emb-key",
            base_url="https://emb.example.com/v1",
            model="text-embedding-3-large",
        )

        vectors = await provider.embed(["alpha", "beta"])

        assert vectors == [[0.7, 0.8], [0.7, 0.8]]
        assert constructed["api_key"] == "emb-key"
        assert constructed["base_url"] == "https://emb.example.com/v1"
        assert constructed["model"] == "text-embedding-3-large"
        assert constructed["tiktoken_enabled"] is False

    asyncio.run(_run())


def test_registry_registers_default_embedding_provider_only_when_embedding_config_is_usable(monkeypatch) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("EMBEDDING_API_KEY", "emb-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://emb.example.com/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("DENSE_EMBEDDING_DIM", "1536")
    monkeypatch.setenv("MILVUS_URI", "http://milvus.example.com:19530")

    registry = get_extension_registry()

    assert registry.get_embedding("embedding-default") is not None

    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    registry = get_extension_registry()

    assert registry.get_embedding("embedding-default") is None
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
