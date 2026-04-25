from __future__ import annotations

import asyncio

import pytest

from app.common.exceptions import AppError
from app.common.config import Settings, get_settings
from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
from app.extensions.langchain_embedding_providers import OpenAIEmbeddingProvider
from app.extensions.registry import ExtensionRegistry
from app.extensions.registry import get_extension_registry
from app.infra.milvus_document_index import MilvusDocumentIndex
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
        self.search_calls: list[tuple[str, list[float], int, str, list[str] | None]] = []

    async def ensure_collection(self, *, collection_name: str, dimension: int) -> None:
        self.ensure_calls.append((collection_name, dimension))

    async def upsert_generation(self, *, collection_name: str, rows: list[dict[str, object]]) -> None:
        self.upsert_calls.append((collection_name, rows))

    async def delete_generation(self, *, collection_name: str, document_id: str, generation: int) -> None:
        self.delete_calls.append((collection_name, document_id, generation))

    async def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        filter: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict[str, object]]:
        self.search_calls.append((collection_name, vector, limit, filter, output_fields))
        return []


class _FakeMilvusClient:
    def __init__(self) -> None:
        self.has_collection_result = True
        self.has_collection_calls: list[str] = []
        self.describe_collection_result: dict[str, object] = {
            "fields": [
                {"name": "vector", "params": {"dim": 3}},
            ]
        }
        self.delete_calls: list[tuple[str, object, object, object]] = []
        self.delete_error: Exception | None = None
        self.search_calls: list[tuple[str, list[list[float]], int, str, list[str] | None]] = []
        self.search_result: list[list[dict[str, object]]] = [[{"id": "row-1", "distance": 0.1}]]

    def has_collection(self, collection_name: str) -> bool:
        self.has_collection_calls.append(collection_name)
        return self.has_collection_result

    def create_collection(self, *args, **kwargs) -> None:
        raise AssertionError("unexpected create_collection")

    def describe_collection(self, collection_name: str) -> dict[str, object]:
        return self.describe_collection_result

    def delete(self, collection_name: str, ids=None, timeout=None, filter=None, **kwargs) -> dict[str, int]:
        self.delete_calls.append((collection_name, ids, timeout, filter))
        if self.delete_error is not None:
            raise self.delete_error
        return {"delete_count": 1}

    def search(
        self,
        collection_name: str,
        data=None,
        filter: str = "",
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[dict[str, object]]]:
        assert isinstance(data, list)
        self.search_calls.append((collection_name, data, limit, filter, output_fields))
        return self.search_result


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


def test_dense_index_service_raises_when_dense_mode_is_active_but_provider_is_missing(monkeypatch) -> None:
    async def _run() -> None:
        settings = _dense_settings()
        monkeypatch.setattr("app.documents.dense_index_service.get_extension_registry", lambda: ExtensionRegistry())
        service = DenseIndexService(
            settings=settings,
            embedding_provider=None,
            document_index=_FakeDocumentIndex(),
        )
        chunks = [
            DocumentChunk(document_id="doc-1", generation=2, chunk_index=0, content="alpha", chunk_metadata={}),
        ]

        with pytest.raises(AppError) as exc_info:
            await service.index_candidate_generation(document_id="doc-1", generation=2, chunks=chunks)

        assert exc_info.value.code == "DENSE_INDEX_PROVIDER_UNAVAILABLE"

    asyncio.run(_run())


def test_milvus_document_index_delete_generation_is_benign_when_collection_is_missing() -> None:
    async def _run() -> None:
        client = _FakeMilvusClient()
        client.has_collection_result = False
        index = MilvusDocumentIndex(client=client)

        await index.delete_generation(collection_name="document_chunks_fp", document_id="doc-1", generation=2)

        assert client.has_collection_calls == ["document_chunks_fp"]
        assert client.delete_calls == []

    asyncio.run(_run())


def test_milvus_document_index_delete_generation_swallows_missing_collection_race() -> None:
    async def _run() -> None:
        client = _FakeMilvusClient()
        client.delete_error = RuntimeError("collection not found")
        index = MilvusDocumentIndex(client=client)

        await index.delete_generation(collection_name="document_chunks_fp", document_id="doc-1", generation=2)

        assert client.has_collection_calls == ["document_chunks_fp"]
        assert client.delete_calls == [("document_chunks_fp", None, None, 'document_id == "doc-1" and generation == 2')]

    asyncio.run(_run())


def test_milvus_document_index_search_returns_first_result_page() -> None:
    async def _run() -> None:
        client = _FakeMilvusClient()
        index = MilvusDocumentIndex(client=client)

        results = await index.search(
            collection_name="document_chunks_fp",
            vector=[0.1, 0.2, 0.3],
            limit=5,
            filter='document_id == "doc-1"',
            output_fields=["document_id", "generation"],
        )

        assert results == [{"id": "row-1", "distance": 0.1}]
        assert client.search_calls == [
            (
                "document_chunks_fp",
                [[0.1, 0.2, 0.3]],
                5,
                'document_id == "doc-1"',
                ["document_id", "generation"],
            )
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
