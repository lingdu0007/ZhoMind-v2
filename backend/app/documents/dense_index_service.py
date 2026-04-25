from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.common.config import Settings, get_settings
from app.extensions.registry import get_extension_registry
from app.infra.milvus_document_index import MilvusDocumentIndex
from app.model.document import DocumentChunk
from app.rag.dense_contract import DenseEmbeddingContract, build_embedding_contract_fingerprint, build_milvus_collection_name
from app.rag.interfaces import EmbeddingProvider

_DEFAULT_EMBEDDING_PROVIDER = "embedding-default"


class DocumentIndex(Protocol):
    async def ensure_collection(self, *, collection_name: str, dimension: int) -> None: ...

    async def upsert_generation(self, *, collection_name: str, rows: list[dict[str, object]]) -> None: ...

    async def delete_generation(self, *, collection_name: str, document_id: str, generation: int) -> None: ...


@dataclass(frozen=True)
class DenseIndexResult:
    active: bool
    fingerprint: str | None


class DenseIndexService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        document_index: DocumentIndex | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._document_index = document_index

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks: list[DocumentChunk],
    ) -> DenseIndexResult:
        contract = DenseEmbeddingContract.from_settings(self._settings)
        if not contract.active:
            return DenseIndexResult(active=False, fingerprint=None)

        embedding_provider = self._resolve_embedding_provider()
        document_index = self._resolve_document_index()
        if embedding_provider is None or document_index is None:
            return DenseIndexResult(active=False, fingerprint=None)

        fingerprint = build_embedding_contract_fingerprint(self._settings)
        collection_name = build_milvus_collection_name(fingerprint)
        await document_index.ensure_collection(collection_name=collection_name, dimension=contract.dimension)

        vectors = await embedding_provider.embed([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding provider returned mismatched vector count")

        rows = [
            {
                "document_id": document_id,
                "generation": generation,
                "chunk_index": chunk.chunk_index,
                "content_sha256": chunk.content_sha256,
                "embedding_contract_fingerprint": fingerprint,
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        await document_index.upsert_generation(collection_name=collection_name, rows=rows)
        return DenseIndexResult(active=True, fingerprint=fingerprint)

    async def delete_candidate_generation(self, *, document_id: str, generation: int | None) -> None:
        if generation is None:
            return
        contract = DenseEmbeddingContract.from_settings(self._settings)
        if not contract.active:
            return

        document_index = self._resolve_document_index()
        if document_index is None:
            return

        fingerprint = build_embedding_contract_fingerprint(self._settings)
        await document_index.delete_generation(
            collection_name=build_milvus_collection_name(fingerprint),
            document_id=document_id,
            generation=generation,
        )

    def _resolve_embedding_provider(self) -> EmbeddingProvider | None:
        if self._embedding_provider is not None:
            return self._embedding_provider
        self._embedding_provider = get_extension_registry().get_embedding(_DEFAULT_EMBEDDING_PROVIDER)
        return self._embedding_provider

    def _resolve_document_index(self) -> DocumentIndex | None:
        if self._document_index is None:
            self._document_index = MilvusDocumentIndex()
        return self._document_index
