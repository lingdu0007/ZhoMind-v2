from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pymilvus import MilvusClient

from app.common.exceptions import AppError
from app.common.config import Settings, get_settings
from app.extensions.langchain_embedding_providers import OpenAIEmbeddingProvider
from app.infra.milvus_document_index import MilvusDocumentIndex
from app.model.document import DocumentChunk
from app.rag.dense_contract import DenseEmbeddingContract, build_embedding_contract_fingerprint, build_milvus_collection_name
from app.rag.interfaces import EmbeddingProvider


class DocumentIndex(Protocol):
    async def ensure_collection(self, *, collection_name: str, dimension: int) -> None: ...

    async def upsert_generation(self, *, collection_name: str, rows: list[dict[str, object]]) -> None: ...

    async def delete_generation(self, *, collection_name: str, document_id: str, generation: int) -> None: ...

    async def delete_document(self, *, collection_name: str, document_id: str) -> None: ...

    async def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        filter: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict[str, object]]: ...


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

        embedding_provider = self._require_embedding_provider()
        document_index = self._require_document_index()

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

    async def delete_document_current_fingerprint(self, *, document_id: str) -> None:
        contract = DenseEmbeddingContract.from_settings(self._settings)
        if not contract.active:
            return

        document_index = self._resolve_document_index()
        if document_index is None:
            return

        fingerprint = build_embedding_contract_fingerprint(self._settings)
        await document_index.delete_document(
            collection_name=build_milvus_collection_name(fingerprint),
            document_id=document_id,
        )

    def _resolve_embedding_provider(self) -> EmbeddingProvider | None:
        if self._embedding_provider is not None:
            return self._embedding_provider
        contract = DenseEmbeddingContract.from_settings(self._settings)
        if contract.active:
            self._embedding_provider = OpenAIEmbeddingProvider(
                api_key=self._settings.embedding_api_key,
                base_url=self._settings.embedding_base_url_normalized,
                model=self._settings.embedding_model_normalized,
            )
        return self._embedding_provider

    def _resolve_document_index(self) -> DocumentIndex | None:
        if self._document_index is None:
            client_kwargs: dict[str, str] = {"uri": self._settings.milvus_uri}
            if self._settings.milvus_token:
                client_kwargs["token"] = self._settings.milvus_token
            self._document_index = MilvusDocumentIndex(client=MilvusClient(**client_kwargs))
        return self._document_index

    def _require_embedding_provider(self) -> EmbeddingProvider:
        provider = self._resolve_embedding_provider()
        if provider is None:
            raise AppError(
                status_code=503,
                code="DENSE_INDEX_PROVIDER_UNAVAILABLE",
                message="dense embedding provider is unavailable",
            )
        return provider

    def _require_document_index(self) -> DocumentIndex:
        document_index = self._resolve_document_index()
        if document_index is None:
            raise AppError(
                status_code=503,
                code="DENSE_INDEX_BACKEND_UNAVAILABLE",
                message="dense document index is unavailable",
            )
        return document_index
