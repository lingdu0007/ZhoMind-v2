from __future__ import annotations

import re
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import Settings, get_settings
from app.extensions.registry import get_extension_registry
from app.infra.milvus_document_index import MilvusDocumentIndex
from app.model.document import Document, DocumentChunk
from app.rag.dense_contract import DenseEmbeddingContract, build_embedding_contract_fingerprint, build_milvus_collection_name
from app.rag.interfaces import EmbeddingProvider, RetrieveResult

_DEFAULT_EMBEDDING_PROVIDER = "embedding-default"


def _normalize_provider_error(exc: Exception) -> dict[str, str]:
    return {
        "code": "PROVIDER_EXEC_FAILED",
        "message": str(exc),
        "type": type(exc).__name__,
    }


class MixedModeDocumentRetrieverService:
    name = "inmemory-mixed-mode-retriever"

    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        document_index: MilvusDocumentIndex | None = None,
        candidate_limit: int = 200,
        dense_search_multiplier: int = 4,
    ) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._document_index = document_index
        self._candidate_limit = candidate_limit
        self._dense_search_multiplier = max(1, dense_search_multiplier)

    async def retrieve(self, query: str, top_k: int) -> RetrieveResult:
        normalized_query = query.strip()
        strategy = "dense_plus_lexical_migration" if DenseEmbeddingContract.from_settings(self._settings).active else "sparse_only"
        if not normalized_query:
            return RetrieveResult(items=[], strategy=strategy, merged_count=0)

        contract = DenseEmbeddingContract.from_settings(self._settings)
        if not contract.active:
            lexical_items = await self._lexical_search(normalized_query, top_k=top_k, lexical_scope="full_published_live")
            return RetrieveResult(
                items=lexical_items,
                strategy="sparse_only",
                lexical_candidate_count=len(lexical_items),
                merged_count=len(lexical_items),
                dense_query_failed=False,
                lexical_scope="full_published_live",
            )

        fingerprint = build_embedding_contract_fingerprint(self._settings)
        dense_candidates: list[dict[str, Any]] = []
        dense_items: list[dict[str, Any]] = []
        dense_query_failed = False
        dense_provider_error: dict[str, str] | None = None

        try:
            dense_candidates = await self._dense_search(normalized_query, top_k=top_k, fingerprint=fingerprint)
            dense_items = await self._hydrate_dense_hits(dense_candidates=dense_candidates, fingerprint=fingerprint)
        except Exception as exc:
            dense_candidates = []
            dense_items = []
            dense_query_failed = True
            dense_provider_error = _normalize_provider_error(exc)

        lexical_scope = "full_published_live" if dense_query_failed else "not_dense_ready_published"
        lexical_items = await self._lexical_search(
            normalized_query,
            top_k=top_k,
            lexical_scope=lexical_scope,
            fingerprint=fingerprint,
        )
        merged_items = self._merge_items(dense_items=dense_items, lexical_items=lexical_items, top_k=top_k)
        return RetrieveResult(
            items=merged_items,
            strategy="dense_plus_lexical_migration",
            dense_candidate_count=len(dense_candidates),
            dense_hydrated_count=len(dense_items),
            lexical_candidate_count=len(lexical_items),
            merged_count=len(merged_items),
            dense_query_failed=dense_query_failed,
            lexical_scope=lexical_scope,
            fallback_used=dense_query_failed,
            provider_error=dense_provider_error,
        )

    def _tokenize(self, text: str) -> list[str]:
        return [item for item in re.split(r"[\s\W_]+", text.lower()) if len(item) >= 2]

    def _compact(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _bigram_overlap(self, a: str, b: str) -> int:
        if len(a) < 2 or len(b) < 2:
            return 0
        a_set = {a[i : i + 2] for i in range(len(a) - 1)}
        b_set = {b[i : i + 2] for i in range(len(b) - 1)}
        return len(a_set & b_set)

    def _score_chunk(self, query: str, content: str) -> float:
        query_norm = query.strip().lower()
        content_norm = (content or "").strip().lower()
        if not query_norm or not content_norm:
            return 0.0

        score = 0.0
        query_compact = self._compact(query_norm)
        content_compact = self._compact(content_norm)

        if query_compact and query_compact in content_compact:
            score += 5.0

        query_tokens = self._tokenize(query_norm)
        if query_tokens:
            content_tokens = set(self._tokenize(content_norm))
            overlap = sum(1 for token in query_tokens if token in content_tokens)
            score += float(overlap)

        score += min(self._bigram_overlap(query_compact, content_compact), 6) * 0.8
        return score

    async def _lexical_search(
        self,
        query: str,
        *,
        top_k: int,
        lexical_scope: str,
        fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.deleted_at.is_(None),
                Document.published_generation > 0,
                DocumentChunk.generation == Document.published_generation,
            )
            .limit(self._candidate_limit)
        )
        if lexical_scope == "not_dense_ready_published" and fingerprint:
            stmt = stmt.where(
                or_(
                    Document.dense_ready_generation != Document.published_generation,
                    Document.dense_ready_fingerprint.is_(None),
                    Document.dense_ready_fingerprint != fingerprint,
                )
            )

        result = await self._session.execute(stmt)
        candidates = list(result.scalars().all())

        ranked: list[dict[str, Any]] = []
        for chunk in candidates:
            score = self._score_chunk(query=query, content=chunk.content)
            if score <= 0:
                continue
            ranked.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "generation": chunk.generation,
                    "chunk_index": chunk.chunk_index,
                    "score": round(score, 4),
                    "content_preview": chunk.content[:160],
                    "metadata": chunk.chunk_metadata,
                    "retrieval_source": "lexical",
                }
            )

        ranked.sort(key=lambda item: (-float(item["score"]), int(item["chunk_index"]), str(item["chunk_id"])))
        return ranked[:top_k]

    async def _dense_search(
        self,
        query: str,
        *,
        top_k: int,
        fingerprint: str,
    ) -> list[dict[str, Any]]:
        embedding_provider = self._resolve_embedding_provider()
        if embedding_provider is None:
            raise RuntimeError("dense embedding provider is unavailable")

        vectors = await embedding_provider.embed([query])
        if not vectors:
            return []

        search_results = await self._resolve_document_index().search(
            collection_name=build_milvus_collection_name(fingerprint),
            vector=vectors[0],
            limit=max(top_k, top_k * self._dense_search_multiplier),
            output_fields=["document_id", "generation", "chunk_index", "content_sha256"],
        )
        return [candidate for candidate in (self._normalize_dense_candidate(row) for row in search_results) if candidate is not None]

    async def _hydrate_dense_hits(
        self,
        *,
        dense_candidates: list[dict[str, Any]],
        fingerprint: str,
    ) -> list[dict[str, Any]]:
        if not dense_candidates:
            return []

        document_ids = sorted({str(item["document_id"]) for item in dense_candidates})
        result = await self._session.execute(
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.id.in_(document_ids),
                Document.deleted_at.is_(None),
                Document.published_generation > 0,
                Document.dense_ready_generation == Document.published_generation,
                Document.dense_ready_fingerprint == fingerprint,
                DocumentChunk.generation == Document.published_generation,
            )
        )
        chunks = list(result.scalars().all())
        chunk_map = {
            (chunk.document_id, chunk.generation, chunk.chunk_index, chunk.content_sha256): chunk
            for chunk in chunks
        }

        hydrated: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for candidate in dense_candidates:
            key = (
                str(candidate["document_id"]),
                int(candidate["generation"]),
                int(candidate["chunk_index"]),
                str(candidate["content_sha256"]),
            )
            chunk = chunk_map.get(key)
            if chunk is None:
                continue
            merged_key = (chunk.document_id, chunk.generation, chunk.chunk_index)
            if merged_key in seen:
                continue
            seen.add(merged_key)
            hydrated.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "generation": chunk.generation,
                    "chunk_index": chunk.chunk_index,
                    "score": round(float(candidate.get("score") or 0.0), 4),
                    "content_preview": chunk.content[:160],
                    "metadata": chunk.chunk_metadata,
                    "retrieval_source": "dense",
                }
            )
        return hydrated

    def _merge_items(
        self,
        *,
        dense_items: list[dict[str, Any]],
        lexical_items: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, int, int]] = set()
        for item in [*dense_items, *lexical_items]:
            key = (
                str(item.get("document_id") or ""),
                int(item.get("generation") or 0),
                int(item.get("chunk_index") or 0),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= top_k:
                break
        return merged

    def _resolve_embedding_provider(self) -> EmbeddingProvider | None:
        if self._embedding_provider is not None:
            return self._embedding_provider
        self._embedding_provider = get_extension_registry().get_embedding(_DEFAULT_EMBEDDING_PROVIDER)
        return self._embedding_provider

    def _resolve_document_index(self) -> MilvusDocumentIndex:
        if self._document_index is None:
            self._document_index = MilvusDocumentIndex()
        return self._document_index

    def _normalize_dense_candidate(self, row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        payload = row.get("entity") if isinstance(row.get("entity"), dict) else row
        if not isinstance(payload, dict):
            return None

        document_id = payload.get("document_id")
        generation = payload.get("generation")
        chunk_index = payload.get("chunk_index")
        content_sha256 = payload.get("content_sha256")
        if not all(item is not None for item in (document_id, generation, chunk_index, content_sha256)):
            return None

        return {
            "document_id": str(document_id),
            "generation": int(generation),
            "chunk_index": int(chunk_index),
            "content_sha256": str(content_sha256),
            "score": self._extract_dense_score(row),
        }

    def _extract_dense_score(self, row: dict[str, Any]) -> float:
        for key in ("distance", "score", "similarity"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        entity = row.get("entity")
        if isinstance(entity, dict):
            for key in ("distance", "score", "similarity"):
                value = entity.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0
