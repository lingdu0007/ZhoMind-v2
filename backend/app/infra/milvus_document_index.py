from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from pymilvus import MilvusClient

from app.infra.milvus import get_milvus_client


class MilvusDocumentIndex:
    def __init__(self, client: MilvusClient | None = None) -> None:
        self._client = client or get_milvus_client()

    async def ensure_collection(self, *, collection_name: str, dimension: int) -> None:
        exists = await asyncio.to_thread(self._client.has_collection, collection_name)
        if not exists:
            await asyncio.to_thread(
                self._client.create_collection,
                collection_name,
                dimension,
                "id",
                "string",
                "vector",
                "COSINE",
                False,
                None,
                None,
                max_length=256,
                enable_dynamic_field=True,
            )
            return

        description = await asyncio.to_thread(self._client.describe_collection, collection_name)
        current_dimension = self._extract_vector_dimension(description)
        if current_dimension != dimension:
            raise ValueError(
                f"milvus collection dimension mismatch for {collection_name}: expected {dimension}, got {current_dimension}"
            )

    async def upsert_generation(self, *, collection_name: str, rows: list[dict[str, Any]]) -> None:
        payload = [self._normalize_row(row) for row in rows]
        if not payload:
            return
        await asyncio.to_thread(self._client.upsert, collection_name, payload)

    async def delete_generation(self, *, collection_name: str, document_id: str, generation: int) -> None:
        exists = await asyncio.to_thread(self._client.has_collection, collection_name)
        if not exists:
            return
        try:
            await asyncio.to_thread(
                self._client.delete,
                collection_name,
                None,
                None,
                self._delete_filter(document_id=document_id, generation=generation),
            )
        except Exception as exc:
            if self._is_missing_collection_error(exc):
                return
            raise

    async def delete_document(self, *, collection_name: str, document_id: str) -> None:
        exists = await asyncio.to_thread(self._client.has_collection, collection_name)
        if not exists:
            return
        try:
            await asyncio.to_thread(
                self._client.delete,
                collection_name,
                None,
                None,
                self._delete_document_filter(document_id=document_id),
            )
        except Exception as exc:
            if self._is_missing_collection_error(exc):
                return
            raise

    async def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        filter: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        results = await asyncio.to_thread(
            self._client.search,
            collection_name,
            [vector],
            filter,
            limit,
            output_fields,
        )
        if not results:
            return []
        return list(results[0])

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized.setdefault(
            "id",
            f"{row['document_id']}:{row['generation']}:{row['chunk_index']}:{row['content_sha256']}",
        )
        return normalized

    @staticmethod
    def _delete_filter(*, document_id: str, generation: int) -> str:
        escaped_document_id = document_id.replace("\\", "\\\\").replace('"', '\\"')
        return f'document_id == "{escaped_document_id}" and generation == {generation}'

    @staticmethod
    def _delete_document_filter(*, document_id: str) -> str:
        escaped_document_id = document_id.replace("\\", "\\\\").replace('"', '\\"')
        return f'document_id == "{escaped_document_id}"'

    @staticmethod
    def _is_missing_collection_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            needle in message
            for needle in (
                "collection not found",
                "can't find collection",
                "cannot find collection",
                "collection doesn't exist",
                "collection does not exist",
            )
        )

    @classmethod
    def _extract_vector_dimension(cls, description: Any) -> int | None:
        if not isinstance(description, dict):
            return None
        fields = description.get("fields")
        if not isinstance(fields, Iterable):
            return None
        for field in fields:
            if not isinstance(field, dict):
                continue
            if field.get("name") != "vector":
                continue
            params = field.get("params")
            if isinstance(params, dict) and "dim" in params:
                try:
                    return int(params["dim"])
                except (TypeError, ValueError):
                    return None
            if "dim" in field:
                try:
                    return int(field["dim"])
                except (TypeError, ValueError):
                    return None
        return None
