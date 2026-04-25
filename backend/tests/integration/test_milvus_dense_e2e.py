import asyncio
from collections.abc import Generator
import json
import os
import subprocess
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import Settings, get_settings
from app.extensions.registry import get_extension_registry
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.model.base import Base
from app.model.document import Document, DocumentChunk
from app.rag.dense_contract import build_embedding_contract_fingerprint, build_milvus_collection_name
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

pytestmark = pytest.mark.skipif(not os.getenv("RUN_MILVUS_E2E"), reason="requires RUN_MILVUS_E2E=1")

_TEST_ADMIN_CODE = "test-admin-code"
_TEST_QUERY = "s3 dense milvus sentinel"


class _InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = dict(mapping)
        return len(mapping)

    async def expire(self, key: str, ttl: int) -> bool:
        return key in self._store

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0

    async def get(self, key: str) -> str | None:
        value = self._store.get(key)
        return value if isinstance(value, str) else None

    async def set(self, key: str, value: str) -> bool:
        self._store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += 1 if self._store.pop(key, None) is not None else 0
        return removed


class _StubEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for_text(text) for text in texts]

    @staticmethod
    def _vector_for_text(text: str) -> list[float]:
        normalized = text.lower()
        if _TEST_QUERY in normalized:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


def _extract_data(payload: dict) -> dict:
    return payload.get("data") or payload


def _admin_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _create_admin_token(client: TestClient, username: str = "milvus-admin") -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "password": "secret-123",
            "role": "admin",
            "admin_code": _TEST_ADMIN_CODE,
        },
    )
    assert response.status_code == 200
    return response.json()["data"]["access_token"]


def _poll_job_until_terminal(
    client: TestClient,
    *,
    headers: dict[str, str],
    job_id: str,
    timeout_seconds: float = 10.0,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_data: dict | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/documents/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        last_data = _extract_data(response.json())
        if last_data["status"] in {"succeeded", "failed", "canceled"}:
            return last_data
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach terminal state: {last_data}")


def _drop_collection_if_exists(collection_name: str) -> None:
    settings = get_settings()
    _run_milvus_admin(
        operation="drop_if_exists",
        collection_name=collection_name,
        settings=settings,
    )


def _query_milvus_rows(*, collection_name: str, document_id: str, settings: Settings) -> list[dict]:
    raw = _run_milvus_admin(
        operation="query_doc",
        collection_name=collection_name,
        settings=settings,
        document_id=document_id,
    )
    payload = json.loads(raw) if raw else []
    assert isinstance(payload, list)
    return payload


def _run_milvus_admin(
    *,
    operation: str,
    collection_name: str,
    settings: Settings,
    document_id: str | None = None,
) -> str:
    script = """
from contextlib import suppress
import json
import sys
from pymilvus import MilvusClient

uri = sys.argv[1]
token = sys.argv[2]
operation = sys.argv[3]
collection_name = sys.argv[4]
kwargs = {"uri": uri}
if token:
    kwargs["token"] = token
client = MilvusClient(**kwargs)

if operation == "drop_if_exists":
    collections = set(client.list_collections(timeout=5))
    if collection_name in collections:
        with suppress(Exception):
            client.release_collection(collection_name, timeout=5)
        client.drop_collection(collection_name, timeout=5)
    print(json.dumps({"present_before_drop": collection_name in collections}))
elif operation == "query_doc":
    document_id = sys.argv[5]
    rows = client.query(
        collection_name,
        filter=f'document_id == "{document_id}"',
        output_fields=[
            "document_id",
            "generation",
            "chunk_index",
            "content_sha256",
            "embedding_contract_fingerprint",
        ],
        timeout=5,
    )
    print(json.dumps(rows))
else:
    raise SystemExit(f"unsupported operation: {operation}")
"""
    command = [
        "uv",
        "run",
        "python",
        "-c",
        script,
        settings.milvus_uri_normalized,
        settings.milvus_token or "",
        operation,
        collection_name,
    ]
    if document_id is not None:
        command.append(document_id)
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    return completed.stdout.strip()


async def _load_document_state(session_factory, *, document_id: str) -> tuple[Document, list[DocumentChunk]]:
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        assert document is not None
        result = await session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return document, list(result.scalars().all())


async def _retrieve_chunks(session_factory, *, query: str) -> tuple[object, DocumentChunk | None]:
    async with session_factory() as session:
        service = MixedModeDocumentRetrieverService(session)
        result = await service.retrieve(query, top_k=5)
        chunk = None
        if result.items:
            chunk = await session.get(DocumentChunk, result.items[0]["chunk_id"])
        return result, chunk


def _build_settings_without_api_key(settings: Settings) -> Settings:
    return Settings(
        EMBEDDING_API_KEY="different-key",
        EMBEDDING_BASE_URL=settings.embedding_base_url_normalized,
        EMBEDDING_MODEL=settings.embedding_model_normalized,
        DENSE_EMBEDDING_DIM=settings.dense_embedding_dim,
        MILVUS_URI=settings.milvus_uri_normalized,
    )


def test_milvus_dense_upload_and_retrieval_e2e(tmp_path) -> None:
    from app.main import app

    settings = get_settings()
    assert settings.embedding_api_key_configured, "EMBEDDING_API_KEY must be configured for RUN_MILVUS_E2E"
    assert settings.embedding_base_url_normalized, "EMBEDDING_BASE_URL must be configured for RUN_MILVUS_E2E"
    assert settings.embedding_model_normalized, "EMBEDDING_MODEL must be configured for RUN_MILVUS_E2E"
    assert settings.dense_embedding_dim == 3, "RUN_MILVUS_E2E expects DENSE_EMBEDDING_DIM=3"
    assert settings.milvus_uri_normalized, "MILVUS_URI must be configured for RUN_MILVUS_E2E"

    fingerprint = build_embedding_contract_fingerprint(settings)
    assert fingerprint == build_embedding_contract_fingerprint(_build_settings_without_api_key(settings))
    collection_name = build_milvus_collection_name(fingerprint)

    db_path = tmp_path / "milvus-dense-e2e.db"
    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    registry = get_extension_registry()
    registry.register_embedding("embedding-default", _StubEmbeddingProvider())
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    original_admin_code = settings.admin_invite_code
    settings.admin_invite_code = _TEST_ADMIN_CODE
    _drop_collection_if_exists(collection_name)

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            upload_response = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={
                    "file": (
                        "milvus-e2e.txt",
                        f"{_TEST_QUERY}\nretrieval hydration proof".encode("utf-8"),
                        "text/plain",
                    )
                },
            )
            assert upload_response.status_code == 200
            upload_data = _extract_data(upload_response.json())
            document_id = upload_data["document_id"]

            terminal_job = _poll_job_until_terminal(client, headers=headers, job_id=upload_data["job_id"])
            assert terminal_job["status"] == "succeeded"

        document, chunks = asyncio.run(_load_document_state(session_factory, document_id=document_id))
        assert document.published_generation == 1
        assert document.dense_ready_generation == 1
        assert document.dense_ready_fingerprint == fingerprint
        assert chunks

        milvus_rows = _query_milvus_rows(
            collection_name=collection_name,
            document_id=document_id,
            settings=settings,
        )
        assert milvus_rows
        assert all(row["document_id"] == document_id for row in milvus_rows)
        assert all(int(row["generation"]) == document.published_generation for row in milvus_rows)
        assert {row["embedding_contract_fingerprint"] for row in milvus_rows} == {fingerprint}

        retrieved, hydrated_chunk = asyncio.run(_retrieve_chunks(session_factory, query=_TEST_QUERY))
        assert retrieved.strategy == "dense_plus_lexical_migration"
        assert retrieved.dense_query_failed is False
        assert retrieved.dense_candidate_count >= 1
        assert retrieved.dense_hydrated_count >= 1
        assert retrieved.items
        assert retrieved.items[0]["document_id"] == document_id
        assert retrieved.items[0]["retrieval_source"] == "dense"
        assert hydrated_chunk is not None
        assert retrieved.items[0]["chunk_id"] == hydrated_chunk.id
        assert retrieved.items[0]["content_preview"] == hydrated_chunk.content[:160]
        assert _TEST_QUERY in hydrated_chunk.content
    finally:
        settings.admin_invite_code = original_admin_code
        app.dependency_overrides.clear()
        _drop_collection_if_exists(collection_name)
        asyncio.run(db_engine.dispose())
