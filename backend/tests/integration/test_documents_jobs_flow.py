import asyncio
from collections.abc import Generator
from datetime import datetime, timezone
import hashlib
import os
import tempfile
import threading
import time

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import Settings
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.document import Document, DocumentChunk, DocumentJob


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


async def _create_admin_token(client: TestClient, username: str = "admin") -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "password": "secret-123",
            "role": "admin",
            "admin_code": "test-admin-code",
        },
    )
    assert response.status_code == 200
    return response.json()["data"]["access_token"]


def _admin_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _extract_data(payload: dict) -> dict:
    return payload.get("data") or payload


def _poll_job_until_terminal(
    client: TestClient,
    *,
    headers: dict[str, str],
    job_id: str,
    timeout_seconds: float = 2.0,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_data: dict | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/documents/jobs/{job_id}", headers=headers)
        assert response.status_code == 200
        data = _extract_data(response.json())
        last_data = data
        if data["status"] in {"succeeded", "failed", "canceled"}:
            return data
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach terminal state, last={last_data}")


def _get_document_item(client: TestClient, *, headers: dict[str, str], document_id: str) -> dict:
    response = client.get("/api/v1/documents?page=1&page_size=50", headers=headers)
    assert response.status_code == 200
    items = _extract_data(response.json())["items"]
    return next(item for item in items if item["document_id"] == document_id)


def _get_chunk_snapshot(client: TestClient, *, headers: dict[str, str], document_id: str) -> list[tuple[int, str, dict]]:
    response = client.get(
        f"/api/v1/documents/{document_id}/chunks?page=1&page_size=50",
        headers=headers,
    )
    assert response.status_code == 200
    items = _extract_data(response.json())["items"]
    return [(item["chunk_index"], item["content"], item["metadata"]) for item in items]


async def _load_document_record(session_factory, *, document_id: str) -> Document | None:
    async with session_factory() as session:
        return await session.get(Document, document_id)


async def _load_document_job_record(session_factory, *, job_id: str) -> DocumentJob | None:
    async with session_factory() as session:
        return await session.get(DocumentJob, job_id)


async def _tombstone_document_record(session_factory, *, document_id: str) -> None:
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        assert document is not None
        document.deleted_at = datetime.now(timezone.utc)
        document.status = "pending"
        document.latest_requested_generation = document.published_generation
        await session.commit()


class _StubEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.6, 0.4] for _ in texts]


class _FakeDenseDocumentIndex:
    def __init__(self, rows: list[dict] | None = None, *, error: Exception | None = None) -> None:
        self._rows = list(rows or [])
        self._error = error

    async def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        filter: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        if self._error is not None:
            raise self._error
        return self._rows[:limit]


class _DenseMaintenanceIndexResult:
    def __init__(self, *, fingerprint: str) -> None:
        self.active = True
        self.fingerprint = fingerprint


class _DenseMaintenanceIndexSpy:
    def __init__(self, *, result_fingerprint: str) -> None:
        self.result_fingerprint = result_fingerprint
        self.index_calls: list[tuple[str, int, list[DocumentChunk]]] = []
        self.delete_calls: list[str] = []

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks: list[DocumentChunk],
    ) -> _DenseMaintenanceIndexResult:
        self.index_calls.append((document_id, generation, list(chunks)))
        return _DenseMaintenanceIndexResult(fingerprint=self.result_fingerprint)

    async def delete_document_current_fingerprint(self, *, document_id: str) -> None:
        self.delete_calls.append(document_id)


class _ScalarListResult:
    def __init__(self, rows: list[DocumentChunk]) -> None:
        self._rows = rows

    def scalars(self) -> "_ScalarListResult":
        return self

    def all(self) -> list[DocumentChunk]:
        return list(self._rows)


class _LimitAwareLexicalSession:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self.limit_history: list[int | None] = []

    async def execute(self, stmt):
        limit_clause = stmt._limit_clause
        limit_value = None if limit_clause is None else int(limit_clause.value)
        self.limit_history.append(limit_value)
        rows = self._chunks if limit_value is None else self._chunks[:limit_value]
        return _ScalarListResult(rows)


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


async def _seed_mixed_mode_retrieval_documents(
    session_factory,
    *,
    fingerprint: str,
) -> None:
    async with session_factory() as session:
        dense_chunk = DocumentChunk(
            id="chunk-dense-published",
            document_id="doc-dense-published",
            generation=1,
            chunk_index=0,
            content="alpha dense published evidence",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "dense"},
        )
        lexical_chunk = DocumentChunk(
            id="chunk-lexical-published",
            document_id="doc-lexical-published",
            generation=1,
            chunk_index=0,
            content="beta lexical published evidence",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "lexical"},
        )
        superseded_published_chunk = DocumentChunk(
            id="chunk-superseded-published",
            document_id="doc-superseded",
            generation=1,
            chunk_index=0,
            content="qqq qqq qqq",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "published"},
        )
        superseded_candidate_chunk = DocumentChunk(
            id="chunk-superseded-candidate",
            document_id="doc-superseded",
            generation=2,
            chunk_index=0,
            content="alpha superseded dense candidate should stay hidden",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "candidate"},
        )
        inflight_published_chunk = DocumentChunk(
            id="chunk-inflight-published",
            document_id="doc-inflight",
            generation=1,
            chunk_index=0,
            content="zzz zzz zzz",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "published"},
        )
        inflight_candidate_chunk = DocumentChunk(
            id="chunk-inflight-candidate",
            document_id="doc-inflight",
            generation=2,
            chunk_index=0,
            content="beta inflight candidate should stay hidden",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "candidate"},
        )
        tombstoned_chunk = DocumentChunk(
            id="chunk-tombstoned",
            document_id="doc-tombstoned",
            generation=1,
            chunk_index=0,
            content="alpha tombstoned content should stay hidden",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "deleted"},
        )

        session.add_all(
            [
                Document(
                    id="doc-dense-published",
                    filename="dense-published.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=fingerprint,
                    next_generation=2,
                    latest_requested_generation=1,
                ),
                Document(
                    id="doc-lexical-published",
                    filename="lexical-published.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=0,
                    dense_ready_fingerprint=None,
                    next_generation=2,
                    latest_requested_generation=1,
                ),
                Document(
                    id="doc-superseded",
                    filename="superseded.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=fingerprint,
                    next_generation=3,
                    latest_requested_generation=2,
                ),
                Document(
                    id="doc-inflight",
                    filename="inflight.txt",
                    file_type="txt",
                    file_size=10,
                    status="pending",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=0,
                    dense_ready_fingerprint=None,
                    next_generation=3,
                    latest_requested_generation=2,
                    active_build_generation=2,
                    active_build_job_id="job-inflight",
                ),
                Document(
                    id="doc-tombstoned",
                    filename="tombstoned.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=0,
                    dense_ready_fingerprint=None,
                    next_generation=2,
                    latest_requested_generation=1,
                    deleted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
                dense_chunk,
                lexical_chunk,
                superseded_published_chunk,
                superseded_candidate_chunk,
                inflight_published_chunk,
                inflight_candidate_chunk,
                tombstoned_chunk,
            ]
        )
        await session.commit()


def test_documents_and_jobs_flow(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-flow-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    previous_admin_code = app.state.__dict__.get("_test_admin_invite_code", None)

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    original_build_document_runner = documents_api._build_document_runner
    delayed_rebuild_gate = threading.Event()
    delay_next_build = {"enabled": False}

    def _build_document_runner_with_gate(*, bind_url: str, document_id: str, job_id: str, content=None, gate=None):
        if delay_next_build["enabled"] and gate is None:
            delay_next_build["enabled"] = False
            gate = delayed_rebuild_gate
        return original_build_document_runner(
            bind_url=bind_url,
            document_id=document_id,
            job_id=job_id,
            content=content,
            gate=gate,
        )

    monkeypatch.setattr(documents_api, "_build_document_runner", _build_document_runner_with_gate)

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            unsupported_upload_response = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("demo.docx", b"binary", "application/octet-stream")},
            )
            assert unsupported_upload_response.status_code == 415
            assert unsupported_upload_response.json()["code"] == "DOC_FILE_TYPE_NOT_SUPPORTED"

            upload_response = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("demo.txt", b"line 1\nline 2\nline 3", "text/plain")},
            )
            assert upload_response.status_code == 200
            upload_data = _extract_data(upload_response.json())
            document_id = upload_data["document_id"]
            job_id = upload_data["job_id"]

            upload_chunk_not_ready = client.get(
                f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                headers=headers,
            )
            assert upload_chunk_not_ready.status_code == 409
            assert upload_chunk_not_ready.json()["code"] == "DOC_CHUNK_RESULT_NOT_READY"

            upload_job_poll = _poll_job_until_terminal(client, headers=headers, job_id=job_id)
            assert upload_job_poll["status"] == "succeeded"
            assert upload_job_poll["stage"] == "completed"

            list_docs_response = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert list_docs_response.status_code == 200
            docs_data = _extract_data(list_docs_response.json())
            assert docs_data["items"]
            first_doc = docs_data["items"][0]
            assert first_doc["document_id"] == document_id
            assert first_doc["status"] == "ready"
            assert first_doc["chunk_strategy"] == "general"
            assert first_doc["chunk_count"] > 0
            upload_chunk_count = first_doc["chunk_count"]

            get_job_response = client.get(f"/api/v1/documents/jobs/{job_id}", headers=headers)
            assert get_job_response.status_code == 200
            job_data = _extract_data(get_job_response.json())
            assert job_data["status"] == "succeeded"
            assert job_data["stage"] == "completed"

            list_jobs_response = client.get("/api/v1/documents/jobs?page=1&page_size=20", headers=headers)
            assert list_jobs_response.status_code == 200
            jobs_data = _extract_data(list_jobs_response.json())
            assert any(item["job_id"] == job_id for item in jobs_data["items"])

            chunk_response = client.get(
                f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                headers=headers,
            )
            assert chunk_response.status_code == 200
            chunks_data = _extract_data(chunk_response.json())
            assert chunks_data["items"]
            assert "metadata" in chunks_data["items"][0]
            upload_chunks = _get_chunk_snapshot(client, headers=headers, document_id=document_id)
            assert any("line 1" in content and "line 3" in content for _, content, _ in upload_chunks)

            delay_next_build["enabled"] = True
            build_response = client.post(
                f"/api/v1/documents/{document_id}/build",
                headers=headers,
                json={"chunk_strategy": "paper"},
            )
            assert build_response.status_code == 200
            rebuilt_job = _extract_data(build_response.json())
            rebuilt_job_id = rebuilt_job["job_id"]
            assert rebuilt_job["status"] == "queued"
            assert rebuilt_job["stage"] == "queued"
            assert rebuilt_job["document_id"] == document_id

            pending_summary_error: AssertionError | None = None
            try:
                rebuilt_doc_during_pending = _get_document_item(client, headers=headers, document_id=document_id)
                assert rebuilt_doc_during_pending["status"] == "pending"
                assert rebuilt_doc_during_pending["chunk_strategy"] == "general"
                assert rebuilt_doc_during_pending["chunk_count"] == upload_chunk_count

                not_ready_chunk_response = client.get(
                    f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                    headers=headers,
                )
                assert not_ready_chunk_response.status_code == 409
                assert not_ready_chunk_response.json()["code"] == "DOC_CHUNK_RESULT_NOT_READY"
            except AssertionError as exc:
                pending_summary_error = exc
            finally:
                delayed_rebuild_gate.set()
                if pending_summary_error is not None:
                    _poll_job_until_terminal(client, headers=headers, job_id=rebuilt_job_id)

            if pending_summary_error is not None:
                raise pending_summary_error

            rebuilt_job_data = _poll_job_until_terminal(client, headers=headers, job_id=rebuilt_job_id)
            assert rebuilt_job_data["status"] == "succeeded"
            assert rebuilt_job_data["stage"] == "completed"
            assert rebuilt_job_data["progress"] == 100

            chunk_response = client.get(
                f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                headers=headers,
            )
            assert chunk_response.status_code == 200
            chunks_data = _extract_data(chunk_response.json())
            assert chunks_data["items"]
            assert "metadata" in chunks_data["items"][0]
            rebuilt_chunks = _get_chunk_snapshot(client, headers=headers, document_id=document_id)
            assert any("line 1" in content and "line 3" in content for _, content, _ in rebuilt_chunks)
            assert all("demo.txt\npaper" not in content for _, content, _ in rebuilt_chunks)

            batch_build_response = client.post(
                "/api/v1/documents/batch-build",
                headers=headers,
                json={"document_ids": [document_id], "chunk_strategy": "qa"},
            )
            assert batch_build_response.status_code == 200
            batch_build_data = _extract_data(batch_build_response.json())
            assert len(batch_build_data["items"]) == 1
            batch_job_id = batch_build_data["items"][0]["job_id"]
            assert batch_build_data["items"][0]["status"] == "queued"

            batch_job_poll_data = _poll_job_until_terminal(client, headers=headers, job_id=batch_job_id)
            assert batch_job_poll_data["status"] == "succeeded"
            batch_chunks = _get_chunk_snapshot(client, headers=headers, document_id=document_id)
            assert any("line 1" in content and "line 3" in content for _, content, _ in batch_chunks)
            assert all("demo.txt\nqa" not in content for _, content, _ in batch_chunks)

            invalid_batch_strategy_response = client.post(
                "/api/v1/documents/batch-build",
                headers=headers,
                json={"document_ids": [document_id], "chunk_strategy": "outline"},
            )
            assert invalid_batch_strategy_response.status_code == 422

            cancel_completed_job_response = client.post(
                f"/api/v1/documents/jobs/{job_id}/cancel",
                headers=headers,
            )
            assert cancel_completed_job_response.status_code == 200
            assert _extract_data(cancel_completed_job_response.json())["status"] == "succeeded"

            cancel_target_build_response = client.post(
                f"/api/v1/documents/{document_id}/build",
                headers=headers,
                json={"chunk_strategy": "paper"},
            )
            assert cancel_target_build_response.status_code == 200
            cancel_target_job = _extract_data(cancel_target_build_response.json())
            cancel_target_job_id = cancel_target_job["job_id"]
            assert cancel_target_job["status"] == "queued"

            cancel_response = client.post(
                f"/api/v1/documents/jobs/{cancel_target_job_id}/cancel",
                headers=headers,
            )
            assert cancel_response.status_code == 200
            cancel_data = _extract_data(cancel_response.json())
            assert cancel_data["status"] == "canceled"
            assert cancel_data["stage"] == "failed"

            canceled_docs_response = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert canceled_docs_response.status_code == 200
            canceled_docs_data = _extract_data(canceled_docs_response.json())
            canceled_doc = next(item for item in canceled_docs_data["items"] if item["document_id"] == document_id)
            assert canceled_doc["status"] == "pending"

            canceled_chunk_response = client.get(
                f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                headers=headers,
            )
            assert canceled_chunk_response.status_code == 409
            assert canceled_chunk_response.json()["code"] == "DOC_CHUNK_RESULT_NOT_READY"

            list_jobs_response = client.get("/api/v1/documents/jobs?page=1&page_size=20", headers=headers)
            assert list_jobs_response.status_code == 200
            jobs_data = _extract_data(list_jobs_response.json())
            assert any(item["job_id"] == job_id for item in jobs_data["items"])
            assert any(item["job_id"] == rebuilt_job_id for item in jobs_data["items"])
            assert any(item["job_id"] == cancel_target_job_id for item in jobs_data["items"])

            async def _insert_legacy_document() -> str:
                async with session_factory() as session:
                    legacy_document = Document(
                        filename="legacy.txt",
                        file_type="txt",
                        file_size=18,
                        source_content=None,
                        status="ready",
                        chunk_strategy="general",
                    )
                    session.add(legacy_document)
                    await session.commit()
                    return legacy_document.id

            legacy_document_id = asyncio.run(_insert_legacy_document())

            legacy_build_response = client.post(
                f"/api/v1/documents/{legacy_document_id}/build",
                headers=headers,
                json={"chunk_strategy": "paper"},
            )
            assert legacy_build_response.status_code == 200
            legacy_build_job = _extract_data(legacy_build_response.json())
            legacy_build_job_id = legacy_build_job["job_id"]

            legacy_job_poll = _poll_job_until_terminal(client, headers=headers, job_id=legacy_build_job_id)
            assert legacy_job_poll["status"] == "failed"
            assert legacy_job_poll["stage"] == "failed"
            assert "DOC_SOURCE_CONTENT_MISSING" in legacy_job_poll["message"]
            assert "document source content is missing" in legacy_job_poll["message"]

            legacy_document_item = _get_document_item(client, headers=headers, document_id=legacy_document_id)
            assert legacy_document_item["status"] == "failed"

            batch_delete_response = client.post(
                "/api/v1/documents/batch-delete",
                headers=headers,
                json={"document_ids": [document_id, legacy_document_id]},
            )
            assert batch_delete_response.status_code == 200
            batch_delete_data = _extract_data(batch_delete_response.json())
            assert document_id in batch_delete_data["success_ids"]
            assert legacy_document_id in batch_delete_data["success_ids"]

            post_delete_docs = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert post_delete_docs.status_code == 200
            post_delete_data = _extract_data(post_delete_docs.json())
            assert post_delete_data["items"] == []
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_and_jobs_flow_tombstone_visibility_after_delete() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-delete-visibility-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            upload_response = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("demo.txt", b"line 1\nline 2\nline 3", "text/plain")},
            )
            assert upload_response.status_code == 200
            upload_data = _extract_data(upload_response.json())
            document_id = upload_data["document_id"]
            assert _poll_job_until_terminal(client, headers=headers, job_id=upload_data["job_id"])["status"] == "succeeded"

            delete_response = client.delete("/api/v1/documents/demo.txt", headers=headers)
            assert delete_response.status_code == 200
            delete_data = _extract_data(delete_response.json())
            assert delete_data["success_ids"] == [document_id]
            assert delete_data["failed_items"] == []

            hidden_docs = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert hidden_docs.status_code == 200
            hidden_docs_data = _extract_data(hidden_docs.json())
            hidden_docs_items = hidden_docs_data["items"]
            assert hidden_docs_data["pagination"]["total"] == 0
            assert all(item["document_id"] != document_id for item in hidden_docs_items)

            hidden_chunks = client.get(
                f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                headers=headers,
            )
            assert hidden_chunks.status_code == 404
            assert hidden_chunks.json()["code"] == "RESOURCE_NOT_FOUND"

            tombstoned_document = asyncio.run(_load_document_record(session_factory, document_id=document_id))
            assert tombstoned_document is not None
            assert tombstoned_document.deleted_at is not None
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_migration_drain_status_and_resume(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-drain-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 1)

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            drain_response = client.post("/api/v1/documents/ops/migration-drain", headers=headers)
            assert drain_response.status_code == 200
            drain_data = _extract_data(drain_response.json())
            assert drain_data["drain_enabled"] is True
            assert drain_data["active_dispatcher_tasks"] == 1
            assert drain_data["ready_for_migration"] is False

            blocked_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("blocked.txt", b"body", "text/plain")},
            )
            assert blocked_upload.status_code == 503
            assert blocked_upload.json()["code"] == "DOC_MIGRATION_DRAIN_ACTIVE"

            monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 0)
            status_response = client.get("/api/v1/documents/ops/migration-status", headers=headers)
            assert status_response.status_code == 200
            status_data = _extract_data(status_response.json())
            assert status_data["drain_enabled"] is True

            resume_response = client.post("/api/v1/documents/ops/migration-resume", headers=headers)
            assert resume_response.status_code == 200
            resume_data = _extract_data(resume_response.json())
            assert resume_data["drain_enabled"] is False

            unblocked_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("after-resume.txt", b"body", "text/plain")},
            )
            assert unblocked_upload.status_code == 200
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_migration_drain_blocks_cancel_job(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-drain-cancel-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 0)

    async def _seed_cancel_target() -> None:
        async with session_factory() as session:
            session.add(
                Document(
                    id="doc-drain-cancel",
                    filename="drain-cancel.txt",
                    file_type="txt",
                    file_size=12,
                    source_content=b"hello world",
                    status="pending",
                    chunk_strategy="general",
                    chunk_count=0,
                    published_generation=0,
                    next_generation=2,
                    latest_requested_generation=1,
                    active_build_generation=1,
                    active_build_job_id="job-drain-cancel",
                )
            )
            session.add(
                DocumentJob(
                    id="job-drain-cancel",
                    document_id="doc-drain-cancel",
                    build_generation=1,
                    requested_chunk_strategy="general",
                    status="queued",
                    stage="queued",
                    progress=0,
                    message="queued",
                )
            )
            await session.commit()

    asyncio.run(_seed_cancel_target())

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            drain_response = client.post("/api/v1/documents/ops/migration-drain", headers=headers)
            assert drain_response.status_code == 200

            cancel_response = client.post("/api/v1/documents/jobs/job-drain-cancel/cancel", headers=headers)
            assert cancel_response.status_code == 503
            assert cancel_response.json()["code"] == "DOC_MIGRATION_DRAIN_ACTIVE"

            persisted_job = asyncio.run(_load_document_job_record(session_factory, job_id="job-drain-cancel"))
            assert persisted_job is not None
            assert persisted_job.status == "queued"
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_migration_reconcile_only_cancels_queued_jobs(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-reconcile-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    from app.api.v1 import documents as documents_api

    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 0)
    claimed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first_updated_at = datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)
    second_updated_at = datetime(2026, 1, 2, 0, 1, tzinfo=timezone.utc)
    running_updated_at = datetime(2026, 1, 2, 0, 2, tzinfo=timezone.utc)

    async def _seed_reconcile_state() -> None:
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-live",
                        filename="live.txt",
                        file_type="txt",
                        file_size=12,
                        status="pending",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=2,
                        next_generation=4,
                        latest_requested_generation=3,
                        active_build_generation=3,
                        active_build_job_id="job-live-queued",
                        active_build_heartbeat_at=claimed_at,
                    ),
                    Document(
                        id="doc-tombstoned",
                        filename="tombstoned.txt",
                        file_type="txt",
                        file_size=12,
                        status="pending",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        next_generation=3,
                        latest_requested_generation=2,
                        deleted_at=datetime.now(timezone.utc),
                    ),
                ]
            )
            session.add_all(
                [
                    DocumentJob(
                        id="job-live-queued",
                        document_id="doc-live",
                        build_generation=3,
                        requested_chunk_strategy="paper",
                        status="queued",
                        stage="queued",
                        progress=0,
                        message="queued",
                        updated_at=first_updated_at,
                    ),
                    DocumentJob(
                        id="job-tombstoned-queued",
                        document_id="doc-tombstoned",
                        build_generation=2,
                        requested_chunk_strategy="paper",
                        status="queued",
                        stage="queued",
                        progress=0,
                        message="queued",
                        updated_at=second_updated_at,
                    ),
                    DocumentJob(
                        id="job-running",
                        document_id="doc-live",
                        build_generation=4,
                        requested_chunk_strategy="paper",
                        status="running",
                        stage="chunking",
                        progress=60,
                        message="running",
                        updated_at=running_updated_at,
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_seed_reconcile_state())

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            drain_response = client.post("/api/v1/documents/ops/migration-drain", headers=headers)
            assert drain_response.status_code == 200

            reconcile_response = client.post("/api/v1/documents/ops/migration-reconcile", headers=headers)
            assert reconcile_response.status_code == 200
            reconcile_data = _extract_data(reconcile_response.json())
            assert reconcile_data["reconciled_job_ids"] == ["job-live-queued", "job-tombstoned-queued"]
            assert reconcile_data["queued_jobs"] == 0
            assert reconcile_data["running_jobs"] == 1
            assert reconcile_data["active_dispatcher_tasks"] == 0

            live_job = client.get("/api/v1/documents/jobs/job-live-queued", headers=headers)
            live_job_data = _extract_data(live_job.json())
            assert live_job_data["status"] == "canceled"
            assert live_job_data["stage"] == "failed"
            assert live_job_data["progress"] == 0
            assert live_job_data["message"] == "canceled during migration drain"

            tombstoned_job = client.get("/api/v1/documents/jobs/job-tombstoned-queued", headers=headers)
            tombstoned_job_data = _extract_data(tombstoned_job.json())
            assert tombstoned_job_data["status"] == "canceled"
            assert tombstoned_job_data["stage"] == "failed"
            assert tombstoned_job_data["progress"] == 0
            assert tombstoned_job_data["message"] == "canceled during migration drain"

            running_job = client.get("/api/v1/documents/jobs/job-running", headers=headers)
            running_job_data = _extract_data(running_job.json())
            assert running_job_data["status"] == "running"
            assert running_job_data["stage"] == "chunking"
            assert running_job_data["progress"] == 60
            assert running_job_data["message"] == "running"

            docs_data = _extract_data(client.get("/api/v1/documents?page=1&page_size=20", headers=headers).json())
            live_doc = next(item for item in docs_data["items"] if item["document_id"] == "doc-live")
            assert live_doc["status"] == "ready"
            assert all(item["document_id"] != "doc-tombstoned" for item in docs_data["items"])

            persisted_live_doc = asyncio.run(_load_document_record(session_factory, document_id="doc-live"))
            assert persisted_live_doc is not None
            assert persisted_live_doc.status == "ready"
            assert persisted_live_doc.latest_requested_generation == 2
            assert persisted_live_doc.active_build_generation is None
            assert persisted_live_doc.active_build_job_id is None
            assert persisted_live_doc.active_build_heartbeat_at is None

            persisted_tombstoned_doc = asyncio.run(
                _load_document_record(session_factory, document_id="doc-tombstoned")
            )
            assert persisted_tombstoned_doc is not None
            assert persisted_tombstoned_doc.deleted_at is not None
            assert persisted_tombstoned_doc.status == "pending"
            assert persisted_tombstoned_doc.latest_requested_generation == 2
            assert persisted_tombstoned_doc.published_generation == 1

            persisted_live_job = asyncio.run(_load_document_job_record(session_factory, job_id="job-live-queued"))
            assert persisted_live_job is not None
            assert persisted_live_job.status == "canceled"
            assert persisted_live_job.stage == "failed"
            assert persisted_live_job.progress == 0
            assert persisted_live_job.message == "canceled during migration drain"

            persisted_tombstoned_job = asyncio.run(
                _load_document_job_record(session_factory, job_id="job-tombstoned-queued")
            )
            assert persisted_tombstoned_job is not None
            assert persisted_tombstoned_job.status == "canceled"
            assert persisted_tombstoned_job.stage == "failed"
            assert persisted_tombstoned_job.progress == 0
            assert persisted_tombstoned_job.message == "canceled during migration drain"

            persisted_running_job = asyncio.run(_load_document_job_record(session_factory, job_id="job-running"))
            assert persisted_running_job is not None
            assert persisted_running_job.status == "running"
            assert persisted_running_job.stage == "chunking"
            assert persisted_running_job.progress == 60
            assert persisted_running_job.message == "running"
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_migration_reconcile_requires_active_drain() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-reconcile-inactive-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            reconcile_response = client.post("/api/v1/documents/ops/migration-reconcile", headers=headers)
            assert reconcile_response.status_code == 409
            assert reconcile_response.json()["code"] == "DOC_MIGRATION_DRAIN_INACTIVE"
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_dense_status_merges_operator_and_dense_counts(monkeypatch) -> None:
    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from app.rag.dense_contract import build_embedding_contract_fingerprint

    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-status-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    dense_settings = _dense_settings()
    current_fingerprint = build_embedding_contract_fingerprint(dense_settings)
    other_fingerprint = build_embedding_contract_fingerprint(_dense_settings(EMBEDDING_MODEL="text-embedding-3-small"))

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed_documents() -> None:
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-ready-current",
                        filename="doc-ready-current.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=2,
                        dense_ready_generation=2,
                        dense_ready_fingerprint=current_fingerprint,
                    ),
                    Document(
                        id="doc-not-ready",
                        filename="doc-not-ready.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=0,
                        dense_ready_fingerprint=None,
                    ),
                    Document(
                        id="doc-stale-current",
                        filename="doc-stale-current.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=3,
                        dense_ready_generation=2,
                        dense_ready_fingerprint=current_fingerprint,
                    ),
                    Document(
                        id="doc-tombstoned-current",
                        filename="doc-tombstoned-current.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=4,
                        dense_ready_generation=4,
                        dense_ready_fingerprint=current_fingerprint,
                        deleted_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                    ),
                    Document(
                        id="doc-ready-other",
                        filename="doc-ready-other.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=5,
                        dense_ready_generation=5,
                        dense_ready_fingerprint=other_fingerprint,
                    ),
                ]
            )
            session.add_all(
                [
                    DocumentJob(
                        id="job-queued-dense-status",
                        document_id="doc-not-ready",
                        build_generation=2,
                        requested_chunk_strategy="general",
                        status="queued",
                        stage="queued",
                        progress=0,
                        message="queued",
                    ),
                    DocumentJob(
                        id="job-running-dense-status",
                        document_id="doc-ready-current",
                        build_generation=3,
                        requested_chunk_strategy="general",
                        status="running",
                        stage="chunking",
                        progress=10,
                        message="running",
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())
    asyncio.run(_seed_documents())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 2)
    monkeypatch.setattr(
        documents_api,
        "DenseMaintenanceService",
        lambda: DenseMaintenanceService(settings=dense_settings),
    )

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            response = client.get("/api/v1/documents/ops/dense-status", headers=headers)
            assert response.status_code == 200
            data = _extract_data(response.json())
            assert data["drain_enabled"] is False
            assert data["drain_started_at"] is None
            assert data["queued_jobs"] == 1
            assert data["running_jobs"] == 1
            assert data["active_dispatcher_tasks"] == 2
            assert data["ready_for_migration"] is False
            assert data["current_embedding_contract_fingerprint"] == current_fingerprint
            assert data["dense_mode_active"] is True
            assert data["published_live_documents"] == 4
            assert data["published_live_dense_ready_documents"] == 1
            assert data["published_live_not_dense_ready_documents"] == 3
            assert data["published_live_stale_generation_documents"] == 1
            assert data["tombstoned_current_fingerprint_documents"] == 1
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_dense_backfill_and_dense_reconcile_require_active_drain(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-backfill-inactive-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 0)

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            backfill_response = client.post("/api/v1/documents/ops/dense-backfill", headers=headers, json={})
            assert backfill_response.status_code == 409
            assert backfill_response.json()["code"] == "DOC_MIGRATION_DRAIN_INACTIVE"

            reconcile_response = client.post("/api/v1/documents/ops/dense-reconcile", headers=headers, json={})
            assert reconcile_response.status_code == 409
            assert reconcile_response.json()["code"] == "DOC_MIGRATION_DRAIN_INACTIVE"
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_dense_backfill_indexes_eligible_published_docs(monkeypatch) -> None:
    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from app.rag.dense_contract import build_embedding_contract_fingerprint

    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-backfill-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    dense_settings = _dense_settings()
    current_fingerprint = build_embedding_contract_fingerprint(dense_settings)
    spy = _DenseMaintenanceIndexSpy(result_fingerprint="fingerprint-from-maintenance-spy")

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed_documents() -> None:
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-backfill-a",
                        filename="doc-backfill-a.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=2,
                        published_generation=2,
                        dense_ready_generation=0,
                        dense_ready_fingerprint=None,
                        next_generation=3,
                        latest_requested_generation=2,
                        uploaded_at=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                    Document(
                        id="doc-backfill-b",
                        filename="doc-backfill-b.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=0,
                        dense_ready_fingerprint=None,
                        next_generation=2,
                        latest_requested_generation=1,
                        uploaded_at=datetime(2026, 4, 1, 0, 1, tzinfo=timezone.utc),
                    ),
                    Document(
                        id="doc-backfill-stale",
                        filename="doc-backfill-stale.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=3,
                        dense_ready_generation=2,
                        dense_ready_fingerprint=current_fingerprint,
                        next_generation=4,
                        latest_requested_generation=3,
                        uploaded_at=datetime(2026, 4, 1, 0, 2, tzinfo=timezone.utc),
                    ),
                    Document(
                        id="doc-backfill-active",
                        filename="doc-backfill-active.txt",
                        file_type="txt",
                        file_size=10,
                        status="pending",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=4,
                        dense_ready_generation=0,
                        dense_ready_fingerprint=None,
                        next_generation=5,
                        latest_requested_generation=4,
                        active_build_generation=5,
                        uploaded_at=datetime(2026, 4, 1, 0, 3, tzinfo=timezone.utc),
                    ),
                ]
            )
            session.add_all(
                [
                    DocumentChunk(
                        id="chunk-backfill-a-0",
                        document_id="doc-backfill-a",
                        generation=2,
                        chunk_index=0,
                        content="doc-backfill-a chunk 0",
                        chunk_metadata={"source": "published"},
                    ),
                    DocumentChunk(
                        id="chunk-backfill-a-1",
                        document_id="doc-backfill-a",
                        generation=2,
                        chunk_index=1,
                        content="doc-backfill-a chunk 1",
                        chunk_metadata={"source": "published"},
                    ),
                    DocumentChunk(
                        id="chunk-backfill-b-0",
                        document_id="doc-backfill-b",
                        generation=1,
                        chunk_index=0,
                        content="doc-backfill-b chunk 0",
                        chunk_metadata={"source": "published"},
                    ),
                    DocumentChunk(
                        id="chunk-backfill-stale-0",
                        document_id="doc-backfill-stale",
                        generation=3,
                        chunk_index=0,
                        content="doc-backfill-stale chunk 0",
                        chunk_metadata={"source": "published"},
                    ),
                    DocumentChunk(
                        id="chunk-backfill-active-0",
                        document_id="doc-backfill-active",
                        generation=4,
                        chunk_index=0,
                        content="doc-backfill-active chunk 0",
                        chunk_metadata={"source": "published"},
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())
    asyncio.run(_seed_documents())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 0)
    monkeypatch.setattr(
        documents_api,
        "DenseMaintenanceService",
        lambda: DenseMaintenanceService(settings=dense_settings, dense_index_service=spy),
    )

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            drain_response = client.post("/api/v1/documents/ops/migration-drain", headers=headers)
            assert drain_response.status_code == 200

            response = client.post("/api/v1/documents/ops/dense-backfill", headers=headers, json={})
            assert response.status_code == 200
            data = _extract_data(response.json())
            assert data["processed_documents"] == 4
            assert data["indexed_documents"] == 2
            assert data["skipped_documents"] == 2
            assert data["failed_documents"] == 0
            assert data["current_embedding_contract_fingerprint"] == current_fingerprint
            assert data["dense_mode_active"] is True
            assert [(item["document_id"], item["outcome"], item["reason"]) for item in data["documents"]] == [
                ("doc-backfill-a", "indexed", None),
                ("doc-backfill-b", "indexed", None),
                ("doc-backfill-stale", "skipped", "stale_current_fingerprint"),
                ("doc-backfill-active", "skipped", "active_build_in_progress"),
            ]
            assert [(document_id, generation) for document_id, generation, _chunks in spy.index_calls] == [
                ("doc-backfill-a", 2),
                ("doc-backfill-b", 1),
            ]

            persisted_a = asyncio.run(_load_document_record(session_factory, document_id="doc-backfill-a"))
            persisted_b = asyncio.run(_load_document_record(session_factory, document_id="doc-backfill-b"))
            persisted_stale = asyncio.run(_load_document_record(session_factory, document_id="doc-backfill-stale"))
            persisted_active = asyncio.run(_load_document_record(session_factory, document_id="doc-backfill-active"))
            assert persisted_a is not None
            assert persisted_a.dense_ready_generation == 2
            assert persisted_a.dense_ready_fingerprint == "fingerprint-from-maintenance-spy"
            assert persisted_b is not None
            assert persisted_b.dense_ready_generation == 1
            assert persisted_b.dense_ready_fingerprint == "fingerprint-from-maintenance-spy"
            assert persisted_stale is not None
            assert persisted_stale.dense_ready_generation == 2
            assert persisted_stale.dense_ready_fingerprint == current_fingerprint
            assert persisted_active is not None
            assert persisted_active.dense_ready_generation == 0
            assert persisted_active.dense_ready_fingerprint is None
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_dense_reconcile_clears_tombstoned_and_stale_current_fingerprint_docs(monkeypatch) -> None:
    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings
    from app.documents.dense_maintenance_service import DenseMaintenanceService
    from app.rag.dense_contract import build_embedding_contract_fingerprint

    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-reconcile-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    dense_settings = _dense_settings()
    current_fingerprint = build_embedding_contract_fingerprint(dense_settings)
    spy = _DenseMaintenanceIndexSpy(result_fingerprint=current_fingerprint)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed_documents() -> None:
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-reconcile-stale",
                        filename="doc-reconcile-stale.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=3,
                        dense_ready_generation=2,
                        dense_ready_fingerprint=current_fingerprint,
                        next_generation=4,
                        latest_requested_generation=3,
                        uploaded_at=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                    ),
                    Document(
                        id="doc-reconcile-tombstoned",
                        filename="doc-reconcile-tombstoned.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=2,
                        dense_ready_generation=2,
                        dense_ready_fingerprint=current_fingerprint,
                        next_generation=3,
                        latest_requested_generation=2,
                        deleted_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
                        uploaded_at=datetime(2026, 4, 1, 0, 1, tzinfo=timezone.utc),
                    ),
                    Document(
                        id="doc-reconcile-keep",
                        filename="doc-reconcile-keep.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=1,
                        dense_ready_fingerprint=current_fingerprint,
                        next_generation=2,
                        latest_requested_generation=1,
                        uploaded_at=datetime(2026, 4, 1, 0, 2, tzinfo=timezone.utc),
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())
    asyncio.run(_seed_documents())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 0)
    monkeypatch.setattr(
        documents_api,
        "DenseMaintenanceService",
        lambda: DenseMaintenanceService(settings=dense_settings, dense_index_service=spy),
    )

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            drain_response = client.post("/api/v1/documents/ops/migration-drain", headers=headers)
            assert drain_response.status_code == 200

            response = client.post("/api/v1/documents/ops/dense-reconcile", headers=headers, json={"limit": 10})
            assert response.status_code == 200
            data = _extract_data(response.json())
            assert data["processed_documents"] == 2
            assert data["reconciled_documents"] == 2
            assert data["failed_documents"] == 0
            assert data["current_embedding_contract_fingerprint"] == current_fingerprint
            assert data["dense_mode_active"] is True
            assert [(item["document_id"], item["outcome"], item["reason"]) for item in data["documents"]] == [
                ("doc-reconcile-stale", "reconciled", None),
                ("doc-reconcile-tombstoned", "reconciled", None),
            ]
            assert spy.delete_calls == ["doc-reconcile-stale", "doc-reconcile-tombstoned"]

            persisted_stale = asyncio.run(_load_document_record(session_factory, document_id="doc-reconcile-stale"))
            persisted_tombstoned = asyncio.run(
                _load_document_record(session_factory, document_id="doc-reconcile-tombstoned")
            )
            persisted_keep = asyncio.run(_load_document_record(session_factory, document_id="doc-reconcile-keep"))
            assert persisted_stale is not None
            assert persisted_stale.dense_ready_generation == 0
            assert persisted_stale.dense_ready_fingerprint is None
            assert persisted_tombstoned is not None
            assert persisted_tombstoned.dense_ready_generation == 0
            assert persisted_tombstoned.dense_ready_fingerprint is None
            assert persisted_keep is not None
            assert persisted_keep.dense_ready_generation == 1
            assert persisted_keep.dense_ready_fingerprint == current_fingerprint
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_dense_backfill_and_dense_reconcile_return_migration_not_ready_when_drain_not_quiescent(
    monkeypatch,
) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-not-ready-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed_job() -> None:
        async with session_factory() as session:
            session.add(
                Document(
                    id="doc-dense-not-ready",
                    filename="doc-dense-not-ready.txt",
                    file_type="txt",
                    file_size=10,
                    status="pending",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    next_generation=2,
                    latest_requested_generation=1,
                )
            )
            session.add(
                DocumentJob(
                    id="job-dense-not-ready",
                    document_id="doc-dense-not-ready",
                    build_generation=2,
                    requested_chunk_strategy="general",
                    status="queued",
                    stage="queued",
                    progress=0,
                    message="queued",
                )
            )
            await session.commit()

    asyncio.run(_init_db())
    asyncio.run(_seed_job())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"
    monkeypatch.setattr(documents_api, "_get_active_dispatcher_tasks", lambda: 1)

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            drain_response = client.post("/api/v1/documents/ops/migration-drain", headers=headers)
            assert drain_response.status_code == 200

            backfill_response = client.post("/api/v1/documents/ops/dense-backfill", headers=headers, json={"limit": 5})
            assert backfill_response.status_code == 409
            assert backfill_response.json()["code"] == "DOC_MIGRATION_NOT_READY"

            reconcile_response = client.post(
                "/api/v1/documents/ops/dense-reconcile",
                headers=headers,
                json={"limit": 5},
            )
            assert reconcile_response.status_code == 409
            assert reconcile_response.json()["code"] == "DOC_MIGRATION_NOT_READY"
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_chat_lexical_retrieval_reads_only_published_generation_on_live_documents() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="chat-lexical-live-published-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed_documents() -> None:
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-live",
                        filename="live.txt",
                        file_type="txt",
                        file_size=11,
                        status="pending",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=2,
                        next_generation=4,
                        latest_requested_generation=3,
                    ),
                    Document(
                        id="doc-zero",
                        filename="zero.txt",
                        file_type="txt",
                        file_size=11,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=0,
                        published_generation=0,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    Document(
                        id="doc-deleted",
                        filename="deleted.txt",
                        file_type="txt",
                        file_size=11,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        next_generation=2,
                        latest_requested_generation=1,
                        deleted_at=datetime.now(timezone.utc),
                    ),
                ]
            )
            session.add_all(
                [
                    DocumentChunk(
                        id="chunk-live-old",
                        document_id="doc-live",
                        generation=1,
                        chunk_index=0,
                        content="retrieval sentinel old generation should stay hidden",
                    ),
                    DocumentChunk(
                        id="chunk-live-published",
                        document_id="doc-live",
                        generation=2,
                        chunk_index=0,
                        content="retrieval sentinel published generation stays visible",
                    ),
                    DocumentChunk(
                        id="chunk-live-candidate",
                        document_id="doc-live",
                        generation=3,
                        chunk_index=0,
                        content="retrieval sentinel candidate generation must stay hidden",
                    ),
                    DocumentChunk(
                        id="chunk-zero-published",
                        document_id="doc-zero",
                        generation=1,
                        chunk_index=0,
                        content="retrieval sentinel zero published generation must stay hidden",
                    ),
                    DocumentChunk(
                        id="chunk-deleted",
                        document_id="doc-deleted",
                        generation=1,
                        chunk_index=0,
                        content="retrieval sentinel tombstoned document must stay hidden",
                    ),
                ]
            )
            await session.commit()

    async def _retrieve_chunks() -> list[dict]:
        from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

        async with session_factory() as session:
            service = MixedModeDocumentRetrieverService(session)
            return (await service.retrieve("retrieval sentinel", top_k=10)).items

    asyncio.run(_init_db())

    try:
        asyncio.run(_seed_documents())
        retrieved = asyncio.run(_retrieve_chunks())
        assert [item["chunk_id"] for item in retrieved] == ["chunk-live-published"]
        assert retrieved[0]["document_id"] == "doc-live"
        assert "published generation stays visible" in retrieved[0]["content_preview"]
    finally:
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_enqueue_failure_compensation(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-enqueue-fail-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    original_enqueue_task = documents_api._enqueue_document_task
    fail_mode = {"enabled": True}

    async def _maybe_fail_enqueue_task(name: str, payload: dict):
        if fail_mode["enabled"]:
            raise RuntimeError("enqueue failed")
        return await original_enqueue_task(name=name, payload=payload)

    monkeypatch.setattr(documents_api, "_enqueue_document_task", _maybe_fail_enqueue_task)

    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            failed_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("retry.txt", b"content", "text/plain")},
            )
            assert failed_upload.status_code == 500
            assert failed_upload.json()["code"] == "INTERNAL_ERROR"

            empty_docs = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert empty_docs.status_code == 200
            assert _extract_data(empty_docs.json())["items"] == []

            fail_mode["enabled"] = False
            successful_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("retry.txt", b"content", "text/plain")},
            )
            assert successful_upload.status_code == 200
            upload_data = _extract_data(successful_upload.json())
            document_id = upload_data["document_id"]
            upload_job_id = upload_data["job_id"]
            assert _poll_job_until_terminal(client, headers=headers, job_id=upload_job_id)["status"] == "succeeded"

            fail_mode["enabled"] = True
            failed_build = client.post(
                f"/api/v1/documents/{document_id}/build",
                headers=headers,
                json={"chunk_strategy": "paper"},
            )
            assert failed_build.status_code == 500
            assert failed_build.json()["code"] == "INTERNAL_ERROR"

            docs_after_build_fail = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert docs_after_build_fail.status_code == 200
            doc_item = next(item for item in _extract_data(docs_after_build_fail.json())["items"] if item["document_id"] == document_id)
            assert doc_item["status"] == "ready"

            jobs_after_build_fail = client.get("/api/v1/documents/jobs?page=1&page_size=20", headers=headers)
            assert jobs_after_build_fail.status_code == 200
            build_failed_job = next(
                item
                for item in _extract_data(jobs_after_build_fail.json())["items"]
                if item["document_id"] == document_id and item["status"] == "failed"
            )
            assert build_failed_job["stage"] == "failed"

            failed_batch_build = client.post(
                "/api/v1/documents/batch-build",
                headers=headers,
                json={"document_ids": [document_id], "chunk_strategy": "qa"},
            )
            assert failed_batch_build.status_code == 500
            assert failed_batch_build.json()["code"] == "INTERNAL_ERROR"

            docs_after_batch_fail = client.get("/api/v1/documents?page=1&page_size=20", headers=headers)
            assert docs_after_batch_fail.status_code == 200
            doc_item = next(item for item in _extract_data(docs_after_batch_fail.json())["items"] if item["document_id"] == document_id)
            assert doc_item["status"] == "ready"
            assert doc_item["chunk_strategy"] == "general"

            chunks_after_batch_fail = client.get(
                f"/api/v1/documents/{document_id}/chunks?page=1&page_size=5",
                headers=headers,
            )
            assert chunks_after_batch_fail.status_code == 200

            jobs_after_batch_fail = client.get("/api/v1/documents/jobs?page=1&page_size=20", headers=headers)
            assert jobs_after_batch_fail.status_code == 200
            failed_jobs = [
                item
                for item in _extract_data(jobs_after_batch_fail.json())["items"]
                if item["document_id"] == document_id and item["status"] == "failed"
            ]
            assert len(failed_jobs) >= 2

            fail_mode["enabled"] = False
            tombstone_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("demo.txt", b"initial body", "text/plain")},
            )
            assert tombstone_upload.status_code == 200
            tombstone_upload_data = _extract_data(tombstone_upload.json())
            tombstone_document_id = tombstone_upload_data["document_id"]
            assert _poll_job_until_terminal(client, headers=headers, job_id=tombstone_upload_data["job_id"])["status"] == "succeeded"

            asyncio.run(_tombstone_document_record(session_factory, document_id=tombstone_document_id))

            reupload_after_tombstone = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("demo.txt", b"replacement body", "text/plain")},
            )
            assert reupload_after_tombstone.status_code == 200
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_batch_enqueue_failure_compensates_all_items(monkeypatch) -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-batch-enqueue-fail-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    from app.api.v1 import documents as documents_api
    from app.common.config import get_settings

    original_enqueue_task = documents_api._enqueue_document_task
    original_enqueue_runner = documents_api._enqueue_document_runner
    settings = get_settings()
    original_code = settings.admin_invite_code
    settings.admin_invite_code = "test-admin-code"

    try:
        with TestClient(app) as client:
            token = asyncio.run(_create_admin_token(client))
            headers = _admin_headers(token)

            first_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("batch-a.txt", b"alpha", "text/plain")},
            )
            assert first_upload.status_code == 200
            first_data = _extract_data(first_upload.json())
            first_document_id = first_data["document_id"]
            assert _poll_job_until_terminal(client, headers=headers, job_id=first_data["job_id"])["status"] == "succeeded"

            second_upload = client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("batch-b.txt", b"beta", "text/plain")},
            )
            assert second_upload.status_code == 200
            second_data = _extract_data(second_upload.json())
            second_document_id = second_data["document_id"]
            assert _poll_job_until_terminal(client, headers=headers, job_id=second_data["job_id"])["status"] == "succeeded"

            first_doc_before = _get_document_item(client, headers=headers, document_id=first_document_id)
            second_doc_before = _get_document_item(client, headers=headers, document_id=second_document_id)
            first_chunks_before = _get_chunk_snapshot(client, headers=headers, document_id=first_document_id)
            second_chunks_before = _get_chunk_snapshot(client, headers=headers, document_id=second_document_id)

            jobs_before_batch = client.get("/api/v1/documents/jobs?page=1&page_size=50", headers=headers)
            assert jobs_before_batch.status_code == 200
            jobs_before_ids = {item["job_id"] for item in _extract_data(jobs_before_batch.json())["items"]}

            enqueue_calls = {"count": 0}
            submitted_runner_job_ids: list[str] = []

            async def _fail_on_second_enqueue_task(name: str, payload: dict):
                enqueue_calls["count"] += 1
                if enqueue_calls["count"] == 2:
                    # Keep a real race window open: vulnerable code could let item-1 finish here.
                    await asyncio.sleep(0.1)
                    raise RuntimeError("enqueue failed on second")
                return await original_enqueue_task(name=name, payload=payload)

            def _track_enqueue_runner(job_id: str, runner):
                submitted_runner_job_ids.append(job_id)
                return original_enqueue_runner(job_id, runner)

            monkeypatch.setattr(documents_api, "_enqueue_document_task", _fail_on_second_enqueue_task)
            monkeypatch.setattr(documents_api, "_enqueue_document_runner", _track_enqueue_runner)

            batch_response = client.post(
                "/api/v1/documents/batch-build",
                headers=headers,
                json={"document_ids": [first_document_id, second_document_id], "chunk_strategy": "qa"},
            )
            assert batch_response.status_code == 500
            assert batch_response.json()["code"] == "INTERNAL_ERROR"
            assert enqueue_calls["count"] == 2
            assert submitted_runner_job_ids == []

            first_doc_after = _get_document_item(client, headers=headers, document_id=first_document_id)
            second_doc_after = _get_document_item(client, headers=headers, document_id=second_document_id)
            assert first_doc_after["status"] == first_doc_before["status"]
            assert first_doc_after["chunk_strategy"] == first_doc_before["chunk_strategy"]
            assert second_doc_after["status"] == second_doc_before["status"]
            assert second_doc_after["chunk_strategy"] == second_doc_before["chunk_strategy"]
            assert _get_chunk_snapshot(client, headers=headers, document_id=first_document_id) == first_chunks_before
            assert _get_chunk_snapshot(client, headers=headers, document_id=second_document_id) == second_chunks_before

            jobs_response = client.get("/api/v1/documents/jobs?page=1&page_size=50", headers=headers)
            assert jobs_response.status_code == 200
            jobs_items = _extract_data(jobs_response.json())["items"]
            target_docs = {first_document_id, second_document_id}
            created_batch_jobs = [
                item for item in jobs_items if item["job_id"] not in jobs_before_ids and item["document_id"] in target_docs
            ]

            assert len(created_batch_jobs) == 2
            assert not any(item["status"] in {"queued", "running"} for item in created_batch_jobs)
            assert not any(item["status"] == "succeeded" for item in created_batch_jobs)
            assert all(
                item["status"] in {"failed", "canceled"} and item["message"] == "failed to enqueue document build"
                for item in created_batch_jobs
            )
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_requires_admin_role() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="documents-auth-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    try:
        with TestClient(app) as client:
            register_response = client.post(
                "/api/v1/auth/register",
                json={"username": "user1", "password": "secret-123", "role": "user"},
            )
            assert register_response.status_code == 200
            token = register_response.json()["data"]["access_token"]
            headers = _admin_headers(token)

            response = client.get("/api/v1/documents", headers=headers)
            assert response.status_code == 403
            body = response.json()
            assert body["code"] == "AUTH_FORBIDDEN"
            assert "request_id" in body
    finally:
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
        os.remove(db_path)


def test_documents_dense_and_lexical_retrieval_respects_published_generation_visibility() -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-read-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())
    asyncio.run(_seed_mixed_mode_retrieval_documents(session_factory, fingerprint=fingerprint))

    dense_row = {
        "document_id": "doc-dense-published",
        "generation": 1,
        "chunk_index": 0,
        "content_sha256": hashlib.sha256("alpha dense published evidence".encode("utf-8")).hexdigest(),
        "distance": 0.98,
    }
    stale_dense_row = {
        "document_id": "doc-superseded",
        "generation": 2,
        "chunk_index": 0,
        "content_sha256": hashlib.sha256(
            "alpha superseded dense candidate should stay hidden".encode("utf-8")
        ).hexdigest(),
        "distance": 0.97,
    }

    async def _run() -> dict:
        async with session_factory() as session:
            service = MixedModeDocumentRetrieverService(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(rows=[dense_row, stale_dense_row]),
            )
            result = await service.retrieve("alpha beta", top_k=5)
            assert "doc-tombstoned" not in {item["document_id"] for item in result.items}
            return {
                "items": result.items,
                "strategy": result.strategy,
                "dense_candidate_count": result.dense_candidate_count,
                "dense_hydrated_count": result.dense_hydrated_count,
                "lexical_candidate_count": result.lexical_candidate_count,
                "merged_count": result.merged_count,
                "dense_query_failed": result.dense_query_failed,
                "lexical_scope": result.lexical_scope,
            }

    result = asyncio.run(_run())

    assert result["strategy"] == "dense_plus_lexical_migration"
    assert result["dense_candidate_count"] == 2
    assert result["dense_hydrated_count"] == 1
    assert result["lexical_candidate_count"] == 1
    assert result["merged_count"] == 2
    assert result["dense_query_failed"] is False
    assert result["lexical_scope"] == "not_dense_ready_published"
    assert [(item["document_id"], item["retrieval_source"]) for item in result["items"]] == [
        ("doc-dense-published", "dense"),
        ("doc-lexical-published", "lexical"),
    ]

    asyncio.run(db_engine.dispose())
    os.remove(db_path)


def test_documents_dense_query_failure_falls_back_to_full_published_lexical_corpus() -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-fallback-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())
    asyncio.run(_seed_mixed_mode_retrieval_documents(session_factory, fingerprint=fingerprint))

    async def _run() -> dict:
        async with session_factory() as session:
            service = MixedModeDocumentRetrieverService(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(error=RuntimeError("milvus unavailable")),
            )
            result = await service.retrieve("alpha beta", top_k=5)
            return {
                "items": result.items,
                "strategy": result.strategy,
                "dense_candidate_count": result.dense_candidate_count,
                "dense_hydrated_count": result.dense_hydrated_count,
                "lexical_candidate_count": result.lexical_candidate_count,
                "merged_count": result.merged_count,
                "dense_query_failed": result.dense_query_failed,
                "lexical_scope": result.lexical_scope,
            }

    result = asyncio.run(_run())

    assert result["strategy"] == "dense_plus_lexical_migration"
    assert result["dense_candidate_count"] == 0
    assert result["dense_hydrated_count"] == 0
    assert result["lexical_candidate_count"] == 2
    assert result["merged_count"] == 2
    assert result["dense_query_failed"] is True
    assert result["lexical_scope"] == "full_published_live"
    assert {item["document_id"] for item in result["items"]} == {
        "doc-dense-published",
        "doc-lexical-published",
    }
    assert {item["retrieval_source"] for item in result["items"]} == {"lexical"}

    asyncio.run(db_engine.dispose())
    os.remove(db_path)


def test_documents_dense_query_failure_searches_full_published_live_corpus_beyond_candidate_limit() -> None:
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    lexical_chunks = [
        DocumentChunk(
            id=f"chunk-filler-{idx:03d}",
            document_id=f"doc-filler-{idx:03d}",
            generation=1,
            chunk_index=0,
            content=f"aaaaa bbbbb ccccc {idx:03d}",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "filler"},
        )
        for idx in range(205)
    ]
    lexical_chunks.append(
        DocumentChunk(
            id="chunk-dense-tail-match",
            document_id="doc-dense-tail-match",
            generation=1,
            chunk_index=0,
            content="xqvzjk dense-ready tail evidence",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "tail-match"},
        )
    )
    session = _LimitAwareLexicalSession(lexical_chunks)

    async def _run() -> dict:
        service = MixedModeDocumentRetrieverService(
            session,
            settings=settings,
            embedding_provider=_StubEmbeddingProvider(),
            document_index=_FakeDenseDocumentIndex(error=RuntimeError("milvus unavailable")),
        )
        result = await service.retrieve("xqvzjk", top_k=5)
        return {
            "items": result.items,
            "lexical_candidate_count": result.lexical_candidate_count,
            "dense_query_failed": result.dense_query_failed,
            "lexical_scope": result.lexical_scope,
        }

    result = asyncio.run(_run())

    assert result["dense_query_failed"] is True
    assert result["lexical_scope"] == "full_published_live"
    assert session.limit_history == [None]
    assert result["lexical_candidate_count"] == 1
    assert [(item["document_id"], item["retrieval_source"]) for item in result["items"]] == [
        ("doc-dense-tail-match", "lexical")
    ]


def test_documents_sparse_only_searches_full_published_live_corpus_beyond_candidate_limit() -> None:
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    lexical_chunks = [
        DocumentChunk(
            id=f"chunk-filler-{idx:03d}",
            document_id=f"doc-filler-{idx:03d}",
            generation=1,
            chunk_index=0,
            content=f"aaaaa bbbbb ccccc {idx:03d}",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "filler"},
        )
        for idx in range(205)
    ]
    lexical_chunks.append(
        DocumentChunk(
            id="chunk-sparse-tail-match",
            document_id="doc-sparse-tail-match",
            generation=1,
            chunk_index=0,
            content="xqvzjk sparse-only tail evidence",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "tail-match"},
        )
    )
    session = _LimitAwareLexicalSession(lexical_chunks)

    async def _run() -> dict:
        service = MixedModeDocumentRetrieverService(session)
        result = await service.retrieve("xqvzjk", top_k=5)
        return {
            "items": result.items,
            "strategy": result.strategy,
            "lexical_candidate_count": result.lexical_candidate_count,
            "lexical_scope": result.lexical_scope,
        }

    result = asyncio.run(_run())

    assert result["strategy"] == "sparse_only"
    assert result["lexical_scope"] == "full_published_live"
    assert session.limit_history == [None]
    assert result["lexical_candidate_count"] == 1
    assert [(item["document_id"], item["retrieval_source"]) for item in result["items"]] == [
        ("doc-sparse-tail-match", "lexical")
    ]


def test_documents_dense_success_keeps_lexical_fallback_scoped_to_not_dense_ready_subset() -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_fd, db_path = tempfile.mkstemp(prefix="documents-dense-narrow-scope-", suffix=".db")
    os.close(db_fd)
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed() -> None:
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-dense-hit",
                        filename="dense-hit.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=1,
                        dense_ready_fingerprint=fingerprint,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    Document(
                        id="doc-not-dense-tail",
                        filename="not-dense-tail.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=0,
                        dense_ready_fingerprint=None,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    Document(
                        id="doc-dense-tail",
                        filename="dense-tail.txt",
                        file_type="txt",
                        file_size=10,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=1,
                        dense_ready_fingerprint=fingerprint,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    DocumentChunk(
                        id="chunk-dense-hit",
                        document_id="doc-dense-hit",
                        generation=1,
                        chunk_index=0,
                        content="anchor dense hit",
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={"source": "dense"},
                    ),
                    DocumentChunk(
                        id="chunk-not-dense-tail",
                        document_id="doc-not-dense-tail",
                        generation=1,
                        chunk_index=0,
                        content="xqvzjk not dense ready tail evidence",
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={"source": "not-dense"},
                    ),
                    DocumentChunk(
                        id="chunk-dense-tail",
                        document_id="doc-dense-tail",
                        generation=1,
                        chunk_index=0,
                        content="xqvzjk dense ready tail evidence",
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={"source": "dense-ready"},
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())
    asyncio.run(_seed())

    dense_row = {
        "document_id": "doc-dense-hit",
        "generation": 1,
        "chunk_index": 0,
        "content_sha256": hashlib.sha256("anchor dense hit".encode("utf-8")).hexdigest(),
        "distance": 0.99,
    }

    async def _run() -> dict:
        async with session_factory() as session:
            service = MixedModeDocumentRetrieverService(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(rows=[dense_row]),
            )
            result = await service.retrieve("xqvzjk", top_k=5)
            return {
                "items": result.items,
                "lexical_candidate_count": result.lexical_candidate_count,
                "lexical_scope": result.lexical_scope,
            }

    result = asyncio.run(_run())

    assert result["lexical_scope"] == "not_dense_ready_published"
    assert result["lexical_candidate_count"] == 1
    assert [item["document_id"] for item in result["items"]] == ["doc-dense-hit", "doc-not-dense-tail"]

    asyncio.run(db_engine.dispose())
    os.remove(db_path)
