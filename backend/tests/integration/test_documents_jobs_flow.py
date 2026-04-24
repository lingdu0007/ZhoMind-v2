import asyncio
from collections.abc import Generator
from datetime import datetime, timezone
import os
import tempfile
import threading
import time

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.document import Document


class _InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = dict(mapping)
        return len(mapping)

    async def expire(self, key: str, ttl: int) -> bool:
        return key in self._store

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0


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


async def _tombstone_document_record(session_factory, *, document_id: str) -> None:
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        assert document is not None
        document.deleted_at = datetime.now(timezone.utc)
        document.status = "pending"
        document.latest_requested_generation = document.published_generation
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
            hidden_docs_items = _extract_data(hidden_docs.json())["items"]
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
            assert tombstoned_document.status == "pending"
            assert tombstoned_document.latest_requested_generation == tombstoned_document.published_generation
    finally:
        settings.admin_invite_code = original_code
        app.dependency_overrides.clear()
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
