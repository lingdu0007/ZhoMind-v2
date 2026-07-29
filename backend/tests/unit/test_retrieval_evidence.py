from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from app.common.config import Settings
from app.rag.dense_contract import build_embedding_contract_fingerprint
from app.retrieval_evidence import HttpResponse, RetrievalEvidenceSmoke


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._job_polls = 0

    async def request(self, method: str, path: str, **kwargs) -> HttpResponse:
        self.calls.append((method, path))
        if path == "/api/v1/health":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"status": "up"}})
        if path == "/api/v1/auth/register":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"username": "smoke-admin"}})
        if path == "/api/v1/auth/login":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"access_token": "login-token"}})
        if path == "/api/v1/documents/upload":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"document_id": "doc-ingested", "job_id": "job-1"}})
        if path == "/api/v1/documents/jobs/job-1":
            self._job_polls += 1
            status = "running" if self._job_polls == 1 else "succeeded"
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"status": status, "stage": "completed"}})
        if path == "/api/v1/documents/doc-ingested/chunks?page=1&page_size=1":
            return HttpResponse(
                status_code=200,
                payload={"code": "OK", "data": {"items": [{"chunk_id": "chunk-1"}], "pagination": {"total": 1}}},
            )
        raise AssertionError(f"unexpected request: {method} {path}")


class _DenseResult:
    dense_candidate_count = 1
    dense_hydrated_count = 1
    dense_query_failed = False
    items = [{"document_id": "doc-ingested", "chunk_id": "chunk-1", "retrieval_source": "dense"}]


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "JWT_SECRET": "test-jwt-value",
        "ADMIN_INVITE_CODE": "test-admin-code",
        "EMBEDDING_API_KEY": "test-qwen-api-key",
        "EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
        "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
        "DENSE_EMBEDDING_DIM": 1024,
        "MILVUS_URI": "http://milvus:19530",
    }
    values.update(overrides)
    return Settings(**values)


def test_retrieval_evidence_smoke_writes_non_sensitive_success_manifest(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(),
            http_client=_FakeHttpClient(),
            retrieve=lambda query: _return_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="run-001",
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run()

    manifest = asyncio.run(_run())

    assert manifest["outcome"] == "passed"
    assert manifest["checks"]["document_build"] == {
        "document_id": "doc-ingested",
        "job_id": "job-1",
        "status": "succeeded",
        "chunk_count": 1,
    }
    assert manifest["checks"]["live_embedding"] == {
        "provider": "qwen",
        "status": "completed",
        "embedding_contract_fingerprint": build_embedding_contract_fingerprint(_settings()),
    }
    assert manifest["checks"]["indexing"] == {"dense_candidate_count": 1, "dense_hydrated_count": 1}
    assert manifest["checks"]["retrieval"] == {
        "candidate_document_id": "doc-ingested",
        "candidate_chunk_id": "chunk-1",
        "candidate_belongs_to_ingested_document": True,
    }
    assert manifest["checks"]["chat_model"] == {"invoked": False}

    persisted = json.loads((tmp_path / "run-001" / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest
    serialized = json.dumps(manifest)
    assert "test-qwen-api-key" not in serialized
    assert "https://embedding.example.test/v1" not in serialized
    assert "Qwen/Qwen3-Embedding-8B" not in serialized
    assert "test-admin-code" not in serialized
    assert "test-jwt-value" not in serialized


def test_retrieval_evidence_smoke_fails_safely_for_incomplete_runtime_configuration(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ADMIN_INVITE_CODE="", EMBEDDING_API_KEY=""),
            http_client=_FakeHttpClient(),
            retrieve=lambda query: _return_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="run-002",
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run()

    manifest = asyncio.run(_run())

    assert manifest["outcome"] == "failed"
    assert manifest["failed_check"] == "runtime_configuration"
    assert set(manifest["missing_configuration"]) == {"ADMIN_INVITE_CODE", "EMBEDDING_API_KEY"}
    assert "test-qwen-api-key" not in (tmp_path / "run-002" / "manifest.json").read_text(encoding="utf-8")


async def _return_dense_result(query: str) -> _DenseResult:
    assert query.startswith("retrieval-evidence-")
    return _DenseResult()


async def _return_none() -> None:
    return None
