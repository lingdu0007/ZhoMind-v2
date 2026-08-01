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
        self.request_details: list[tuple[str, str, dict]] = []
        self._job_polls = 0

    async def request(self, method: str, path: str, **kwargs) -> HttpResponse:
        self.calls.append((method, path))
        self.request_details.append((method, path, kwargs))
        if path == "/api/v1/health":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"status": "up"}})
        if path == "/api/v1/auth/login":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"access_token": "login-token"}})
        if path == "/api/v1/documents/upload":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"document_id": "doc-ingested", "job_id": "job-1"}})
        if path == "/api/v1/documents/jobs/job-1":
            self._job_polls += 1
            status = "running" if self._job_polls == 1 else "succeeded"
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"status": status, "stage": "completed"}})
        if path == "/api/v1/documents/doc-ingested/publish":
            return HttpResponse(
                status_code=200,
                payload={"code": "OK", "data": {"document_id": "doc-ingested", "status": "ready", "published_generation": 1}},
            )
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


class _GenerationFakeHttpClient(_FakeHttpClient):
    async def request(self, method: str, path: str, **kwargs) -> HttpResponse:
        if path == "/api/v1/chat":
            self.calls.append((method, path))
            self.request_details.append((method, path, kwargs))
            return HttpResponse(
                status_code=200,
                payload={
                    "code": "OK",
                    "data": {
                        "answer": "基于已发布来源的回答",
                        "message": {
                            "evidence_summary": {
                                "coverage": "sufficient",
                                "source_count": 1,
                                "sources": [
                                    {
                                        "source_id": "chunk-1",
                                        "metadata": {"title": "smoke.md", "publication_version": "v1"},
                                        "excerpt": "retrieval evidence excerpt",
                                    }
                                ],
                            }
                        },
                    },
                },
            )
        if path == "/api/v1/chat/stream":
            self.calls.append((method, path))
            self.request_details.append((method, path, kwargs))
            return HttpResponse(
                status_code=200,
                payload={},
                body=(
                    'event: content\\ndata: {"content": "基于已发布来源的回答"}\\n\\n'
                    'event: evidence_summary\\ndata: {"evidence_summary": {"coverage": "sufficient"}}\\n\\n'
                    "event: done\\ndata: [DONE]\\n\\n"
                ),
            )
        return await super().request(method, path, **kwargs)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "JWT_SECRET": "test-jwt-value",
        "BOOTSTRAP_ADMIN_USERNAME": "test-bootstrap-admin",
        "BOOTSTRAP_ADMIN_PASSWORD": "test-bootstrap-password",
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
    assert manifest["checks"]["document_publication"] == {
        "document_id": "doc-ingested",
        "published_generation": 1,
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
    assert "test-bootstrap-password" not in serialized
    assert "test-jwt-value" not in serialized


def test_retrieval_evidence_smoke_fails_safely_for_incomplete_runtime_configuration(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(BOOTSTRAP_ADMIN_USERNAME="", EMBEDDING_API_KEY=""),
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
    assert set(manifest["missing_configuration"]) == {"BOOTSTRAP_ADMIN_USERNAME", "EMBEDDING_API_KEY"}
    assert "test-qwen-api-key" not in (tmp_path / "run-002" / "manifest.json").read_text(encoding="utf-8")


def test_generation_smoke_proves_cited_normal_and_streaming_chat_without_recording_content_or_secrets(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ARK_API_KEY="test-ark-api-key", BASE_URL="https://llm.example.test/v1", MODEL="test-model"),
            http_client=_GenerationFakeHttpClient(),
            retrieve=lambda query: _return_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="generation-run-001",
            include_generation=True,
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run()

    manifest = asyncio.run(_run())

    assert manifest["outcome"] == "passed"
    assert manifest["command"] == "retrieval-evidence generation-smoke"
    assert manifest["checks"]["chat_model"] == {
        "invoked": True,
        "normal_contract": "passed",
        "stream_contract": "passed",
        "citation_source_count": 1,
    }
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "test-ark-api-key" not in serialized
    assert "https://llm.example.test/v1" not in serialized
    assert "基于已发布来源的回答" not in serialized
    assert "retrieval evidence excerpt" not in serialized


def test_generation_smoke_uses_a_natural_language_question_for_the_published_fixture(tmp_path) -> None:
    async def _run() -> tuple[dict, _GenerationFakeHttpClient]:
        client = _GenerationFakeHttpClient()
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ARK_API_KEY="test-ark-api-key", BASE_URL="https://llm.example.test/v1", MODEL="test-model"),
            http_client=client,
            retrieve=lambda query: _return_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="generation-run-002",
            include_generation=True,
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run(), client

    manifest, client = asyncio.run(_run())

    assert manifest["outcome"] == "passed"
    upload = next(details for method, path, details in client.request_details if (method, path) == ("POST", "/api/v1/documents/upload"))
    filename, content = upload["upload"]
    assert filename.endswith(".md")
    assert "已发布知识版本" in content.decode("utf-8")

    normal_chat = next(details for method, path, details in client.request_details if (method, path) == ("POST", "/api/v1/chat"))
    question = normal_chat["json_body"]["message"]
    assert question == "生成带引用的回答应依据哪个知识版本？"
    assert "retrieval-evidence-" not in question


def test_retrieval_evidence_production_run_ids_do_not_create_administrators(tmp_path) -> None:
    async def _run() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        first_client = _FakeHttpClient()
        second_client = _FakeHttpClient()
        for client, run_id in (
            (first_client, "production-20260731T173054Z-101-1"),
            (second_client, "production-20260731T173054Z-102-2"),
        ):
            runner = RetrievalEvidenceSmoke(
                settings=_settings(),
                http_client=client,
                retrieve=lambda query: _return_dense_result(query),
                output_dir=tmp_path,
                source_revision="abc123",
                run_id=run_id,
                now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
                sleep=lambda _: _return_none(),
            )
            manifest = await runner.run()
            assert manifest["outcome"] == "passed"
        return first_client.calls, second_client.calls

    first_calls, second_calls = asyncio.run(_run())

    assert ("POST", "/api/v1/auth/register") not in first_calls
    assert ("POST", "/api/v1/auth/register") not in second_calls
    assert ("POST", "/api/v1/auth/login") in first_calls
    assert ("POST", "/api/v1/auth/login") in second_calls


async def _return_dense_result(query: str) -> _DenseResult:
    assert query == "生成带引用的回答应依据哪个知识版本？"
    return _DenseResult()


async def _return_none() -> None:
    return None
