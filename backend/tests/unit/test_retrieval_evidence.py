from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from app.common.config import Settings
from app.rag.dense_contract import build_embedding_contract_fingerprint
from app.retrieval_evidence import HttpResponse, RetrievalEvidenceSmoke, UrllibHttpClient


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
        if path == "/api/v1/members/invitations":
            return HttpResponse(
                status_code=200,
                payload={"code": "OK", "data": {"id": "invitation-1", "invitation_code": "team-invitation-code"}},
            )
        if path == "/api/v1/auth/register":
            return HttpResponse(status_code=200, payload={"code": "OK", "data": {"access_token": "knowledge-user-token"}})
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


class _ExistingPublishedDenseResult:
    dense_candidate_count = 1
    dense_hydrated_count = 1
    dense_query_failed = False
    items = [{"document_id": "doc-existing", "chunk_id": "chunk-1", "retrieval_source": "dense"}]


class _DifferentDiagnosticChunkResult:
    dense_candidate_count = 1
    dense_hydrated_count = 1
    dense_query_failed = False
    items = [{"document_id": "doc-ingested", "chunk_id": "diagnostic-chunk", "retrieval_source": "dense"}]


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
                        "outcome": "evidence_gated_answer",
                        "answer": "基于已发布来源的回答",
                        "message": {
                            "outcome": "evidence_gated_answer",
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
                    'event: outcome\ndata: {"outcome": "evidence_gated_answer"}\n\n'
                    'event: content\ndata: {"content": "基于已发布来源的回答"}\n\n'
                    'event: evidence_summary\ndata: {"evidence_summary": {"coverage": "sufficient", "source_count": 1, "sources": [{"source_id": "chunk-1", "metadata": {"title": "smoke.md", "publication_version": "v1"}, "excerpt": "retrieval evidence excerpt"}]}}\n\n'
                    "event: done\ndata: [DONE]\n\n"
                ),
            )
        if path.startswith("/api/v1/sessions/generation-"):
            self.calls.append((method, path))
            self.request_details.append((method, path, kwargs))
            evidence_summary = {
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
            return HttpResponse(
                status_code=200,
                payload={
                    "code": "OK",
                    "data": {
                        "messages": [
                            {"type": "assistant", "outcome": "evidence_gated_answer", "evidence_summary": evidence_summary},
                            {"type": "assistant", "outcome": "evidence_gated_answer", "evidence_summary": evidence_summary},
                        ]
                    },
                },
            )
        return await super().request(method, path, **kwargs)


class _StreamCitationMissingFakeHttpClient(_GenerationFakeHttpClient):
    async def request(self, method: str, path: str, **kwargs) -> HttpResponse:
        if path == "/api/v1/chat/stream":
            self.calls.append((method, path))
            self.request_details.append((method, path, kwargs))
            return HttpResponse(
                status_code=200,
                payload={},
                body=(
                    'event: outcome\ndata: {"outcome": "evidence_gated_answer"}\n\n'
                    'event: content\ndata: {"content": "基于已发布来源的回答"}\n\n'
                    'event: evidence_summary\ndata: {"evidence_summary": {"coverage": "sufficient", "source_count": 0, "sources": []}}\n\n'
                    "event: done\ndata: [DONE]\n\n"
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


def test_generation_smoke_rejects_a_dense_candidate_from_another_published_document(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ARK_API_KEY="test-ark-api-key", BASE_URL="https://llm.example.test/v1", MODEL="test-model"),
            http_client=_GenerationFakeHttpClient(),
            retrieve=lambda query: _return_existing_published_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="generation-run-existing-published-source",
            include_generation=True,
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run()

    manifest = asyncio.run(_run())

    assert manifest["outcome"] == "failed"
    assert manifest["failed_check"] == "retrieval"
    assert manifest["failure_code"] == "INGESTED_DOCUMENT_NOT_RETRIEVED"


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
        "history_contract": "passed",
        "citation_source_count": 1,
    }
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "test-ark-api-key" not in serialized
    assert "https://llm.example.test/v1" not in serialized
    assert "基于已发布来源的回答" not in serialized
    assert "retrieval evidence excerpt" not in serialized


def test_generation_smoke_does_not_use_direct_retrieval_as_a_citation_oracle(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ARK_API_KEY="test-ark-api-key", BASE_URL="https://llm.example.test/v1", MODEL="test-model"),
            http_client=_GenerationFakeHttpClient(),
            retrieve=lambda query: _return_different_diagnostic_chunk(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="generation-run-independent-chat-seam",
            include_generation=True,
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run()

    manifest = asyncio.run(_run())

    assert manifest["outcome"] == "passed"
    assert manifest["checks"]["retrieval"]["candidate_chunk_id"] == "diagnostic-chunk"
    assert manifest["checks"]["chat_model"]["history_contract"] == "passed"


def test_generation_smoke_admits_a_knowledge_user_before_calling_chat(tmp_path) -> None:
    async def _run() -> tuple[dict, _GenerationFakeHttpClient]:
        client = _GenerationFakeHttpClient()
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ARK_API_KEY="test-ark-api-key", BASE_URL="https://llm.example.test/v1", MODEL="test-model"),
            http_client=client,
            retrieve=lambda query: _return_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="generation-run-knowledge-user",
            include_generation=True,
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run(), client

    manifest, client = asyncio.run(_run())

    assert manifest["outcome"] == "passed"
    invitation = next(details for method, path, details in client.request_details if (method, path) == ("POST", "/api/v1/members/invitations"))
    assert invitation["headers"] == {"Authorization": "Bearer login-token"}
    registration = next(details for method, path, details in client.request_details if (method, path) == ("POST", "/api/v1/auth/register"))
    assert registration["json_body"]["invitation_code"] == "team-invitation-code"
    assert registration["json_body"]["username"].startswith("generation-smoke-")
    assert not registration["json_body"]["password"].startswith("generation-smoke-")
    assert registration["json_body"]["password"] not in json.dumps(manifest)
    for method, path, details in client.request_details:
        if (method, path) in {("POST", "/api/v1/chat"), ("POST", "/api/v1/chat/stream")}:
            assert details["headers"] == {"Authorization": "Bearer knowledge-user-token"}


def test_production_acceptance_rejects_a_missing_generation_configuration(tmp_path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "deploy" / "production" / "acceptance.sh"
    app_directory = tmp_path / "app"
    fake_bin = tmp_path / "bin"
    app_directory.mkdir()
    fake_bin.mkdir()
    (app_directory / ".env").write_text(
        "\n".join(
            (
                "BOOTSTRAP_ADMIN_USERNAME=test-admin",
                "BOOTSTRAP_ADMIN_PASSWORD=test-password",
                "EMBEDDING_API_KEY=test-embedding-key",
                "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
                "EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B",
                "DENSE_EMBEDDING_DIM=1024",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "sudo",
        "#!/usr/bin/env bash\nshift\nexec \"$@\"\n",
    )
    _write_executable(
        fake_bin / "docker",
        "#!/usr/bin/env bash\n"
        "if [[ \" $* \" == *\" ps \"* ]]; then\n"
        "  printf '%s\\n' postgres redis etcd minio milvus backend caddy\n"
        "fi\n",
    )
    _write_executable(fake_bin / "openssl", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\n"
        "url=${!#}\n"
        "case \"$url\" in\n"
        "  */api/health) printf '%s' '{\"data\":{\"status\":\"up\"}}' ;;\n"
        "  */api/auth/login) printf '%s' '{\"data\":{\"access_token\":\"test-token\"}}' ;;\n"
        "  */api/auth/me) printf '%s' '{\"data\":{\"role\":\"admin\"}}' ;;\n"
        "  *) printf '%s' '<div id=\"app\"></div>' ;;\n"
        "esac\n",
    )

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        check=False,
        env={
            "DEPLOY_APP_DIR": str(app_directory),
            "DEPLOY_CADDY_SITE_ADDRESS": "zhomind.example.test",
            "PATH": f"{fake_bin}:{Path('/usr/bin')}:{Path('/bin')}",
            "SOURCE_REVISION": "test-revision",
        },
        text=True,
    )

    assert result.returncode != 0
    assert "live Generation Smoke requires complete approved-provider configuration" in result.stderr


def test_production_acceptance_closes_compose_run_stdin() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = (repository_root / "deploy" / "production" / "acceptance.sh").read_text(encoding="utf-8")

    assert script.count(">/dev/null </dev/null") == 2
    assert "generation-smoke \\\n" in script
    assert "--timeout-seconds 420" in script


def test_retrieval_evidence_http_client_normalizes_timeouts(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr("app.retrieval_evidence.urlopen", raise_timeout)
    client = UrllibHttpClient(base_url="http://backend:8000", timeout_seconds=1)

    response = asyncio.run(client.request("GET", "/api/v1/health"))

    assert response.status_code == 0
    assert response.payload == {}


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


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
    assert "蓝松石版本" in content.decode("utf-8")
    assert "2026年7月30日0时00分00秒" in content.decode("utf-8")
    assert "银杏晨雾云雀验收" in content.decode("utf-8")

    normal_chat = next(details for method, path, details in client.request_details if (method, path) == ("POST", "/api/v1/chat"))
    question = normal_chat["json_body"]["message"]
    assert question == "根据银杏晨雾云雀验收（2026年7月30日0时00分00秒）的验收事实，蓝松石版本在生成带引用回答时具有什么作用？"
    assert "retrieval-evidence-" not in question


def test_generation_smoke_fails_when_the_stream_omits_the_cited_source(tmp_path) -> None:
    async def _run() -> dict:
        runner = RetrievalEvidenceSmoke(
            settings=_settings(ARK_API_KEY="test-ark-api-key", BASE_URL="https://llm.example.test/v1", MODEL="test-model"),
            http_client=_StreamCitationMissingFakeHttpClient(),
            retrieve=lambda query: _return_dense_result(query),
            output_dir=tmp_path,
            source_revision="abc123",
            run_id="generation-run-003",
            include_generation=True,
            now=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
            sleep=lambda _: _return_none(),
        )
        return await runner.run()

    manifest = asyncio.run(_run())

    assert manifest["outcome"] == "failed"
    assert manifest["failed_check"] == "generation_stream"
    assert manifest["failure_code"] == "STREAM_CITATION_MISSING"


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
    assert "蓝松石版本" in query
    assert "retrieval-evidence-" not in query
    return _DenseResult()


async def _return_existing_published_dense_result(query: str) -> _ExistingPublishedDenseResult:
    assert "蓝松石版本" in query
    assert "retrieval-evidence-" not in query
    return _ExistingPublishedDenseResult()


async def _return_different_diagnostic_chunk(query: str) -> _DifferentDiagnosticChunkResult:
    assert "蓝松石版本" in query
    return _DifferentDiagnosticChunkResult()


async def _return_none() -> None:
    return None
