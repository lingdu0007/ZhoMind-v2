from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import Settings, get_settings
from app.rag.dense_contract import build_embedding_contract_fingerprint
from app.rag.interfaces import RetrieveResult
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

_POLL_ATTEMPTS = 120
_POLL_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    payload: Mapping[str, Any]
    body: str = ""


class EvidenceHttpClient(Protocol):
    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        upload: tuple[str, bytes] | None = None,
    ) -> HttpResponse: ...


class _SmokeFailure(Exception):
    def __init__(self, check: str, code: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(code)
        self.check = check
        self.code = code
        self.details = dict(details or {})


class UrllibHttpClient:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        upload: tuple[str, bytes] | None = None,
    ) -> HttpResponse:
        return await asyncio.to_thread(
            self._request_sync,
            method,
            path,
            json_body=json_body,
            headers=headers,
            upload=upload,
        )

    def _request_sync(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        upload: tuple[str, bytes] | None,
    ) -> HttpResponse:
        request_headers = dict(headers or {})
        request_body: bytes | None = None
        if json_body is not None:
            request_body = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        elif upload is not None:
            filename, content = upload
            boundary = f"retrieval-evidence-{uuid4().hex}"
            request_body = self._multipart_body(boundary=boundary, filename=filename, content=content)
            request_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        request = Request(
            f"{self._base_url}{path}",
            data=request_body,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - localhost service URL is explicit input.
                raw = response.read()
                return HttpResponse(status_code=response.status, payload=self._decode_payload(raw), body=self._decode_body(raw))
        except HTTPError as exc:
            raw = exc.read()
            return HttpResponse(status_code=exc.code, payload=self._decode_payload(raw), body=self._decode_body(raw))
        except URLError:
            return HttpResponse(status_code=0, payload={})

    @staticmethod
    def _multipart_body(*, boundary: str, filename: str, content: bytes) -> bytes:
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/markdown\r\n\r\n"
        ).encode("utf-8")
        return b"".join((prefix, content, f"\r\n--{boundary}--\r\n".encode("utf-8")))

    @staticmethod
    def _decode_payload(raw: bytes) -> Mapping[str, Any]:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, Mapping) else {}

    @staticmethod
    def _decode_body(raw: bytes) -> str:
        return raw.decode("utf-8", errors="replace")


class RetrievalEvidenceSmoke:
    """Run the Retrieval Smoke through the existing application and retrieval paths."""

    def __init__(
        self,
        *,
        settings: Settings,
        http_client: EvidenceHttpClient,
        retrieve: Callable[[str], Awaitable[RetrieveResult]],
        output_dir: Path,
        source_revision: str,
        run_id: str | None = None,
        include_generation: bool = False,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._retrieve = retrieve
        self._output_dir = output_dir
        self._source_revision = source_revision
        self._run_id = run_id or uuid4().hex
        self._include_generation = include_generation
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep

    async def run(self) -> dict[str, Any]:
        started_at = self._now()
        try:
            missing = self._missing_configuration()
            if missing:
                raise _SmokeFailure(
                    "runtime_configuration",
                    "REQUIRED_CONFIGURATION_MISSING",
                    {"missing_configuration": missing},
                )

            await self._expect_ok("health", "GET", "/api/v1/health")
            access_token = await self._login_bootstrap_administrator()
            headers = {"Authorization": f"Bearer {access_token}"}
            document_id, job_id = await self._upload_markdown(headers=headers)
            job = await self._wait_for_job(job_id=job_id, headers=headers)
            chunk_count = await self._verify_chunks(document_id=document_id, headers=headers)
            published_generation = await self._publish_document(document_id=document_id, headers=headers)
            result = await self._retrieve(f"retrieval-evidence-{self._run_id}")
            candidate = self._verify_dense_retrieval(result=result, document_id=document_id)
            chat_model_check = {"invoked": False}
            if self._include_generation:
                chat_model_check = await self._verify_generation_loop(
                    headers=headers,
                    question=f"retrieval-evidence-{self._run_id}",
                    expected_source_id=str(candidate["chunk_id"]),
                )

            manifest = self._manifest_base(started_at=started_at)
            manifest.update(
                {
                    "outcome": "passed",
                    "checks": {
                        "health": {"status": "ready"},
                        "document_build": {
                            "document_id": document_id,
                            "job_id": job_id,
                            "status": str(job["status"]),
                            "chunk_count": chunk_count,
                        },
                        "document_publication": {
                            "document_id": document_id,
                            "published_generation": published_generation,
                        },
                        "live_embedding": {
                            "provider": "qwen",
                            "status": "completed",
                            "embedding_contract_fingerprint": build_embedding_contract_fingerprint(self._settings),
                        },
                        "indexing": {
                            "dense_candidate_count": result.dense_candidate_count,
                            "dense_hydrated_count": result.dense_hydrated_count,
                        },
                        "retrieval": {
                            "candidate_document_id": str(candidate["document_id"]),
                            "candidate_chunk_id": str(candidate["chunk_id"]),
                            "candidate_belongs_to_ingested_document": True,
                        },
                        "chat_model": chat_model_check,
                    },
                }
            )
        except _SmokeFailure as failure:
            manifest = self._manifest_base(started_at=started_at)
            manifest.update(
                {
                    "outcome": "failed",
                    "failed_check": failure.check,
                    "failure_code": failure.code,
                    **failure.details,
                }
            )
        except Exception:
            manifest = self._manifest_base(started_at=started_at)
            manifest.update(
                {
                    "outcome": "failed",
                    "failed_check": "command",
                    "failure_code": "UNEXPECTED_FAILURE",
                }
            )

        manifest["finished_at"] = self._now().isoformat()
        self._write_manifest(manifest)
        return manifest

    def _missing_configuration(self) -> list[str]:
        missing: list[str] = []
        if not self._settings.bootstrap_admin_username.strip():
            missing.append("BOOTSTRAP_ADMIN_USERNAME")
        if not self._settings.bootstrap_admin_password.strip():
            missing.append("BOOTSTRAP_ADMIN_PASSWORD")
        if not self._settings.jwt_secret.strip() or self._settings.jwt_secret == "change-me":
            missing.append("JWT_SECRET")
        if not self._settings.embedding_api_key_configured:
            missing.append("EMBEDDING_API_KEY")
        if not self._settings.embedding_base_url_normalized:
            missing.append("EMBEDDING_BASE_URL")
        if not self._settings.embedding_model_normalized:
            missing.append("EMBEDDING_MODEL")
        elif "qwen" not in self._settings.embedding_model_normalized.lower():
            missing.append("EMBEDDING_MODEL_QWEN")
        if self._settings.dense_embedding_dim <= 0:
            missing.append("DENSE_EMBEDDING_DIM")
        if not self._settings.milvus_uri_normalized:
            missing.append("MILVUS_URI")
        if self._include_generation:
            if self._settings.rag_primary_llm_provider != "ark":
                missing.append("RAG_PRIMARY_LLM_PROVIDER_ARK")
            if not self._settings.ark_api_key.strip():
                missing.append("ARK_API_KEY")
            if not self._settings.llm_base_url.strip():
                missing.append("BASE_URL")
            if not self._settings.llm_model.strip():
                missing.append("MODEL")
        return missing

    async def _login_bootstrap_administrator(self) -> str:
        login = await self._expect_ok(
            "administrator_login",
            "POST",
            "/api/v1/auth/login",
            json_body={
                "username": self._settings.bootstrap_admin_username,
                "password": self._settings.bootstrap_admin_password,
            },
        )
        access_token = login.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise _SmokeFailure("administrator_login", "ACCESS_TOKEN_MISSING")
        return access_token

    async def _upload_markdown(self, *, headers: Mapping[str, str]) -> tuple[str, str]:
        sentinel = f"retrieval-evidence-{self._run_id}"
        upload = await self._expect_ok(
            "document_ingestion",
            "POST",
            "/api/v1/documents/upload",
            headers=headers,
            upload=(f"{sentinel}.md", f"# Retrieval Smoke\n\n{sentinel}\n".encode("utf-8")),
        )
        document_id = upload.get("document_id")
        job_id = upload.get("job_id")
        if not isinstance(document_id, str) or not isinstance(job_id, str):
            raise _SmokeFailure("document_ingestion", "DOCUMENT_JOB_IDENTIFIERS_MISSING")
        return document_id, job_id

    async def _wait_for_job(self, *, job_id: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
        for _ in range(_POLL_ATTEMPTS):
            job = await self._expect_ok("document_build", "GET", f"/api/v1/documents/jobs/{job_id}", headers=headers)
            status = job.get("status")
            if status == "succeeded":
                return job
            if status in {"failed", "canceled"}:
                raise _SmokeFailure("document_build", "DOCUMENT_BUILD_NOT_SUCCEEDED")
            await self._sleep(_POLL_INTERVAL_SECONDS)
        raise _SmokeFailure("document_build", "DOCUMENT_BUILD_TIMED_OUT")

    async def _verify_chunks(self, *, document_id: str, headers: Mapping[str, str]) -> int:
        payload = await self._expect_ok(
            "document_build",
            "GET",
            f"/api/v1/documents/{document_id}/chunks?page=1&page_size=1",
            headers=headers,
        )
        pagination = payload.get("pagination")
        chunk_count = pagination.get("total") if isinstance(pagination, Mapping) else None
        if not isinstance(chunk_count, int) or chunk_count <= 0:
            raise _SmokeFailure("document_build", "DOCUMENT_CHUNKS_MISSING")
        return chunk_count

    async def _publish_document(self, *, document_id: str, headers: Mapping[str, str]) -> int:
        payload = await self._expect_ok(
            "document_publication",
            "POST",
            f"/api/v1/documents/{document_id}/publish",
            headers=headers,
        )
        published_generation = payload.get("published_generation")
        if isinstance(published_generation, bool) or not isinstance(published_generation, int) or published_generation < 1:
            raise _SmokeFailure("document_publication", "DOCUMENT_PUBLICATION_INVALID")
        return published_generation

    def _verify_dense_retrieval(self, *, result: RetrieveResult, document_id: str) -> Mapping[str, Any]:
        if result.dense_query_failed:
            raise _SmokeFailure("live_embedding", "DENSE_QUERY_FAILED")
        if result.dense_candidate_count <= 0 or result.dense_hydrated_count <= 0:
            raise _SmokeFailure("indexing", "DENSE_INDEX_CANDIDATES_MISSING")
        for item in result.items:
            if item.get("retrieval_source") == "dense" and item.get("document_id") == document_id:
                return item
        raise _SmokeFailure("retrieval", "INGESTED_DOCUMENT_NOT_RETRIEVED")

    async def _verify_generation_loop(
        self,
        *,
        headers: Mapping[str, str],
        question: str,
        expected_source_id: str,
    ) -> dict[str, Any]:
        session_id = f"generation-{sha256(self._run_id.encode('utf-8')).hexdigest()[:24]}"
        response = await self._expect_ok(
            "generation_normal",
            "POST",
            "/api/v1/chat",
            headers=headers,
            json_body={"message": question, "session_id": session_id},
        )
        answer = response.get("answer")
        if not isinstance(answer, str) or not answer.strip() or answer.startswith("【生成不可用】"):
            raise _SmokeFailure("generation_normal", "GENERATION_RESPONSE_INVALID")
        citation_source_count = self._verify_cited_response(response, expected_source_id=expected_source_id)

        stream = await self._http_client.request(
            "POST",
            "/api/v1/chat/stream",
            headers=headers,
            json_body={"message": question, "session_id": session_id},
        )
        if stream.status_code < 200 or stream.status_code >= 300:
            raise _SmokeFailure("generation_stream", "APPLICATION_REQUEST_FAILED")
        if not all(marker in stream.body for marker in ("event: content", "event: evidence_summary", "event: done", '"coverage": "sufficient"')):
            raise _SmokeFailure("generation_stream", "STREAM_CONTRACT_INVALID")
        return {
            "invoked": True,
            "normal_contract": "passed",
            "stream_contract": "passed",
            "citation_source_count": citation_source_count,
        }

    @staticmethod
    def _verify_cited_response(response: Mapping[str, Any], *, expected_source_id: str) -> int:
        message = response.get("message")
        summary = message.get("evidence_summary") if isinstance(message, Mapping) else None
        if not isinstance(summary, Mapping) or summary.get("coverage") != "sufficient":
            raise _SmokeFailure("generation_normal", "CITATION_MISSING")
        sources = summary.get("sources")
        if not isinstance(sources, list) or not sources:
            raise _SmokeFailure("generation_normal", "CITATION_MISSING")
        if not any(isinstance(source, Mapping) and source.get("source_id") == expected_source_id for source in sources):
            raise _SmokeFailure("generation_normal", "INGESTED_SOURCE_NOT_CITED")
        for source in sources:
            metadata = source.get("metadata") if isinstance(source, Mapping) else None
            if (
                not isinstance(metadata, Mapping)
                or not isinstance(metadata.get("title"), str)
                or not metadata["title"].strip()
                or not isinstance(metadata.get("publication_version"), str)
                or not metadata["publication_version"].strip()
                or not isinstance(source.get("excerpt"), str)
                or not source["excerpt"].strip()
            ):
                raise _SmokeFailure("generation_normal", "CITATION_MISSING")
        return len(sources)

    async def _expect_ok(
        self,
        check: str,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        upload: tuple[str, bytes] | None = None,
    ) -> Mapping[str, Any]:
        response = await self._http_client.request(
            method,
            path,
            json_body=json_body,
            headers=headers,
            upload=upload,
        )
        if response.status_code < 200 or response.status_code >= 300:
            raise _SmokeFailure(check, "APPLICATION_REQUEST_FAILED")
        data = response.payload.get("data")
        if not isinstance(data, Mapping):
            raise _SmokeFailure(check, "APPLICATION_RESPONSE_INVALID")
        return data

    def _manifest_base(self, *, started_at: datetime) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self._run_id,
            "command": "retrieval-evidence generation-smoke" if self._include_generation else "retrieval-evidence smoke",
            "source_revision": self._source_revision,
            "started_at": started_at.isoformat(),
            "runtime_configuration": {
                "path": "backend/.env",
                "values_recorded": False,
            },
        }

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        run_directory = self._output_dir / self._run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


async def _retrieve_from_application(settings: Settings, query: str) -> RetrieveResult:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as session:
            return await MixedModeDocumentRetrieverService(session, settings=settings).retrieve(query, top_k=5)
    finally:
        await engine.dispose()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="retrieval-evidence")
    subparsers = parser.add_subparsers(dest="profile", required=True)
    for profile in ("smoke", "generation-smoke"):
        command = subparsers.add_parser(profile)
        command.add_argument("--base-url", required=True)
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--source-revision", required=True)
        command.add_argument("--run-id")
        command.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    http_client = UrllibHttpClient(base_url=args.base_url, timeout_seconds=args.timeout_seconds)
    runner = RetrievalEvidenceSmoke(
        settings=settings,
        http_client=http_client,
        retrieve=lambda query: _retrieve_from_application(settings, query),
        output_dir=args.output_dir,
        source_revision=args.source_revision,
        run_id=args.run_id,
        include_generation=args.profile == "generation-smoke",
    )
    manifest = asyncio.run(runner.run())
    print(json.dumps({"outcome": manifest["outcome"], "run_id": manifest["run_id"]}, sort_keys=True))
    return 0 if manifest["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
