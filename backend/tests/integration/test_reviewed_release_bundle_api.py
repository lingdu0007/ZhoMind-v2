from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1 import reviewed_bundles as reviewed_bundles_api
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.canonical import CanonicalEventModel
from app.model.document import Document
from app.reviewed_bundles.models import CandidateBuildJob
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = {str(name): str(value) for name, value in mapping.items()}
        return len(mapping)

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self._store and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self._store)

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                deleted += 1
        return deleted


class _ApprovedExportVerifier:
    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        assert artifact_sha256 == _sha256(artifact)
        return artifact


def _sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _approved_export() -> dict:
    return {
        "schema": "editorial_export/v1",
        "entry_identity": "entry:bundle-api-entry-001",
        "entry_id": "bundle-api-entry-001",
        "editorial_revision_identity": "editorial_revision:bundle-api-entry-001.r1",
        "revision_number": 1,
        "revision_sha256": "a" * 64,
        "roles": {
            "author_identity": "member:bundle-api-author-001",
            "approving_reviewer_identity": "member:bundle-api-reviewer-001",
            "accountable_maintainer_identity": "member:bundle-api-maintainer-001",
        },
        "approval": {"status": "approved", "reviewer_identity": "member:bundle-api-reviewer-001"},
        "entry": {
            "schema_version": 1,
            "entry_id": "bundle-api-entry-001",
            "title": "Inspect approved bundle intake",
            "coverage_position": "rag_source_admission_and_chunking",
            "assurance_level": "source_grounded",
            "chunk_strategy": {
                "strategy_id": "section-aware-900-120",
                "max_characters": 900,
                "overlap_characters": 120,
                "preserve_section_boundaries": True,
            },
            "acceptance_material": {
                "supported_queries": [
                    {
                        "query_id": "supported-bundle-api",
                        "query": "How is a reviewed bundle admitted?",
                        "expected_outcome": "supported",
                    }
                ],
                "boundary_queries": [
                    {
                        "query_id": "boundary-bundle-api",
                        "query": "Can bundle intake publish a Candidate?",
                        "expected_outcome": "insufficient_evidence",
                    }
                ],
            },
            "body": {
                "decision_query": "Which reviewed exports can become Candidate Build inputs?",
                "recommendation_or_reviewed_branches": "Only retained approved exports may enter Candidate Build.",
            },
            "sources": [
                {
                    "source_id": "source-bundle-api-001",
                    "source_tier": "primary_evidence_source",
                    "authority": "ZhoMind architecture group",
                    "access_scope": "public",
                    "public_url": "https://example.com/bundle-api",
                    "availability": "verified_usable",
                }
            ],
        },
        "sources": [
            {
                "source_identity": "source:source-bundle-api-001",
                "availability": "verified_usable",
                "source_definition_sha256": "b" * 64,
                "availability_event": {"event_id": "event:bundle-api-source-availability-001"},
            }
        ],
        "release_assurance_snapshot": None,
        "editorial_audit": [{"sequence": 1, "action": "revision_approved"}],
    }


def _bundle_manifest() -> dict:
    artifact = _approved_export()
    item = {
        "bundle_item_id": "bundle-api-item-001",
        "operation": "create",
        "artifact_sha256": _sha256(artifact),
        "artifact": artifact,
    }
    item["bundle_item_sha256"] = _sha256(item)
    manifest = {
        "schema": "reviewed_release_bundle/v1",
        "schema_version": 1,
        "bundle_id": "bundle-api-001",
        "editorial_source_revision": artifact["revision_sha256"],
        "exported_at": "2026-09-05T12:00:00Z",
        "items": [item],
    }
    return {**manifest, "bundle_sha256": _sha256(manifest)}


@pytest.fixture
def client(tmp_path) -> Generator[TestClient, None, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'reviewed-bundle-api.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()

    async def initialize_database() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(initialize_database())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.settings_session_factory = session_factory
    app.state.test_auth_session_factory = session_factory
    app.state.test_auth_redis = redis
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = SessionLocal
    asyncio.run(engine.dispose())


async def _headers(client: TestClient, *, username: str, role: str) -> dict[str, str]:
    token = await create_authenticated_test_token(
        client.app.state.test_auth_session_factory,
        client.app.state.test_auth_redis,
        username=username,
        role=role,
    )
    return {"Authorization": f"Bearer {token}"}


def _data(response) -> dict:
    return response.json()["data"]


def test_reviewed_bundle_intake_api_is_admin_only_and_exposes_a_recoverable_non_publishing_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueued_job_ids: list[str] = []
    canceled_job_ids: list[str] = []

    async def enqueue(_session, job_id: str) -> None:
        enqueued_job_ids.append(job_id)

    async def cancel(job_id: str) -> bool:
        canceled_job_ids.append(job_id)
        return True

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    monkeypatch.setattr(reviewed_bundles_api.candidate_build_runtime, "enqueue", enqueue)
    monkeypatch.setattr(reviewed_bundles_api.candidate_build_runtime, "cancel", cancel)

    user_headers = asyncio.run(_headers(client, username="bundle-api-user", role="user"))
    admin_headers = asyncio.run(_headers(client, username="bundle-api-admin", role="admin"))
    manifest = _bundle_manifest()

    assert client.post("/api/v1/reviewed-release-bundles/import", json=manifest).status_code == 401
    denied = client.post("/api/v1/reviewed-release-bundles/import", headers=user_headers, json=manifest)
    assert denied.status_code == 403

    imported = client.post("/api/v1/reviewed-release-bundles/import", headers=admin_headers, json=manifest)
    assert imported.status_code == 200
    imported_data = _data(imported)
    item = imported_data["items"][0]
    job_id = item["job_id"]
    assert imported_data == {
        "bundle_id": "bundle-api-001",
        "state": "processing",
        "schema_version": 1,
        "editorial_source_revision": "a" * 64,
        "exported_at": "2026-09-05T12:00:00Z",
        "bundle_sha256": manifest["bundle_sha256"],
        "items": [
            {
                "bundle_item_id": "bundle-api-item-001",
                "entry_identity": "entry:bundle-api-entry-001",
                "operation": "create",
                "state": "admitted",
                "artifact_sha256": manifest["items"][0]["artifact_sha256"],
                "bundle_item_sha256": manifest["items"][0]["bundle_item_sha256"],
                "allowed_next_action": "dispatch_candidate_build",
                "job_id": job_id,
            }
        ],
    }
    assert enqueued_job_ids == []

    listed = client.get("/api/v1/reviewed-release-bundles", headers=admin_headers)
    assert listed.status_code == 200
    assert _data(listed)["items"] == [imported_data]
    fetched = client.get("/api/v1/reviewed-release-bundles/bundle-api-001", headers=admin_headers)
    assert fetched.status_code == 200
    assert _data(fetched) == imported_data

    job = client.get(f"/api/v1/reviewed-release-bundles/jobs/{job_id}", headers=admin_headers)
    assert job.status_code == 200
    job_data = _data(job)
    assert job_data["job_id"] == job_id
    assert job_data["bundle_id"] == "bundle:bundle-api-001"
    assert job_data["entry_identity"] == "entry:bundle-api-entry-001"
    assert job_data["editorial_source_revision"] == "a" * 64
    assert job_data["input_sha256"] == manifest["items"][0]["artifact_sha256"]
    assert job_data["status"] == "queued"
    assert job_data["attempt"] == 1
    assert job_data["candidate_id"] is None
    assert job_data["derived_cleanup_pending"] is False
    assert job_data["dispatched_at"] is None
    assert job_data["events"][0]["to_state"] == "queued"
    assert "api_key" not in json.dumps(job_data["embedding_configuration"]).lower()

    dispatched = client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/dispatch", headers=admin_headers)
    assert dispatched.status_code == 200
    dispatched_data = _data(dispatched)
    assert dispatched_data["status"] == "queued"
    assert dispatched_data["dispatched_at"] is not None
    assert dispatched_data["allowed_next_action"] == "cancel_or_await_candidate_build"
    assert dispatched_data["events"][-1]["payload"]["action"] == "dispatched"
    assert dispatched_data["events"][-1]["recorded_by"].startswith("member:")
    assert enqueued_job_ids == [job_id]

    repeated_dispatch = client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/dispatch", headers=admin_headers)
    assert repeated_dispatch.status_code == 200
    assert _data(repeated_dispatch)["dispatched_at"] == dispatched_data["dispatched_at"]
    assert enqueued_job_ids == [job_id]

    canceled = client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/cancel", headers=admin_headers)
    assert canceled.status_code == 200
    canceled_data = _data(canceled)
    assert canceled_job_ids == [job_id]
    assert canceled_data["status"] == "canceled"
    assert canceled_data["terminal_state"] == "canceled"
    assert canceled_data["failure_reason"]["code"] == "CANDIDATE_BUILD_CANCELED"
    assert canceled_data["allowed_next_action"] == "retry_fixed_inputs"

    retried = client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/retry", headers=admin_headers)
    assert retried.status_code == 200
    retried_data = _data(retried)
    assert retried_data["status"] == "queued"
    assert retried_data["attempt"] == 2
    assert retried_data["events"][-1]["payload"]["action"] == "retry_dispatched"
    assert retried_data["events"][-1]["recorded_by"].startswith("member:")
    assert enqueued_job_ids == [job_id, job_id]

    repeated = client.post("/api/v1/reviewed-release-bundles/import", headers=admin_headers, json=manifest)
    assert repeated.status_code == 200
    assert _data(repeated)["items"][0]["job_id"] == job_id
    assert client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/publish", headers=admin_headers).status_code == 404

    async def counts() -> tuple[int, int]:
        async with client.app.state.test_auth_session_factory() as session:
            candidate_jobs = await session.scalar(select(func.count()).select_from(CandidateBuildJob))
            documents = await session.scalar(select(func.count()).select_from(Document))
            return int(candidate_jobs or 0), int(documents or 0)

    assert asyncio.run(counts()) == (1, 0)


def test_non_object_bundle_manifest_records_a_safe_admission_attempt_audit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    admin_headers = asyncio.run(_headers(client, username="bundle-api-invalid-manifest-admin", role="admin"))

    rejected = client.post(
        "/api/v1/reviewed-release-bundles/import",
        headers=admin_headers,
        json=["not", "a", "reviewed-release-bundle"],
    )

    assert rejected.status_code == 422
    assert rejected.json()["code"] == "BUNDLE_INTEGRITY_REJECTED"

    async def audit() -> tuple[CanonicalEventModel | None, int]:
        async with client.app.state.test_auth_session_factory() as session:
            event = await session.scalar(
                select(CanonicalEventModel)
                .where(CanonicalEventModel.aggregate_kind == "admission_attempt")
                .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
            )
            jobs = await session.scalar(select(func.count()).select_from(CandidateBuildJob))
            return event, int(jobs or 0)

    event, job_count = asyncio.run(audit())
    assert event is not None
    assert event.aggregate_id.startswith("admission_attempt:")
    assert event.payload == {
        "schema": "reviewed_release_bundle_import_audit/v1",
        "action": "integrity_rejected",
        "reasons": [
            {
                "field": "manifest",
                "code": "invalid",
                "message": "bundle integrity validation failed",
            }
        ],
    }
    assert job_count == 0


@pytest.mark.parametrize("worker_result", [False, RuntimeError("worker control plane unavailable")])
def test_cancel_request_failure_is_retained_as_a_recoverable_candidate_job_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    worker_result: bool | Exception,
) -> None:
    async def enqueue(_session, _job_id: str) -> None:
        return None

    async def cancel(_job_id: str) -> bool:
        if isinstance(worker_result, Exception):
            raise worker_result
        return worker_result

    monkeypatch.setattr(
        reviewed_bundles_api,
        "CanonicalEditorialExportVerifier",
        lambda _session: _ApprovedExportVerifier(),
    )
    monkeypatch.setattr(reviewed_bundles_api.candidate_build_runtime, "enqueue", enqueue)
    monkeypatch.setattr(reviewed_bundles_api.candidate_build_runtime, "cancel", cancel)
    admin_headers = asyncio.run(_headers(client, username="bundle-api-cancel-failure-admin", role="admin"))

    imported = client.post(
        "/api/v1/reviewed-release-bundles/import",
        headers=admin_headers,
        json=_bundle_manifest(),
    )
    assert imported.status_code == 200
    job_id = _data(imported)["items"][0]["job_id"]
    assert client.post(
        f"/api/v1/reviewed-release-bundles/jobs/{job_id}/dispatch",
        headers=admin_headers,
    ).status_code == 200

    canceled = client.post(
        f"/api/v1/reviewed-release-bundles/jobs/{job_id}/cancel",
        headers=admin_headers,
    )

    assert canceled.status_code == 200
    job = _data(canceled)
    assert job["status"] == "failed"
    assert job["terminal_state"] == "failed"
    assert job["failure_reason"] == {
        "code": "CANDIDATE_CANCELLATION_REQUEST_FAILED",
        "stage": "queued",
        "message": "Candidate Build cancellation request could not reach the worker",
    }
    assert job["derived_cleanup_pending"] is True
    assert job["allowed_next_action"] == "reconcile_derived_data_then_retry"
    assert job["events"][-1]["payload"]["action"] == "cancellation_request_failed"
