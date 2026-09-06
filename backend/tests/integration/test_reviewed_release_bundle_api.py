from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1 import reviewed_bundles as reviewed_bundles_api
from app.common.canonical_json import canonical_json_sha256
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.document import Document
from app.reviewed_bundles.inputs import frozen_candidate_build_input_sha256
from app.reviewed_bundles.models import CandidateBuildChunk, CandidateBuildJob
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

    @asynccontextmanager
    async def verify_for_candidate_finalization(
        self,
        artifact: dict,
        artifact_sha256: str,
    ) -> AsyncIterator[dict]:
        yield await self.verify(artifact, artifact_sha256)


def _sha256(value: object) -> str:
    return canonical_json_sha256(value)


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
            "applicability_conditions": [{"condition_id": "bundle-api-applicability"}],
            "freshness_triggers": [{"trigger_id": "bundle-api-freshness"}],
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
            "section_source_relationships": [
                {"section_id": "decision_query", "source_ids": ["source-bundle-api-001"]},
                {
                    "section_id": "recommendation_or_reviewed_branches",
                    "source_ids": ["source-bundle-api-001"],
                },
            ],
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
                "source": {
                    "source_id": "source-bundle-api-001",
                    "source_tier": "primary_evidence_source",
                    "authority": "ZhoMind architecture group",
                    "access_scope": "public",
                    "public_url": "https://example.com/bundle-api",
                    "availability": "verified_usable",
                },
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


def test_candidate_preview_is_admin_only_and_never_marks_candidate_content_as_answer_evidence(
    client: TestClient,
) -> None:
    job_id = "preview-candidate-job-001"
    candidate_id = f"candidate:{job_id}-attempt-1"

    async def seed() -> None:
        session_factory = client.app.state.test_auth_session_factory
        async with session_factory() as session:
            artifact = _approved_export()
            entry = artifact["entry"]
            bundle_id = "bundle:preview-bundle-001"
            bundle_item_id = "bundle_item:preview-item-001"
            document_identity = "document:bundle-api-entry-001"
            chunk_strategy = entry["chunk_strategy"]
            embedding_configuration = {"active": False}
            input_sha256 = _sha256(artifact)
            bundle_item_sha256 = _sha256(
                {
                    "bundle_item_id": "preview-item-001",
                    "operation": "create",
                    "artifact_sha256": input_sha256,
                    "artifact": artifact,
                }
            )
            manifest = {
                "schema": "reviewed_release_bundle/v1",
                "schema_version": 1,
                "bundle_id": "preview-bundle-001",
                "editorial_source_revision": artifact["revision_sha256"],
                "exported_at": "2026-09-06T12:00:00Z",
                "items": [
                    {
                        "bundle_item_id": "preview-item-001",
                        "operation": "create",
                        "artifact_sha256": input_sha256,
                        "artifact": artifact,
                        "bundle_item_sha256": bundle_item_sha256,
                    }
                ],
            }
            bundle_sha256 = _sha256(manifest)
            manifest["bundle_sha256"] = bundle_sha256
            input_payload = {
                "schema": "candidate_build_input/v1",
                "bundle_id": bundle_id,
                "bundle_sha256": bundle_sha256,
                "bundle_item_id": bundle_item_id,
                "bundle_item_sha256": bundle_item_sha256,
                "entry_identity": artifact["entry_identity"],
                "document_identity": document_identity,
                "requested_generation": 1,
                "editorial_source_revision": artifact["revision_sha256"],
                "input_sha256": input_sha256,
                "chunk_strategy": chunk_strategy,
                "embedding_configuration": embedding_configuration,
            }
            frozen_input_sha256 = frozen_candidate_build_input_sha256(input_payload)
            input_payload["frozen_input_sha256"] = frozen_input_sha256
            content = "Candidate-only Sparse BM25 preview must remain isolated."
            content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            chunk_metadata = {
                "entry_id": artifact["entry_id"],
                "entry_identity": artifact["entry_identity"],
                "editorial_revision_identity": artifact["editorial_revision_identity"],
                "section_id": "recommendation_or_reviewed_branches",
                "source_relationships": [
                    {
                        "source_identity": "source:source-bundle-api-001",
                        "availability": "verified_usable",
                        "access_scope": "public",
                    }
                ],
                "assurance_level": "source_grounded",
                "applicability_conditions": [{"condition_id": "bundle-api-applicability"}],
                "freshness_triggers": [{"trigger_id": "bundle-api-freshness"}],
                "candidate_build": True,
            }
            session.add_all(
                [
                    CanonicalRecordModel(
                        stable_id=bundle_id,
                        identity_kind="bundle",
                        identity_value="preview-bundle-001",
                        state="processing",
                        record_class="immutable",
                        payload={
                            "schema": "reviewed_release_bundle/v1",
                            "bundle_sha256": bundle_sha256,
                            "manifest": manifest,
                        },
                    ),
                    CanonicalRecordModel(
                        stable_id=bundle_item_id,
                        identity_kind="bundle_item",
                        identity_value="preview-item-001",
                        state="admitted",
                        record_class="immutable",
                        payload={
                            "schema": "reviewed_release_bundle_item/v1",
                            "bundle_id": bundle_id,
                            "entry_identity": artifact["entry_identity"],
                            "operation": "create",
                            "artifact_sha256": input_sha256,
                            "artifact": artifact,
                            "bundle_item_sha256": bundle_item_sha256,
                        },
                    ),
                    CanonicalRecordModel(
                        stable_id=f"build_generation:{job_id}",
                        identity_kind="build_generation",
                        identity_value=job_id,
                        state="frozen",
                        record_class="immutable",
                        payload=input_payload,
                    ),
                    CanonicalRecordModel(
                        stable_id=candidate_id,
                        identity_kind="candidate",
                        identity_value=f"{job_id}-attempt-1",
                        state="candidate_ready",
                        record_class="immutable",
                        payload={
                            "schema": "candidate_build_candidate/v1",
                            "bundle_id": bundle_id,
                            "bundle_sha256": bundle_sha256,
                            "bundle_item_id": bundle_item_id,
                            "bundle_item_sha256": bundle_item_sha256,
                            "build_generation_id": f"build_generation:{job_id}",
                            "entry_identity": artifact["entry_identity"],
                            "document_identity": document_identity,
                            "requested_generation": 1,
                            "editorial_source_revision": artifact["revision_sha256"],
                            "input_sha256": input_sha256,
                            "frozen_input_sha256": frozen_input_sha256,
                            "chunk_strategy": chunk_strategy,
                            "embedding_configuration": embedding_configuration,
                            "attempt": 1,
                            "chunk_count": 1,
                            "chunk_sha256s": [content_sha256],
                        },
                    ),
                    CandidateBuildJob(
                        id=job_id,
                        bundle_id=bundle_id,
                        bundle_item_id=bundle_item_id,
                        entry_identity=artifact["entry_identity"],
                        document_identity=document_identity,
                        requested_generation=1,
                        editorial_source_revision=artifact["revision_sha256"],
                        input_sha256=input_sha256,
                        frozen_input_sha256=frozen_input_sha256,
                        chunk_strategy=chunk_strategy,
                        embedding_configuration=embedding_configuration,
                        status="candidate_ready",
                        stage="indexing",
                        progress=100,
                        attempt=1,
                        terminal_state="candidate_ready",
                        allowed_next_action="await_candidate_inspection",
                        candidate_id=candidate_id,
                    ),
                    CandidateBuildChunk(
                        id="preview-candidate-chunk-001",
                        job_id=job_id,
                        candidate_id=candidate_id,
                        document_identity=document_identity,
                        generation=1,
                        attempt=1,
                        chunk_index=0,
                        content=content,
                        content_sha256=content_sha256,
                        chunk_metadata=chunk_metadata,
                    ),
                ]
            )
            await session.commit()

    asyncio.run(seed())
    user_headers = asyncio.run(_headers(client, username="candidate-preview-user", role="user"))
    admin_headers = asyncio.run(_headers(client, username="candidate-preview-admin", role="admin"))
    url = f"/api/v1/reviewed-release-bundles/candidates/{candidate_id}/preview?query=Sparse%20BM25"

    assert client.get(url).status_code == 401
    assert client.get(url, headers=user_headers).status_code == 403

    response = client.get(url, headers=admin_headers)

    assert response.status_code == 200
    payload = _data(response)
    assert payload["profile_identity"] == "retrieval-answer-policy/pilot-v1"
    assert payload["candidate_pool_scope"] == "candidate_preview"
    assert payload["items"][0]["candidate_id"] == candidate_id
    assert payload["items"][0]["candidate_version"] == f"{candidate_id}:generation:1:attempt:1"
    assert payload["items"][0]["answer_evidence_eligible"] is False
    assert payload["items"][0]["diagnostic_only"] is True
    assert payload["items"][0]["content_preview"].startswith("Candidate-only")
    assert payload["candidate_exclusions"] == []


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
    assert len(job_data["frozen_input_sha256"]) == 64
    assert job_data["events"][0]["payload"]["frozen_input_sha256"] == job_data["frozen_input_sha256"]
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
    assert dispatched_data["events"][-1]["payload"]["frozen_input_sha256"] == job_data["frozen_input_sha256"]
    assert dispatched_data["events"][-1]["recorded_by"].startswith("member:")
    assert enqueued_job_ids == [job_id]

    repeated_dispatch = client.post(f"/api/v1/reviewed-release-bundles/jobs/{job_id}/dispatch", headers=admin_headers)
    assert repeated_dispatch.status_code == 200
    assert _data(repeated_dispatch)["dispatched_at"] == dispatched_data["dispatched_at"]
    assert enqueued_job_ids == [job_id, job_id]

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
    assert enqueued_job_ids == [job_id, job_id, job_id]

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
