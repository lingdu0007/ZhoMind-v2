from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError

from app.common.config import Settings
from app.common.exceptions import AppError
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.document import Document
from app.reviewed_bundles import build_service as candidate_build_module
from app.reviewed_bundles import lifecycle as candidate_lifecycle
from app.reviewed_bundles import runtime as reviewed_bundles_runtime
from app.reviewed_bundles.build_service import CandidateBuildService
from app.reviewed_bundles.models import CandidateBuildChunk, CandidateBuildJob
from app.reviewed_bundles.recovery import CandidateBuildRecoveryService
from app.reviewed_bundles.runtime import CandidateBuildRuntime
from app.reviewed_bundles.service import ReviewedReleaseBundleService, candidate_embedding_configuration


def _sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _approved_export(entry_id: str = "source-admission-001") -> dict:
    return {
        "schema": "editorial_export/v1",
        "entry_identity": f"entry:{entry_id}",
        "entry_id": entry_id,
        "editorial_revision_identity": f"editorial_revision:{entry_id}-r1",
        "revision_number": 1,
        "revision_sha256": "a" * 64,
        "roles": {
            "author_identity": "member:author-001",
            "approving_reviewer_identity": "member:reviewer-001",
            "accountable_maintainer_identity": "member:maintainer-001",
        },
        "approval": {
            "status": "approved",
            "reviewer_identity": "member:reviewer-001",
        },
        "entry": {
            "schema_version": 1,
            "entry_id": entry_id,
            "title": "Choose a source admission boundary",
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
                        "query_id": "supported-source-admission",
                        "query": "How should a RAG source be admitted?",
                        "expected_outcome": "supported",
                    }
                ],
                "boundary_queries": [
                    {
                        "query_id": "boundary-source-admission",
                        "query": "Should an unknown public URL be treated as verified?",
                        "expected_outcome": "insufficient_evidence",
                    }
                ],
            },
            "body": {
                "decision_query": "Which source admission conditions are required before indexing?",
                "recommendation_or_reviewed_branches": "Require a reviewed source record before indexing.",
            },
            "sources": [
                {
                    "source_id": "source-rag-admission-001",
                    "source_tier": "primary_evidence_source",
                    "authority": "ZhoMind architecture group",
                    "access_scope": "public",
                    "public_url": "https://example.com/rag/source-admission",
                    "availability": "verified_usable",
                }
            ],
        },
        "sources": [
            {
                "source_identity": "source:source-rag-admission-001",
                "availability": "verified_usable",
                "source_definition_sha256": "b" * 64,
                "availability_event": {"event_id": "event:source-availability-001"},
            }
        ],
        "release_assurance_snapshot": None,
        "editorial_audit": [{"sequence": 1, "action": "revision_approved"}],
    }


class _ApprovedExportVerifier:
    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        assert artifact_sha256 == _sha256(artifact)
        return artifact


class _StructuredSourceFailureVerifier:
    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        assert artifact_sha256 == _sha256(artifact)
        if artifact["entry_identity"] == "entry:authority-blocking-field-entry-002":
            raise AppError(
                status_code=409,
                code="EDITORIAL_SOURCE_UNAVAILABLE",
                message="the approved export has an unavailable source",
                detail={
                    "reasons": [
                        {
                            "field": "sources[0].availability",
                            "code": "source_unavailable",
                            "message": "the retained source availability is not verified_usable",
                        }
                    ]
                },
            )
        return artifact


class _SourceWithdrawsDuringBuildVerifier:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        self.calls += 1
        assert artifact_sha256 == _sha256(artifact)
        if self.calls >= 3:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SOURCE_UNAVAILABLE",
                message="source authority became unavailable during Candidate Build",
            )
        return artifact


class _SourceWithdrawsDuringIndexingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        self.calls += 1
        assert artifact_sha256 == _sha256(artifact)
        if self.calls >= 4:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SOURCE_UNAVAILABLE",
                message="source authority became unavailable while Candidate Build indexing completed",
            )
        return artifact


class _SourceWithdrawsDuringFinalizationVerifier:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, artifact: dict, artifact_sha256: str) -> dict:
        self.calls += 1
        assert artifact_sha256 == _sha256(artifact)
        if self.calls >= 5:
            raise AppError(
                status_code=409,
                code="EDITORIAL_SOURCE_UNAVAILABLE",
                message="source authority became unavailable during Candidate finalization",
            )
        return artifact


class _DenseIndexSpy:
    def __init__(self) -> None:
        self.index_calls: list[tuple[str, int, int]] = []
        self.index_fingerprints: list[str | None] = []

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        self.index_calls.append((document_id, generation, len(chunks)))
        self.index_fingerprints.append(embedding_fingerprint)
        return type("DenseIndexResult", (), {"active": True, "fingerprint": "embedding-profile-001"})()

    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None:
        return None


class _SlowDenseIndex(_DenseIndexSpy):
    def __init__(self) -> None:
        super().__init__()
        self.delete_calls: list[tuple[str, int | None, str | None]] = []

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        await asyncio.sleep(0.15)
        return await super().index_candidate_generation(
            document_id=document_id,
            generation=generation,
            chunks=chunks,
            embedding_fingerprint=embedding_fingerprint,
        )

    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None:
        self.delete_calls.append((document_id, generation, embedding_fingerprint))


class _CancellationAwareDenseIndex(_DenseIndexSpy):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True
        return await super().index_candidate_generation(
            document_id=document_id,
            generation=generation,
            chunks=chunks,
            embedding_fingerprint=embedding_fingerprint,
        )


class _FailingDenseIndex:
    def __init__(self) -> None:
        self.delete_calls: list[tuple[str, int | None, str | None]] = []

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        raise AppError(
            status_code=503,
            code="DENSE_INDEX_BACKEND_UNAVAILABLE",
            message="dense index is unavailable",
        )

    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None:
        self.delete_calls.append((document_id, generation, embedding_fingerprint))


class _CleanupFailingDenseIndex(_FailingDenseIndex):
    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None:
        self.delete_calls.append((document_id, generation, embedding_fingerprint))
        raise RuntimeError("candidate vector cleanup is unavailable")


class _RecoveryFenceDenseIndex(_FailingDenseIndex):
    def __init__(self, job: CandidateBuildJob) -> None:
        super().__init__()
        self._job = job
        self.observed_fence: tuple[str, bool, str | None] | None = None

    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None:
        self.observed_fence = (
            self._job.status,
            self._job.derived_cleanup_pending,
            self._job.lease_owner,
        )
        await super().delete_candidate_generation(
            document_id=document_id,
            generation=generation,
            embedding_fingerprint=embedding_fingerprint,
        )


class _SupersedingDenseIndex(_DenseIndexSpy):
    def __init__(self, intake: ReviewedReleaseBundleService, replacement_manifest: dict) -> None:
        super().__init__()
        self._intake = intake
        self._replacement_manifest = replacement_manifest
        self.delete_calls: list[tuple[str, int | None, str | None]] = []

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        await self._intake.import_bundle(
            self._replacement_manifest,
            actor_identity="member:administrator-001",
        )
        return await super().index_candidate_generation(
            document_id=document_id,
            generation=generation,
            chunks=chunks,
            embedding_fingerprint=embedding_fingerprint,
        )

    async def delete_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int | None,
        embedding_fingerprint: str | None = None,
    ) -> None:
        self.delete_calls.append((document_id, generation, embedding_fingerprint))


def _bundle_manifest(*, bundle_id: str = "reviewed-bundle-001") -> dict:
    artifact = _approved_export()
    item = {
        "bundle_item_id": "reviewed-bundle-item-001",
        "operation": "create",
        "artifact_sha256": _sha256(artifact),
        "artifact": artifact,
    }
    item["bundle_item_sha256"] = _sha256(item)
    manifest = {
        "schema": "reviewed_release_bundle/v1",
        "schema_version": 1,
        "bundle_id": bundle_id,
        "editorial_source_revision": artifact["revision_sha256"],
        "exported_at": "2026-09-05T12:00:00Z",
        "items": [item],
    }
    return {**manifest, "bundle_sha256": _sha256(manifest)}


def _rehash_manifest(manifest: dict) -> None:
    for item in manifest["items"]:
        item["artifact_sha256"] = _sha256(item["artifact"])
        item["bundle_item_sha256"] = _sha256({key: value for key, value in item.items() if key != "bundle_item_sha256"})
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    manifest["bundle_sha256"] = _sha256(unsigned)


def _active_dense_settings() -> Settings:
    return Settings(
        EMBEDDING_API_KEY="candidate-build-test-key",
        EMBEDDING_BASE_URL="https://token:do-not-retain@embeddings.example.test/v1?access_token=do-not-retain",
        EMBEDDING_MODEL="candidate-build-test-model",
        DENSE_EMBEDDING_DIM=3,
        MILVUS_URI="http://milvus.example.test:19530",
    )


async def _intake_record_count(db_session) -> int:
    count = await db_session.scalar(
        select(func.count())
        .select_from(CanonicalRecordModel)
        .where(
            CanonicalRecordModel.identity_kind.in_(
                ["bundle", "bundle_item", "build_generation"]
            )
        )
    )
    return int(count or 0)


async def _dispatch_candidate_build(db_session, job_id: str) -> None:
    dispatched = await CandidateBuildService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).dispatch_job(job_id, actor_identity="member:administrator-001")
    assert dispatched is True


async def test_importing_an_approved_bundle_persists_immutable_inputs_and_is_idempotent(db_session) -> None:
    published_document = Document(
        id="published-document-001",
        filename="published-document-001.md",
        file_type="md",
        file_size=1,
        status="ready",
        published_generation=7,
        next_generation=8,
        latest_requested_generation=7,
    )
    db_session.add(published_document)
    await db_session.commit()

    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest()

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["bundle_id"] == "reviewed-bundle-001"
    assert accepted["state"] == "processing"
    assert accepted["editorial_source_revision"] == "a" * 64
    assert accepted["schema_version"] == 1
    assert accepted["exported_at"] == "2026-09-05T12:00:00Z"
    assert accepted["bundle_sha256"] == manifest["bundle_sha256"]
    assert accepted["items"] == [
        {
            "bundle_item_id": "reviewed-bundle-item-001",
            "entry_identity": "entry:source-admission-001",
            "operation": "create",
            "state": "admitted",
            "artifact_sha256": manifest["items"][0]["artifact_sha256"],
            "bundle_item_sha256": manifest["items"][0]["bundle_item_sha256"],
            "allowed_next_action": "dispatch_candidate_build",
            "job_id": accepted["items"][0]["job_id"],
        }
    ]

    assert await db_session.get(CanonicalRecordModel, "bundle:reviewed-bundle-001") is not None
    assert await db_session.get(CanonicalRecordModel, "bundle_item:reviewed-bundle-item-001") is not None
    assert await db_session.get(
        CanonicalRecordModel,
        f"build_generation:{accepted['items'][0]['job_id']}",
    ) is not None
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 1
    queued_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{accepted['items'][0]['job_id']}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    assert queued_event is not None
    assert queued_event.payload["action"] == "queued"
    assert queued_event.payload["editorial_source_revision"] == "a" * 64
    assert queued_event.payload["input_sha256"] == manifest["items"][0]["artifact_sha256"]

    again = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert again == accepted
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 1
    await db_session.refresh(published_document)
    assert published_document.published_generation == 7


async def test_same_hash_bundle_insert_race_returns_the_existing_immutable_projection(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="same-hash-insert-race-001")
    original_commit = db_session.commit
    injected = False

    async def commit_after_lost_insert_race() -> None:
        nonlocal injected
        if not injected:
            injected = True
            await original_commit()
            raise IntegrityError("INSERT", {}, RuntimeError("simulated unique collision"))
        await original_commit()

    monkeypatch.setattr(db_session, "commit", commit_after_lost_insert_race)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["bundle_id"] == "same-hash-insert-race-001"
    assert accepted["bundle_sha256"] == manifest["bundle_sha256"]
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 1
    assert await db_session.scalar(
        select(func.count())
        .select_from(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_kind == "admission_attempt")
    ) == 0


async def test_generation_collision_retries_the_immutable_bundle_with_a_fresh_candidate_generation(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    allocated_generations = iter([1, 2])
    original_commit = db_session.commit
    commit_attempts = 0

    async def next_generation(_: str) -> int:
        return next(allocated_generations)

    async def commit_with_first_generation_collision() -> None:
        nonlocal commit_attempts
        commit_attempts += 1
        if commit_attempts == 1:
            raise IntegrityError(
                "INSERT INTO candidate_build_jobs",
                {},
                Exception("uq_candidate_build_jobs_entry_generation"),
            )
        await original_commit()

    monkeypatch.setattr(service, "_next_generation", next_generation)
    monkeypatch.setattr(db_session, "commit", commit_with_first_generation_collision)

    accepted = await service.import_bundle(
        _bundle_manifest(bundle_id="generation-retry-bundle-001"),
        actor_identity="member:administrator-001",
    )

    job = await db_session.get(CandidateBuildJob, accepted["items"][0]["job_id"])
    assert job is not None
    assert job.requested_generation == 2
    assert commit_attempts == 2


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "unsupported schema",
            lambda manifest: (
                manifest.__setitem__("schema_version", 2),
                _rehash_manifest(manifest),
            ),
        ),
        (
            "tampered artifact",
            lambda manifest: manifest["items"][0]["artifact"]["entry"].__setitem__("title", "tampered export"),
        ),
        (
            "duplicate entry operation",
            lambda manifest: (
                manifest["items"].append(
                    {
                        **deepcopy(manifest["items"][0]),
                        "bundle_item_id": "reviewed-bundle-item-002",
                    }
                ),
                _rehash_manifest(manifest),
            ),
        ),
        (
            "credential",
            lambda manifest: (
                manifest["items"][0]["artifact"]["entry"].__setitem__("api_token", "not-a-real-token"),
                _rehash_manifest(manifest),
            ),
        ),
        (
            "automatic publication instruction",
            lambda manifest: (
                manifest["items"][0]["artifact"].__setitem__(
                    "instruction",
                    "Publish this artifact to production.",
                ),
                _rehash_manifest(manifest),
            ),
        ),
    ],
)
async def test_bundle_integrity_failures_reject_the_whole_bundle_before_dispatch(
    db_session,
    label: str,
    mutate,
) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id=f"rejected-bundle-{label.replace(' ', '-')}")
    mutate(manifest)

    with pytest.raises(AppError) as exc_info:
        await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert exc_info.value.code == "BUNDLE_INTEGRITY_REJECTED"
    assert await _intake_record_count(db_session) == 0
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 0


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "bundle item hash",
            lambda manifest: manifest["items"][0].__setitem__("bundle_item_sha256", "0" * 64),
        ),
        (
            "source revision",
            lambda manifest: manifest.__setitem__("editorial_source_revision", "b" * 64),
        ),
    ],
)
async def test_bundle_integrity_binds_each_item_hash_and_the_authoritative_source_revision(
    db_session,
    label: str,
    mutate,
) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id=f"bound-{label.replace(' ', '-')}-bundle-001")
    mutate(manifest)
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    manifest["bundle_sha256"] = _sha256(unsigned)

    with pytest.raises(AppError) as exc_info:
        await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert exc_info.value.code == "BUNDLE_INTEGRITY_REJECTED"
    assert await _intake_record_count(db_session) == 0
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 0


async def test_bundle_integrity_rejects_a_short_source_revision_before_candidate_dispatch(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="short-source-revision-bundle-001")
    manifest["editorial_source_revision"] = "c3afbf1"
    manifest["items"][0]["artifact"]["revision_sha256"] = "c3afbf1"
    _rehash_manifest(manifest)

    with pytest.raises(AppError) as exc_info:
        await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert exc_info.value.code == "BUNDLE_INTEGRITY_REJECTED"
    assert await _intake_record_count(db_session) == 0
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 0


async def test_bundle_integrity_rejection_records_a_safe_import_audit_event_without_retaining_the_manifest(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="rejected-bundle-audit-001")
    manifest["items"][0]["artifact"]["entry"]["sensitive-audit-only-token"] = "never-persist-this-token"
    _rehash_manifest(manifest)

    with pytest.raises(AppError) as exc_info:
        await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert exc_info.value.code == "BUNDLE_INTEGRITY_REJECTED"
    audit_event = await db_session.scalar(select(CanonicalEventModel))
    assert audit_event is not None
    assert audit_event.aggregate_id.startswith("admission_attempt:")
    assert audit_event.aggregate_kind == "admission_attempt"
    assert audit_event.to_state == "rejected"
    assert audit_event.recorded_by == "member:administrator-001"
    assert audit_event.payload["schema"] == "reviewed_release_bundle_import_audit/v1"
    assert audit_event.payload["reasons"] == [
        {
            "field": "items",
            "code": "unsafe",
            "message": "bundle integrity validation failed",
        }
    ]
    assert "never-persist-this-token" not in json.dumps(audit_event.payload)
    assert "sensitive-audit-only-token" not in json.dumps(audit_event.payload)
    assert await _intake_record_count(db_session) == 0


async def test_item_validation_rejects_only_the_invalid_item_and_keeps_a_valid_sibling_admitted(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="mixed-validation-bundle-001")
    invalid_artifact = _approved_export(entry_id="invalid-approval-001")
    invalid_artifact["approval"]["status"] = "pending"
    manifest["items"].append(
        {
            "bundle_item_id": "mixed-validation-item-002",
            "operation": "create",
            "artifact_sha256": _sha256(invalid_artifact),
            "artifact": invalid_artifact,
        }
    )
    _rehash_manifest(manifest)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["state"] == "processing"
    assert accepted["items"][0]["state"] == "admitted"
    assert accepted["items"][0]["job_id"]
    assert accepted["items"][1] == {
        "bundle_item_id": "mixed-validation-item-002",
        "entry_identity": "entry:invalid-approval-001",
        "operation": "create",
        "state": "rejected",
        "artifact_sha256": manifest["items"][1]["artifact_sha256"],
        "bundle_item_sha256": manifest["items"][1]["bundle_item_sha256"],
        "allowed_next_action": "correct_item_in_new_bundle",
        "failure_reason": {
            "code": "EDITORIAL_APPROVAL_REQUIRED",
            "field": "approval.status",
            "message": "item must contain an approved editorial export",
        },
    }
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 1
    rejected_item = await db_session.get(CanonicalRecordModel, "bundle_item:mixed-validation-item-002")
    assert rejected_item is not None
    assert rejected_item.state == "rejected"


async def test_mixed_item_bundle_finishes_with_rejections_after_its_valid_candidate_is_ready(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="mixed-completion-bundle-001")
    invalid_artifact = _approved_export(entry_id="mixed-completion-invalid-entry-002")
    invalid_artifact["approval"]["status"] = "pending"
    manifest["items"].append(
        {
            "bundle_item_id": "mixed-completion-item-002",
            "operation": "create",
            "artifact_sha256": _sha256(invalid_artifact),
            "artifact": invalid_artifact,
        }
    )
    _rehash_manifest(manifest)

    accepted = await intake.import_bundle(manifest, actor_identity="member:administrator-001")
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    candidate = await CandidateBuildService(
        db_session,
        dense_index_service=_DenseIndexSpy(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    assert candidate["status"] == "candidate_ready"
    observed = await intake.get_bundle(accepted["bundle_id"])
    assert observed["state"] == "completed_with_rejections"
    completion_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"bundle:{accepted['bundle_id']}")
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
    )
    assert completion_event is not None
    assert completion_event.to_state == "completed_with_rejections"
    assert completion_event.payload["action"] == "candidate_work_completed_with_rejections"


async def test_independent_valid_items_each_receive_their_own_candidate_build_job(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="independent-valid-items-bundle-001")
    second_artifact = _approved_export(entry_id="independent-valid-entry-002")
    manifest["items"].append(
        {
            "bundle_item_id": "independent-valid-item-002",
            "operation": "create",
            "artifact_sha256": _sha256(second_artifact),
            "artifact": second_artifact,
        }
    )
    _rehash_manifest(manifest)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert [item["state"] for item in accepted["items"]] == ["admitted", "admitted"]
    job_ids = [item["job_id"] for item in accepted["items"]]
    assert len(set(job_ids)) == 2
    jobs = (
        await db_session.execute(
            select(CandidateBuildJob)
            .where(CandidateBuildJob.id.in_(job_ids))
            .order_by(CandidateBuildJob.entry_identity.asc())
        )
    ).scalars().all()
    assert [(job.entry_identity, job.status, job.requested_generation) for job in jobs] == [
        ("entry:independent-valid-entry-002", "queued", 1),
        ("entry:source-admission-001", "queued", 1),
    ]


@pytest.mark.parametrize(
    ("label", "mutate", "code", "field"),
    [
        (
            "metadata",
            lambda artifact: artifact["entry"].__setitem__("title", ""),
            "EDITORIAL_METADATA_INVALID",
            "entry.title",
        ),
        (
            "approval",
            lambda artifact: artifact["approval"].__setitem__("status", "pending"),
            "EDITORIAL_APPROVAL_REQUIRED",
            "approval.status",
        ),
        (
            "assurance",
            lambda artifact: artifact["entry"].__setitem__("assurance_level", "unrecognized"),
            "EDITORIAL_ASSURANCE_INVALID",
            "entry.assurance_level",
        ),
        (
            "access",
            lambda artifact: artifact["entry"]["sources"][0].__setitem__("access_scope", "private"),
            "EDITORIAL_SOURCE_UNAVAILABLE",
            "entry.sources",
        ),
        (
            "chunking",
            lambda artifact: artifact["entry"]["chunk_strategy"].__setitem__("overlap_characters", 900),
            "EDITORIAL_CHUNK_STRATEGY_INVALID",
            "entry.chunk_strategy",
        ),
        (
            "acceptance",
            lambda artifact: artifact["entry"]["acceptance_material"].__setitem__("supported_queries", []),
            "EDITORIAL_ACCEPTANCE_INVALID",
            "entry.acceptance_material.supported_queries",
        ),
    ],
)
async def test_item_validation_keeps_every_domain_failure_local_to_its_item(
    db_session,
    label: str,
    mutate,
    code: str,
    field: str,
) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id=f"isolated-item-{label}-bundle-001")
    invalid_artifact = _approved_export(entry_id=f"isolated-item-{label}-entry-002")
    mutate(invalid_artifact)
    manifest["items"].append(
        {
            "bundle_item_id": f"isolated-item-{label}-002",
            "operation": "create",
            "artifact_sha256": _sha256(invalid_artifact),
            "artifact": invalid_artifact,
        }
    )
    _rehash_manifest(manifest)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["items"][0]["state"] == "admitted"
    assert accepted["items"][1]["state"] == "rejected"
    assert accepted["items"][1]["failure_reason"]["code"] == code
    assert accepted["items"][1]["failure_reason"]["field"] == field
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 1


async def test_author_declared_source_availability_does_not_override_verified_authority(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="author-declared-availability-bundle-001")
    manifest["items"][0]["artifact"]["entry"]["sources"][0]["availability"] = "unavailable_for_new_evidence"
    _rehash_manifest(manifest)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["items"][0]["state"] == "admitted"
    assert accepted["items"][0]["allowed_next_action"] == "dispatch_candidate_build"


async def test_item_rejection_preserves_the_authority_blocking_field(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_StructuredSourceFailureVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="authority-blocking-field-bundle-001")
    unavailable_artifact = _approved_export(entry_id="authority-blocking-field-entry-002")
    manifest["items"].append(
        {
            "bundle_item_id": "authority-blocking-field-item-002",
            "operation": "create",
            "artifact_sha256": _sha256(unavailable_artifact),
            "artifact": unavailable_artifact,
        }
    )
    _rehash_manifest(manifest)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["items"][0]["state"] == "admitted"
    assert accepted["items"][1]["state"] == "rejected"
    assert accepted["items"][1]["failure_reason"] == {
        "code": "EDITORIAL_SOURCE_UNAVAILABLE",
        "field": "sources[0].availability",
        "message": "the retained source availability is not verified_usable",
    }


async def test_all_invalid_items_and_non_build_operations_remain_explicit_without_dispatching_jobs(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="explicit-item-plan-bundle-001")
    manifest["items"][0]["artifact"]["approval"]["status"] = "pending"

    no_op_artifact = _approved_export(entry_id="explicit-no-op-entry-002")
    withdrawal_artifact = _approved_export(entry_id="explicit-withdrawal-entry-003")
    manifest["items"].extend(
        [
            {
                "bundle_item_id": "explicit-no-op-item-002",
                "operation": "no_op",
                "artifact_sha256": _sha256(no_op_artifact),
                "artifact": no_op_artifact,
            },
            {
                "bundle_item_id": "explicit-withdrawal-item-003",
                "operation": "proposed_withdrawal",
                "artifact_sha256": _sha256(withdrawal_artifact),
                "artifact": withdrawal_artifact,
            },
        ]
    )
    _rehash_manifest(manifest)

    accepted = await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert accepted["state"] == "completed_with_rejections"
    assert accepted["items"][0]["state"] == "rejected"
    assert accepted["items"][0]["allowed_next_action"] == "correct_item_in_new_bundle"
    assert accepted["items"][1] == {
        "bundle_item_id": "explicit-no-op-item-002",
        "entry_identity": "entry:explicit-no-op-entry-002",
        "operation": "no_op",
        "state": "no_op",
        "artifact_sha256": manifest["items"][1]["artifact_sha256"],
        "bundle_item_sha256": manifest["items"][1]["bundle_item_sha256"],
        "allowed_next_action": "review_explicit_no_op",
    }
    assert accepted["items"][2] == {
        "bundle_item_id": "explicit-withdrawal-item-003",
        "entry_identity": "entry:explicit-withdrawal-entry-003",
        "operation": "proposed_withdrawal",
        "state": "proposed_withdrawal",
        "artifact_sha256": manifest["items"][2]["artifact_sha256"],
        "bundle_item_sha256": manifest["items"][2]["bundle_item_sha256"],
        "allowed_next_action": "requires_t04_publication_workflow",
    }
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 0


async def test_queued_item_builds_a_hidden_candidate_from_frozen_inputs_without_touching_published_documents(
    db_session,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="candidate-build-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    dense_index = _DenseIndexSpy()
    await _dispatch_candidate_build(db_session, job_id)

    candidate = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.status == "candidate_ready"
    assert job.stage == "indexing"
    assert job.progress == 100
    assert job.attempt == 1
    assert job.started_at is not None
    assert job.completed_at is not None
    assert candidate["candidate_id"].startswith("candidate:")
    assert candidate["entry_identity"] == "entry:source-admission-001"
    assert candidate["input_sha256"] == _bundle_manifest()["items"][0]["artifact_sha256"]
    assert dense_index.index_calls == [(job.document_identity, job.requested_generation, candidate["chunk_count"])]
    assert candidate["chunk_count"] > 0

    candidate_record = await db_session.get(CanonicalRecordModel, candidate["candidate_id"])
    assert candidate_record is not None
    assert candidate_record.state == "candidate_ready"
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == candidate["chunk_count"]
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0
    completed_bundle = await intake.get_bundle("candidate-build-bundle-001")
    assert completed_bundle["state"] == "completed"

    event_result = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    assert {event.to_state for event in event_result.scalars().all()} == {
        "queued",
        "parsing",
        "chunking",
        "indexing",
        "candidate_ready",
    }


async def test_bundle_completion_locks_the_bundle_before_reconstructing_candidate_status(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="serialized-bundle-completion-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.status = "candidate_ready"
    job.terminal_state = "candidate_ready"
    await db_session.commit()

    statements = []
    original_scalar = db_session.scalar

    async def capture_scalar(statement, *args, **kwargs):
        statements.append(statement)
        return await original_scalar(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "scalar", capture_scalar)

    assert await candidate_lifecycle.complete_bundle_when_candidate_work_is_finished(db_session, job.bundle_id)
    assert any(getattr(statement, "_for_update_arg", None) is not None for statement in statements)


async def test_parser_failure_records_a_retryable_candidate_job_without_indexing_or_publication(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="candidate-parser-failure-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    dense_index = _DenseIndexSpy()
    build = CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None

    def parser_failure(*_args, **_kwargs):
        assert job.stage == "parsing"
        raise AppError(
            status_code=422,
            code="CANDIDATE_PARSER_FAILED",
            message="editorial export could not be parsed for Candidate Build",
        )

    monkeypatch.setattr(build, "_parse_artifact", parser_failure, raising=False)
    await _dispatch_candidate_build(db_session, job_id)

    failed = await build.process_job(job_id)

    assert failed["status"] == "failed"
    assert failed["failure_reason"] == {
        "code": "CANDIDATE_PARSER_FAILED",
        "stage": "parsing",
        "message": "editorial export could not be parsed for Candidate Build",
    }
    assert failed["allowed_next_action"] == "retry_fixed_inputs"
    assert dense_index.index_calls == []
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_empty_candidate_chunks_record_a_retryable_chunking_failure_without_publication(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id="candidate-empty-chunks-bundle-001")
    manifest["items"][0]["artifact"]["entry"]["body"] = {"decision_query": "   "}
    _rehash_manifest(manifest)
    accepted = await intake.import_bundle(manifest, actor_identity="member:administrator-001")
    job_id = accepted["items"][0]["job_id"]
    dense_index = _DenseIndexSpy()
    await _dispatch_candidate_build(db_session, job_id)

    failed = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    assert failed["status"] == "failed"
    assert failed["failure_reason"] == {
        "code": "CANDIDATE_CHUNKING_EMPTY",
        "stage": "chunking",
        "message": "accepted editorial export did not produce any candidate chunks",
    }
    assert failed["allowed_next_action"] == "retry_fixed_inputs"
    assert dense_index.index_calls == []
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_index_failure_cleans_derived_assets_and_retry_reuses_only_the_immutable_input(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="retryable-candidate-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    failing_dense_index = _FailingDenseIndex()
    build = CandidateBuildService(
        db_session,
        dense_index_service=failing_dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    await _dispatch_candidate_build(db_session, job_id)

    failed = await build.process_job(job_id)

    assert failed == {
        "job_id": job_id,
        "status": "failed",
        "stage": "indexing",
        "progress": 75,
        "attempt": 1,
        "terminal_state": "failed",
        "failure_reason": {
            "code": "DENSE_INDEX_BACKEND_UNAVAILABLE",
            "stage": "indexing",
            "message": "dense index is unavailable",
        },
        "allowed_next_action": "retry_fixed_inputs",
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert failing_dense_index.delete_calls == [(job.document_identity, job.requested_generation, None)]
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0

    failure_event_result = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    failure_event = next(
        event for event in failure_event_result.scalars().all() if event.payload.get("action") == "failed"
    )
    assert failure_event.payload["failure_reason"] == failed["failure_reason"]
    assert failure_event.payload["allowed_next_action"] == "retry_fixed_inputs"

    retried = await build.retry_job(job_id, actor_identity="member:administrator-001")

    assert retried == {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "attempt": 2,
        "terminal_state": None,
        "failure_reason": None,
        "allowed_next_action": "cancel_or_await_candidate_build",
    }
    await db_session.refresh(job)
    assert job.input_sha256 == _bundle_manifest()["items"][0]["artifact_sha256"]
    assert job.candidate_id is None
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_retry_rechecks_the_current_locked_job_state_before_requeuing(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="retry-current-state-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    build = CandidateBuildService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    assert (await build.process_job(job_id))["status"] == "failed"

    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    await db_session.execute(
        update(CandidateBuildJob)
        .where(CandidateBuildJob.id == job_id)
        .values(
            status="superseded",
            terminal_state="superseded",
            allowed_next_action="none",
        ),
        execution_options={"synchronize_session": False},
    )
    await db_session.commit()
    assert job.status == "failed"

    with pytest.raises(AppError) as exc_info:
        await build.retry_job(job_id, actor_identity="member:administrator-001")

    assert exc_info.value.code == "CANDIDATE_RETRY_NOT_ALLOWED"
    await db_session.refresh(job)
    assert job.status == "superseded"
    assert job.attempt == 1


async def test_cancellation_reconciles_candidate_assets_and_preserves_the_existing_published_pointer(db_session) -> None:
    published_document = Document(
        id="published-document-cancel-001",
        filename="published-document-cancel-001.md",
        file_type="md",
        file_size=1,
        status="ready",
        published_generation=4,
        next_generation=5,
        latest_requested_generation=4,
    )
    db_session.add(published_document)
    await db_session.commit()

    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="canceled-candidate-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    dense_index = _FailingDenseIndex()
    build = CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )

    canceled = await build.cancel_job(job_id)

    assert canceled == {
        "job_id": job_id,
        "status": "canceled",
        "stage": "queued",
        "progress": 0,
        "attempt": 1,
        "terminal_state": "canceled",
        "failure_reason": {
            "code": "CANDIDATE_BUILD_CANCELED",
            "stage": "queued",
            "message": "Candidate Build was canceled by an administrator",
        },
        "allowed_next_action": "retry_fixed_inputs",
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert dense_index.delete_calls == [(job.document_identity, job.requested_generation, None)]
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    await db_session.refresh(published_document)
    assert published_document.published_generation == 4


async def test_candidate_worker_leaves_undispatched_work_queued_without_creating_derived_data(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="worker-undispatched-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    dense_index = _DenseIndexSpy()

    queued = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    assert queued == {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "attempt": 1,
        "terminal_state": None,
        "failure_reason": None,
        "allowed_next_action": "dispatch_candidate_build",
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.started_at is None
    assert dense_index.index_calls == []
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0


async def test_source_withdrawal_during_build_fails_the_item_without_creating_or_publishing_a_candidate(db_session) -> None:
    verifier = _SourceWithdrawsDuringBuildVerifier()
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=verifier,
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="source-withdrawal-during-build-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    failed = await CandidateBuildService(
        db_session,
        dense_index_service=_DenseIndexSpy(),
        editorial_export_verifier=verifier,
    ).process_job(job_id)

    assert failed == {
        "job_id": job_id,
        "status": "failed",
        "stage": "chunking",
        "progress": 45,
        "attempt": 1,
        "terminal_state": "failed",
        "failure_reason": {
            "code": "EDITORIAL_SOURCE_UNAVAILABLE",
            "stage": "chunking",
            "message": "source authority became unavailable during Candidate Build",
        },
        "allowed_next_action": "retry_fixed_inputs",
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.candidate_id is None
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_source_withdrawal_while_indexing_finishes_fails_before_candidate_persistence(db_session) -> None:
    verifier = _SourceWithdrawsDuringIndexingVerifier()
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=verifier,
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="source-withdrawal-while-indexing-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    dense_index = _DenseIndexSpy()
    await _dispatch_candidate_build(db_session, job_id)

    failed = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=verifier,
    ).process_job(job_id)

    assert failed["status"] == "failed"
    assert failed["terminal_state"] == "failed"
    assert failed["failure_reason"] == {
        "code": "EDITORIAL_SOURCE_UNAVAILABLE",
        "stage": "indexing",
        "message": "source authority became unavailable while Candidate Build indexing completed",
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.candidate_id is None
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_source_withdrawal_in_finalization_window_fails_before_candidate_persistence(db_session) -> None:
    verifier = _SourceWithdrawsDuringFinalizationVerifier()
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=verifier,
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="source-withdrawal-during-finalization-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    failed = await CandidateBuildService(
        db_session,
        dense_index_service=_DenseIndexSpy(),
        editorial_export_verifier=verifier,
    ).process_job(job_id)

    assert verifier.calls == 5
    assert failed["status"] == "failed"
    assert failed["terminal_state"] == "failed"
    assert failed["failure_reason"] == {
        "code": "EDITORIAL_SOURCE_UNAVAILABLE",
        "stage": "indexing",
        "message": "source authority became unavailable during Candidate finalization",
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.candidate_id is None
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("bundle_id", "bundle:mutated-bundle-002"),
        ("bundle_item_id", "bundle_item:mutated-item-002"),
        ("entry_identity", "entry:mutated-entry-002"),
        ("document_identity", "runtime-document:mutated-entry-002"),
        ("requested_generation", 99),
        ("editorial_source_revision", "b" * 64),
        ("input_sha256", "c" * 64),
        (
            "chunk_strategy",
            {
                "strategy_id": "mutated",
                "max_characters": 100,
                "overlap_characters": 0,
                "preserve_section_boundaries": True,
            },
        ),
        (
            "embedding_configuration",
            {
                "schema": "candidate_embedding_configuration/v1",
                "active": False,
                "embedding_model": "mutated-model",
                "dense_embedding_dim": 0,
                "fingerprint": "d" * 64,
            },
        ),
    ],
)
async def test_candidate_build_refuses_every_mutated_job_binding_and_never_uses_it_for_execution(
    db_session,
    field_name: str,
    replacement: object,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id=f"mutated-job-{field_name}-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    setattr(job, field_name, replacement)
    await db_session.commit()
    dense_index = _DenseIndexSpy()

    failed = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    assert failed["status"] == "failed"
    assert failed["failure_reason"]["code"] == "CANDIDATE_INPUT_INTEGRITY_FAILED"
    assert dense_index.index_calls == []
    await db_session.refresh(job)
    assert job.candidate_id is None


async def test_failed_cleanup_targets_the_frozen_embedding_collection_and_snapshot_never_retains_an_endpoint(db_session) -> None:
    settings = _active_dense_settings()
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
        settings=settings,
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="frozen-embedding-cleanup-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.embedding_configuration["active"] is True
    assert "embedding_base_url" not in job.embedding_configuration
    assert "do-not-retain" not in json.dumps(job.embedding_configuration)
    await _dispatch_candidate_build(db_session, job_id)

    dense_index = _FailingDenseIndex()
    await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
        settings=settings,
    ).process_job(job_id)

    assert dense_index.delete_calls == [
        (
            job.document_identity,
            job.requested_generation,
            job.embedding_configuration["fingerprint"],
        )
    ]


async def test_newer_bundle_generation_supersedes_a_ready_candidate_without_publishing_or_rewriting_it(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    first = await intake.import_bundle(
        _bundle_manifest(bundle_id="superseded-candidate-bundle-001"),
        actor_identity="member:administrator-001",
    )
    first_job_id = first["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, first_job_id)
    first_candidate = await CandidateBuildService(
        db_session,
        dense_index_service=_DenseIndexSpy(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(first_job_id)

    replacement_manifest = _bundle_manifest(bundle_id="superseding-candidate-bundle-002")
    replacement_manifest["items"][0]["bundle_item_id"] = "superseding-candidate-item-002"
    replacement_manifest["items"][0]["operation"] = "replace"
    _rehash_manifest(replacement_manifest)
    second = await intake.import_bundle(
        replacement_manifest,
        actor_identity="member:administrator-001",
    )

    first_job = await db_session.get(CandidateBuildJob, first_job_id)
    assert first_job is not None
    assert first_job.status == "superseded"
    assert first_job.terminal_state == "superseded"
    assert first_job.allowed_next_action == "none"
    assert first_job.candidate_id == first_candidate["candidate_id"]
    assert await db_session.get(CanonicalRecordModel, first_candidate["candidate_id"]) is not None

    second_job_id = second["items"][0]["job_id"]
    second_job = await db_session.get(CandidateBuildJob, second_job_id)
    assert second_job is not None
    assert second_job.requested_generation == 2
    assert second_job.status == "queued"
    assert second_job.candidate_id is None
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0

    candidate_events = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == first_candidate["candidate_id"])
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    assert candidate_events.scalars().all()[-1].to_state == "superseded"
    supersession_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{first_job_id}")
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
    )
    assert supersession_event is not None
    assert supersession_event.payload["action"] == "superseded"
    assert supersession_event.payload["stage"] == first_job.stage
    assert supersession_event.payload["status"] == "superseded"
    assert supersession_event.payload["progress"] == first_job.progress
    assert supersession_event.payload["editorial_source_revision"] == first_job.editorial_source_revision
    assert supersession_event.payload["input_sha256"] == first_job.input_sha256


async def test_supersession_while_indexing_cannot_restore_the_older_job_or_leave_derived_assets(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    first = await intake.import_bundle(
        _bundle_manifest(bundle_id="inflight-superseded-candidate-bundle-001"),
        actor_identity="member:administrator-001",
    )
    first_job_id = first["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, first_job_id)
    replacement_manifest = _bundle_manifest(bundle_id="inflight-superseding-candidate-bundle-002")
    replacement_manifest["items"][0]["bundle_item_id"] = "inflight-superseding-candidate-item-002"
    replacement_manifest["items"][0]["operation"] = "replace"
    _rehash_manifest(replacement_manifest)
    dense_index = _SupersedingDenseIndex(intake, replacement_manifest)

    result = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(first_job_id)

    first_job = await db_session.get(CandidateBuildJob, first_job_id)
    assert first_job is not None
    assert result["status"] == "superseded"
    assert first_job.status == "superseded"
    assert first_job.candidate_id is None
    assert first_job.derived_cleanup_pending is True
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) > 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0
    assert dense_index.delete_calls == []

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=dense_index,
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["reconciled_cleanup_job_ids"] == [first_job_id]
    await db_session.refresh(first_job)
    assert first_job.derived_cleanup_pending is False
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert dense_index.delete_calls == [(first_job.document_identity, first_job.requested_generation, None)]


async def test_startup_requeues_valid_work_and_marks_expired_running_leases_interrupted_and_retryable(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    queued_bundle = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-queued-bundle-001"),
        actor_identity="member:administrator-001",
    )
    queued_job_id = queued_bundle["items"][0]["job_id"]
    queued_job = await db_session.get(CandidateBuildJob, queued_job_id)
    assert queued_job is not None
    await _dispatch_candidate_build(db_session, queued_job_id)

    running_manifest = _bundle_manifest(bundle_id="startup-running-bundle-002")
    running_manifest["items"][0]["bundle_item_id"] = "startup-running-item-002"
    running_manifest["items"][0]["artifact"] = _approved_export(entry_id="startup-running-entry-002")
    _rehash_manifest(running_manifest)
    running_bundle = await intake.import_bundle(
        running_manifest,
        actor_identity="member:administrator-001",
    )
    running_job_id = running_bundle["items"][0]["job_id"]
    running_job = await db_session.get(CandidateBuildJob, running_job_id)
    assert running_job is not None
    running_job.status = "running"
    running_job.stage = "indexing"
    running_job.progress = 80
    running_job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db_session.add(
        CandidateBuildChunk(
            job_id=running_job.id,
            candidate_id=f"candidate:{running_job.id}-abandoned",
            document_identity=running_job.document_identity,
            generation=running_job.requested_generation,
            attempt=running_job.attempt,
            chunk_index=0,
            content="abandoned candidate chunk",
            content_sha256=hashlib.sha256(b"abandoned candidate chunk").hexdigest(),
            chunk_metadata={},
        )
    )
    await db_session.commit()
    dense_index = _FailingDenseIndex()
    enqueued_job_ids: list[str] = []

    async def enqueue(job_id: str) -> None:
        enqueued_job_ids.append(job_id)

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=dense_index,
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered == {
        "requeued_job_ids": [queued_job_id],
        "interrupted_job_ids": [running_job_id],
        "invalid_queued_job_ids": [],
        "invalid_running_job_ids": [],
        "reconciled_cleanup_job_ids": [running_job_id],
        "cleanup_pending_job_ids": [],
        "enqueue_failed_job_ids": [],
    }
    assert enqueued_job_ids == [queued_job_id]
    requeue_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{queued_job_id}")
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
    )
    assert requeue_event is not None
    assert requeue_event.payload["action"] == "requeued_on_startup"
    assert requeue_event.payload["editorial_source_revision"] == queued_job.editorial_source_revision
    await db_session.refresh(running_job)
    assert running_job.status == "interrupted_retryable"
    assert running_job.terminal_state == "interrupted_retryable"
    assert running_job.allowed_next_action == "retry_fixed_inputs"
    assert running_job.failure_reason == {
        "code": "CANDIDATE_WORKER_LEASE_INTERRUPTED",
        "stage": "indexing",
        "message": "startup found an expired Candidate Build worker lease",
    }
    assert dense_index.delete_calls == [(running_job.document_identity, running_job.requested_generation, None)]
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Document)) == 0


async def test_recovery_persists_the_interruption_fence_before_external_cleanup(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="recovery-fence-before-cleanup-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.status = "running"
    job.stage = "indexing"
    job.progress = 80
    job.lease_owner = "crashed-worker"
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db_session.add(
        CandidateBuildChunk(
            job_id=job.id,
            candidate_id=f"candidate:{job.id}-abandoned",
            document_identity=job.document_identity,
            generation=job.requested_generation,
            attempt=job.attempt,
            chunk_index=0,
            content="abandoned candidate chunk",
            content_sha256=hashlib.sha256(b"abandoned candidate chunk").hexdigest(),
            chunk_metadata={},
        )
    )
    await db_session.commit()
    dense_index = _RecoveryFenceDenseIndex(job)

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=dense_index,
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["interrupted_job_ids"] == [job_id]
    assert recovered["reconciled_cleanup_job_ids"] == [job_id]
    assert dense_index.observed_fence == ("interrupted_retryable", True, None)
    await db_session.refresh(job)
    assert job.status == "interrupted_retryable"
    assert job.derived_cleanup_pending is False


async def test_startup_does_not_report_or_audit_a_requeue_after_the_job_is_canceled(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-canceled-requeue-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    async def enqueue(_: str) -> None:
        await CandidateBuildService(
            db_session,
            dense_index_service=_FailingDenseIndex(),
            editorial_export_verifier=_ApprovedExportVerifier(),
        ).cancel_job(job_id)

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["requeued_job_ids"] == []
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.status == "canceled"
    events = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    assert "requeued_on_startup" not in [event.payload.get("action") for event in events.scalars().all()]


async def test_startup_does_not_requeue_candidate_work_before_an_administrator_dispatches_it(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-undispatched-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    enqueued_job_ids: list[str] = []

    async def enqueue(job_id: str) -> None:
        enqueued_job_ids.append(job_id)

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered == {
        "requeued_job_ids": [],
        "interrupted_job_ids": [],
        "invalid_queued_job_ids": [],
        "invalid_running_job_ids": [],
        "reconciled_cleanup_job_ids": [],
        "cleanup_pending_job_ids": [],
        "enqueue_failed_job_ids": [],
    }
    assert enqueued_job_ids == []
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.status == "queued"
    assert job.dispatched_at is None
    assert job.allowed_next_action == "dispatch_candidate_build"
    events = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    assert "requeued_on_startup" not in [event.payload.get("action") for event in events.scalars().all()]


async def test_runtime_and_recovery_refuse_a_mutable_dispatch_timestamp_without_append_only_admin_evidence(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="dispatch-evidence-required-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.dispatched_at = datetime.now(UTC)
    job.allowed_next_action = "cancel_or_await_candidate_build"
    await db_session.commit()

    def should_not_enqueue(_name: str):
        raise AssertionError("a mutable timestamp without a dispatch audit event must not enqueue Candidate work")

    monkeypatch.setattr(reviewed_bundles_runtime, "get_task_backend", should_not_enqueue)
    with pytest.raises(AppError) as exc_info:
        await CandidateBuildRuntime().enqueue(db_session, job_id)

    assert exc_info.value.code == "CANDIDATE_DISPATCH_REQUIRED"
    enqueued_job_ids: list[str] = []

    async def enqueue(candidate_job_id: str) -> None:
        enqueued_job_ids.append(candidate_job_id)

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["requeued_job_ids"] == []
    assert enqueued_job_ids == []


async def test_explicit_dispatch_repairs_a_timestamp_only_queued_job_with_current_attempt_evidence(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="dispatch-repair-timestamp-only-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.dispatched_at = datetime.now(UTC)
    job.allowed_next_action = "cancel_or_await_candidate_build"
    await db_session.commit()

    assert await CandidateBuildService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).dispatch_job(job_id, actor_identity="member:administrator-004") is True

    events = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    dispatch_events = [event for event in events.scalars().all() if event.payload["action"] == "dispatched"]
    assert len(dispatch_events) == 1
    assert dispatch_events[0].recorded_by == "member:administrator-004"
    assert dispatch_events[0].payload["attempt"] == 1


async def test_explicit_dispatch_and_retry_record_administrator_authority_for_the_current_attempt(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="dispatch-audit-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    build = CandidateBuildService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    )

    assert await build.dispatch_job(job_id, actor_identity="member:administrator-002") is True
    await build.cancel_job(job_id)
    retried = await build.retry_job(job_id, actor_identity="member:administrator-003")

    events = await db_session.execute(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    by_action = {event.payload["action"]: event for event in events.scalars().all()}
    assert by_action["dispatched"].recorded_by == "member:administrator-002"
    assert by_action["dispatched"].payload["attempt"] == 1
    assert by_action["retry_dispatched"].recorded_by == "member:administrator-003"
    assert by_action["retry_dispatched"].payload["attempt"] == retried["attempt"] == 2


async def test_runtime_refuses_to_enqueue_candidate_work_without_administrator_dispatch_evidence(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="runtime-undispatched-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]

    with pytest.raises(AppError) as exc_info:
        await CandidateBuildRuntime().enqueue(db_session, job_id)

    assert exc_info.value.code == "CANDIDATE_DISPATCH_REQUIRED"
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.status == "queued"
    assert job.dispatched_at is None


async def test_runtime_refreshes_current_job_state_before_enqueuing(db_session, monkeypatch: pytest.MonkeyPatch) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="runtime-current-state-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None

    await db_session.execute(
        update(CandidateBuildJob)
        .where(CandidateBuildJob.id == job_id)
        .values(status="canceled", terminal_state="canceled"),
        execution_options={"synchronize_session": False},
    )
    await db_session.commit()
    assert job.status == "queued"

    def should_not_enqueue(_name: str):
        raise AssertionError("a stale Candidate Build must not reach the task backend")

    monkeypatch.setattr(reviewed_bundles_runtime, "get_task_backend", should_not_enqueue)

    with pytest.raises(AppError) as exc_info:
        await CandidateBuildRuntime().enqueue(db_session, job_id)

    assert exc_info.value.code == "CANDIDATE_ENQUEUE_NOT_ALLOWED"


async def test_startup_marks_a_previous_worker_owner_interrupted_even_before_its_lease_expires(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-previous-owner-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.status = "running"
    job.stage = "indexing"
    job.progress = 80
    job.lease_owner = "previous-process-owner"
    job.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    await db_session.commit()

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(
        enqueue=enqueue,
        now=datetime.now(UTC),
        recovery_owner="current-process-owner",
    )

    assert recovered["interrupted_job_ids"] == [job_id]
    await db_session.refresh(job)
    assert job.status == "interrupted_retryable"
    assert job.failure_reason["code"] == "CANDIDATE_WORKER_LEASE_INTERRUPTED"


async def test_startup_does_not_interrupt_a_job_that_completed_before_its_locked_recovery_check(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-current-running-state-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.status = "running"
    job.stage = "indexing"
    job.progress = 80
    job.lease_owner = "crashed-worker"
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await db_session.commit()

    original_refresh = db_session.refresh
    state_changed = False

    async def refresh_with_completed_state(instance, *args, **kwargs):
        nonlocal state_changed
        if (
            not state_changed
            and isinstance(instance, CandidateBuildJob)
            and instance.id == job_id
            and kwargs.get("with_for_update") is True
        ):
            state_changed = True
            await db_session.execute(
                update(CandidateBuildJob)
                .where(CandidateBuildJob.id == job_id)
                .values(
                    status="candidate_ready",
                    terminal_state="candidate_ready",
                    lease_owner=None,
                    lease_expires_at=None,
                ),
                execution_options={"synchronize_session": False},
            )
            await db_session.commit()
        await original_refresh(instance, *args, **kwargs)

    monkeypatch.setattr(db_session, "refresh", refresh_with_completed_state)

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["interrupted_job_ids"] == []
    await db_session.refresh(job)
    assert job.status == "candidate_ready"


async def test_startup_does_not_reconcile_cleanup_after_the_locked_job_has_been_retried(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-current-cleanup-state-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.status = "failed"
    job.stage = "indexing"
    job.terminal_state = "failed"
    job.derived_cleanup_pending = True
    job.failure_reason = {
        "code": "DENSE_INDEX_BACKEND_UNAVAILABLE",
        "stage": "indexing",
        "message": "dense index is unavailable",
    }
    db_session.add(
        CandidateBuildChunk(
            job_id=job.id,
            candidate_id=f"candidate:{job.id}-cleanup-pending",
            document_identity=job.document_identity,
            generation=job.requested_generation,
            attempt=job.attempt,
            chunk_index=0,
            content="candidate cleanup must remain untouched after retry",
            content_sha256=hashlib.sha256(b"candidate cleanup must remain untouched after retry").hexdigest(),
            chunk_metadata={},
        )
    )
    await db_session.commit()

    original_refresh = db_session.refresh
    state_changed = False

    async def refresh_with_retried_state(instance, *args, **kwargs):
        nonlocal state_changed
        if (
            not state_changed
            and isinstance(instance, CandidateBuildJob)
            and instance.id == job_id
            and kwargs.get("with_for_update") is True
        ):
            state_changed = True
            await db_session.execute(
                update(CandidateBuildJob)
                .where(CandidateBuildJob.id == job_id)
                .values(
                    status="queued",
                    terminal_state=None,
                    derived_cleanup_pending=False,
                    allowed_next_action="await_candidate_build",
                ),
                execution_options={"synchronize_session": False},
            )
            await db_session.commit()
        await original_refresh(instance, *args, **kwargs)

    monkeypatch.setattr(db_session, "refresh", refresh_with_retried_state)
    dense_index = _FailingDenseIndex()

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=dense_index,
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["reconciled_cleanup_job_ids"] == []
    assert dense_index.delete_calls == []
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 1


async def test_retry_requires_a_new_bundle_when_immutable_input_records_are_missing(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="retry-corrupted-input-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    build = CandidateBuildService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    await build.process_job(job_id)

    await db_session.execute(
        text("DELETE FROM canonical_records WHERE stable_id = :stable_id"),
        {"stable_id": f"build_generation:{job_id}"},
    )
    await db_session.commit()

    with pytest.raises(AppError) as exc_info:
        await build.retry_job(job_id, actor_identity="member:administrator-001")

    assert exc_info.value.code == "CANDIDATE_RETRY_REQUIRES_NEW_BUNDLE"
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.allowed_next_action == "import_new_bundle"
    retry_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
    )
    assert retry_event is not None
    assert retry_event.payload["action"] == "retry_requires_new_bundle"
    assert retry_event.payload["allowed_next_action"] == "import_new_bundle"
    assert retry_event.payload["failure_reason"]["code"] == "CANDIDATE_RETRY_REQUIRES_NEW_BUNDLE"


async def test_retry_cleanup_refusal_is_retained_in_the_candidate_event_audit(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="retry-cleanup-audit-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    build = CandidateBuildService(
        db_session,
        dense_index_service=_CleanupFailingDenseIndex(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    assert (await build.process_job(job_id))["status"] == "failed"

    with pytest.raises(AppError) as exc_info:
        await build.retry_job(job_id, actor_identity="member:administrator-001")

    assert exc_info.value.code == "CANDIDATE_DERIVED_CLEANUP_REQUIRED"
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.allowed_next_action == "reconcile_derived_data_then_retry"
    retry_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_id == f"build_generation:{job_id}")
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
    )
    assert retry_event is not None
    assert retry_event.payload["action"] == "retry_cleanup_required"
    assert retry_event.payload["allowed_next_action"] == "reconcile_derived_data_then_retry"
    assert retry_event.payload["failure_reason"]["code"] == "CANDIDATE_DERIVED_CLEANUP_REQUIRED"


async def test_startup_marks_failed_reenqueue_with_a_structured_retryable_reason(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="startup-enqueue-failure-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    await _dispatch_candidate_build(db_session, job_id)

    async def enqueue(_: str) -> None:
        raise RuntimeError("queue unavailable")

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered == {
        "requeued_job_ids": [],
        "interrupted_job_ids": [],
        "invalid_queued_job_ids": [],
        "invalid_running_job_ids": [],
        "reconciled_cleanup_job_ids": [],
        "cleanup_pending_job_ids": [],
        "enqueue_failed_job_ids": [job_id],
    }
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.failure_reason == {
        "code": "CANDIDATE_ENQUEUE_FAILED",
        "stage": "queued",
        "message": "Candidate Build could not be returned to the queue",
    }
    assert job.allowed_next_action == "retry_fixed_inputs"


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "boolean schema version",
            lambda manifest: manifest.__setitem__("schema_version", True),
        ),
        (
            "numeric bundle identity",
            lambda manifest: manifest.__setitem__("bundle_id", 123456),
        ),
        (
            "numeric bundle item identity",
            lambda manifest: manifest["items"][0].__setitem__("bundle_item_id", 123456),
        ),
        (
            "whitespace padded bundle identity",
            lambda manifest: manifest.__setitem__("bundle_id", " strict-types-bundle-001 "),
        ),
        (
            "whitespace padded bundle item identity",
            lambda manifest: manifest["items"][0].__setitem__("bundle_item_id", " reviewed-bundle-item-001 "),
        ),
    ],
)
async def test_bundle_integrity_requires_exact_schema_and_stable_identity_types(
    db_session,
    label: str,
    mutate,
) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    manifest = _bundle_manifest(bundle_id=f"strict-types-{label.replace(' ', '-')}-001")
    mutate(manifest)
    _rehash_manifest(manifest)

    with pytest.raises(AppError) as exc_info:
        await service.import_bundle(manifest, actor_identity="member:administrator-001")

    assert exc_info.value.code == "BUNDLE_INTEGRITY_REJECTED"
    assert await _intake_record_count(db_session) == 0
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 0


async def test_bundle_item_identity_collision_is_a_safe_whole_bundle_conflict(db_session) -> None:
    service = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    await service.import_bundle(
        _bundle_manifest(bundle_id="identity-collision-original-001"),
        actor_identity="member:administrator-001",
    )
    conflicting = _bundle_manifest(bundle_id="identity-collision-conflict-002")
    conflicting["items"][0]["artifact"] = _approved_export(entry_id="identity-collision-entry-002")
    _rehash_manifest(conflicting)

    with pytest.raises(AppError) as exc_info:
        await service.import_bundle(conflicting, actor_identity="member:administrator-001")

    assert exc_info.value.code == "BUNDLE_ITEM_ID_CONFLICT"
    assert await _intake_record_count(db_session) == 3
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildJob)) == 1
    audit_event = await db_session.scalar(
        select(CanonicalEventModel)
        .where(CanonicalEventModel.aggregate_kind == "admission_attempt")
        .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
    )
    assert audit_event is not None
    assert audit_event.aggregate_id.startswith("admission_attempt:")
    assert audit_event.payload == {
        "schema": "reviewed_release_bundle_import_audit/v1",
        "action": "identity_conflict",
        "reasons": [
            {
                "field": "items",
                "code": "conflict",
                "message": "bundle integrity validation failed",
            }
        ],
    }


async def test_candidate_embedding_identity_excludes_endpoint_and_uses_a_frozen_candidate_collection(
    db_session,
) -> None:
    original_settings = _active_dense_settings()
    rotated_endpoint_settings = Settings(
        EMBEDDING_API_KEY="different-candidate-build-test-key",
        EMBEDDING_BASE_URL="https://different-token@different-embeddings.example.test/v1?access_token=different",
        EMBEDDING_MODEL="candidate-build-test-model",
        DENSE_EMBEDDING_DIM=3,
        MILVUS_URI="http://milvus.example.test:19530",
    )
    assert candidate_embedding_configuration(original_settings) == candidate_embedding_configuration(
        rotated_endpoint_settings
    )

    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
        settings=original_settings,
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="endpoint-independent-candidate-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    dense_index = _DenseIndexSpy()
    await _dispatch_candidate_build(db_session, job_id)

    candidate = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
        settings=rotated_endpoint_settings,
    ).process_job(job_id)

    assert candidate["status"] == "candidate_ready"
    assert dense_index.index_fingerprints == [job.embedding_configuration["fingerprint"]]
    stored_configuration = json.dumps(job.embedding_configuration)
    assert "embeddings.example.test" not in stored_configuration
    assert "different-token" not in stored_configuration
    assert "candidate-build-test-key" not in stored_configuration


async def test_long_running_indexing_renews_the_candidate_worker_lease(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_build_module, "_LEASE_DURATION", timedelta(milliseconds=50))
    monkeypatch.setattr(candidate_build_module, "_LEASE_HEARTBEAT_INTERVAL_SECONDS", 0.01, raising=False)
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="long-indexing-lease-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    candidate = await CandidateBuildService(
        db_session,
        dense_index_service=_SlowDenseIndex(),
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert candidate["status"] == "candidate_ready"
    assert job.status == "candidate_ready"
    assert job.heartbeat_at is not None
    assert job.started_at is not None
    assert job.heartbeat_at >= job.started_at


async def test_lost_indexing_lease_leaves_derived_data_for_fenced_recovery(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_build_module, "_LEASE_DURATION", timedelta(milliseconds=50))
    monkeypatch.setattr(candidate_build_module, "_LEASE_HEARTBEAT_INTERVAL_SECONDS", 0.1, raising=False)
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="lost-indexing-lease-bundle-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    dense_index = _SlowDenseIndex()
    result = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert result["status"] == "running"
    assert job.status == "running"
    assert job.failure_reason is None
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) > 0
    assert dense_index.delete_calls == []

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=dense_index,
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["interrupted_job_ids"] == [job_id]
    await db_session.refresh(job)
    assert job.status == "interrupted_retryable"
    assert job.failure_reason["code"] == "CANDIDATE_WORKER_LEASE_INTERRUPTED"
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 0
    assert dense_index.delete_calls == [(job.document_identity, job.requested_generation, None)]


async def test_lease_heartbeat_loss_cancels_inflight_indexing_before_it_can_write_or_clean_up(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(candidate_build_module, "_LEASE_HEARTBEAT_INTERVAL_SECONDS", 0.01, raising=False)
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="heartbeat-fence-candidate-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    dense_index = _CancellationAwareDenseIndex()
    build = CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
        lease_owner="heartbeat-owner",
    )

    async def lose_lease(*, job_id: str, attempt: int) -> bool:
        await db_session.execute(
            update(CandidateBuildJob)
            .where(CandidateBuildJob.id == job_id, CandidateBuildJob.attempt == attempt)
            .values(
                lease_owner="replacement-worker",
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
            ),
            execution_options={"synchronize_session": False},
        )
        await db_session.commit()
        return False

    monkeypatch.setattr(build, "_renew_lease", lose_lease)
    result = await asyncio.wait_for(build.process_job(job_id), timeout=0.5)

    assert result["status"] == "running"
    assert dense_index.started.is_set()
    assert dense_index.cancelled is True
    assert dense_index.finished is False
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert job.status == "running"
    assert job.lease_owner == "replacement-worker"
    assert job.candidate_id is None
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) > 0


class _LeaseStolenDuringIndexingDenseIndex(_DenseIndexSpy):
    def __init__(self, db_session, job_id: str) -> None:
        super().__init__()
        self._session = db_session
        self._job_id = job_id

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        job = await self._session.get(CandidateBuildJob, self._job_id)
        assert job is not None
        job.attempt += 1
        job.lease_owner = "replacement-worker"
        job.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)
        await self._session.commit()
        return await super().index_candidate_generation(
            document_id=document_id,
            generation=generation,
            chunks=chunks,
            embedding_fingerprint=embedding_fingerprint,
        )


async def test_stale_worker_cannot_finalize_after_another_worker_claims_the_attempt(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="lease-fence-candidate-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    dense_index = _LeaseStolenDuringIndexingDenseIndex(db_session, job_id)
    await _dispatch_candidate_build(db_session, job_id)

    result = await CandidateBuildService(
        db_session,
        dense_index_service=dense_index,
        editorial_export_verifier=_ApprovedExportVerifier(),
        lease_owner="original-worker",
    ).process_job(job_id)

    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert result["status"] == "running"
    assert job.status == "running"
    assert job.attempt == 2
    assert job.lease_owner == "replacement-worker"
    assert job.candidate_id is None
    assert await db_session.scalar(
        select(func.count())
        .select_from(CanonicalRecordModel)
        .where(CanonicalRecordModel.identity_kind == "candidate")
    ) == 0


class _BindingMutatingDenseIndex(_DenseIndexSpy):
    def __init__(self, db_session, job_id: str) -> None:
        super().__init__()
        self._session = db_session
        self._job_id = job_id

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks,
        embedding_fingerprint: str | None = None,
    ) -> object:
        job = await self._session.get(CandidateBuildJob, self._job_id)
        assert job is not None
        job.document_identity = "runtime-document:tampered-after-indexing"
        await self._session.commit()
        return await super().index_candidate_generation(
            document_id=document_id,
            generation=generation,
            chunks=chunks,
            embedding_fingerprint=embedding_fingerprint,
        )


async def test_candidate_finalization_rechecks_frozen_inputs_before_persisting_a_candidate(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="candidate-finalization-recheck-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)

    failed = await CandidateBuildService(
        db_session,
        dense_index_service=_BindingMutatingDenseIndex(db_session, job_id),
        editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id)

    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    assert failed["status"] == "failed"
    assert failed["stage"] == "indexing"
    assert failed["failure_reason"]["code"] == "CANDIDATE_INPUT_INTEGRITY_FAILED"
    assert job.candidate_id is None
    assert job.derived_cleanup_pending is True
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) > 0


async def test_recovery_preserves_derived_assets_when_the_frozen_input_no_longer_matches(db_session) -> None:
    intake = ReviewedReleaseBundleService(
        db_session,
        editorial_export_verifier=_ApprovedExportVerifier(),
    )
    accepted = await intake.import_bundle(
        _bundle_manifest(bundle_id="recovery-input-mismatch-001"),
        actor_identity="member:administrator-001",
    )
    job_id = accepted["items"][0]["job_id"]
    job = await db_session.get(CandidateBuildJob, job_id)
    assert job is not None
    job.status = "running"
    job.stage = "indexing"
    job.progress = 80
    job.lease_owner = "crashed-worker"
    job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    job.document_identity = "runtime-document:tampered-before-recovery"
    db_session.add(
        CandidateBuildChunk(
            job_id=job.id,
            candidate_id=f"candidate:{job.id}-abandoned",
            document_identity="runtime-document:source-admission-001",
            generation=job.requested_generation,
            attempt=job.attempt,
            chunk_index=0,
            content="candidate data awaiting manual reconciliation",
            content_sha256=hashlib.sha256(b"candidate data awaiting manual reconciliation").hexdigest(),
            chunk_metadata={},
        )
    )
    await db_session.commit()

    async def enqueue(_: str) -> None:
        return None

    recovered = await CandidateBuildRecoveryService(
        db_session,
        dense_index_service=_FailingDenseIndex(),
    ).recover(enqueue=enqueue, now=datetime.now(UTC))

    assert recovered["invalid_running_job_ids"] == [job_id]
    await db_session.refresh(job)
    assert job.status == "failed"
    assert job.stage == "indexing"
    assert job.failure_reason["code"] == "CANDIDATE_INPUT_INTEGRITY_FAILED"
    assert job.derived_cleanup_pending is True
    assert job.allowed_next_action == "import_new_bundle"
    assert await db_session.scalar(select(func.count()).select_from(CandidateBuildChunk)) == 1
