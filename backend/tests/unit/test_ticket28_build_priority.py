import asyncio

import pytest

from app.common.exceptions import AppError
from app.operations.chat_capacity import get_chat_admission_gate
from app.operations.service import OperationsService
from app.reviewed_bundles.build_service import CandidateBuildService
from app.reviewed_bundles.service import ReviewedReleaseBundleService
from tests.unit.test_reviewed_release_bundle_intake import (
    _ApprovedExportVerifier,
    _bundle_manifest,
    _DenseIndexSpy,
    _dispatch_candidate_build,
)


async def test_operations_counts_reviewed_candidate_queue_without_export_content(db_session):
    service = ReviewedReleaseBundleService(db_session, editorial_export_verifier=_ApprovedExportVerifier())
    await service.import_bundle(_bundle_manifest(), actor_identity="member:administrator-001")
    projection = await OperationsService(db_session).read()
    assert projection["documents"]["queued_builds"] == 1
    assert projection["documents"]["running_builds"] == 0
    assert "artifact" not in str(projection)


async def test_operations_exposes_only_supported_candidate_retry_without_failure_body(db_session):
    service = ReviewedReleaseBundleService(db_session, editorial_export_verifier=_ApprovedExportVerifier())
    bundle = await service.import_bundle(_bundle_manifest(), actor_identity="member:administrator-001")
    job_id = bundle["items"][0]["job_id"]
    await CandidateBuildService(
        db_session, editorial_export_verifier=_ApprovedExportVerifier(),
    ).mark_enqueue_failed(job_id)
    projection = await OperationsService(db_session).read()
    assert any(item["kind"] == "candidate_build" for item in projection["failures"])
    assert {
        "action": "retry_candidate_build", "job_id": job_id, "method": "POST",
        "path": f"/api/v1/reviewed-release-bundles/jobs/{job_id}/retry",
    } in projection["retry_actions"]
    assert "failure_reason" not in str(projection)


async def test_dispatched_build_waits_while_interactive_answer_is_executing(db_session):
    service = ReviewedReleaseBundleService(db_session, editorial_export_verifier=_ApprovedExportVerifier())
    bundle = await service.import_bundle(_bundle_manifest(), actor_identity="member:administrator-001")
    job_id = bundle["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    gate = get_chat_admission_gate()
    reservation = gate.reserve(member_id="interactive-member")
    task = asyncio.create_task(CandidateBuildService(
        db_session, editorial_export_verifier=_ApprovedExportVerifier(),
    ).process_job(job_id))
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        gate.finish(reservation)
        result = await asyncio.wait_for(task, timeout=5)
        assert result["status"] == "candidate_ready"
    finally:
        gate.finish(reservation)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_running_build_yields_at_stage_boundary_when_interactive_work_arrives(db_session):
    service = ReviewedReleaseBundleService(db_session, editorial_export_verifier=_ApprovedExportVerifier())
    bundle = await service.import_bundle(_bundle_manifest(), actor_identity="member:administrator-001")
    job_id = bundle["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    gate = get_chat_admission_gate()
    entered = asyncio.Event()
    reservations = []

    class InteractiveArrival(_ApprovedExportVerifier):
        async def verify(self, artifact, artifact_sha256):
            result = await super().verify(artifact, artifact_sha256)
            if not reservations:
                reservations.append(gate.reserve(member_id="arriving-member"))
                entered.set()
            return result

    task = asyncio.create_task(CandidateBuildService(
        db_session, editorial_export_verifier=InteractiveArrival(),
    ).process_job(job_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        gate.finish(reservations[0])
        result = await asyncio.wait_for(task, timeout=5)
        assert result["status"] == "candidate_ready"
    finally:
        for reservation in reservations:
            gate.finish(reservation)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_build_rechecks_revoked_authority_after_yield_before_indexing(db_session):
    service = ReviewedReleaseBundleService(db_session, editorial_export_verifier=_ApprovedExportVerifier())
    bundle = await service.import_bundle(_bundle_manifest(), actor_identity="member:administrator-001")
    job_id = bundle["items"][0]["job_id"]
    await _dispatch_candidate_build(db_session, job_id)
    gate = get_chat_admission_gate()
    entered = asyncio.Event()
    held = []

    class RevocableVerifier(_ApprovedExportVerifier):
        revoked = False
        calls = 0

        async def verify(self, artifact, artifact_sha256):
            if self.revoked:
                raise AppError(status_code=409, code="EDITORIAL_AUTHORITY_REVOKED", message="review authority is unavailable")
            result = await super().verify(artifact, artifact_sha256)
            self.calls += 1
            if self.calls == 2:
                held.append(gate.reserve(member_id="interactive-revocation"))
                entered.set()
            return result

    verifier = RevocableVerifier()
    indexer = _DenseIndexSpy()
    task = asyncio.create_task(CandidateBuildService(
        db_session, editorial_export_verifier=verifier, dense_index_service=indexer,
    ).process_job(job_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
        verifier.revoked = True
        gate.finish(held[0])
        result = await asyncio.wait_for(task, timeout=5)
        assert result["status"] == "failed"
        assert indexer.index_calls == []
    finally:
        for reservation in held:
            gate.finish(reservation)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
