from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import require_admin
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.contracts.canonical import StableIdentityKind
from app.infra.db import get_db_session
from app.model.canonical import CanonicalEventModel
from app.retrieval.candidate_pool import AuthorizedRetrievalCandidatePool
from app.reviewed_bundles.build_service import CandidateBuildService
from app.reviewed_bundles.models import CandidateBuildJob
from app.reviewed_bundles.publication import CandidatePublicationService
from app.reviewed_bundles.runtime import candidate_build_runtime
from app.reviewed_bundles.service import ReviewedReleaseBundleService
from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier
from app.service.identity_audit_service import IdentityAuditService
from app.settings.runtime import get_runtime_settings

router = APIRouter(prefix="/reviewed-release-bundles", tags=["reviewed-release-bundles"])


def _ok(data: dict) -> dict:
    return ok_response(data=data, request_id=get_request_id())


def _candidate_publication_service(session: AsyncSession) -> CandidatePublicationService:
    return CandidatePublicationService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
        settings=get_runtime_settings(),
    )


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


async def _serialize_job(session: AsyncSession, job: CandidateBuildJob) -> dict:
    event_result = await session.execute(
        select(CanonicalEventModel)
        .where(
            CanonicalEventModel.aggregate_id == f"build_generation:{job.id}",
            CanonicalEventModel.aggregate_kind == StableIdentityKind.BUILD_GENERATION.value,
        )
        .order_by(CanonicalEventModel.occurred_at.asc(), CanonicalEventModel.id.asc())
    )
    return {
        "job_id": job.id,
        "bundle_id": job.bundle_id,
        "bundle_item_id": job.bundle_item_id,
        "entry_identity": job.entry_identity,
        "document_identity": job.document_identity,
        "requested_generation": job.requested_generation,
        "editorial_source_revision": job.editorial_source_revision,
        "input_sha256": job.input_sha256,
        "frozen_input_sha256": job.frozen_input_sha256,
        "chunk_strategy": job.chunk_strategy,
        "embedding_configuration": job.embedding_configuration,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "attempt": job.attempt,
        "terminal_state": job.terminal_state,
        "failure_reason": job.failure_reason,
        "allowed_next_action": job.allowed_next_action,
        "candidate_id": job.candidate_id,
        "derived_cleanup_pending": job.derived_cleanup_pending,
        "dispatched_at": _timestamp(job.dispatched_at),
        "started_at": _timestamp(job.started_at),
        "heartbeat_at": _timestamp(job.heartbeat_at),
        "lease_expires_at": _timestamp(job.lease_expires_at),
        "completed_at": _timestamp(job.completed_at),
        "created_at": _timestamp(job.created_at),
        "updated_at": _timestamp(job.updated_at),
        "events": [
            {
                "event_id": event.id,
                "event_type": event.event_type,
                "from_state": event.from_state,
                "to_state": event.to_state,
                "occurred_at": _timestamp(event.occurred_at),
                "payload": event.payload,
                "recorded_by": event.recorded_by,
            }
            for event in event_result.scalars().all()
        ],
    }


async def _get_job(session: AsyncSession, job_id: str) -> CandidateBuildJob:
    job = await session.get(CandidateBuildJob, job_id)
    if job is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="Candidate Build job not found")
    return job


async def _enqueue_or_record_failure(session: AsyncSession, job_id: str) -> None:
    try:
        await candidate_build_runtime.enqueue(session, job_id)
    except Exception:
        await CandidateBuildService(
            session,
            editorial_export_verifier=CanonicalEditorialExportVerifier(session),
        ).mark_enqueue_failed(job_id)


@router.post("/import")
async def import_reviewed_release_bundle(
    manifest: object = Body(...),
    current_user=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    actor_identity = await IdentityAuditService(session).ensure_member_record(
        current_user,
        admission_path="reviewed_release_bundle_intake",
    )
    service = ReviewedReleaseBundleService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
    )
    result = await service.import_bundle(manifest, actor_identity=actor_identity)
    return _ok(await service.get_bundle(result["bundle_id"]))


@router.get("")
async def list_reviewed_release_bundles(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    items = await ReviewedReleaseBundleService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
    ).list_bundles()
    return _ok({"items": items})


@router.get("/jobs/{job_id}")
async def get_candidate_build_job(
    job_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    return _ok(await _serialize_job(session, await _get_job(session, job_id)))


@router.get("/candidates/{candidate_id}/preview")
async def preview_candidate(
    candidate_id: str,
    query: str = Query(..., min_length=1, max_length=4000),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await AuthorizedRetrievalCandidatePool(
        session,
        settings=get_runtime_settings(),
    ).preview_candidate(candidate_id, query)
    return _ok(
        {
            "items": result.items,
            "strategy": result.strategy,
            "profile_identity": result.profile_identity,
            "candidate_pool_scope": result.candidate_pool_scope,
            "candidate_exclusions": result.candidate_exclusions,
        }
    )


@router.get("/candidates/{candidate_id}/inspection")
async def get_candidate_inspection(
    candidate_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await _candidate_publication_service(session).get_inspection(candidate_id)
    return _ok(result)


@router.post("/candidates/{candidate_id}/inspection")
async def inspect_candidate(
    candidate_id: str,
    current_user=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    actor_identity = await IdentityAuditService(session).ensure_member_record(
        current_user,
        admission_path="candidate_inspection",
    )
    result = await _candidate_publication_service(session).inspect(candidate_id, actor_identity=actor_identity)
    return _ok(result)


@router.post("/candidates/{candidate_id}/acceptance")
async def accept_candidate(
    candidate_id: str,
    current_user=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    actor_identity = await IdentityAuditService(session).ensure_member_record(
        current_user,
        admission_path="candidate_acceptance",
    )
    result = await _candidate_publication_service(session).accept(candidate_id, actor_identity=actor_identity)
    return _ok(result)


@router.get("/candidates/{candidate_id}/publication-eligibility")
async def candidate_publication_eligibility(
    candidate_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await _candidate_publication_service(session).publication_eligibility(candidate_id)
    return _ok(result)


@router.get("/candidates/{candidate_id}/publication")
async def get_candidate_publication(
    candidate_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await _candidate_publication_service(session).get_publication(candidate_id)
    return _ok(result)


@router.post("/publication-batches")
async def publish_candidate_batch(
    payload: object = Body(...),
    current_user=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    actor_identity = await IdentityAuditService(session).ensure_member_record(
        current_user,
        admission_path="candidate_publication",
    )
    result = await _candidate_publication_service(session).confirm_publication_batch(
        payload,
        actor_identity=actor_identity,
    )
    return _ok(result)


@router.post("/jobs/{job_id}/dispatch")
async def dispatch_candidate_build_job(
    job_id: str,
    current_user=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    actor_identity = await IdentityAuditService(session).ensure_member_record(
        current_user,
        admission_path="reviewed_candidate_build_dispatch",
    )
    service = CandidateBuildService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
    )
    await service.dispatch_job(job_id, actor_identity=actor_identity)
    await _enqueue_or_record_failure(session, job_id)
    return _ok(await _serialize_job(session, await _get_job(session, job_id)))


@router.post("/jobs/{job_id}/retry")
async def retry_candidate_build_job(
    job_id: str,
    current_user=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    actor_identity = await IdentityAuditService(session).ensure_member_record(
        current_user,
        admission_path="reviewed_candidate_build_retry",
    )
    service = CandidateBuildService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
    )
    await service.retry_job(job_id, actor_identity=actor_identity)
    await _enqueue_or_record_failure(session, job_id)
    return _ok(await _serialize_job(session, await _get_job(session, job_id)))


@router.post("/jobs/{job_id}/cancel")
async def cancel_candidate_build_job(
    job_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    service = CandidateBuildService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
    )
    try:
        reached_worker = await candidate_build_runtime.cancel(job_id)
    except Exception:
        await service.mark_cancellation_request_failed(job_id)
    else:
        if reached_worker:
            await service.cancel_job(job_id)
        else:
            await service.mark_cancellation_request_failed(job_id)
    return _ok(await _serialize_job(session, await _get_job(session, job_id)))


@router.get("/{bundle_id}")
async def get_reviewed_release_bundle(
    bundle_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await ReviewedReleaseBundleService(
        session,
        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
    ).get_bundle(bundle_id)
    return _ok(result)
