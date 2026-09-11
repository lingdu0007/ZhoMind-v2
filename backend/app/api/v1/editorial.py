from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import get_current_user
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.editorial_authority.schemas import (
    CreateEditorialEntryRequest,
    RecordIntegrityReview,
    RecordSourceAvailabilityRequest,
    RequestFreshnessReview,
    ReviseEditorialEntryRequest,
)
from app.editorial_authority.service import EditorialAuthorityService
from app.infra.db import get_db_session

router = APIRouter(prefix="/editorial", tags=["editorial"])


@router.post("/entries")
async def create_editorial_entry(
    payload: CreateEditorialEntryRequest,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).create_draft(payload, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.get("/entries/{entry_id}")
async def get_editorial_entry(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).get_private_projection(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/evidence-collected")
async def collect_editorial_evidence(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).collect_evidence(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/editorial-review")
async def request_editorial_review(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).request_editorial_review(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/approve")
async def approve_editorial_revision(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).approve_current_revision(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/revisions")
async def revise_editorial_entry(
    entry_id: str,
    payload: ReviseEditorialEntryRequest,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).revise_entry(entry_id, payload, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/lightweight-accept")
async def accept_wording_revision(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).accept_wording_revision(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/maintainer-acceptance")
async def accept_editorial_maintainer_responsibility(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).accept_maintainer_responsibility(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/sources/{source_id}/availability")
async def record_editorial_source_availability(
    entry_id: str,
    source_id: str,
    payload: RecordSourceAvailabilityRequest,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).record_source_availability(
        entry_id,
        source_id,
        payload.availability,
        current_user,
    )
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/export")
async def export_editorial_revision(
    entry_id: str,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).export_approved_revision(entry_id, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/freshness-review")
async def request_freshness_review(
    entry_id: str,
    payload: RequestFreshnessReview,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).request_freshness_review(entry_id, payload, current_user)
    return ok_response(data=result, request_id=get_request_id())


@router.post("/entries/{entry_id}/integrity-review")
async def record_integrity_review(
    entry_id: str,
    payload: RecordIntegrityReview,
    current_user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await EditorialAuthorityService(session).record_integrity_review(entry_id, payload, current_user)
    return ok_response(data=result, request_id=get_request_id())
