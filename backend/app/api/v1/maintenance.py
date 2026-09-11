from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import get_current_user, require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.maintenance.schemas import (
    AdministratorJoin,
    CadenceInput,
    DiagnosisInput,
    FindingInput,
    MaintenanceAssignment,
    MaintenanceConsolidation,
    MaintenanceCreate,
    MaintenanceTransition,
    ProviderVerificationAuthorizationInput,
    ReplayInput,
    ReproductionInput,
    ResolutionInput,
    RoadmapInput,
    RoadmapReview,
)
from app.maintenance.service import MaintenanceService

router = APIRouter(prefix="/maintenance", tags=["knowledge-maintenance"])


@router.get("/context")
async def context(actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).context(actor), request_id=get_request_id())


@router.post("/items/{identity}/provider-verification-authorizations")
async def authorize_provider_verification(
    identity: str, payload: ProviderVerificationAuthorizationInput,
    actor=Depends(require_admin), session: AsyncSession = Depends(get_db_session),
):
    return ok_response(
        data=await MaintenanceService(session).authorize_provider_verification(identity, payload, actor),
        request_id=get_request_id(),
    )


@router.get("/provider-verification-authorizations/{identity}")
async def provider_verification_authorization(
    identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session),
):
    return ok_response(
        data=await MaintenanceService(session).provider_verification_authorization(identity, actor),
        request_id=get_request_id(),
    )


@router.post("/items/{identity}/administrator")
async def join_administrator(
    identity: str, payload: AdministratorJoin, actor=Depends(require_admin), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).join_administrator(identity, payload, actor), request_id=get_request_id())


@router.post("/assignments")
async def assign(payload: MaintenanceAssignment, actor=Depends(require_admin), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).assign(payload.username, actor), request_id=get_request_id())


@router.post("/assignments/{identity}/accept")
async def accept(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).accept(identity, actor), request_id=get_request_id())


@router.post("/items")
async def create(payload: MaintenanceCreate, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).create(payload, actor), request_id=get_request_id())


@router.get("/inbox")
async def inbox(actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).inbox(actor), request_id=get_request_id())


@router.get("/items")
async def list_items(actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).list_items(actor), request_id=get_request_id())


@router.get("/items/{identity}")
async def get(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get(identity, actor), request_id=get_request_id())


@router.post("/items/{identity}/transition")
async def transition(
    identity: str, payload: MaintenanceTransition, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).transition(identity, payload, actor), request_id=get_request_id())


@router.post("/items/{identity}/signals")
async def consolidate(
    identity: str, payload: MaintenanceConsolidation, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).consolidate(identity, payload, actor), request_id=get_request_id())


@router.post("/items/{identity}/reproductions")
async def reproduce(
    identity: str, payload: ReproductionInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).reproduce(identity, payload, actor), request_id=get_request_id())


@router.get("/items/{identity}/reproduction-inputs")
async def reproduction_inputs(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).reproduction_inputs(identity, actor), request_id=get_request_id())


@router.get("/fixtures/{identity}")
async def get_fixture(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_fixture(identity, actor), request_id=get_request_id())


@router.post("/items/{identity}/diagnosis")
async def diagnose(
    identity: str, payload: DiagnosisInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).diagnose(identity, payload, actor), request_id=get_request_id())


@router.post("/items/{identity}/replays")
async def replay(identity: str, payload: ReplayInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).replay(identity, payload, actor), request_id=get_request_id())


@router.get("/replays/{identity}")
async def get_replay(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_replay(identity, actor), request_id=get_request_id())


@router.post("/items/{identity}/findings")
async def approve_finding(
    identity: str, payload: FindingInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).approve_finding(identity, payload, actor), request_id=get_request_id())


@router.get("/findings/{identity}")
async def get_finding(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_finding(identity, actor), request_id=get_request_id())


@router.post("/items/{identity}/roadmap")
async def qualify_roadmap(
    identity: str, payload: RoadmapInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).qualify_roadmap(identity, payload, actor), request_id=get_request_id())


@router.get("/roadmap")
async def list_roadmap(actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).list_roadmap(actor), request_id=get_request_id())


@router.get("/roadmap/{identity}")
async def get_roadmap(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_roadmap(identity, actor), request_id=get_request_id())


@router.post("/roadmap/{identity}/review")
async def review_roadmap(
    identity: str, payload: RoadmapReview, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).review_roadmap(identity, payload, actor), request_id=get_request_id())


@router.get("/maps/{identity}")
async def get_map(identity: str, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_map(identity, actor), request_id=get_request_id())


@router.post("/cadence")
async def record_cadence(payload: CadenceInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).record_cadence(payload, actor), request_id=get_request_id())


@router.get("/review-context")
async def review_context(actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_review_context(actor), request_id=get_request_id())


@router.get("/dashboard")
async def get_dashboard(actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)):
    return ok_response(data=await MaintenanceService(session).get_dashboard(actor), request_id=get_request_id())


@router.post("/items/{identity}/resolution")
async def resolve(
    identity: str, payload: ResolutionInput, actor=Depends(get_current_user), session: AsyncSession = Depends(get_db_session)
):
    return ok_response(data=await MaintenanceService(session).resolve(identity, payload, actor), request_id=get_request_id())
