from uuid import UUID

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.schemas import CreatedTeamInvitationData, CreateTeamInvitationRequest, MemberData, TeamInvitationData
from app.common.deps import require_admin
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.service.member_admission_service import MemberAdmissionService

router = APIRouter(prefix="/members", tags=["members"])


def _member_data(member) -> MemberData:
    return MemberData(
        username=member.username,
        role=member.role,
        is_active=member.is_active,
        created_at=member.created_at,
        updated_at=member.updated_at,
    )


def _invitation_data(invitation) -> TeamInvitationData:
    return TeamInvitationData(
        id=invitation.id,
        expires_at=invitation.expires_at,
        revoked_at=invitation.revoked_at,
        created_at=invitation.created_at,
    )


@router.post("/invitations")
async def create_invitation(
    payload: CreateTeamInvitationRequest,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    invitation, invitation_code = await MemberAdmissionService(session, redis).issue_invitation(
        administrator=administrator,
        expires_at=payload.expires_at,
    )
    data = CreatedTeamInvitationData(**_invitation_data(invitation).model_dump(), invitation_code=invitation_code)
    return ok_response(data=data.model_dump(mode="json"), request_id=get_request_id())


@router.get("/invitations")
async def list_invitations(
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    invitations = await MemberAdmissionService(session, redis).list_invitations()
    data = [_invitation_data(invitation).model_dump(mode="json") for invitation in invitations]
    return ok_response(data=data, request_id=get_request_id())


@router.post("/invitations/{invitation_id}/revoke")
async def revoke_invitation(
    invitation_id: UUID,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    invitation = await MemberAdmissionService(session, redis).revoke_invitation(invitation_id)
    return ok_response(data=_invitation_data(invitation).model_dump(mode="json"), request_id=get_request_id())


@router.get("")
async def list_members(
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    members = await MemberAdmissionService(session, redis).list_members()
    data = [_member_data(member).model_dump(mode="json") for member in members]
    return ok_response(data=data, request_id=get_request_id())


@router.post("/{username}/promote")
async def promote_member(
    username: str,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    member = await MemberAdmissionService(session, redis).promote_member(username)
    return ok_response(data=_member_data(member).model_dump(mode="json"), request_id=get_request_id())


@router.post("/{username}/deactivate")
async def deactivate_member(
    username: str,
    administrator=Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    member = await MemberAdmissionService(session, redis).deactivate_member(username)
    return ok_response(data=_member_data(member).model_dump(mode="json"), request_id=get_request_id())
