from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    invitation_code: str = Field(min_length=1, max_length=512)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class AuthTokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class WorkspaceCapabilities(BaseModel):
    system_settings: bool


class MeData(BaseModel):
    username: str
    role: str
    capabilities: WorkspaceCapabilities


class CreateTeamInvitationRequest(BaseModel):
    expires_at: datetime | None = None


class TeamInvitationData(BaseModel):
    id: UUID
    expires_at: datetime
    revoked_at: datetime | None
    consumed_at: datetime | None
    created_at: datetime


class CreatedTeamInvitationData(TeamInvitationData):
    invitation_code: str


class MemberData(BaseModel):
    username: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class IdentityAuditEventData(BaseModel):
    id: str
    action: str
    outcome: str
    reason: str
    actor_identity: str
    target_identity: str
    reference_identity: str
    occurred_at: datetime
