from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.canonical import CanonicalEventModel
from app.retention.policy import REGISTRY_ID, DataClass


class RetentionFailureObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["retention_cleanup_failure/v1"] = Field(alias="schema")
    data_class: DataClass
    status: Literal["failed"]
    attempt: int | None = Field(ge=1)
    invalidated_attempt: int = Field(ge=0)
    policy_identity: str | None = Field(pattern=r"^configuration:[0-9a-f]{64}$")
    checked_at: str
    deleted_count: int = Field(ge=0)
    remaining_expired: int | None = Field(ge=0)
    normalized_error: Literal["PRIVACY_EXPIRY_SURVIVED", "RETENTION_CLEANUP_FAILED", "RETENTION_CLEANUP_INTERRUPTED"]
    pending_committed: bool | None = None
    affected_acceptance_identities: list[Annotated[str, Field(pattern=r"^delivery_acceptance_record:[a-z0-9._:-]+$")]]

    @field_validator("checked_at")
    @classmethod
    def validate_time(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value

async def read_failure_evidence(session: AsyncSession) -> list[tuple[str, RetentionFailureObservation]]:
    events = (await session.scalars(select(CanonicalEventModel).where(
        CanonicalEventModel.aggregate_id == REGISTRY_ID,
        CanonicalEventModel.event_type == "retention_cleanup_failed",
    ).order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc()))).all()
    result = []
    for event in events:
        try:
            observation = RetentionFailureObservation.model_validate(event.payload)
            if event.aggregate_kind != "configuration" or event.recorded_by != "system:retention-cleanup":
                raise ValueError
        except ValueError as exc:
            raise AppError(
                status_code=409, code="RETENTION_FAILURE_EVIDENCE_INVALID",
                message="retention failure evidence cannot be verified",
            ) from exc
        result.append((event.id, observation))
    return result
