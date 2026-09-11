from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User


def denied() -> AppError:
    return AppError(status_code=403, code="MAINTENANCE_AUTHORITY_REQUIRED", message="maintenance responsibility required")


class AssignmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["maintenance_assignment/v1"] = Field(alias="schema")
    member_id: UUID
    assigned_by: UUID


async def assignment_state(session: AsyncSession, identity: str, member: User) -> str:
    record = await session.get(CanonicalRecordModel, identity)
    if (
        record is None
        or identity != f"maintenance_item:assignment-{member.id}"
        or record.identity_kind != "maintenance_item"
        or record.identity_value != f"assignment-{member.id}"
        or record.record_class != "authoritative"
        or record.state != "assigned"
    ):
        raise denied()
    try:
        fields = AssignmentRecord.model_validate(record.payload)
    except ValidationError as exc:
        raise denied() from exc
    administrator = await session.get(User, fields.assigned_by, populate_existing=True)
    if fields.member_id != member.id or administrator is None or administrator.role != "admin":
        raise denied()
    events = list((await session.scalars(
        select(CanonicalEventModel).where(CanonicalEventModel.aggregate_id == identity),
    )).all())
    if len(events) > 1:
        raise denied()
    if not events:
        return "assigned"
    event = events[0]
    if (
        event.aggregate_kind != "maintenance_item"
        or event.event_type != "responsibility_accepted"
        or event.from_state is not None
        or event.to_state != "accepted"
        or event.recorded_by != f"member:{member.id.hex}"
        or event.payload != {"schema": "maintenance_assignment_event/v1"}
    ):
        raise denied()
    return "accepted"
