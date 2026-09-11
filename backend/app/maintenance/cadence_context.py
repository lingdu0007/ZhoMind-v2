from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.contracts.canonical import EntryLifecycleState
from app.delivery_acceptance.schemas import AcceptanceCheckVerificationInput, AffectedScopeInput
from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.editorial_authority.service import EditorialAuthorityService
from app.maintenance.fixtures import PublicationIdentity, SourceState, evidence_required
from app.maintenance.history import ItemIdentity, MemberIdentity
from app.maintenance.schemas import RoadmapOutcome, RoadmapReason
from app.model.canonical import CanonicalRecordModel
from app.reviewed_bundles.models import PublishedKnowledgePointer

Revision = Annotated[int, Field(ge=1, strict=True)]


def utcnow() -> datetime:
    return datetime.now(UTC)


def in_current_quarter(verified_at: datetime, now: datetime) -> bool:
    now = now.astimezone(UTC)
    start = datetime(now.year, ((now.month - 1) // 3) * 3 + 1, 1, tzinfo=UTC)
    return start <= verified_at <= now


class SampledCheck(AcceptanceCheckVerificationInput):
    result: Literal["passed", "carried_forward"]


class AcceptanceSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_identity: str = Field(pattern=r"^delivery_acceptance_record:[a-z0-9][a-z0-9._:-]{2,159}$")
    status_event_id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")
    accepted_scope: AffectedScopeInput
    verified_at: AwareDatetime
    verified_by: MemberIdentity
    verified_checks: list[SampledCheck] = Field(min_length=1, max_length=50)


def current_acceptance_sample(accepted: dict, now: datetime) -> dict | None:
    if accepted["current_status"] != "active":
        return None
    activation = accepted["status_history"][-1]
    try:
        verified_at = datetime.fromisoformat(activation["occurred_at"])
        if verified_at.tzinfo is None or not in_current_quarter(verified_at, now):
            return None
        if activation["reason_code"] != "checks_verified":
            return None
        results = {
            check["check_id"]: check["result"]
            for check in accepted["checks"] if check["result"] in {"passed", "carried_forward"}
        }
        attachments = activation["verified_checks"]
        attached_ids = [check["check_id"] for check in attachments]
        if len(set(attached_ids)) != len(attached_ids) or set(attached_ids) != set(results):
            return None
        return AcceptanceSample.model_validate({
            "record_identity": accepted["record_id"],
            "status_event_id": activation["event_id"],
            "accepted_scope": accepted["accepted_scope"],
            "verified_at": verified_at,
            "verified_by": activation["recorded_by"],
            "verified_checks": [{**check, "result": results[check["check_id"]]} for check in attachments],
        }).model_dump(mode="json")
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


class PublishedHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_identity: str = Field(pattern=r"^entry:[a-z0-9][a-z0-9._:-]{2,159}$")
    publication_identity: PublicationIdentity
    published_revision_identity: str = Field(pattern=r"^editorial_revision:[a-z0-9][a-z0-9._:-]{2,159}$")
    current_revision_identity: str = Field(pattern=r"^editorial_revision:[a-z0-9][a-z0-9._:-]{2,159}$")
    current_lifecycle_state: EntryLifecycleState
    current_revision_answer_eligible: bool = Field(strict=True)
    source_states: list[SourceState | Literal["unknown"]]


class KnowledgeHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    published_entries: list[PublishedHealth]


class DeferralSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: ItemIdentity
    item_identity: ItemIdentity
    revision: Revision
    owner_identity: MemberIdentity
    review_date: date
    state: Literal["deferred"]
    desired_outcome: RoadmapOutcome
    bounded_work_reason: RoadmapReason
    overdue_review: bool = Field(strict=True)


class ReviewSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maintainer_identity: MemberIdentity
    item_revisions: dict[ItemIdentity, Revision]
    knowledge_health: KnowledgeHealth
    deferrals: list[DeferralSnapshot]
    sample_options: list[AcceptanceSample]


async def build_review_context(session: AsyncSession, items: list[dict], candidates: list[dict], actor_identity: str) -> dict:
    owned = [item for item in items if item["accountable_maintainer"] == actor_identity]
    owned_ids = {item["id"] for item in owned}
    deferrals = [
        {key: candidate[key] for key in DeferralSnapshot.model_fields}
        for candidate in candidates
        if candidate["state"] == "deferred" and (candidate["item_identity"] in owned_ids or candidate["owner_identity"] == actor_identity)
    ]
    entries = []
    pointers = (
        await session.scalars(
            select(PublishedKnowledgePointer)
            .order_by(
                PublishedKnowledgePointer.entry_identity,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).all()
    for pointer in pointers:
        record = await session.get(CanonicalRecordModel, pointer.current_version_id)
        if (
            record is None
            or record.payload.get("schema") != "published_knowledge_version/v1"
            or record.payload.get("entry_identity") != pointer.entry_identity
        ):
            raise evidence_required()
        editorial = await EditorialAuthorityService(session).get_projection(pointer.entry_identity.removeprefix("entry:"))
        entries.append(
            {
                "entry_identity": pointer.entry_identity,
                "publication_identity": pointer.current_version_id,
                "published_revision_identity": record.payload["editorial_revision_identity"],
                "current_revision_identity": editorial["revision_identity"],
                "current_lifecycle_state": editorial["lifecycle_state"],
                "current_revision_answer_eligible": editorial["answer_eligible"],
                "source_states": sorted({source["availability"] for source in editorial["sources"]}),
            }
        )
    acceptances = (
        await session.scalars(
            select(CanonicalRecordModel)
            .where(
                CanonicalRecordModel.identity_kind == "delivery_acceptance_record",
                CanonicalRecordModel.payload["schema"].as_string() == "delivery_acceptance_record/v1",
            )
            .order_by(CanonicalRecordModel.stable_id)
            .with_for_update()
        )
    ).all()
    samples = []
    now = utcnow()
    for record in acceptances:
        accepted = await DeliveryAcceptanceService(session).get_projection(record.stable_id)
        sample = current_acceptance_sample(accepted, now)
        if sample is not None:
            samples.append(sample)
    context = ReviewSnapshot.model_validate(
        {
            "maintainer_identity": actor_identity,
            "item_revisions": {
                item["id"]: item["revision"] for item in owned if item["state"] in {"open", "triaged", "in_progress", "deferred"}
            },
            "knowledge_health": {"published_entries": entries},
            "deferrals": deferrals,
            "sample_options": samples,
        }
    ).model_dump(mode="json")
    return {**context, "context_sha256": canonical_json_sha256(context)}
