from calendar import monthrange
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from pydantic import Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.maintenance.cadence_context import AcceptanceSample, ReviewSnapshot, in_current_quarter
from app.maintenance.fixtures import evidence_required
from app.maintenance.history import project_item
from app.maintenance.schemas import CadenceInput
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink
from app.retention.policy import read_policy


def utcnow() -> datetime:
    return datetime.now(UTC)


def next_review(last: date, period: str) -> date:
    if period == "weekly":
        return last + timedelta(days=7)
    index = last.year * 12 + last.month - 1 + (1 if period == "monthly" else 3)
    year, month_index = divmod(index, 12)
    month = month_index + 1
    return date(year, month, min(last.day, monthrange(year, month)[1]))


class CadenceRecord(CadenceInput):
    schema_name: str = Field(alias="schema", pattern=r"^maintenance_cadence/v1$")
    review_date: date
    review_snapshot: ReviewSnapshot
    sampled_acceptances: list[AcceptanceSample]


async def record_review(session: AsyncSession, context: dict, payload: CadenceInput) -> dict:
    if payload.evidence_links != [f"evidence://maintenance/{payload.period}-review"]:
        raise AppError(
            status_code=422, code="MAINTENANCE_REVIEW_REFERENCE_INVALID", message="review reference must match the cadence period"
        )
    if context["item_revisions"] != payload.item_revisions or context["context_sha256"] != payload.context_sha256:
        raise AppError(status_code=409, code="MAINTENANCE_STALE", message="review the current open item revisions")
    if payload.period == "quarterly" and not payload.sample_acceptance_identities:
        raise AppError(status_code=422, code="MAINTENANCE_SAMPLE_REQUIRED", message="quarterly review requires sampled acceptance")
    now = utcnow()
    options = {
        sample["record_identity"]: sample for sample in context["sample_options"]
        if in_current_quarter(datetime.fromisoformat(sample["verified_at"]), now)
    }
    if len(set(payload.sample_acceptance_identities)) != len(payload.sample_acceptance_identities) or not set(
        payload.sample_acceptance_identities
    ).issubset(options):
        raise AppError(status_code=409, code="MAINTENANCE_SAMPLE_UNACCEPTED", message="current distinct sampled acceptance required")
    identity = f"maintenance_item:{uuid4().hex}"
    fields = {
        "schema": "maintenance_cadence/v1",
        **payload.model_dump(mode="json"),
        "review_date": now.date().isoformat(),
        "review_snapshot": {key: value for key, value in context.items() if key != "context_sha256"},
        "sampled_acceptances": [options[identity] for identity in payload.sample_acceptance_identities],
    }
    session.add(
        CanonicalRecordModel(
            stable_id=identity,
            identity_kind="maintenance_item",
            identity_value=identity.split(":", 1)[1],
            state="reviewed",
            record_class="immutable",
            payload=fields,
        )
    )
    return {"id": identity, **fields}


async def dashboard(session: AsyncSession, items: list[dict], *, actor_identity: str) -> dict:
    now = utcnow()
    records = (
        (
            await session.scalars(
                select(CanonicalRecordModel)
                .join(
                    CanonicalEventModel,
                    CanonicalEventModel.aggregate_id == CanonicalRecordModel.stable_id,
                )
                .where(
                    CanonicalRecordModel.identity_kind == "maintenance_item",
                    CanonicalEventModel.aggregate_kind == "maintenance_item",
                    CanonicalEventModel.event_type == "cadence_reviewed",
                    CanonicalEventModel.to_state == "reviewed",
                    CanonicalEventModel.recorded_by == actor_identity,
                    CanonicalEventModel.payload["schema"].as_string() == "maintenance_cadence_event/v1",
                )
                .order_by(CanonicalRecordModel.created_at, CanonicalRecordModel.stable_id)
            )
        )
        .unique()
        .all()
    )
    reviews = []
    for record in records:
        try:
            review = CadenceRecord.model_validate(record.payload).model_dump(mode="json", by_alias=True)
            events = (
                await session.scalars(
                    select(CanonicalEventModel).where(
                        CanonicalEventModel.aggregate_id == record.stable_id,
                    )
                )
            ).all()
            if len(events) != 1:
                raise ValueError("ambiguous review trail")
            event = events[0]
            if (
                record.record_class != "immutable"
                or record.state != "reviewed"
                or review["review_snapshot"]["maintainer_identity"] != actor_identity
                or review["context_sha256"] != canonical_json_sha256(review["review_snapshot"])
                or event.aggregate_kind != "maintenance_item"
                or event.event_type != "cadence_reviewed"
                or event.to_state != "reviewed"
                or event.recorded_by != actor_identity
                or event.from_state is not None
                or event.payload != {"schema": "maintenance_cadence_event/v1", "record_sha256": canonical_json_sha256(record.payload)}
            ):
                raise ValueError("review binding")
            reviews.append({"id": record.stable_id, **review})
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            raise evidence_required() from exc
    dates = {
        period: max(
            (date.fromisoformat(review["review_date"]) for review in reviews if review["period"] == period),
            default=None,
        )
        for period in ("weekly", "monthly", "quarterly")
    }
    retention_days = (await read_policy(session))["days"]["feedback_signals"]
    pending_signals = (
        await session.scalars(
            select(KnowledgeFeedbackSignal).where(
                KnowledgeFeedbackSignal.created_at <= now - timedelta(days=7),
                KnowledgeFeedbackSignal.created_at > now - timedelta(days=retention_days),
                KnowledgeFeedbackSignal.expires_at > now,
                KnowledgeFeedbackSignal.label != "helpful",
            )
        )
    ).all()
    linked_records = (
        (
            await session.scalars(
                select(CanonicalRecordModel)
                .join(MaintenanceSignalLink, MaintenanceSignalLink.item_id == CanonicalRecordModel.stable_id)
                .where(MaintenanceSignalLink.signal_id.in_([signal.id for signal in pending_signals]))
            )
        )
        .unique()
        .all()
    )
    trails: dict[str, list[CanonicalEventModel]] = {record.stable_id: [] for record in linked_records}
    for event in (await session.scalars(select(CanonicalEventModel).where(CanonicalEventModel.aggregate_id.in_(trails)))).all():
        trails[event.aggregate_id].append(event)
    triaged_ids = {record.stable_id for record in linked_records if project_item(record, trails[record.stable_id])["state"] != "open"}
    reviewed_signals = set(
        (
            await session.scalars(
                select(MaintenanceSignalLink.signal_id).where(
                    MaintenanceSignalLink.item_id.in_(triaged_ids),
                )
            )
        ).all()
    )
    return {
        "items": items,
        "overdue_triage_count": sum(signal.id not in reviewed_signals for signal in pending_signals),
        "cadence_due": {
            period: last is None or last > now.date() or next_review(last, period) <= now.date() for period, last in dates.items()
        },
        "cadence_records": reviews,
    }
