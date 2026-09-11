from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.maintenance.fixtures import evidence_required
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.settings.generation_routes import GenerationRouteService


async def verify_reacceptance(session: AsyncSession, item: dict, records: list[str], replays: list[dict]) -> str:
    if len(records) != 1 or not replays or item.get("containment_record_identity") is None:
        raise evidence_required()
    # Share acceptance status locks until the enclosing maintenance transition commits.
    await session.execute(
        select(CanonicalRecordModel.stable_id)
        .where(CanonicalRecordModel.stable_id.in_([item["containment_record_identity"], *records]))
        .order_by(CanonicalRecordModel.stable_id.asc())
        .with_for_update(),
    )
    service = DeliveryAcceptanceService(session)
    original = await service.get_projection(item["containment_record_identity"])
    containment = next(
        (event for event in original["status_history"] if event["event_id"] == item.get("containment_event_id")), None,
    )
    accepted = await service.get_projection(records[0])
    activation = accepted["status_history"][-1]
    failure = containment.get("status_failure", {}) if containment else {}
    scope = item.get("blocking_scope") or {}
    scope_kind = scope.get("scope")
    if not isinstance(scope_kind, str):
        raise evidence_required()
    scope_field = {
        "entry_version": "entry_identities",
        "collection": "collection_identities",
        "deployment": "deployment_identity",
        "public_claim": "public_claim_identities",
    }.get(scope_kind)
    accepted_scope = accepted["accepted_scope"].get(scope_field) if scope_field else None
    in_scope = (
        scope.get("identity") == accepted_scope
        if isinstance(accepted_scope, str) else scope.get("identity") in (accepted_scope or [])
    )
    if (
        containment is None
        or containment["status"] != "suspended"
        or containment["reason_code"] != "integrity_failure"
        or failure.get("blocking_scope") != scope
        or accepted["current_status"] != "active"
        or accepted["record_id"] == original["record_id"]
        or accepted["stage"] != original["stage"]
        or accepted["affected_scope"].get("deployment_identity") != original["affected_scope"].get("deployment_identity")
        or not in_scope
        or datetime.fromisoformat(activation["occurred_at"]) <= datetime.fromisoformat(containment["occurred_at"])
    ):
        raise evidence_required()
    check = next((check for check in accepted["checks"] if check["check_id"] == failure.get("check_id")), None)
    if check is None or check["result"] != "passed":
        raise evidence_required()
    for replay in replays:
        link = f"evidence://maintenance/artifacts/{replay['id']}"
        event = await session.scalar(select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == item["id"],
            CanonicalEventModel.event_type == "fixture_replayed",
            CanonicalEventModel.payload["changes"]["replay_identity"].as_string() == replay["id"],
        ))
        if event is None or link not in accepted["evidence_links"] or link not in check["evidence_links"]:
            raise evidence_required()
        replay_time = event.occurred_at.replace(tzinfo=UTC) if event.occurred_at.tzinfo is None else event.occurred_at
        if datetime.fromisoformat(activation["occurred_at"]) <= replay_time:
            raise evidence_required()
    return f"evidence://maintenance/artifacts/{accepted['record_id']}:{activation['event_id']}"


async def verify_provider_reactivation(session: AsyncSession, item: dict, replay: dict, acceptance_identity: str) -> dict:
    verified = replay["generation_context"]
    active = await GenerationRouteService(session).verification_context()
    if (
        verified is None or "authorization_identity" not in verified
        or active is None or active["route_identity"] != verified["route_identity"]
        or active["acceptance_record_identity"] != acceptance_identity
        or acceptance_identity == verified["acceptance_record_identity"]
    ):
        raise evidence_required()
    activation = await session.get(CanonicalEventModel, active["activation_event_id"])
    replay_event = await session.scalar(select(CanonicalEventModel).where(
        CanonicalEventModel.aggregate_id == item["id"],
        CanonicalEventModel.event_type == "fixture_replayed",
        CanonicalEventModel.payload["changes"]["replay_identity"].as_string() == replay["id"],
    ))
    if (
        activation is None or replay_event is None
        or not isinstance(activation.recorded_by, str) or not activation.recorded_by.startswith("member:")
        or activation.aggregate_id != verified["route_identity"]
        or activation.event_type != "generation_route_activated"
    ):
        raise evidence_required()
    try:
        administrator_identity = f"member:{UUID(activation.recorded_by.removeprefix('member:')).hex}"
    except ValueError as exc:
        raise evidence_required() from exc
    if administrator_identity != item["administrator_identity"]:
        raise evidence_required()
    activation_time = activation.occurred_at.replace(tzinfo=UTC) if activation.occurred_at.tzinfo is None else activation.occurred_at
    replay_time = replay_event.occurred_at.replace(tzinfo=UTC) if replay_event.occurred_at.tzinfo is None else replay_event.occurred_at
    if activation_time <= replay_time:
        raise evidence_required()
    return active
