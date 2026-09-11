from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.contracts.canonical import AnswerOutcome
from app.maintenance.containment import ContainmentVerification, qualify_verification, verification_receipt
from app.maintenance.diagnosis import active_publications, current_publication_review, evidence_required, execute_fresh, owned_execution
from app.maintenance.findings import load_finding
from app.maintenance.fixtures import (
    GenerationContext,
    Hash,
    PublicationIdentity,
    PublicationReview,
    VerificationGenerationContext,
    load_fixture,
)
from app.maintenance.history import ItemIdentity
from app.maintenance.provider_verification import load_authorization, qualify_execution_authorization, verify_current_admission
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.retention.policy import lock_registry
from app.retrieval.policy import get_retrieval_policy
from app.settings.generation_routes import GenerationRouteService
from app.settings.runtime import get_runtime_settings


class ReplayRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["maintenance_replay/v1"] = Field(alias="schema")
    item_identity: ItemIdentity
    fixture_identity: ItemIdentity
    request_sha256: Hash
    query_condition_set_identity: Hash
    expected_outcome: AnswerOutcome
    observed_outcome: AnswerOutcome | None
    observed_state: Literal["completed", "failed"]
    active_publication_identities: list[PublicationIdentity]
    evidence_publication_identities: list[PublicationIdentity]
    retrieval_profile_identity: Literal["retrieval-answer-policy/pilot-v1"]
    passed: bool = Field(strict=True)
    verification_method: Literal["independent_authenticated_replay"]
    publication_review: PublicationReview | None
    generation_context: GenerationContext | VerificationGenerationContext | None
    containment_verification: ContainmentVerification | None = None


async def approved_fixture(session: AsyncSession, item: dict, identity: str) -> dict:
    fixture = await load_fixture(session, item, identity)
    findings = [await load_finding(session, item, finding_id) for finding_id in item.get("finding_identities", [])]
    if not any(finding["fixture_identity"] == identity for finding in findings):
        raise evidence_required()
    return fixture


async def latest_replay_identity(session: AsyncSession, item: dict, fixture_identity: str) -> str | None:
    events = (await session.scalars(
        select(CanonicalEventModel)
        .where(
            CanonicalEventModel.aggregate_id == item["id"],
            CanonicalEventModel.event_type == "fixture_replayed",
        )
        .order_by(CanonicalEventModel.payload["revision"].as_integer().desc())
    )).all()
    for event in events:
        changes = event.payload["changes"]
        identity = changes.get("replay_identity")
        if not isinstance(identity, str):
            raise evidence_required()
        record = await session.get(CanonicalRecordModel, identity)
        if (
            record is None or event.recorded_by != item["work_owner"]
            or changes.get("replay_sha256") != canonical_json_sha256(record.payload)
        ):
            raise evidence_required()
        replay = await load_replay(session, item, identity)
        if replay["fixture_identity"] == fixture_identity:
            return identity
    return None


async def replay_fixture(
    session: AsyncSession, item: dict, identity: str, actor: User, *, answer_id: str, authorization_identity: str | None = None,
) -> dict:
    fixture = await approved_fixture(session, item, identity)
    selected = await owned_execution(session, answer_id, actor)
    request = {"message": selected["question"], "query_conditions": selected["query_condition_set"]["conditions"]}
    if (
        canonical_json_sha256(request) != fixture["request_sha256"]
        or selected["query_condition_set"]["identity"] != fixture["query_condition_set_identity"]
    ):
        raise evidence_required()
    publications = await active_publications(session)
    review = (
        await current_publication_review(session, fixture["publication_review"]["entry_identity"])
        if fixture["publication_review"]
        else None
    )
    if authorization_identity:
        authorization = await load_authorization(session, authorization_identity)
        if (
            authorization["item_identity"] != item["id"] or fixture["verified_observation"] != "provider_failure"
            or authorization["approved_fixture_identity"] != identity
        ):
            raise evidence_required()
    execution = await execute_fresh(session, request, actor, authorization_identity=authorization_identity)
    await lock_registry(session)
    projection = execution["projection"]
    if (
        publications != await active_publications(session)
        or projection["query_condition_set"]["identity"] != fixture["query_condition_set_identity"]
        or not set(projection.get("knowledge_version_identities", [])).issubset(publications)
        or (review is not None and review != await current_publication_review(session, review["entry_identity"]))
    ):
        raise evidence_required()
    result = ReplayRecord.model_validate(
        {
            "schema": "maintenance_replay/v1",
            "item_identity": item["id"],
            "fixture_identity": identity,
            "request_sha256": fixture["request_sha256"],
            "query_condition_set_identity": fixture["query_condition_set_identity"],
            "expected_outcome": fixture["expected_outcome"],
            "observed_outcome": projection.get("outcome"),
            "observed_state": projection["state"],
            "active_publication_identities": publications,
            "evidence_publication_identities": projection.get("knowledge_version_identities", []),
            "retrieval_profile_identity": execution["retrieval_profile_identity"],
            "passed": projection["state"] == "completed" and projection.get("outcome") == fixture["expected_outcome"],
            "verification_method": "independent_authenticated_replay",
            "publication_review": review,
            "generation_context": execution["generation_context"],
            "containment_verification": verification_receipt(),
        }
    ).model_dump(mode="json", by_alias=True)
    replay_identity = f"maintenance_item:{uuid4().hex}"
    session.add(
        CanonicalRecordModel(
            stable_id=replay_identity,
            identity_kind="maintenance_item",
            identity_value=replay_identity.split(":", 1)[1],
            state="verified",
            record_class="immutable",
            payload=result,
        )
    )
    return {"id": replay_identity, **result}


async def load_replay(session: AsyncSession, item: dict, identity: str, *, current: bool = False) -> dict:
    record = await session.get(CanonicalRecordModel, identity)
    if record is None or record.record_class != "immutable" or record.state != "verified":
        raise evidence_required()
    try:
        replay = ReplayRecord.model_validate(record.payload).model_dump(mode="json", by_alias=True)
    except ValidationError as exc:
        raise evidence_required() from exc
    qualifying = await session.scalar(
        select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == item["id"],
            CanonicalEventModel.event_type == "fixture_replayed",
            CanonicalEventModel.recorded_by == item["work_owner"],
            CanonicalEventModel.payload["changes"]["replay_identity"].as_string() == identity,
        )
    )
    if (
        replay["item_identity"] != item["id"]
        or qualifying is None
        or qualifying.payload.get("changes", {}).get("replay_sha256") != canonical_json_sha256(record.payload)
    ):
        raise evidence_required()
    await qualify_execution_authorization(session, replay, qualifying)
    qualify_verification(item, replay.get("containment_verification"), qualifying)
    fixture = await approved_fixture(session, item, replay["fixture_identity"])
    if (
        any(replay[key] != fixture[key] for key in ("request_sha256", "query_condition_set_identity", "expected_outcome"))
        or replay["passed"] != (replay["observed_state"] == "completed" and replay["observed_outcome"] == replay["expected_outcome"])
        or not set(replay["evidence_publication_identities"]).issubset(replay["active_publication_identities"])
        or (replay["observed_state"] == "failed" and replay["observed_outcome"] is not None)
        or (replay["observed_state"] == "completed" and replay["observed_outcome"] is None)
        or (fixture["publication_review"] is None) != (replay["publication_review"] is None)
    ):
        raise evidence_required()
    context = replay["generation_context"]
    if current and context is not None:
        if "authorization_identity" in context:
            await verify_current_admission(session, context)
        elif context != await GenerationRouteService(session).verification_context():
            raise evidence_required()
    latest_identity = identity
    if current:
        latest_identity = await latest_replay_identity(session, item, replay["fixture_identity"])
    if current and (
        latest_identity != identity
        or replay["active_publication_identities"] != await active_publications(session)
        or replay["retrieval_profile_identity"] != get_retrieval_policy(get_runtime_settings()).identity
        or (
            replay["publication_review"] is not None
            and replay["publication_review"] != await current_publication_review(session, replay["publication_review"]["entry_identity"])
        )
    ):
        raise evidence_required()
    return {"id": identity, **replay}
