import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schemas import ChatRequest
from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.documents.content_admission import recognizable_private_material
from app.editorial_authority.service import EditorialAuthorityService
from app.extensions.provider_router import ProviderRouter
from app.maintenance.affected_scope import retain_affected_scope
from app.maintenance.containment import verification_receipt
from app.maintenance.diagnosis_policy import diagnosis_route
from app.maintenance.fixtures import evidence_required, load_fixture, qualify_fixture_scope
from app.maintenance.schemas import ReproductionInput
from app.model.answer_execution import AnswerExecutionModel
from app.model.canonical import CanonicalRecordModel
from app.model.chat import ChatMessage, ChatSession
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink
from app.model.user import User
from app.operations.chat_capacity import get_chat_admission_gate
from app.rag.answer_execution import QueryConditionLossError
from app.rag.evidence_sufficiency import QueryConditionSet
from app.repository.chat_repository import ChatRepository
from app.retention.policy import lock_registry, read_policy
from app.retrieval.policy import get_retrieval_policy
from app.reviewed_bundles.models import PublishedKnowledgePointer
from app.service.answer_execution_store import AnswerExecutionHandle, AnswerExecutionStore
from app.service.chat_service import ChatService
from app.settings.generation_routes import GenerationRouteService
from app.settings.runtime import get_runtime_settings


async def active_publications(session: AsyncSession) -> list[str]:
    pointers = (
        await session.scalars(
            select(PublishedKnowledgePointer)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).all()
    identities = []
    for pointer in pointers:
        record = await session.get(CanonicalRecordModel, pointer.current_version_id)
        if (
            record is None
            or record.payload.get("schema") != "published_knowledge_version/v1"
            or record.payload.get("entry_identity") != pointer.entry_identity
        ):
            raise evidence_required()
        identities.append(record.stable_id)
    return sorted(identities)


async def create_reproduction(session: AsyncSession, item: dict, payload: ReproductionInput, actor: User) -> dict:
    signal = await session.get(KnowledgeFeedbackSignal, payload.signal_id)
    link = await session.get(MaintenanceSignalLink, (item["id"], payload.signal_id))
    if signal is None or link is None or signal.user_id == actor.username:
        raise evidence_required()
    reported_entry = signal.entry_id
    if payload.entry_identity and reported_entry is not None and payload.entry_identity != f"entry:{reported_entry}":
        raise evidence_required()
    affected_scope = await retain_affected_scope(session, [signal])
    projection = await owned_execution(session, payload.answer_id, actor)
    conditions = projection["query_condition_set"]
    conditions_sha256 = canonical_json_sha256(conditions["conditions"])
    if conditions_sha256 != signal.normalized_metadata.get("query_conditions_sha256"):
        raise evidence_required()
    replay = {"message": projection["question"], "query_conditions": conditions["conditions"]}
    encoded = json.dumps(replay, ensure_ascii=False)
    if recognizable_private_material(encoded) or (signal.note and signal.note in encoded):
        raise AppError(
            status_code=422, code="MAINTENANCE_PRIVATE_CONTENT", message="fixture must be independently authored non-personal material"
        )
    publications = await active_publications(session)
    if not set(projection.get("knowledge_version_identities", [])).issubset(publications):
        raise evidence_required()
    authorization_identity = payload.provider_verification_authorization_identity
    if authorization_identity:
        from app.maintenance.provider_verification import load_authorization

        authorization = await load_authorization(session, authorization_identity)
        if (
            authorization["item_identity"] != item["id"] or payload.verified_observation != "provider_failure"
            or authorization["approved_fixture_identity"] is not None
        ):
            raise evidence_required()
    execution = await execute_fresh(session, replay, actor, authorization_identity=authorization_identity)
    await lock_registry(session)
    policy = await read_policy(session)
    now = datetime.now(UTC)
    signal = await session.scalar(
        select(KnowledgeFeedbackSignal)
        .where(
            KnowledgeFeedbackSignal.id == payload.signal_id,
            KnowledgeFeedbackSignal.expires_at > now,
            KnowledgeFeedbackSignal.created_at > now - timedelta(days=policy["days"]["feedback_signals"]),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    retained_link = await session.scalar(
        select(MaintenanceSignalLink.signal_id).where(
            MaintenanceSignalLink.item_id == item["id"],
            MaintenanceSignalLink.signal_id == payload.signal_id,
        )
    )
    projection = execution["projection"]
    if (
        signal is None
        or retained_link is None
        or signal.entry_id != reported_entry
        or await retain_affected_scope(session, [signal]) != affected_scope
        or projection["query_condition_set"]["identity"] != conditions["identity"]
        or signal.normalized_metadata.get("query_conditions_sha256") != conditions_sha256
        or publications != await active_publications(session)
        or not set(projection.get("knowledge_version_identities", [])).issubset(publications)
    ):
        raise evidence_required()
    facts = {
        "outcome": projection.get("outcome"),
        "state": projection["state"],
        "expected_outcome": payload.expected_outcome.value,
        "reason": (projection.get("insufficient_evidence_reply") or {}).get("reason"),
        "frozen_evidence": projection.get("evidence_set_identity") is not None,
    }
    if payload.verified_observation == "citation_drift":
        facts["verified_difference"] = execution["verified_citation_failure"]
    if payload.verified_observation == "condition_loss":
        facts["verified_difference"] = execution["verified_condition_loss"]
    reference = None
    reference_reviews = []
    if payload.reference_answer_id:
        reference = await owned_execution(session, payload.reference_answer_id, actor)
        if (
            reference["query_condition_set"]["identity"] != conditions["identity"]
            or reference.get("outcome") != "evidence_gated_answer"
            or not reference.get("knowledge_version_identities")
            or not set(reference["knowledge_version_identities"]).issubset(publications)
        ):
            raise evidence_required()
        reference_reviews = await eligible_reference_reviews(session, reference["knowledge_version_identities"])
        facts["reference_supported"] = True
    publication_review = None
    if payload.entry_identity:
        publication_review = await current_publication_review(session, payload.entry_identity)
        states = publication_review["source_states"]
        facts["source_state"] = next((state for state in states if state != "verified_usable"), "verified_usable")
        facts["verified_publication"] = True
        facts["verified_integrity_review"] = publication_review["integrity_review_event_id"] is not None
    diagnosis_route(payload.verified_observation, facts)
    if payload.verified_observation == "provider_failure" and execution["generation_context"] is None:
        raise evidence_required()
    identity = f"maintenance_item:{uuid4().hex}"
    fixture = {
        "schema": "maintenance_fixture/v1",
        "item_identity": item["id"],
        "affected_scope": affected_scope,
        "request_sha256": canonical_json_sha256(replay),
        "query_condition_set_identity": conditions["identity"],
        "expected_outcome": payload.expected_outcome.value,
        "observed_outcome": projection.get("outcome"),
        "observed_state": projection["state"],
        "verified_observation": payload.verified_observation,
        "diagnosis_facts": facts,
        "publication_review": publication_review,
        "reference_evidence_publication_identities": reference["knowledge_version_identities"] if reference else [],
        "reference_publication_reviews": reference_reviews,
        "active_publication_identities": publications,
        "evidence_publication_identities": projection.get("knowledge_version_identities", []),
        "retrieval_profile_identity": execution["retrieval_profile_identity"],
        "observed_reason": (projection.get("insufficient_evidence_reply") or {}).get("reason"),
        "verification_method": "independent_authenticated_replay",
        "generation_context": execution["generation_context"],
        "containment_verification": verification_receipt(),
    }
    qualify_fixture_scope(item, fixture)
    session.add(
        CanonicalRecordModel(
            stable_id=identity,
            identity_kind="maintenance_item",
            identity_value=identity.split(":", 1)[1],
            state="reproduced",
            record_class="immutable",
            payload=fixture,
        )
    )
    return {"id": identity, **fixture}


async def eligible_reference_reviews(session: AsyncSession, identities: list[str]) -> list[dict]:
    reviews = []
    for identity in sorted(identities):
        publication = await session.get(CanonicalRecordModel, identity)
        if publication is None or not isinstance(publication.payload.get("entry_identity"), str):
            raise evidence_required()
        review = await current_publication_review(session, publication.payload["entry_identity"])
        if review["publication_identity"] != identity:
            raise evidence_required()
        try:
            authority = await EditorialAuthorityService(session).get_retrieval_authority_for_revision(
                review["entry_identity"].removeprefix("entry:"), review["revision_identity"]
            )
        except AppError as exc:
            raise evidence_required() from exc
        if authority["answer_eligible"] is not True:
            raise evidence_required()
        reviews.append(review)
    return reviews


async def current_publication_review(session: AsyncSession, entry_identity: str) -> dict:
    pointer = await session.get(PublishedKnowledgePointer, entry_identity, populate_existing=True)
    if pointer is None:
        raise evidence_required()
    publication = await session.get(CanonicalRecordModel, pointer.current_version_id)
    if (
        publication is None
        or publication.payload.get("schema") != "published_knowledge_version/v1"
        or publication.payload.get("entry_identity") != entry_identity
        or not isinstance(publication.payload.get("editorial_revision_identity"), str)
    ):
        raise evidence_required()
    editorial = await EditorialAuthorityService(session).get_source_review_for_revision(
        entry_identity.removeprefix("entry:"), publication.payload["editorial_revision_identity"]
    )
    return {
        "entry_identity": entry_identity,
        "publication_identity": publication.stable_id,
        "revision_identity": editorial["revision_identity"],
        "integrity_review_event_id": await EditorialAuthorityService(session).get_integrity_review_for_revision(
            entry_identity.removeprefix("entry:"), editorial["revision_identity"],
        ),
        "source_states": sorted({source["availability"] for source in editorial["sources"]}),
        "source_facts": [
            {key: source[key] for key in ("source_identity", "availability", "status_event_id", "event_trail_sha256")}
            for source in editorial["sources"]
        ],
    }


class _VerificationChatService(ChatService):
    def __init__(self, session: AsyncSession, router: ProviderRouter) -> None:
        super().__init__(session)
        self.router = router

    def _provider_router(self) -> ProviderRouter:
        return self.router


async def execute_fresh(
    session: AsyncSession, replay: dict, actor: User, *, authorization_identity: str | None = None,
) -> dict:
    if set(replay) != {"message", "query_conditions"}:
        raise evidence_required()
    request = ChatRequest.model_validate(replay)
    policy = get_retrieval_policy(get_runtime_settings())
    if policy.diagnostic_or_migration_only:
        raise evidence_required()
    gate = get_chat_admission_gate()
    if not gate.try_admit():
        raise AppError(
            status_code=429,
            code="CHAT_CONCURRENCY_LIMIT_REACHED",
            message="the first-release concurrent chat limit has been reached",
        )
    admitted: AnswerExecutionHandle | None = None

    async def capture(handle: AnswerExecutionHandle) -> None:
        nonlocal admitted
        admitted = handle

    try:
        authorization = None
        chat = ChatService(session)
        if authorization_identity:
            from app.maintenance.service import MaintenanceService

            conditions = QueryConditionSet.from_records(
                normalized_question=request.message, records=replay["query_conditions"],
            )
            authorization, administrator = await MaintenanceService(session).verify_provider_authorization(
                authorization_identity, actor, conditions.identity,
            )
            admission = {key: authorization[key] for key in (
                "route_identity", "acceptance_record_identity", "acceptance_evidence_sha256",
            )}
            router = await GenerationRouteService(session).capture_verification(admission, administrator)
            chat = _VerificationChatService(session, router)
            generation_context = {
                **admission, "authorization_identity": authorization_identity,
                "authorization_sha256": canonical_json_sha256({
                    key: value for key, value in authorization.items() if key != "id"
                }),
            }
        else:
            generation_context = await GenerationRouteService(session).verification_context()
        runtime = {}
        verified_condition_loss = False
        try:
            result = await chat.run_chat(
                user_id=actor.username,
                question=request.message,
                session_id=f"maintenance-{uuid4().hex}",
                query_conditions=[condition.model_dump() for condition in request.query_conditions or []],
                inherit_conditions=False,
                on_admitted=capture,
            )
            projection = await owned_execution(session, result["message"]["id"], actor)
            runtime = (result["message"].get("rag_trace") or {}).get("runtime", {})
            actual_route = runtime.get("route_identity")
        except Exception as exc:
            if admitted is None or admitted.assistant_message_id is None:
                raise
            await session.refresh(actor)
            projection = await owned_execution(session, admitted.assistant_message_id, actor)
            if projection["state"] != "failed":
                raise
            verified_condition_loss = isinstance(exc, QueryConditionLossError)
            actual_route = None
        if policy.identity != get_retrieval_policy(get_runtime_settings()).identity:
            raise evidence_required()
        if authorization is not None:
            current, _ = await MaintenanceService(session).verify_provider_authorization(
                authorization["id"], actor, projection["query_condition_set"]["identity"],
            )
            if current != authorization:
                raise evidence_required()
        if actual_route and (
            generation_context is None
            or actual_route != generation_context["route_identity"]
            or (authorization is None and generation_context != await GenerationRouteService(session).verification_context())
        ):
            raise evidence_required()
        return {
            "projection": projection,
            "retrieval_profile_identity": policy.identity,
            "generation_context": generation_context if actual_route else None,
            "verified_condition_loss": verified_condition_loss,
            "verified_citation_failure": bool(
                actual_route
                and projection.get("outcome") == "generation_unavailable"
                and projection.get("evidence_set_identity")
                and runtime.get("provider_attempts")
                and all(attempt.get("error_code") == "citation_invalid" for attempt in runtime["provider_attempts"])
            ),
        }
    finally:
        gate.release()


async def owned_execution(session: AsyncSession, answer_id: str, actor: User) -> dict:
    policy = await read_policy(session)
    cutoff = datetime.now(UTC) - timedelta(days=policy["days"]["conversations"])
    message = await session.scalar(
        select(ChatMessage).join(ChatSession, ChatSession.id == ChatMessage.session_id).where(
            ChatMessage.id == answer_id,
            ChatMessage.user_id == actor.username,
            ChatMessage.type == "assistant",
            ChatSession.user_id == actor.username,
            ChatSession.created_at > cutoff,
        )
    )
    if message is None:
        raise evidence_required()
    loaded = await AnswerExecutionStore(session, ChatRepository(session)).load_for_message(
        user_id=actor.username,
        session_id=message.session_id,
        message_id=message.id,
        message_type=message.type,
        indexed_execution_id=message.answer_execution_id,
    )
    projection = getattr(loaded, "projection", None)
    if not isinstance(projection, dict) or projection.get("state") not in {"completed", "failed"}:
        raise evidence_required()
    return projection


async def owned_condition_digest(session: AsyncSession, conditions_identity: str, actor: User) -> str:
    answer_id = await session.scalar(
        select(ChatMessage.id)
        .join(AnswerExecutionModel, AnswerExecutionModel.id == ChatMessage.answer_execution_id)
        .where(
            ChatMessage.user_id == actor.username,
            ChatMessage.type == "assistant",
            AnswerExecutionModel.user_id == actor.username,
            AnswerExecutionModel.request["query_condition_set"]["identity"].as_string() == conditions_identity,
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(1)
    )
    if answer_id is None:
        raise evidence_required()
    projection = await owned_execution(session, answer_id, actor)
    conditions = projection["query_condition_set"]
    if conditions["identity"] != conditions_identity:
        raise evidence_required()
    return canonical_json_sha256(conditions["conditions"])


async def diagnose_fixture(session: AsyncSession, item: dict, fixture_identity: str, observation: str) -> dict:
    fixture = await load_fixture(session, item, fixture_identity)
    if (
        fixture.get("schema") != "maintenance_fixture/v1"
        or fixture.get("item_identity") != item["id"]
        or fixture.get("active_publication_identities") != await active_publications(session)
        or fixture.get("retrieval_profile_identity") != get_retrieval_policy(get_runtime_settings()).identity
        or fixture.get("verification_method") != "independent_authenticated_replay"
        or fixture.get("verified_observation") != observation
        or item.get("fixture_identity") != fixture_identity
    ):
        raise evidence_required()
    review = fixture.get("publication_review")
    if review and review != await current_publication_review(session, review["entry_identity"]):
        raise evidence_required()
    if fixture["reference_publication_reviews"] != await eligible_reference_reviews(
        session, fixture["reference_evidence_publication_identities"]
    ):
        raise evidence_required()
    classification, disposition = diagnosis_route(observation, fixture["diagnosis_facts"])
    return {
        "classification": classification,
        "disposition": disposition,
        "diagnosis_fixture_identity": fixture_identity,
        "diagnosed_observation": observation,
    }
