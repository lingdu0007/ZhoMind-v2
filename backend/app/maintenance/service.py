from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.contracts.canonical import MaintenanceState, validate_transition
from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.knowledge_feedback.service import KnowledgeFeedbackService
from app.maintenance.affected_scope import retain_affected_scope
from app.maintenance.assignments import assignment_state, denied
from app.maintenance.cadence import dashboard, record_review
from app.maintenance.cadence_context import build_review_context
from app.maintenance.containment import supported_answer_scope, verification_scope
from app.maintenance.diagnosis import (
    active_publications,
    create_reproduction,
    diagnose_fixture,
    evidence_required,
    owned_condition_digest,
    owned_execution,
)
from app.maintenance.findings import finding_record_identity, load_finding, verification_fingerprint
from app.maintenance.fixtures import load_fixture
from app.maintenance.history import project_item
from app.maintenance.provider_verification import ProviderVerificationAuthorizationRecord, load_authorization
from app.maintenance.replay import approved_fixture, latest_replay_identity, load_replay, replay_fixture
from app.maintenance.resolution import verify_confirmation, verify_resolution
from app.maintenance.roadmap import qualify, review_candidate, roadmap_projection
from app.maintenance.roadmap_history import project_candidate
from app.maintenance.schemas import (
    AdministratorJoin,
    CadenceInput,
    DiagnosisInput,
    FindingInput,
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
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.chat import ChatMessage
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink
from app.model.user import User
from app.repository.chat_repository import ChatRepository
from app.retention.policy import lock_registry, read_policy
from app.settings.generation_routes import GenerationRouteService


class MaintenanceService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def principal(self, username: str) -> User:
        user = await self.session.scalar(select(User).where(User.username == username).execution_options(populate_existing=True))
        if user is None or not user.is_active or user.role != "user":
            raise denied()
        return user

    async def context(self, actor: User) -> dict:
        accepted = False
        assignment = None
        if actor.role == "user":
            identity = f"maintenance_item:assignment-{actor.id}"
            try:
                await self.principal(actor.username)
                state = await assignment_state(self.session, identity, actor)
                accepted = state == "accepted"
                assignment = {"id": identity, "state": state}
            except AppError as exc:
                if exc.code != "MAINTENANCE_AUTHORITY_REQUIRED":
                    raise
        return {
            "member_identity": f"member:{actor.id.hex}",
            "username": actor.username,
            "is_administrator": actor.role == "admin",
            "is_maintainer": accepted,
            "assignment": assignment,
        }

    async def assign(self, username: str, actor: User) -> dict:
        await lock_registry(self.session)
        administrator = await self.session.get(User, actor.id, populate_existing=True)
        if administrator is None or not administrator.is_active or administrator.role != "admin":
            raise denied()
        user = await self.principal(username)
        identity = f"maintenance_item:assignment-{user.id}"
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None:
            self.session.add(
                CanonicalRecordModel(
                    stable_id=identity,
                    identity_kind="maintenance_item",
                    identity_value=f"assignment-{user.id}",
                    state="assigned",
                    record_class="authoritative",
                    payload={"schema": "maintenance_assignment/v1", "member_id": str(user.id), "assigned_by": str(actor.id)},
                )
            )
            await self.session.commit()
        else:
            await assignment_state(self.session, identity, user)
        return {"id": identity, "username": username}

    async def accept(self, identity: str, actor: User) -> dict:
        await lock_registry(self.session)
        await self.principal(actor.username)
        state = await assignment_state(self.session, identity, actor)
        if state == "assigned":
            self.session.add(
                CanonicalEventModel(
                    aggregate_id=identity,
                    aggregate_kind="maintenance_item",
                    event_type="responsibility_accepted",
                    to_state="accepted",
                    recorded_by=f"member:{actor.id.hex}",
                    payload={"schema": "maintenance_assignment_event/v1"},
                )
            )
            await self.session.commit()
        return {"id": identity, "state": "accepted"}

    async def require_maintainer(self, actor: User) -> None:
        await self.principal(actor.username)
        identity = f"maintenance_item:assignment-{actor.id}"
        if await assignment_state(self.session, identity, actor) != "accepted":
            raise denied()

    async def events(self, identity: str) -> list[CanonicalEventModel]:
        return list(
            (
                await self.session.scalars(
                    select(CanonicalEventModel)
                    .where(
                        CanonicalEventModel.aggregate_id == identity,
                    )
                    .order_by(CanonicalEventModel.occurred_at, CanonicalEventModel.id)
                )
            ).all()
        )

    async def inbox(self, actor: User) -> dict:
        await self.require_maintainer(actor)
        await KnowledgeFeedbackService(self.session).purge_expired()
        signals = (
            await self.session.scalars(
                select(KnowledgeFeedbackSignal).order_by(
                    KnowledgeFeedbackSignal.created_at,
                    KnowledgeFeedbackSignal.id,
                )
            )
        ).all()
        projection = [
            {
                "id": signal.id,
                "answer_id": signal.answer_id,
                "answer_execution_id": signal.normalized_metadata.get("answer_execution_id"),
                "query_condition_set_identity": signal.normalized_metadata.get("query_condition_set_identity"),
                "knowledge_version_identities": signal.normalized_metadata.get("knowledge_version_identities"),
                "outcome": signal.normalized_metadata.get("outcome"),
                "entry_id": signal.entry_id,
                "label": signal.label,
                "description": signal.note,
                "submitted_by": signal.user_id,
                "submitted_at": signal.created_at.isoformat(),
            }
            for signal in signals
        ]
        await self.session.commit()
        return {"signals": projection}

    async def list_items(self, actor: User) -> dict:
        records = (
            await self.session.scalars(
                select(CanonicalRecordModel)
                .where(
                    CanonicalRecordModel.identity_kind == "maintenance_item",
                )
                .order_by(CanonicalRecordModel.created_at, CanonicalRecordModel.stable_id)
            )
        ).all()
        items = []
        for record in records:
            if record.payload.get("schema") != "maintenance_item/v1":
                continue
            if actor.role == "admin" or f"member:{actor.id.hex}" in (
                record.payload.get("accountable_maintainer"),
                record.payload.get("work_owner"),
            ):
                items.append(await self.get(record.stable_id, actor))
        return {"items": items}

    async def create(self, payload: MaintenanceCreate, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        worker = await self.principal(payload.work_owner_username)
        await KnowledgeFeedbackService(self.session).purge_expired()
        signals = await self.retained_signals(payload.signal_ids)
        containment = await self.containment(
            severity=payload.severity, classification=payload.classification,
            containment_record_identity=payload.containment_record_identity, signals=signals,
        )
        signals = await self.retained_signals(payload.signal_ids)
        identity = f"maintenance_item:{uuid4().hex}"
        fields = {
            "schema": "maintenance_item/v1",
            "classification": payload.classification,
            "severity": payload.severity,
            "disposition": payload.disposition,
            "coverage_position": payload.coverage_position.value,
            "affected_scope": await retain_affected_scope(self.session, signals),
            "accountable_maintainer": f"member:{actor.id.hex}",
            "work_owner": f"member:{worker.id.hex}",
            "verified_pattern": None,
            "result_links": [],
            **containment,
        }
        self.session.add(
            CanonicalRecordModel(
                stable_id=identity,
                identity_kind="maintenance_item",
                identity_value=identity.split(":", 1)[1],
                state="open",
                record_class="authoritative",
                payload=fields,
            )
        )
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity,
                aggregate_kind="maintenance_item",
                event_type="created",
                to_state="open",
                recorded_by=f"member:{actor.id.hex}",
                payload={"schema": "maintenance_event/v1", "revision": 1},
            )
        )
        await self.session.flush()
        self.session.add_all(MaintenanceSignalLink(item_id=identity, signal_id=signal.id) for signal in signals)
        await self.session.commit()
        return await self.get(identity, actor)

    async def retained_signals(self, identities: list[str]) -> list[KnowledgeFeedbackSignal]:
        policy = await read_policy(self.session)
        now = datetime.now(UTC)
        with self.session.no_autoflush:
            signals = list((await self.session.scalars(
                select(KnowledgeFeedbackSignal).where(
                    KnowledgeFeedbackSignal.id.in_(identities),
                    KnowledgeFeedbackSignal.expires_at > now,
                    KnowledgeFeedbackSignal.created_at > now - timedelta(days=policy["days"]["feedback_signals"]),
                ).with_for_update().execution_options(populate_existing=True),
            )).all())
        if len(signals) != len(set(identities)):
            raise AppError(status_code=409, code="MAINTENANCE_SIGNAL_UNAVAILABLE", message="retained signals required")
        return signals

    async def containment(
        self, *, severity: str, classification: str, containment_record_identity: str | None,
        signals: list[KnowledgeFeedbackSignal], verified_scope: dict | None = None,
    ) -> dict:
        empty = {"blocking_scope": None, "containment_event_id": None, "administrator_identity": None}
        if severity not in {"p0", "p1"} and containment_record_identity is None:
            return empty
        error = AppError(
            status_code=409, code="MAINTENANCE_CONTAINMENT_REQUIRED", message="current administrator-verified containment required"
        )
        if containment_record_identity is None:
            raise error
        projection = await DeliveryAcceptanceService(self.session).get_projection(containment_record_identity)
        event = projection["status_history"][-1]
        failure = event.get("status_failure")
        actor_identity = event.get("recorded_by", "")
        if (
            projection["current_status"] != "suspended"
            or event["status"] != "suspended"
            or event["reason_code"] != "integrity_failure"
            or not isinstance(failure, dict)
            or not actor_identity.startswith("member:")
        ):
            raise error
        if not supported_answer_scope(failure["blocking_scope"]):
            raise error
        try:
            administrator = await self.session.get(User, UUID(actor_identity.removeprefix("member:")))
        except ValueError as exc:
            raise error from exc
        if administrator is None or not administrator.is_active or administrator.role != "admin":
            raise error
        if failure["failure_kind"] == "shared_privacy" and classification != "product-privacy-operations":
            raise error
        if failure["blocking_scope"]["scope"] == "entry_version":
            blocked_identity = failure["blocking_scope"]["identity"]
            if verified_scope is not None and (
                verified_scope["gap_contexts"]
                or not verified_scope["entry_versions"]
                or any(
                    blocked_identity not in {target["entry_identity"], target["publication_identity"]}
                    for target in verified_scope["entry_versions"]
                )
            ):
                raise error
            for signal in signals:
                if signal.entry_id is None:
                    raise error
                entry_identity = f"entry:{signal.entry_id}"
                versions = signal.normalized_metadata.get("knowledge_version_identities")
                if not isinstance(versions, list) or not versions:
                    raise error
                matching_versions = []
                for version in versions:
                    if not isinstance(version, str):
                        raise error
                    record = await self.session.get(CanonicalRecordModel, version)
                    if (
                        record is not None
                        and record.identity_kind == "published_knowledge_version"
                        and record.payload.get("schema") == "published_knowledge_version/v1"
                        and record.payload.get("entry_identity") == entry_identity
                    ):
                        matching_versions.append(version)
                if not matching_versions or blocked_identity not in {entry_identity, *matching_versions}:
                    raise error
        return {
            "blocking_scope": failure["blocking_scope"],
            "containment_record_identity": containment_record_identity,
            "containment_event_id": event["event_id"],
            "administrator_identity": actor_identity,
        }

    async def transition(self, identity: str, payload: MaintenanceTransition, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        item = await self.get(identity, actor)
        if item["accountable_maintainer"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        await self.require_administrator(item)
        try:
            validate_transition(MaintenanceState, item["state"], payload.state)
        except ValueError as exc:
            raise AppError(status_code=409, code="MAINTENANCE_TRANSITION_INVALID", message="illegal maintenance transition") from exc
        if payload.state == MaintenanceState.IN_PROGRESS and item.get("diagnosis_fixture_identity") is None:
            raise AppError(status_code=409, code="MAINTENANCE_EVIDENCE_REQUIRED", message="verified disposition evidence required")
        closing = payload.state == MaintenanceState.CLOSED_CONFIRMATION and (
            item["state"] == "resolved"
            or (item["classification"] == "confirmation" and item.get("finding_identities") and item["severity"] not in {"p0", "p1"})
        )
        if payload.state not in {MaintenanceState.TRIAGED, MaintenanceState.IN_PROGRESS} and not closing:
            raise AppError(status_code=409, code="MAINTENANCE_EVIDENCE_REQUIRED", message="verified disposition evidence required")
        confirmation = closing and item["state"] != "resolved"
        changes = {"result_links": await verify_confirmation(self.session, item, current=True)} if confirmation else None
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity,
                aggregate_kind="maintenance_item",
                event_type="confirmation_closed" if confirmation else "transition",
                from_state=item["state"],
                to_state=payload.state.value,
                recorded_by=f"member:{actor.id.hex}",
                payload={
                    "schema": "maintenance_event/v1", "revision": item["revision"] + 1,
                    **({"changes": changes} if changes is not None else {}),
                },
            )
        )
        await self.session.commit()
        return await self.get(identity, actor)

    async def join_administrator(self, identity: str, payload: AdministratorJoin, actor: User) -> dict:
        await lock_registry(self.session)
        if not actor.is_active or actor.role != "admin":
            raise denied()
        item = await self.get(identity, actor)
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if item["state"] not in {"open", "triaged", "in_progress"}:
            raise AppError(status_code=409, code="MAINTENANCE_TRANSITION_INVALID", message="item is not open work")
        await self.append_observation(item, actor, "administrator_joined", {"administrator_identity": f"member:{actor.id.hex}"})
        await self.session.commit()
        return await self.get(identity, actor)

    async def require_administrator(self, item: dict) -> None:
        if item["severity"] not in {"p0", "p1"} and item["classification"] not in {
            "retrieval-answer-behavior",
            "product-privacy-operations",
        }:
            return
        identity = item.get("administrator_identity")
        administrator = await self.session.get(User, UUID(identity.removeprefix("member:"))) if identity else None
        if administrator is None or not administrator.is_active or administrator.role != "admin":
            raise AppError(
                status_code=409, code="MAINTENANCE_ADMINISTRATOR_REQUIRED", message="current administrator participation required"
            )

    async def authorize_provider_verification(
        self, identity: str, payload: ProviderVerificationAuthorizationInput, actor: User,
    ) -> dict:
        await lock_registry(self.session)
        administrator = await self.session.get(User, actor.id, populate_existing=True)
        if administrator is None or not administrator.is_active or administrator.role != "admin":
            raise denied()
        item = await self.get(identity, actor)
        if (
            item["administrator_identity"] != f"member:{actor.id.hex}"
            or item["severity"] not in {"p0", "p1"} or item["classification"] != "product-privacy-operations"
            or item["state"] not in {"triaged", "in_progress"}
            or item["revision"] != payload.expected_revision
            or payload.admission_acceptance_identity == item.get("containment_record_identity")
        ):
            raise evidence_required()
        worker = await self.session.scalar(
            select(User).where(User.id == UUID(item["work_owner"].removeprefix("member:")))
            .with_for_update().execution_options(populate_existing=True)
        )
        if worker is None or not worker.is_active or worker.role != "user":
            raise evidence_required()
        await self.require_administrator(item)
        fixture_identity = None
        for finding_identity in item.get("finding_identities", []):
            finding = await load_finding(self.session, item, finding_identity)
            fixture = await load_fixture(self.session, item, finding["fixture_identity"])
            if (
                fixture["verified_observation"] == "provider_failure"
                and fixture["query_condition_set_identity"] == payload.query_condition_set_identity
            ):
                fixture_identity = finding["fixture_identity"]
                break
        signals, verified_scope = await self.provider_verification_basis(
            item, worker, payload.query_condition_set_identity, fixture_identity,
        )
        containment = await self.containment(
            severity=item["severity"], classification=item["classification"],
            containment_record_identity=item["containment_record_identity"], signals=signals, verified_scope=verified_scope,
        )
        if any(containment[key] != item[key] for key in ("blocking_scope", "containment_event_id", "administrator_identity")):
            raise evidence_required()
        admission = await GenerationRouteService(self.session).verification_admission(
            payload.route_identity, payload.admission_acceptance_identity, actor,
        )
        body = ProviderVerificationAuthorizationRecord.model_validate({
            "schema": "maintenance_provider_verification_authorization/v1",
            "item_identity": identity, "item_revision": item["revision"],
            "work_owner": item["work_owner"], "administrator_identity": item["administrator_identity"],
            "query_condition_set_identity": payload.query_condition_set_identity,
            "approved_fixture_identity": fixture_identity,
            "active_publication_identities": await active_publications(self.session),
            "containment_record_identity": item["containment_record_identity"],
            "containment_event_id": item["containment_event_id"], **admission,
        }).model_dump(mode="json", by_alias=True)
        authorization_sha256 = canonical_json_sha256(body)
        grant_id = f"maintenance_item:{authorization_sha256[:32]}"
        if await self.session.get(CanonicalRecordModel, grant_id) is not None:
            existing = await load_authorization(self.session, grant_id)
            if existing != {"id": grant_id, **body}:
                raise evidence_required()
            return existing
        self.session.add(CanonicalRecordModel(
            stable_id=grant_id, identity_kind="maintenance_item", identity_value=grant_id.split(":", 1)[1],
            record_class="immutable", state="authorized", payload=body,
        ))
        self.session.add(CanonicalEventModel(
            aggregate_id=grant_id, aggregate_kind="maintenance_item", event_type="created",
            to_state="authorized", recorded_by=item["administrator_identity"],
            payload={
                "schema": "maintenance_provider_verification_authorization_event/v1",
                "authorization_sha256": authorization_sha256,
            },
        ))
        await self.session.commit()
        return await load_authorization(self.session, grant_id)

    async def provider_verification_authorization(self, identity: str, actor: User) -> dict:
        authorization = await load_authorization(self.session, identity)
        await self.get(authorization["item_identity"], actor)
        return authorization

    async def provider_verification_basis(
        self, item: dict, worker: User, conditions_identity: str, fixture_identity: str | None,
    ) -> tuple[list[KnowledgeFeedbackSignal], dict | None]:
        conditions_digest = await owned_condition_digest(self.session, conditions_identity, worker)
        if fixture_identity is not None:
            fixture = await approved_fixture(self.session, item, fixture_identity)
            if (
                fixture["verified_observation"] != "provider_failure"
                or fixture["observed_outcome"] != "generation_unavailable"
                or fixture["generation_context"] is None
                or fixture["query_condition_set_identity"] != conditions_identity
            ):
                raise evidence_required()
            return [], fixture["affected_scope"]
        signal_ids = list((await self.session.scalars(
            select(MaintenanceSignalLink.signal_id).where(MaintenanceSignalLink.item_id == item["id"])
        )).all())
        signals = await self.retained_signals(signal_ids)
        if not any(
            isinstance(signal.normalized_metadata, dict)
            and signal.normalized_metadata.get("query_conditions_sha256") == conditions_digest
            for signal in signals
        ):
            raise evidence_required()
        return signals, None

    async def verify_provider_authorization(self, identity: str, actor: User, conditions_identity: str) -> tuple[dict, User]:
        await lock_registry(self.session)
        await self.principal(actor.username)
        authorization = await load_authorization(self.session, identity)
        item = await self.get(authorization["item_identity"], actor)
        if (
            item["work_owner"] != f"member:{actor.id.hex}"
            or item["work_owner"] != authorization["work_owner"]
            or item["revision"] != authorization["item_revision"]
            or item["state"] not in {"triaged", "in_progress"}
            or item["severity"] not in {"p0", "p1"}
            or item["classification"] != "product-privacy-operations"
            or conditions_identity != authorization["query_condition_set_identity"]
            or await active_publications(self.session) != authorization["active_publication_identities"]
            or any(item[key] != authorization[key] for key in (
                "administrator_identity", "containment_record_identity", "containment_event_id",
            ))
        ):
            raise evidence_required()
        await self.require_administrator(item)
        signals, verified_scope = await self.provider_verification_basis(
            item, actor, conditions_identity, authorization["approved_fixture_identity"],
        )
        containment = await self.containment(
            severity=item["severity"], classification=item["classification"],
            containment_record_identity=item["containment_record_identity"], signals=signals, verified_scope=verified_scope,
        )
        if any(containment[key] != item[key] for key in ("blocking_scope", "containment_event_id", "administrator_identity")):
            raise evidence_required()
        administrator = await self.session.get(
            User, UUID(authorization["administrator_identity"].removeprefix("member:")), populate_existing=True,
        )
        if administrator is None:
            raise evidence_required()
        admission = await GenerationRouteService(self.session).verification_admission(
            authorization["route_identity"], authorization["acceptance_record_identity"], administrator,
        )
        if any(authorization[key] != value for key, value in admission.items()):
            raise evidence_required()
        return authorization, administrator

    async def reproduce(self, identity: str, payload: ReproductionInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.principal(actor.username)
        item = await self.get(identity, actor)
        if item["work_owner"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if item["state"] not in {"triaged", "in_progress"}:
            raise AppError(status_code=409, code="MAINTENANCE_TRANSITION_INVALID", message="triage before reproducing")
        await KnowledgeFeedbackService(self.session).purge_expired()
        with verification_scope(item, actor):
            fixture = await create_reproduction(self.session, item, payload, actor)
        await self.principal(actor.username)
        current = await self.get(identity, actor)
        if current["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if current["work_owner"] != f"member:{actor.id.hex}":
            raise denied()
        await self.append_observation(
            item,
            actor,
            "reproduced",
            {
                "fixture_identity": fixture["id"],
                "fixture_sha256": canonical_json_sha256({key: value for key, value in fixture.items() if key != "id"}),
            },
        )
        await self.session.commit()
        return fixture

    async def reproduction_inputs(self, identity: str, actor: User) -> dict:
        await lock_registry(self.session)
        await self.principal(actor.username)
        item = await self.get(identity, actor)
        if item["work_owner"] != f"member:{actor.id.hex}":
            raise denied()
        await KnowledgeFeedbackService(self.session).purge_expired()
        await ChatRepository(self.session).purge_expired_sessions()
        signals = (
            await self.session.scalars(
                select(KnowledgeFeedbackSignal)
                .join(
                    MaintenanceSignalLink,
                    MaintenanceSignalLink.signal_id == KnowledgeFeedbackSignal.id,
                )
                .where(
                    MaintenanceSignalLink.item_id == identity,
                    KnowledgeFeedbackSignal.user_id != actor.username,
                )
                .order_by(KnowledgeFeedbackSignal.created_at, KnowledgeFeedbackSignal.id)
            )
        ).all()
        targets = [
            {
                "signal_id": signal.id,
                "label": signal.label,
                "entry_id": signal.entry_id,
                "query_condition_set_identity": signal.normalized_metadata.get("query_condition_set_identity"),
                "outcome": signal.normalized_metadata.get("outcome"),
            }
            for signal in signals
            if signal.normalized_metadata.get("query_condition_set_identity")
        ]
        messages = (
            await self.session.scalars(
                select(ChatMessage)
                .where(
                    ChatMessage.user_id == actor.username,
                    ChatMessage.type == "assistant",
                )
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(100)
            )
        ).all()
        answers = []
        for message in messages:
            try:
                execution = await owned_execution(self.session, message.id, actor)
            except AppError as exc:
                if exc.code != "MAINTENANCE_EVIDENCE_REQUIRED":
                    raise
                continue
            answers.append(
                {
                    "answer_id": message.id,
                    "session_id": message.session_id,
                    "query_condition_set_identity": execution["query_condition_set"]["identity"],
                    "outcome": execution.get("outcome"),
                    "state": execution["state"],
                }
            )
        await self.session.commit()
        return {"targets": targets, "answers": answers}

    async def get_fixture(self, identity: str, actor: User) -> dict:
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None or record.payload.get("schema") != "maintenance_fixture/v1":
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="maintenance fixture not found")
        item = await self.get(record.payload.get("item_identity", ""), actor)
        fixture = await load_fixture(self.session, item, identity)
        return {
            "id": identity,
            "latest_replay_identity": await latest_replay_identity(self.session, item, identity),
            **{
                key: fixture[key]
                for key in (
                    "item_identity",
                    "affected_scope",
                    "query_condition_set_identity",
                    "expected_outcome",
                    "observed_outcome",
                    "observed_state",
                    "active_publication_identities",
                    "evidence_publication_identities",
                    "retrieval_profile_identity",
                    "verified_observation",
                    "verification_method",
                    "publication_review",
                )
            },
        }

    async def diagnose(self, identity: str, payload: DiagnosisInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        item = await self.get(identity, actor)
        if item["accountable_maintainer"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        changes = await diagnose_fixture(self.session, item, payload.fixture_identity, payload.observation)
        await self.require_administrator({**item, **changes})
        await self.append_observation(item, actor, "diagnosed", changes)
        await self.session.commit()
        return await self.get(identity, actor)

    async def replay(self, identity: str, payload: ReplayInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.principal(actor.username)
        item = await self.get(identity, actor)
        if item["work_owner"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if item["state"] != "in_progress":
            raise AppError(status_code=409, code="MAINTENANCE_TRANSITION_INVALID", message="start work before repair verification")
        await self.require_administrator(item)
        with verification_scope(item, actor):
            replay = await replay_fixture(
                self.session, item, payload.fixture_identity, actor,
                answer_id=payload.answer_id,
                authorization_identity=payload.provider_verification_authorization_identity,
            )
        await self.principal(actor.username)
        current = await self.get(identity, actor)
        if current["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if current["work_owner"] != f"member:{actor.id.hex}":
            raise denied()
        await self.require_administrator(current)
        await self.append_observation(
            current,
            actor,
            "fixture_replayed",
            {
                "replay_identity": replay["id"],
                "replay_sha256": canonical_json_sha256({key: value for key, value in replay.items() if key != "id"}),
            },
        )
        await self.session.commit()
        return await self.get_replay(replay["id"], actor)

    async def get_replay(self, identity: str, actor: User) -> dict:
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None or record.payload.get("schema") != "maintenance_replay/v1":
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="maintenance replay not found")
        item = await self.get(record.payload.get("item_identity", ""), actor)
        replay = await load_replay(self.session, item, identity)
        publications = []
        for publication_identity in replay["evidence_publication_identities"]:
            publication = await self.session.get(CanonicalRecordModel, publication_identity)
            if publication is None or publication.payload.get("schema") != "published_knowledge_version/v1":
                raise evidence_required()
            publications.append({
                "publication_identity": publication_identity,
                "entry_identity": publication.payload["entry_identity"],
                "revision_identity": publication.payload["editorial_revision_identity"],
            })
        return {**replay, "evidence_publications": publications}

    async def approve_finding(self, identity: str, payload: FindingInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        item = await self.get(identity, actor)
        if item["accountable_maintainer"] != f"member:{actor.id.hex}" or item["work_owner"] == f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if item.get("diagnosis_fixture_identity") != payload.fixture_identity:
            raise AppError(status_code=409, code="MAINTENANCE_EVIDENCE_REQUIRED", message="diagnosed fixture required")
        await diagnose_fixture(self.session, item, payload.fixture_identity, item["diagnosed_observation"])
        fixture_record = await self.session.get(CanonicalRecordModel, payload.fixture_identity)
        if fixture_record is None:
            raise AppError(status_code=409, code="MAINTENANCE_EVIDENCE_REQUIRED", message="fixture required")
        fixture = fixture_record.payload
        fingerprint = verification_fingerprint(fixture)
        finding_identity = finding_record_identity(identity, fixture)
        existing = await self.session.get(CanonicalRecordModel, finding_identity)
        if existing is not None:
            return await load_finding(self.session, item, finding_identity)
        pattern = {
            "classification": item["classification"],
            "coverage_position": item["coverage_position"],
            "observation": item["diagnosed_observation"],
        }
        finding = {
            "schema": "validated_finding/v1",
            "item_identity": identity,
            "pattern": pattern,
            "verification_method": fixture["verification_method"],
            "fixture_identity": payload.fixture_identity,
            "verification_fingerprint": fingerprint,
            "expected_outcome": fixture["expected_outcome"],
            "observed_outcome": fixture["observed_outcome"],
            "result_links": [f"evidence://maintenance/fixtures/{payload.fixture_identity.split(':', 1)[1]}"],
        }
        self.session.add(
            CanonicalRecordModel(
                stable_id=finding_identity,
                identity_kind="maintenance_item",
                identity_value=finding_identity.split(":", 1)[1],
                state="verified",
                record_class="immutable",
                payload=finding,
            )
        )
        await self.append_observation(
            item,
            actor,
            "finding_approved",
            {
                "verified_pattern": pattern,
                "finding_identities": [*item.get("finding_identities", []), finding_identity],
                "finding_sha256": {**item.get("finding_sha256", {}), finding_identity: canonical_json_sha256(finding)},
            },
        )
        await self.session.commit()
        return {"id": finding_identity, **finding}

    async def get_finding(self, identity: str, actor: User) -> dict:
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None or record.payload.get("schema") != "validated_finding/v1":
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="finding not found")
        item = await self.get(record.payload["item_identity"], actor)
        return await load_finding(self.session, item, identity)

    async def qualify_roadmap(self, identity: str, payload: RoadmapInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        item = await self.get(identity, actor)
        if item["accountable_maintainer"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        owner = await self.principal(payload.owner_username)
        await KnowledgeFeedbackService(self.session).purge_expired()
        candidate = await qualify(self.session, item, payload, owner)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity,
                aggregate_kind="maintenance_item",
                event_type="roadmap_qualified",
                from_state=item["state"],
                to_state="deferred",
                recorded_by=f"member:{actor.id.hex}",
                payload={
                    "schema": "maintenance_event/v1",
                    "revision": item["revision"] + 1,
                    "changes": {
                        "roadmap_identity": candidate["id"],
                        "roadmap_sha256": canonical_json_sha256(
                            {key: value for key, value in candidate.items() if key not in {"id", "overdue_review"}}
                        ),
                        "disposition": "roadmap-deferral",
                        "result_links": [f"evidence://maintenance/roadmap/{candidate['id'].split(':', 1)[1]}"],
                    },
                },
            )
        )
        await self.session.commit()
        return candidate

    async def get_roadmap(self, identity: str, actor: User) -> dict:
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None or record.payload.get("schema") != "knowledge_roadmap_candidate/v1":
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="roadmap candidate not found")
        item_record = await self.session.get(CanonicalRecordModel, record.payload.get("item_identity", ""))
        if item_record is None:
            raise evidence_required()
        item_events = await self.events(item_record.stable_id)
        project_item(item_record, item_events)
        qualifications = [
            event
            for event in item_events
            if event.event_type == "roadmap_qualified" and event.payload.get("changes", {}).get("roadmap_identity") == identity
        ]
        if len(qualifications) != 1:
            raise evidence_required()
        qualification = qualifications[0]
        prior = project_item(item_record, [event for event in item_events if event.payload["revision"] < qualification.payload["revision"]])
        candidate = project_candidate(record, await self.events(identity), {"id": item_record.stable_id, **prior}, qualification)
        if candidate["owner_identity"] != f"member:{actor.id.hex}":
            await self.get(candidate["item_identity"], actor)
        return roadmap_projection(candidate)

    async def list_roadmap(self, actor: User) -> dict:
        member = f"member:{actor.id.hex}"
        items = (await self.list_items(actor))["items"]
        statement = select(CanonicalRecordModel).where(
            CanonicalRecordModel.identity_kind == "maintenance_item",
            CanonicalRecordModel.payload["schema"].as_string() == "knowledge_roadmap_candidate/v1",
        )
        if actor.role != "admin":
            transfers = select(CanonicalEventModel.aggregate_id).where(
                CanonicalEventModel.event_type == "roadmap_reviewed",
                CanonicalEventModel.payload["changes"]["owner_identity"].as_string() == member,
            )
            statement = statement.where(
                or_(
                    CanonicalRecordModel.payload["owner_identity"].as_string() == member,
                    CanonicalRecordModel.payload["item_identity"].as_string().in_([item["id"] for item in items]),
                    CanonicalRecordModel.stable_id.in_(transfers),
                )
            )
        records = (
            await self.session.scalars(
                statement.order_by(
                    CanonicalRecordModel.created_at,
                    CanonicalRecordModel.stable_id,
                )
            )
        ).all()
        candidates = []
        for record in records:
            try:
                candidates.append(await self.get_roadmap(record.stable_id, actor))
            except AppError as exc:
                if exc.code != "MAINTENANCE_AUTHORITY_REQUIRED":
                    raise
        return {"candidates": candidates}

    async def review_roadmap(self, identity: str, payload: RoadmapReview, actor: User) -> dict:
        await lock_registry(self.session)
        await self.principal(actor.username)
        candidate = await self.get_roadmap(identity, actor)
        if candidate["owner_identity"] != f"member:{actor.id.hex}":
            raise denied()
        if candidate["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current roadmap revision")
        owner = await self.principal(payload.owner_username)
        changes = await review_candidate(self.session, candidate, payload, owner)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity,
                aggregate_kind="maintenance_item",
                event_type="roadmap_reviewed",
                from_state=candidate["state"],
                to_state=changes["state"],
                recorded_by=f"member:{actor.id.hex}",
                payload={"schema": "roadmap_review/v1", "revision": candidate["revision"] + 1, "changes": changes},
            )
        )
        await self.session.commit()
        return roadmap_projection({**candidate, **changes, "revision": candidate["revision"] + 1})

    async def get_map(self, identity: str, actor: User) -> dict:
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None or record.payload.get("schema") != "maintenance_wayfinder_map/v1":
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="Wayfinder map not found")
        candidate = await self.get_roadmap(record.payload.get("candidate_identity", ""), actor)
        if (
            record.identity_kind != "maintenance_item"
            or record.record_class != "authoritative"
            or record.state != "open"
            or set(record.payload) != {"schema", "candidate_identity", "markdown"}
            or not isinstance(record.payload["markdown"], str)
            or candidate["state"] != "mapped"
            or candidate.get("map_identity") != identity
            or candidate.get("map_sha256") != canonical_json_sha256(record.payload)
        ):
            raise evidence_required()
        return {"id": identity, **record.payload}

    async def record_cadence(self, payload: CadenceInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        context = await self.get_review_context(actor)
        review = await record_review(self.session, context, payload)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=review["id"],
                aggregate_kind="maintenance_item",
                event_type="cadence_reviewed",
                to_state="reviewed",
                recorded_by=f"member:{actor.id.hex}",
                payload={
                    "schema": "maintenance_cadence_event/v1",
                    "record_sha256": canonical_json_sha256({key: value for key, value in review.items() if key != "id"}),
                },
            )
        )
        await self.session.commit()
        return review

    async def get_review_context(self, actor: User) -> dict:
        await self.require_maintainer(actor)
        return await build_review_context(
            self.session,
            (await self.list_items(actor))["items"],
            (await self.list_roadmap(actor))["candidates"],
            f"member:{actor.id.hex}",
        )

    async def get_dashboard(self, actor: User) -> dict:
        await self.require_maintainer(actor)
        return await dashboard(self.session, (await self.list_items(actor))["items"], actor_identity=f"member:{actor.id.hex}")

    async def resolve(self, identity: str, payload: ResolutionInput, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        item = await self.get(identity, actor)
        if item["accountable_maintainer"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        await self.require_administrator(item)
        links = await verify_resolution(self.session, item, payload)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity,
                aggregate_kind="maintenance_item",
                event_type="resolved",
                from_state=item["state"],
                to_state="resolved",
                recorded_by=f"member:{actor.id.hex}",
                payload={
                    "schema": "maintenance_event/v1",
                    "revision": item["revision"] + 1,
                    "changes": {
                        "disposition": payload.disposition,
                        "result_links": links,
                    },
                },
            )
        )
        await self.session.commit()
        return await self.get(identity, actor)

    async def append_observation(self, item: dict, actor: User, event_type: str, changes: dict) -> None:
        self.session.add(
            CanonicalEventModel(
                aggregate_id=item["id"],
                aggregate_kind="maintenance_item",
                event_type=event_type,
                from_state=item["state"],
                to_state=item["state"],
                recorded_by=f"member:{actor.id.hex}",
                payload={"schema": "maintenance_event/v1", "revision": item["revision"] + 1, "changes": changes},
            )
        )
        await self.session.flush()

    async def consolidate(self, identity: str, payload: MaintenanceConsolidation, actor: User) -> dict:
        await lock_registry(self.session)
        await self.require_maintainer(actor)
        item = await self.get(identity, actor)
        if item["accountable_maintainer"] != f"member:{actor.id.hex}":
            raise denied()
        if item["revision"] != payload.expected_revision:
            raise AppError(status_code=409, code="MAINTENANCE_STALE", message="reload the current maintenance revision")
        if item["state"] not in {"open", "triaged", "in_progress"}:
            raise AppError(status_code=409, code="MAINTENANCE_TRANSITION_INVALID", message="item is not open work")
        await KnowledgeFeedbackService(self.session).purge_expired()
        signals = await self.retained_signals(payload.signal_ids)
        containment = await self.containment(
            severity=item["severity"], classification=item["classification"],
            containment_record_identity=item.get("containment_record_identity"), signals=signals,
        )
        if any(containment.get(key) != item.get(key) for key in ("blocking_scope", "containment_event_id")):
            raise AppError(
                status_code=409, code="MAINTENANCE_CONTAINMENT_REQUIRED",
                message="current administrator-verified containment required",
            )
        for signal in signals:
            if await self.session.get(MaintenanceSignalLink, (identity, signal.id)) is None:
                self.session.add(MaintenanceSignalLink(item_id=identity, signal_id=signal.id))
        await self.retained_signals(payload.signal_ids)
        self.session.add(
            CanonicalEventModel(
                aggregate_id=identity,
                aggregate_kind="maintenance_item",
                event_type="signals_consolidated",
                from_state=item["state"],
                to_state=item["state"],
                recorded_by=f"member:{actor.id.hex}",
                payload={
                    "schema": "maintenance_event/v1", "revision": item["revision"] + 1,
                    "changes": {"affected_scope": await retain_affected_scope(self.session, signals, item["affected_scope"])},
                },
            )
        )
        await self.session.commit()
        return await self.get(identity, actor)

    async def get(self, identity: str, actor: User) -> dict:
        record = await self.session.get(CanonicalRecordModel, identity)
        if record is None or record.payload.get("schema") != "maintenance_item/v1":
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="maintenance item not found")
        fields = project_item(record, await self.events(identity))
        if (
            f"member:{actor.id.hex}"
            not in {
                fields["accountable_maintainer"],
                fields["work_owner"],
            }
            and actor.role != "admin"
        ):
            raise denied()
        links = (
            await self.session.scalars(
                select(MaintenanceSignalLink.signal_id)
                .join(
                    KnowledgeFeedbackSignal,
                    KnowledgeFeedbackSignal.id == MaintenanceSignalLink.signal_id,
                )
                .where(
                    MaintenanceSignalLink.item_id == identity,
                    KnowledgeFeedbackSignal.expires_at > datetime.now(UTC),
                )
            )
        ).all()
        result = {
            "id": identity,
            **fields,
            "signal_count": len(links),
        }
        if fields["state"] == "closed_confirmation" and fields["classification"] == "confirmation":
            if fields["result_links"] != await verify_confirmation(self.session, result, current=False):
                raise evidence_required()
        return result
