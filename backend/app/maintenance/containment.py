from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.delivery_acceptance.schemas import FailureBlockingScopeInput
from app.maintenance.history import ItemIdentity, invalid_history, project_item
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.retention.policy import lock_registry
from app.settings.runtime import get_runtime_settings


class ContainmentVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    item_identity: ItemIdentity
    item_revision: int = Field(ge=1, strict=True)
    containment_event_id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")
    scope: FailureBlockingScopeInput


@dataclass(frozen=True)
class _VerificationContext:
    receipt: ContainmentVerification
    actor_identity: str


_verification: ContextVar[_VerificationContext | None] = ContextVar("maintenance_verification", default=None)


def supported_answer_scope(scope: dict) -> bool:
    if scope.get("scope") == "entry_version":
        return True
    if scope.get("scope") == "collection":
        return scope.get("identity") == "collection:production-rag-agent-engineering"
    if scope.get("scope") == "deployment":
        deployment = get_runtime_settings().generation_deployment_identity
        return bool(deployment) and scope.get("identity") == deployment
    return False


@contextmanager
def verification_scope(item: dict, actor: User):
    context = None
    if item["severity"] in {"p0", "p1"}:
        try:
            receipt = ContainmentVerification(
                item_identity=item["id"], item_revision=item["revision"],
                containment_event_id=item["containment_event_id"], scope=item["blocking_scope"],
            )
        except (ValidationError, KeyError, TypeError) as exc:
            raise invalid_history() from exc
        context = _VerificationContext(receipt=receipt, actor_identity=f"member:{actor.id.hex}")
    token = _verification.set(context)
    try:
        yield
    finally:
        _verification.reset(token)


def verification_receipt() -> dict | None:
    context = _verification.get()
    return context.receipt.model_dump(mode="json") if context is not None else None


def qualify_verification(item: dict, receipt: dict | None, event: CanonicalEventModel) -> None:
    if item["severity"] not in {"p0", "p1"}:
        if receipt is not None:
            raise invalid_history()
        return
    if (
        receipt is None
        or receipt["item_identity"] != item["id"]
        or receipt["item_revision"] + 1 != event.payload["revision"]
        or receipt["containment_event_id"] != item["containment_event_id"]
        or receipt["scope"] != item["blocking_scope"]
        or event.recorded_by != item["work_owner"]
    ):
        raise invalid_history()


async def _verify_current_context(session: AsyncSession, context: _VerificationContext, item: dict) -> None:
    from app.delivery_acceptance.service import DeliveryAcceptanceService

    receipt = context.receipt
    if (
        item["severity"] not in {"p0", "p1"}
        or item["state"] not in {"triaged", "in_progress"}
        or item["revision"] != receipt.item_revision
        or item["work_owner"] != context.actor_identity
        or item["containment_event_id"] != receipt.containment_event_id
        or item["blocking_scope"] != receipt.scope.model_dump(mode="json")
        or not supported_answer_scope(item["blocking_scope"])
        or not isinstance(item["administrator_identity"], str)
        or not isinstance(item.get("containment_record_identity"), str)
    ):
        raise invalid_history()
    for identity, role in ((context.actor_identity, "user"), (item["administrator_identity"], "admin")):
        principal = await session.get(User, UUID(identity.removeprefix("member:")), populate_existing=True)
        if principal is None or not principal.is_active or principal.role != role:
            raise invalid_history()
    accepted = await DeliveryAcceptanceService(session).get_projection(item["containment_record_identity"])
    if accepted["current_status"] != "suspended" or accepted["status_history"][-1]["event_id"] != receipt.containment_event_id:
        raise invalid_history()


async def active_answer_blocks(session: AsyncSession) -> list[dict]:
    context = _verification.get()
    records = list((await session.scalars(
        select(CanonicalRecordModel).where(
            CanonicalRecordModel.identity_kind == "maintenance_item",
            CanonicalRecordModel.payload["schema"].as_string() == "maintenance_item/v1",
        ).execution_options(populate_existing=True)
    )).all())
    if not records:
        if context is not None:
            raise invalid_history()
        return []
    events = (await session.scalars(
        select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id.in_([record.stable_id for record in records]),
        ).execution_options(populate_existing=True)
    )).all()
    by_item = defaultdict(list)
    for event in events:
        by_item[event.aggregate_id].append(event)
    blocks = []
    verified_context = False
    for record in records:
        item = project_item(record, by_item[record.stable_id])
        if context is not None and record.stable_id == context.receipt.item_identity:
            await _verify_current_context(session, context, item)
            verified_context = True
            continue
        if item["severity"] not in {"p0", "p1"} or item["state"] in {"resolved", "closed_confirmation"}:
            continue
        try:
            scope = FailureBlockingScopeInput.model_validate(item["blocking_scope"])
        except ValidationError as exc:
            raise invalid_history() from exc
        if not supported_answer_scope(scope.model_dump(mode="json")):
            raise invalid_history()
        blocks.append({"item_identity": record.stable_id, "scope": scope.model_dump(mode="json")})
    if context is not None and not verified_context:
        raise invalid_history()
    return blocks


def publication_is_blocked(blocks: list[dict], *, entry_identity: str, publication_identity: str) -> bool:
    for block in blocks:
        scope = block["scope"]
        if scope["scope"] == "entry_version":
            if scope["identity"] in {entry_identity, publication_identity}:
                return True
        elif supported_answer_scope(scope):
            return True
        else:
            raise invalid_history()
    return False


async def answer_is_blocked(session: AsyncSession, publication_identities: list[str]) -> bool:
    if not publication_identities:
        return False
    await lock_registry(session)
    blocks = await active_answer_blocks(session)
    if not blocks:
        return False
    publications = list((await session.scalars(
        select(CanonicalRecordModel).where(CanonicalRecordModel.stable_id.in_(publication_identities))
        .execution_options(populate_existing=True)
    )).all())
    if {record.stable_id for record in publications} != set(publication_identities):
        raise invalid_history()
    for record in publications:
        if (
            record.identity_kind != "published_knowledge_version"
            or record.payload.get("schema") != "published_knowledge_version/v1"
            or not isinstance(record.payload.get("entry_identity"), str)
        ):
            raise invalid_history()
        if publication_is_blocked(
            blocks, entry_identity=record.payload["entry_identity"], publication_identity=record.stable_id,
        ):
            return True
    return False
