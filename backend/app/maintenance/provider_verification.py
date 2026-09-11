from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.maintenance.fixtures import Hash, PublicationIdentity, evidence_required
from app.maintenance.history import ItemIdentity, MemberIdentity
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.user import User
from app.settings.generation_routes import GenerationRouteService


class ProviderVerificationAuthorizationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["maintenance_provider_verification_authorization/v1"] = Field(alias="schema")
    item_identity: ItemIdentity
    item_revision: int = Field(ge=1, strict=True)
    work_owner: MemberIdentity
    administrator_identity: MemberIdentity
    query_condition_set_identity: Hash
    approved_fixture_identity: ItemIdentity | None = None
    active_publication_identities: list[PublicationIdentity]
    route_identity: str = Field(pattern=r"^provider_route:[a-f0-9]{64}$")
    acceptance_record_identity: str = Field(pattern=r"^delivery_acceptance_record:[A-Za-z0-9._/-]+$")
    acceptance_evidence_sha256: Hash
    containment_record_identity: str = Field(pattern=r"^delivery_acceptance_record:[A-Za-z0-9._/-]+$")
    containment_event_id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")


async def load_authorization(session: AsyncSession, identity: str) -> dict:
    record = await session.get(CanonicalRecordModel, identity)
    if (
        record is None or record.record_class != "immutable" or record.state != "authorized"
        or record.identity_kind != "maintenance_item" or record.stable_id != f"maintenance_item:{record.identity_value}"
    ):
        raise evidence_required()
    try:
        payload = ProviderVerificationAuthorizationRecord.model_validate(record.payload).model_dump(mode="json", by_alias=True)
        events = list((await session.scalars(
            select(CanonicalEventModel).where(CanonicalEventModel.aggregate_id == identity)
        )).all())
        if len(events) != 1:
            raise ValueError("authorization history")
        event = events[0]
        if (
            event.aggregate_kind != "maintenance_item" or event.event_type != "created"
            or event.from_state is not None or event.to_state != "authorized"
            or event.recorded_by != payload["administrator_identity"]
            or event.payload != {
                "schema": "maintenance_provider_verification_authorization_event/v1",
                "authorization_sha256": canonical_json_sha256(payload),
            }
        ):
            raise ValueError("authorization provenance")
    except (ValidationError, TypeError, ValueError) as exc:
        raise evidence_required() from exc
    return {"id": identity, **payload}


async def qualify_execution_authorization(
    session: AsyncSession, artifact: dict, event: CanonicalEventModel,
) -> None:
    context = artifact.get("generation_context")
    if context is None or "authorization_identity" not in context:
        return
    authorization = await load_authorization(session, context["authorization_identity"])
    if (
        context["authorization_sha256"] != canonical_json_sha256({
            key: value for key, value in authorization.items() if key != "id"
        })
        or any(context[key] != authorization[key] for key in (
            "route_identity", "acceptance_record_identity", "acceptance_evidence_sha256",
        ))
        or any(artifact[key] != authorization[key] for key in (
            "item_identity", "query_condition_set_identity", "active_publication_identities",
        ))
        or event.recorded_by != authorization["work_owner"]
        or authorization["approved_fixture_identity"] != artifact.get("fixture_identity")
        or event.payload.get("revision") != authorization["item_revision"] + 1
    ):
        raise evidence_required()


async def verify_current_admission(session: AsyncSession, context: dict) -> None:
    authorization = await load_authorization(session, context["authorization_identity"])
    administrator = await session.get(
        User, UUID(authorization["administrator_identity"].removeprefix("member:")), populate_existing=True,
    )
    if administrator is None or not administrator.is_active or administrator.role != "admin":
        raise evidence_required()
    current = await GenerationRouteService(session).verification_admission(
        authorization["route_identity"], authorization["acceptance_record_identity"], administrator,
    )
    if any(context[key] != value for key, value in current.items()):
        raise evidence_required()
