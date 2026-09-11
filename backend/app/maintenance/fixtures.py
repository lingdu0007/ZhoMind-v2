from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.contracts.canonical import AnswerOutcome
from app.maintenance.containment import ContainmentVerification, qualify_verification
from app.maintenance.history import ItemIdentity
from app.maintenance.schemas import AffectedScope, InsufficientReason, Observation
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel

Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
PublicationIdentity = Annotated[str, Field(pattern=r"^published_knowledge_version:[a-z0-9][a-z0-9._:-]{2,159}$")]
SourceState = Literal["verified_usable", "changed_or_unreachable_awaiting_review", "unavailable_for_new_evidence"]


def evidence_required() -> AppError:
    return AppError(status_code=409, code="MAINTENANCE_EVIDENCE_REQUIRED", message="independent current authenticated evidence required")


class DiagnosisFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: AnswerOutcome | None
    state: Literal["completed", "failed"]
    expected_outcome: AnswerOutcome
    reason: InsufficientReason | None
    frozen_evidence: bool = Field(strict=True)
    reference_supported: bool | None = Field(default=None, strict=True)
    verified_difference: bool | None = Field(default=None, strict=True)
    source_state: SourceState | None = None
    verified_publication: bool | None = Field(default=None, strict=True)
    verified_integrity_review: bool | None = Field(default=None, strict=True)


class SourceReviewFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_identity: str = Field(pattern=r"^source:[a-z0-9][a-z0-9._:-]{2,159}$")
    availability: SourceState
    status_event_id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")
    event_trail_sha256: Hash


class PublicationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_identity: str = Field(pattern=r"^entry:[a-z0-9][a-z0-9._:-]{2,159}$")
    publication_identity: PublicationIdentity
    revision_identity: str = Field(pattern=r"^editorial_revision:[a-z0-9][a-z0-9._:-]{2,159}$")
    integrity_review_event_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{32,36}$")
    source_states: list[SourceState]
    source_facts: list[SourceReviewFact] = Field(min_length=1)


class GenerationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route_identity: str = Field(pattern=r"^provider_route:[a-f0-9]{64}$")
    activation_event_id: str = Field(pattern=r"^[a-f0-9-]{32,36}$")
    acceptance_record_identity: str = Field(pattern=r"^delivery_acceptance_record:[A-Za-z0-9._/-]+$")
    acceptance_evidence_sha256: Hash


class VerificationGenerationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    route_identity: str = Field(pattern=r"^provider_route:[a-f0-9]{64}$")
    authorization_identity: ItemIdentity
    authorization_sha256: Hash
    acceptance_record_identity: str = Field(pattern=r"^delivery_acceptance_record:[A-Za-z0-9._/-]+$")
    acceptance_evidence_sha256: Hash


class FixtureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["maintenance_fixture/v1"] = Field(alias="schema")
    item_identity: ItemIdentity
    affected_scope: AffectedScope
    request_sha256: Hash
    query_condition_set_identity: Hash
    expected_outcome: AnswerOutcome
    observed_outcome: AnswerOutcome | None
    observed_state: Literal["completed", "failed"]
    verified_observation: Observation
    diagnosis_facts: DiagnosisFacts
    publication_review: PublicationReview | None
    reference_evidence_publication_identities: list[PublicationIdentity]
    reference_publication_reviews: list[PublicationReview]
    active_publication_identities: list[PublicationIdentity]
    evidence_publication_identities: list[PublicationIdentity]
    retrieval_profile_identity: Literal["retrieval-answer-policy/pilot-v1"]
    observed_reason: InsufficientReason | None
    verification_method: Literal["independent_authenticated_replay"]
    generation_context: GenerationContext | VerificationGenerationContext | None
    containment_verification: ContainmentVerification | None = None


def qualify_fixture_scope(item: dict, fixture: dict) -> None:
    try:
        scope = AffectedScope.model_validate(fixture["affected_scope"]).model_dump(mode="json")
        retained = AffectedScope.model_validate(item["affected_scope"]).model_dump(mode="json")
        if not any(scope.values()) or any(
            target not in retained[kind] for kind in scope for target in scope[kind]
        ):
            raise ValueError("fixture scope")
        entries = {target["entry_identity"] for target in scope["entry_versions"]}
        review = fixture.get("publication_review")
        if review and entries and review["entry_identity"] not in entries:
            raise ValueError("publication target")
        references = fixture.get("reference_publication_reviews", [])
        if references and not entries.issubset({reference["entry_identity"] for reference in references}):
            raise ValueError("reference target")
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        raise evidence_required() from exc


async def load_fixture(session: AsyncSession, item: dict, identity: str) -> dict:
    record = await session.get(CanonicalRecordModel, identity)
    if record is None or record.record_class != "immutable" or record.state != "reproduced" or record.identity_kind != "maintenance_item":
        raise evidence_required()
    try:
        fixture = FixtureRecord.model_validate(record.payload).model_dump(mode="json", by_alias=True, exclude_unset=True)
        facts = fixture["diagnosis_facts"]
        if (
            fixture["item_identity"] != item["id"]
            or facts["outcome"] != fixture["observed_outcome"]
            or facts["state"] != fixture["observed_state"]
            or facts["expected_outcome"] != fixture["expected_outcome"]
            or facts["reason"] != fixture["observed_reason"]
            or (fixture["observed_state"] == "completed") != (fixture["observed_outcome"] is not None)
            or not set(fixture["evidence_publication_identities"]).issubset(fixture["active_publication_identities"])
            or not set(fixture["reference_evidence_publication_identities"]).issubset(fixture["active_publication_identities"])
            or sorted(fixture["reference_evidence_publication_identities"])
            != sorted(review["publication_identity"] for review in fixture["reference_publication_reviews"])
        ):
            raise ValueError("fixture binding")
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        raise evidence_required() from exc
    event = await session.scalar(
        select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == item["id"],
            CanonicalEventModel.event_type == "reproduced",
            CanonicalEventModel.recorded_by == item["work_owner"],
            CanonicalEventModel.payload["changes"]["fixture_identity"].as_string() == identity,
        )
    )
    if event is None or event.payload.get("changes", {}).get("fixture_sha256") != canonical_json_sha256(record.payload):
        raise evidence_required()
    qualify_fixture_scope(item, fixture)
    qualify_verification(item, fixture.get("containment_verification"), event)
    from app.maintenance.provider_verification import qualify_execution_authorization

    await qualify_execution_authorization(session, fixture, event)
    return fixture
