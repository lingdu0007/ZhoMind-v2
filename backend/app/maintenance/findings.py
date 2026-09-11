from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.contracts.canonical import AnswerOutcome
from app.maintenance.fixtures import Hash, evidence_required, load_fixture
from app.maintenance.history import EvidenceLink, ItemIdentity, Pattern
from app.model.canonical import CanonicalRecordModel


class FindingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["validated_finding/v1"] = Field(alias="schema")
    item_identity: ItemIdentity
    pattern: Pattern
    verification_method: Literal["independent_authenticated_replay"]
    fixture_identity: ItemIdentity
    verification_fingerprint: Hash
    expected_outcome: AnswerOutcome
    observed_outcome: AnswerOutcome | None
    result_links: list[EvidenceLink]


def verification_fingerprint(fixture: dict) -> str:
    return canonical_json_sha256(
        {
            key: fixture[key]
            for key in (
                "request_sha256",
                "active_publication_identities",
                "retrieval_profile_identity",
                "observed_state",
                "observed_outcome",
            )
        }
    )


def finding_record_identity(item_identity: str, fixture: dict) -> str:
    target = {
        key: fixture[key] for key in ("affected_scope", "publication_review", "reference_publication_reviews")
    }
    return f"maintenance_item:{canonical_json_sha256([item_identity, verification_fingerprint(fixture), target])[:32]}"


async def load_finding(session: AsyncSession, item: dict, identity: str) -> dict:
    record = await session.get(CanonicalRecordModel, identity)
    if (
        record is None
        or record.identity_kind != "maintenance_item"
        or record.record_class != "immutable"
        or record.state != "verified"
        or identity not in item.get("finding_identities", [])
        or item.get("finding_sha256", {}).get(identity) != canonical_json_sha256(record.payload)
    ):
        raise evidence_required()
    try:
        finding = FindingRecord.model_validate(record.payload).model_dump(mode="json", by_alias=True)
    except ValidationError as exc:
        raise evidence_required() from exc
    if finding["item_identity"] != item["id"]:
        raise evidence_required()
    fixture = await load_fixture(session, item, finding["fixture_identity"])
    fingerprint = verification_fingerprint(fixture)
    if (
        finding["verification_fingerprint"] != fingerprint
        or identity != finding_record_identity(item["id"], fixture)
        or finding["expected_outcome"] != fixture["expected_outcome"]
        or finding["observed_outcome"] != fixture["observed_outcome"]
        or finding["pattern"]["observation"] != fixture["verified_observation"]
        or finding["result_links"] != [f"evidence://maintenance/fixtures/{finding['fixture_identity'].split(':', 1)[1]}"]
    ):
        raise evidence_required()
    return {"id": identity, **finding}
