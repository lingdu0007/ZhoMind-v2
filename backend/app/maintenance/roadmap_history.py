from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.common.canonical_json import canonical_json_sha256
from app.contracts.canonical import CoveragePosition
from app.maintenance.history import ItemIdentity, MemberIdentity, invalid_history
from app.maintenance.schemas import AffectedScope, Classification, Observation, RoadmapOutcome, RoadmapRationale, RoadmapReason
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel


class RoadmapPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classification: Classification
    coverage_position: CoveragePosition
    observation: Observation | None = None


class Qualification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    distinct_executions_30_days: int = Field(ge=0, strict=True)
    independent_findings: int = Field(ge=0, strict=True)
    direct_scope_deferral: bool = Field(strict=True)


class CandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["knowledge_roadmap_candidate/v1"] = Field(alias="schema")
    item_identity: ItemIdentity
    pattern: RoadmapPattern
    affected_scope: AffectedScope
    scope_boundary: Literal["production_rag_agent_engineering"]
    desired_outcome: RoadmapOutcome
    bounded_work_reason: RoadmapReason
    owner_identity: MemberIdentity
    review_date: date
    qualification: Qualification
    maintenance_decision_links: list[ItemIdentity] = Field(min_length=1, max_length=1)
    state: Literal["deferred"]
    revision: int = Field(ge=1, le=1, strict=True)


class ReviewChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: Literal["deferred", "closed", "mapped"]
    owner_identity: MemberIdentity
    rationale: RoadmapRationale
    review_date: date | None
    map_identity: ItemIdentity | None = None
    map_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["roadmap_review/v1"] = Field(alias="schema")
    revision: int = Field(ge=2, strict=True)
    changes: ReviewChanges


def project_candidate(
    record: CanonicalRecordModel,
    events: list[CanonicalEventModel],
    item: dict,
    qualification: CanonicalEventModel,
) -> dict:
    try:
        candidate = CandidateRecord.model_validate(record.payload).model_dump(mode="json", by_alias=True, exclude_unset=True)
        proof = candidate["qualification"]
        expected_pattern = item.get("verified_pattern") or {
            "classification": item["classification"],
            "coverage_position": item["coverage_position"],
        }
        if (
            record.identity_kind != "maintenance_item"
            or record.record_class != "immutable"
            or record.state != "deferred"
            or candidate["item_identity"] != item["id"]
            or candidate["maintenance_decision_links"] != [item["id"]]
            or candidate["pattern"] != expected_pattern
            or candidate["affected_scope"] != item["affected_scope"]
            or item["severity"] in {"p0", "p1"}
            or proof["direct_scope_deferral"] != (item["classification"] == "scope-roadmap")
            or not (proof["direct_scope_deferral"] or proof["distinct_executions_30_days"] >= 3 or proof["independent_findings"] >= 2)
            or qualification.payload["changes"].get("roadmap_sha256") != canonical_json_sha256(record.payload)
        ):
            raise ValueError("qualification")
        for revision, event in enumerate(sorted(events, key=lambda value: value.payload["revision"]), 2):
            body = ReviewEvent.model_validate(event.payload)
            changes = body.changes.model_dump(mode="json", exclude_unset=True)
            if (
                event.aggregate_id != record.stable_id
                or event.aggregate_kind != "maintenance_item"
                or event.event_type != "roadmap_reviewed"
                or body.revision != revision
                or candidate["state"] != "deferred"
                or event.from_state != candidate["state"]
                or event.to_state != changes["state"]
                or event.recorded_by != candidate["owner_identity"]
                or (changes["state"] == "mapped") != ("map_identity" in changes and changes["map_identity"] is not None)
                or (changes["state"] == "mapped") != ("map_sha256" in changes and changes["map_sha256"] is not None)
                or (changes["state"] != "mapped" and ({"map_identity", "map_sha256"} & changes.keys()))
            ):
                raise ValueError("review authority")
            if changes["state"] == "deferred":
                review_date = body.changes.review_date
                if review_date is None or not event.occurred_at.date() < review_date <= event.occurred_at.date() + timedelta(days=31):
                    raise ValueError("monthly review date")
            elif changes["review_date"] is not None:
                raise ValueError("terminal review")
            candidate.update(changes)
            candidate["revision"] = revision
        return {"id": record.stable_id, **candidate}
    except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise invalid_history() from exc
