import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.common.exceptions import AppError
from app.contracts.canonical import CoveragePosition, MaintenanceState, validate_transition
from app.maintenance.schemas import AffectedScope, Classification, Observation, Severity
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel

ItemIdentity = Annotated[str, Field(pattern=r"^maintenance_item:[a-f0-9]{32}$")]
MemberIdentity = Annotated[str, Field(pattern=r"^member:[a-f0-9]{32}$")]
EvidenceLink = Annotated[str, Field(pattern=r"^evidence://maintenance/(fixtures|roadmap|artifacts)/[a-z0-9:._-]+$")]


class Pattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classification: Classification
    coverage_position: CoveragePosition
    observation: Observation


class ItemFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["maintenance_item/v1"] = Field(alias="schema")
    classification: Classification
    severity: Severity
    disposition: Literal[
        "needs-reproduction",
        "coverage-work",
        "roadmap-deferral",
        "boundary-query",
        "retrieval-experiment",
        "provider-work",
        "product-repair",
        "source-change",
        "entry-revision",
        "confirmation",
    ]
    coverage_position: CoveragePosition
    accountable_maintainer: MemberIdentity
    affected_scope: AffectedScope
    work_owner: MemberIdentity
    verified_pattern: Pattern | None
    result_links: list[EvidenceLink]
    blocking_scope: dict | None
    containment_event_id: str | None
    administrator_identity: MemberIdentity | None
    containment_record_identity: str | None = None
    fixture_identity: ItemIdentity | None = None
    fixture_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    replay_identity: ItemIdentity | None = None
    replay_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    diagnosis_fixture_identity: ItemIdentity | None = None
    finding_identities: list[ItemIdentity] = Field(default_factory=list)
    finding_sha256: dict[ItemIdentity, Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]] = Field(default_factory=dict)
    roadmap_identity: ItemIdentity | None = None
    roadmap_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    diagnosed_observation: Observation | None = None


CHANGE_FIELDS = {
    "created": set(),
    "transition": set(),
    "confirmation_closed": {"result_links"},
    "signals_consolidated": {"affected_scope"},
    "reproduced": {"fixture_identity", "fixture_sha256"},
    "fixture_replayed": {"replay_identity", "replay_sha256"},
    "diagnosed": {"classification", "disposition", "diagnosis_fixture_identity", "diagnosed_observation"},
    "finding_approved": {"verified_pattern", "finding_identities", "finding_sha256"},
    "roadmap_qualified": {"roadmap_identity", "roadmap_sha256", "disposition", "result_links"},
    "resolved": {"disposition", "result_links"},
    "administrator_joined": {"administrator_identity"},
}


def invalid_history() -> AppError:
    return AppError(status_code=409, code="MAINTENANCE_EVENT_INVALID", message="maintenance trail invalid")


def project_item(record: CanonicalRecordModel, events: list[CanonicalEventModel]) -> dict:
    """Replay only the closed maintenance decision vocabulary, never arbitrary JSON."""
    try:
        fields = ItemFields.model_validate(record.payload).model_dump(mode="json", by_alias=True, exclude_unset=True)
        if (
            record.state != "open"
            or record.record_class != "authoritative"
            or record.identity_kind != "maintenance_item"
            or record.stable_id != f"maintenance_item:{record.identity_value}"
            or re.fullmatch(r"[a-f0-9]{32}", record.identity_value) is None
            or not events
        ):
            raise ValueError("initial state")
        ordered = sorted(events, key=lambda event: event.payload["revision"])
        state = None
        for revision, event in enumerate(ordered, 1):
            body = event.payload
            changes = body.get("changes", {})
            if (
                set(body) - {"schema", "revision", "changes"}
                or body.get("schema") != "maintenance_event/v1"
                or type(body.get("revision")) is not int
                or body["revision"] != revision
                or event.aggregate_id != record.stable_id
                or event.aggregate_kind != "maintenance_item"
                or event.from_state != state
                or event.event_type not in CHANGE_FIELDS
                or not isinstance(changes, dict)
                or set(changes) != CHANGE_FIELDS[event.event_type]
            ):
                raise ValueError("event shape")
            owner = fields["work_owner"] if event.event_type in {"reproduced", "fixture_replayed"} else fields["accountable_maintainer"]
            if event.event_type == "administrator_joined":
                owner = changes["administrator_identity"]
            if event.recorded_by != owner:
                raise ValueError("event authority")
            if revision == 1:
                if event.event_type != "created" or event.to_state != "open":
                    raise ValueError("creation")
            elif event.event_type == "created":
                raise ValueError("duplicate creation")
            elif event.to_state != state:
                if state is None:
                    raise ValueError("missing prior state")
                validate_transition(MaintenanceState, state, event.to_state)
                expected = {
                    "transition": {"triaged", "in_progress", "closed_confirmation"},
                    "confirmation_closed": {"closed_confirmation"},
                    "roadmap_qualified": {"deferred"},
                    "resolved": {"resolved"},
                }
                if event.to_state not in expected.get(event.event_type, set()):
                    raise ValueError("transition")
                if (
                    event.to_state == "closed_confirmation"
                    and state != "resolved"
                    and not (
                        fields["classification"] == "confirmation"
                        and fields.get("finding_identities")
                        and fields["severity"] not in {"p0", "p1"}
                        and event.event_type == "confirmation_closed"
                        and changes.get("result_links")
                    )
                ):
                    raise ValueError("unverified closure")
            elif event.event_type in {"transition", "confirmation_closed", "roadmap_qualified", "resolved"}:
                raise ValueError("unchanged transition")
            if event.event_type == "finding_approved" and fields["work_owner"] == owner:
                raise ValueError("self approval")
            if event.event_type == "signals_consolidated":
                scope = AffectedScope.model_validate(changes["affected_scope"]).model_dump(mode="json")
                if any(
                    prior not in scope[kind]
                    for kind in ("entry_versions", "gap_contexts")
                    for prior in fields["affected_scope"][kind]
                ):
                    raise ValueError("lost affected scope")
            if event.event_type == "finding_approved":
                prior = fields.get("finding_identities", [])
                identities = changes["finding_identities"]
                digests = changes["finding_sha256"]
                if (
                    not isinstance(identities, list)
                    or not isinstance(digests, dict)
                    or len(identities) != len(prior) + 1
                    or identities[:-1] != prior
                    or len(set(identities)) != len(identities)
                    or set(digests) != set(identities)
                    or any(digests.get(identity) != digest for identity, digest in fields.get("finding_sha256", {}).items())
                ):
                    raise ValueError("finding approval binding")
            fields = ItemFields.model_validate({**fields, **changes}).model_dump(mode="json", by_alias=True, exclude_unset=True)
            if fields["severity"] in {"p0", "p1"} and event.to_state == "deferred":
                raise ValueError("containment deferral")
            if (
                event.event_type == "transition"
                and fields["administrator_identity"] is None
                and fields["classification"] in {"retrieval-answer-behavior", "product-privacy-operations"}
            ):
                raise ValueError("administrator required")
            state = event.to_state
        return {**fields, "state": state, "revision": len(ordered)}
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        raise invalid_history() from exc
