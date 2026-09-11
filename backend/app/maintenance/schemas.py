from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.contracts.canonical import AnswerOutcome, CoveragePosition, MaintenanceState

Classification = Literal[
    "confirmation",
    "content-integrity",
    "source-freshness",
    "coverage-gap",
    "retrieval-answer-behavior",
    "product-privacy-operations",
    "scope-roadmap",
]
Severity = Literal["p0", "p1", "p2", "p3"]
Observation = Literal[
    "coverage_gap",
    "retrieval_miss",
    "condition_loss",
    "citation_drift",
    "provider_failure",
    "product_failure",
    "stale_source",
    "wrong_content",
    "confirmation",
]
RoadmapOutcome = Literal["expand_coverage", "improve_sources", "evaluate_retrieval", "repair_product_workflow", "clarify_scope"]
RoadmapReason = Literal["requires_separate_scope", "cross_domain_work", "requires_new_capability"]
RoadmapRationale = Literal["still_outside_scope", "separate_discovery_needed", "no_longer_needed", "resolved_elsewhere"]
InsufficientReason = Literal[
    "no_eligible_published_evidence", "decision_not_covered", "decisive_condition_missing",
    "material_evidence_conflict", "assurance_support_missing", "evidence_budget_exceeded", "knowledge_needs_review",
]


class AffectedEntryVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_identity: str = Field(pattern=r"^entry:[a-z0-9][a-z0-9._:-]{2,159}$")
    publication_identity: str = Field(pattern=r"^published_knowledge_version:[a-z0-9][a-z0-9._:-]{2,159}$")
    revision_identity: str = Field(pattern=r"^editorial_revision:[a-z0-9][a-z0-9._:-]{2,159}$")


class AffectedGap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: InsufficientReason
    query_condition_set_identity: str = Field(pattern=r"^[a-f0-9]{64}$")


class AffectedScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entry_versions: list[AffectedEntryVersion]
    gap_contexts: list[AffectedGap]


class MaintenanceAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=64)


class MaintenanceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classification: Classification
    severity: Severity
    disposition: Literal["needs-reproduction"]
    coverage_position: CoveragePosition
    work_owner_username: str = Field(min_length=1, max_length=64)
    signal_ids: list[str] = Field(min_length=1, max_length=50)
    containment_record_identity: str | None = Field(
        default=None,
        pattern=r"^delivery_acceptance_record:[a-z0-9][a-z0-9._:-]{2,159}$",
    )


class MaintenanceTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    state: MaintenanceState


class AdministratorJoin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)


class MaintenanceConsolidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    signal_ids: list[str] = Field(min_length=1, max_length=50)


class ReproductionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    signal_id: str = Field(min_length=1, max_length=64)
    answer_id: str = Field(min_length=1, max_length=64)
    expected_outcome: AnswerOutcome
    confirmed_synthetic_fixture: Literal[True]
    verified_observation: Observation = "coverage_gap"
    reference_answer_id: str | None = Field(default=None, min_length=1, max_length=64)
    entry_identity: str | None = Field(default=None, pattern=r"^entry:[a-z0-9][a-z0-9._:-]{2,159}$")
    provider_verification_authorization_identity: str | None = Field(default=None, pattern=r"^maintenance_item:[a-f0-9]{32}$")

    @field_validator("confirmed_synthetic_fixture", mode="before")
    @classmethod
    def require_explicit_confirmation(cls, value: object) -> Literal[True]:
        if value is not True:
            raise ValueError("explicit confirmation is required")
        return True


class DiagnosisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    fixture_identity: str = Field(pattern=r"^maintenance_item:[a-f0-9]{32}$")
    observation: Observation


class FindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    fixture_identity: str = Field(pattern=r"^maintenance_item:[a-f0-9]{32}$")


class ReplayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    fixture_identity: str = Field(pattern=r"^maintenance_item:[a-f0-9]{32}$")
    answer_id: str = Field(min_length=1, max_length=64)
    provider_verification_authorization_identity: str | None = Field(default=None, pattern=r"^maintenance_item:[a-f0-9]{32}$")


class ProviderVerificationAuthorizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    route_identity: str = Field(pattern=r"^provider_route:[a-f0-9]{64}$")
    admission_acceptance_identity: str = Field(pattern=r"^delivery_acceptance_record:[A-Za-z0-9._/-]+$", max_length=190)
    query_condition_set_identity: str = Field(pattern=r"^[a-f0-9]{64}$")


class RoadmapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    owner_username: str = Field(min_length=1, max_length=64)
    desired_outcome: RoadmapOutcome
    bounded_work_reason: RoadmapReason
    review_date: date


class RoadmapReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    action: Literal["renew", "close", "start_wayfinder"]
    rationale: RoadmapRationale
    owner_username: str = Field(min_length=1, max_length=64)
    review_date: date | None = None


class CadenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period: Literal["weekly", "monthly", "quarterly"]
    item_revisions: dict[str, Annotated[int, Field(ge=1, strict=True)]] = Field(max_length=1000)
    context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sample_acceptance_identities: list[str] = Field(default_factory=list, max_length=50)
    evidence_links: list[
        Literal[
            "evidence://maintenance/weekly-review",
            "evidence://maintenance/monthly-review",
            "evidence://maintenance/quarterly-review",
        ]
    ] = Field(min_length=1, max_length=1)


class ResolutionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=1, strict=True)
    disposition: Literal["boundary-query", "source-change", "provider-work", "retrieval-experiment", "product-repair", "entry-revision"]
    artifact_identities: list[str] = Field(min_length=1, max_length=20)
