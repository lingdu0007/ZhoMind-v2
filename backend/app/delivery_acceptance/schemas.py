from __future__ import annotations

import re
from enum import Enum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.contracts.canonical import (
    AcceptanceStatus,
    DeliveryAcceptanceStage,
    StableIdentity,
    StableIdentityKind,
)

_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,159}$")
_CHECK_ID = re.compile(r"^check:[a-z0-9][a-z0-9._:-]{2,159}$")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|secret|token)\s*[:=]\s*\S+|\bbearer\s+\S+"
)
_JWT = re.compile(r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_URL_ENCODED_SECRET = re.compile(r"(?i)\b(?:token|secret|password|api(?:_|%5f)?key)%?(?:3d|3a|=|:)")
_IMPACT_DEPENDENCY_CHECKS = {
    "product_path_identities": "check:product-path-impact",
    "configuration_identities": "check:configuration-impact",
}
_PRODUCT_DEPENDENCY_CHECKS = {
    "migration": "check:product-revision",
    "product_revision": "check:product-revision",
    "retrieval_profile": "check:retrieval-configuration",
    "embedding_profile": "check:retrieval-configuration",
    "provider_route": "check:answer-contract",
    "prompt_envelope": "check:answer-contract",
    "deployment": "check:deployment-boundary",
    "user_boundary": "check:deployment-boundary",
    "data_boundary": "check:deployment-boundary",
    "host": "check:workload-profile",
    "corpus": "check:workload-profile",
    "concurrency": "check:workload-profile",
}


class AcceptanceBlockingScope(str, Enum):
    ENTRY_VERSION = "entry_version"
    COLLECTION = "collection"
    DEPLOYMENT = "deployment"
    PUBLIC_CLAIM = "public_claim"


class AcceptanceCheckResult(str, Enum):
    REQUIRED = "required"
    PASSED = "passed"
    FAILED = "failed"
    CARRIED_FORWARD = "carried_forward"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


class AcceptanceFailureKind(str, Enum):
    ENTRY_SPECIFIC = "entry_specific"
    COLLECTION_RETRIEVAL = "collection_retrieval"
    COLLECTION_CHUNKING = "collection_chunking"
    COLLECTION_IMPORT = "collection_import"
    COLLECTION_BUNDLE = "collection_bundle"
    COLLECTION_CROSS_ENTRY = "collection_cross_entry"
    SHARED_AUTHORIZATION = "shared_authorization"
    SHARED_WITHDRAWAL = "shared_withdrawal"
    SHARED_EVIDENCE_IDENTITY = "shared_evidence_identity"
    SHARED_PRIVACY = "shared_privacy"
    SHARED_PUBLICATION = "shared_publication"
    SHARED_POLICY_ISOLATION = "shared_policy_isolation"
    SHARED_CLOSED_OUTCOME = "shared_closed_outcome"
    PUBLIC_CLAIM = "public_claim"
    PERFORMANCE = "performance"


class AcceptanceChangeClassification(str, Enum):
    ORDINARY_CONTENT = "ordinary_content"
    PROTECTED_PRODUCT_PATH = "protected_product_path"


class AcceptanceStatusReason(str, Enum):
    RECORD_CREATED_PENDING_VERIFICATION = "record_created_pending_verification"
    RECORD_CREATED_KNOWN_FAILURE = "record_created_known_failure"
    CHECKS_VERIFIED = "checks_verified"
    REACCEPTANCE_DUE = "reacceptance_due"
    PERFORMANCE_OBJECTIVE_MISSED = "performance_objective_missed"
    INTEGRITY_FAILURE = "integrity_failure"
    SUPERSEDED_BY_RECORD = "superseded_by_record"


def _exact_identity(value: str) -> str:
    normalized = value.strip()
    if _IDENTITY.fullmatch(normalized) is None:
        raise ValueError("identity must be a lowercase bounded exact identifier")
    try:
        stable_identity = StableIdentity.from_stable_id(normalized)
    except ValueError as exc:
        raise ValueError("identity must use an accepted canonical identity namespace") from exc
    identity_value = stable_identity.value
    if identity_value in {
        "latest",
        "main",
        "master",
        "head",
        "trunk",
        "current",
        "production",
        "staging",
        "development",
        "preview",
        "pilot",
        "daily",
    }:
        raise ValueError("identity values must not use mutable pointers")
    return normalized


def _exact_identity_of(value: str, allowed_kinds: set[str]) -> str:
    normalized = _exact_identity(value)
    identity_kind, _, _ = normalized.partition(":")
    if identity_kind not in allowed_kinds:
        expected = ", ".join(sorted(allowed_kinds))
        raise ValueError(f"identity must use one of the expected namespaces: {expected}")
    return normalized


def _identity_kind(value: str) -> str:
    return value.partition(":")[0]


def _check_identity(value: str) -> str:
    normalized = value.strip()
    if _CHECK_ID.fullmatch(normalized) is None:
        raise ValueError("check_id must use the check: identity namespace")
    return normalized


def _non_blank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be blank")
    if len(normalized) > 500:
        raise ValueError("value exceeds the maximum retained length")
    if _CONTROL_CHARACTER.search(normalized):
        raise ValueError("value contains unsupported control characters")
    if (
        _SECRET_ASSIGNMENT.search(normalized)
        or _JWT.search(normalized)
        or _AWS_ACCESS_KEY.search(normalized)
        or _URL_ENCODED_SECRET.search(normalized)
    ):
        raise ValueError("value must not contain credentials or secrets")
    return normalized


def _evidence_link(value: str) -> str:
    normalized = _non_blank(value)
    parsed = urlsplit(normalized)
    if (
        parsed.scheme != "evidence"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("evidence links must be credential-free evidence:// references")
    return normalized


def _evidence_links(values: list[str]) -> list[str]:
    return [_evidence_link(value) for value in values]


def _allowed_failure_scopes(failure_kind: AcceptanceFailureKind) -> set[AcceptanceBlockingScope]:
    if failure_kind is AcceptanceFailureKind.ENTRY_SPECIFIC:
        return {AcceptanceBlockingScope.ENTRY_VERSION}
    if failure_kind in {
        AcceptanceFailureKind.COLLECTION_RETRIEVAL,
        AcceptanceFailureKind.COLLECTION_CHUNKING,
        AcceptanceFailureKind.COLLECTION_IMPORT,
        AcceptanceFailureKind.COLLECTION_BUNDLE,
        AcceptanceFailureKind.COLLECTION_CROSS_ENTRY,
    }:
        return {AcceptanceBlockingScope.COLLECTION}
    if failure_kind is AcceptanceFailureKind.PERFORMANCE:
        return {AcceptanceBlockingScope.DEPLOYMENT}
    if failure_kind is AcceptanceFailureKind.PUBLIC_CLAIM:
        return {AcceptanceBlockingScope.PUBLIC_CLAIM}
    return {AcceptanceBlockingScope.COLLECTION, AcceptanceBlockingScope.DEPLOYMENT}


class AffectedScopeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_identities: list[str] = Field(default_factory=list, max_length=50)
    collection_identities: list[str] = Field(default_factory=list, max_length=50)
    product_path_identities: list[str] = Field(default_factory=list, max_length=50)
    configuration_identities: list[str] = Field(default_factory=list, max_length=50)
    protected_capability_identities: list[str] = Field(default_factory=list, max_length=50)
    public_claim_identities: list[str] = Field(default_factory=list, max_length=50)
    deployment_identity: str | None = None
    expected_blocking_scope: AcceptanceBlockingScope
    blocking_scope_identity: str

    @field_validator("entry_identities")
    @classmethod
    def validate_entry_identities(cls, values: list[str]) -> list[str]:
        normalized = [_exact_identity_of(value, {"entry", "published_knowledge_version"}) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity lists must not contain duplicates")
        return normalized

    @field_validator("collection_identities")
    @classmethod
    def validate_collection_identities(cls, values: list[str]) -> list[str]:
        return cls._validate_identity_list(values, {"collection"})

    @field_validator("product_path_identities")
    @classmethod
    def validate_product_path_identities(cls, values: list[str]) -> list[str]:
        return cls._validate_identity_list(values, {"product_path"})

    @field_validator("configuration_identities")
    @classmethod
    def validate_configuration_identities(cls, values: list[str]) -> list[str]:
        return cls._validate_identity_list(values, {"configuration"})

    @field_validator("protected_capability_identities")
    @classmethod
    def validate_capability_identities(cls, values: list[str]) -> list[str]:
        return cls._validate_identity_list(values, {"capability"})

    @field_validator("public_claim_identities")
    @classmethod
    def validate_public_claim_identities(cls, values: list[str]) -> list[str]:
        return cls._validate_identity_list(values, {"bundle", "evidence_set", "public_claim"})

    @field_validator("deployment_identity")
    @classmethod
    def validate_deployment_identity(cls, value: str | None) -> str | None:
        return _exact_identity_of(value, {"deployment"}) if value is not None else None

    @field_validator("blocking_scope_identity")
    @classmethod
    def validate_blocking_scope_identity(cls, value: str) -> str:
        return _exact_identity(value)

    @staticmethod
    def _validate_identity_list(values: list[str], allowed_kinds: set[str]) -> list[str]:
        normalized = [_exact_identity_of(value, allowed_kinds) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity lists must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_declared_scope(self) -> AffectedScopeInput:
        identities = {
            *self.entry_identities,
            *self.collection_identities,
            *self.product_path_identities,
            *self.configuration_identities,
            *self.protected_capability_identities,
            *self.public_claim_identities,
        }
        if self.deployment_identity is not None:
            identities.add(self.deployment_identity)
        if not identities:
            raise ValueError("affected scope must name at least one exact identity")
        if not self.contains(self.expected_blocking_scope, self.blocking_scope_identity):
            raise ValueError("blocking_scope_identity must identify the declared scope")
        return self

    def contains(self, scope: AcceptanceBlockingScope, identity: str) -> bool:
        if scope is AcceptanceBlockingScope.ENTRY_VERSION:
            return identity in self.entry_identities
        if scope is AcceptanceBlockingScope.COLLECTION:
            return identity in self.collection_identities
        if scope is AcceptanceBlockingScope.PUBLIC_CLAIM:
            return identity in self.public_claim_identities
        return identity == self.deployment_identity


class FailureBlockingScopeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: AcceptanceBlockingScope
    identity: str

    @field_validator("identity")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _exact_identity(value)

    @model_validator(mode="after")
    def validate_scope_identity_kind(self) -> FailureBlockingScopeInput:
        allowed_kinds = {
            AcceptanceBlockingScope.ENTRY_VERSION: {"entry", "published_knowledge_version"},
            AcceptanceBlockingScope.COLLECTION: {"collection"},
            AcceptanceBlockingScope.DEPLOYMENT: {"deployment"},
            AcceptanceBlockingScope.PUBLIC_CLAIM: {"bundle", "evidence_set", "public_claim"},
        }[self.scope]
        if _identity_kind(self.identity) not in allowed_kinds:
            raise ValueError("blocking scope identity must match its declared scope")
        return self


class AcceptanceCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    result: AcceptanceCheckResult
    evidence_links: list[str] = Field(default_factory=list, max_length=20)
    applicability_conditions: dict[str, str] = Field(default_factory=dict, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    identity_dependencies: list[str] = Field(default_factory=list, max_length=30)
    reason: str | None = None
    failure_kind: AcceptanceFailureKind | None = None
    blocking_scope: FailureBlockingScopeInput | None = None
    performance_objective_identity: str | None = None
    carried_forward_from: str | None = None

    @field_validator("check_id")
    @classmethod
    def validate_check_id(cls, value: str) -> str:
        return _check_identity(value)

    @field_validator("evidence_links")
    @classmethod
    def validate_evidence_links(cls, values: list[str]) -> list[str]:
        return _evidence_links(values)

    @field_validator("applicability_conditions")
    @classmethod
    def validate_applicability_conditions(cls, values: dict[str, str]) -> dict[str, str]:
        return {_non_blank(key): _non_blank(value) for key, value in values.items()}

    @field_validator("assumptions")
    @classmethod
    def validate_check_assumptions(cls, values: list[str]) -> list[str]:
        normalized = [_non_blank(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("assumptions must not contain duplicates")
        return normalized

    @field_validator("identity_dependencies")
    @classmethod
    def validate_identity_dependencies(cls, values: list[str]) -> list[str]:
        normalized = [_exact_identity(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity dependencies must not contain duplicates")
        return normalized

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        return _non_blank(value) if value is not None else None

    @field_validator("carried_forward_from")
    @classmethod
    def validate_carried_forward_from(cls, value: str | None) -> str | None:
        return _exact_identity_of(value, {"delivery_acceptance_record"}) if value is not None else None

    @field_validator("performance_objective_identity")
    @classmethod
    def validate_performance_objective_identity(cls, value: str | None) -> str | None:
        return _exact_identity_of(value, {"objective"}) if value is not None else None

    @model_validator(mode="after")
    def validate_result_details(self) -> AcceptanceCheckInput:
        requires_reason = self.result in {
            AcceptanceCheckResult.FAILED,
            AcceptanceCheckResult.SKIPPED,
            AcceptanceCheckResult.NOT_APPLICABLE,
        }
        if requires_reason and self.reason is None:
            raise ValueError("failed, skipped, and not applicable checks require a reason")
        if self.result is AcceptanceCheckResult.FAILED:
            if self.failure_kind is None or self.blocking_scope is None or not self.evidence_links:
                raise ValueError("failed checks require failure_kind, blocking_scope, and evidence_links")
        elif self.failure_kind is not None or self.blocking_scope is not None:
            raise ValueError("failure details are only valid for failed checks")
        if self.result in {AcceptanceCheckResult.PASSED, AcceptanceCheckResult.CARRIED_FORWARD} and not self.evidence_links:
            raise ValueError("passed and carried_forward checks require evidence_links")
        if self.failure_kind is AcceptanceFailureKind.PERFORMANCE and self.performance_objective_identity is None:
            raise ValueError("performance failures require an affected objective identity")
        if self.failure_kind is not AcceptanceFailureKind.PERFORMANCE and self.performance_objective_identity is not None:
            raise ValueError("performance objective identity is only valid for performance failures")
        if self.result is AcceptanceCheckResult.CARRIED_FORWARD and self.carried_forward_from is None:
            raise ValueError("carried_forward checks require a source record")
        if self.result is AcceptanceCheckResult.CARRIED_FORWARD and (
            not self.applicability_conditions or not self.assumptions or not self.identity_dependencies
        ):
            raise ValueError("carried_forward checks must declare identities, conditions, and assumptions")
        if self.result is not AcceptanceCheckResult.CARRIED_FORWARD and self.carried_forward_from is not None:
            raise ValueError("source records are only valid for carried_forward checks")
        if self.failure_kind is not None and self.blocking_scope is not None:
            if self.blocking_scope.scope not in _allowed_failure_scopes(self.failure_kind):
                raise ValueError("failure kind must use its minimum sufficient blocking scope")
        return self


class CandidatePublicationBindingInput(BaseModel):
    """Exact immutable identities required to accept a Candidate publication."""

    model_config = ConfigDict(extra="forbid")

    candidate_identity: str
    inspection_record_identity: str
    acceptance_record_identity: str
    published_knowledge_version_identity: str
    entry_identity: str
    configuration_identity: str
    bundle_sha256: str
    frozen_input_sha256: str

    @field_validator("candidate_identity")
    @classmethod
    def validate_candidate_identity(cls, value: str) -> str:
        return _exact_identity_of(value, {"candidate"})

    @field_validator("inspection_record_identity", "acceptance_record_identity")
    @classmethod
    def validate_event_identity(cls, value: str) -> str:
        return _exact_identity_of(value, {"event"})

    @field_validator("published_knowledge_version_identity")
    @classmethod
    def validate_published_version_identity(cls, value: str) -> str:
        return _exact_identity_of(value, {"published_knowledge_version"})

    @field_validator("entry_identity")
    @classmethod
    def validate_entry_identity(cls, value: str) -> str:
        return _exact_identity_of(value, {"entry"})

    @field_validator("configuration_identity")
    @classmethod
    def validate_configuration_identity(cls, value: str) -> str:
        return _exact_identity_of(value, {"configuration"})

    @field_validator("bundle_sha256", "frozen_input_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("publication hashes must be lowercase SHA-256 values")
        return normalized

    @property
    def exact_identities(self) -> set[str]:
        return {
            self.candidate_identity,
            self.inspection_record_identity,
            self.acceptance_record_identity,
            self.published_knowledge_version_identity,
            self.entry_identity,
            self.configuration_identity,
        }

    @property
    def persisted_record_identities(self) -> set[str]:
        return self.exact_identities - {self.entry_identity}


class CreateDeliveryAcceptanceRecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: DeliveryAcceptanceStage
    change_classification: AcceptanceChangeClassification = AcceptanceChangeClassification.ORDINARY_CONTENT
    predecessor_record_identity: str | None = None
    baseline_record_identity: str | None = None
    is_pilot_entry_baseline: bool = False
    replaces_record_identity: str | None = None
    affected_scope: AffectedScopeInput
    content_identities: list[str] = Field(min_length=1, max_length=50)
    product_identities: list[str] = Field(min_length=1, max_length=50)
    conditions: dict[str, str] = Field(default_factory=dict, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    checks: list[AcceptanceCheckInput] = Field(min_length=1, max_length=50)
    candidate_publication_binding: CandidatePublicationBindingInput | None = None
    known_limits: list[str] = Field(default_factory=list, max_length=20)
    risks: list[str] = Field(default_factory=list, max_length=20)
    evidence_links: list[str] = Field(min_length=1, max_length=20)
    reacceptance_triggers: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("predecessor_record_identity")
    @classmethod
    def validate_predecessor_identity(cls, value: str | None) -> str | None:
        return (
            _exact_identity_of(value, {StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value})
            if value is not None
            else None
        )

    @field_validator("baseline_record_identity")
    @classmethod
    def validate_baseline_identity(cls, value: str | None) -> str | None:
        return (
            _exact_identity_of(value, {StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value})
            if value is not None
            else None
        )

    @field_validator("replaces_record_identity")
    @classmethod
    def validate_replacement_identity(cls, value: str | None) -> str | None:
        return (
            _exact_identity_of(value, {StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD.value})
            if value is not None
            else None
        )

    @field_validator("content_identities")
    @classmethod
    def validate_content_identities(cls, values: list[str]) -> list[str]:
        normalized = [
            _exact_identity_of(
                value,
                {
                    "bundle",
                    "bundle_item",
                    "build_generation",
                    "candidate",
                    "editorial_revision",
                    "entry",
                    "event",
                    "evidence_set",
                    "evidence_snapshot",
                    "published_knowledge_version",
                    "public_claim",
                    "source",
                },
            )
            for value in values
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity lists must not contain duplicates")
        return normalized

    @field_validator("product_identities")
    @classmethod
    def validate_product_identities(cls, values: list[str]) -> list[str]:
        normalized = [
            _exact_identity_of(
                value,
                {
                    "capability",
                    "configuration",
                    "corpus",
                    "deployment",
                    "embedding_profile",
                    "host",
                    "migration",
                    "product_path",
                    "product_revision",
                    "prompt_envelope",
                    "provider_route",
                    "retrieval_profile",
                    "user_boundary",
                    "data_boundary",
                    "concurrency",
                },
            )
            for value in values
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity lists must not contain duplicates")
        return normalized

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, values: dict[str, str]) -> dict[str, str]:
        return {_non_blank(key): _non_blank(value) for key, value in values.items()}

    @field_validator("assumptions", "known_limits", "risks", "evidence_links", "reacceptance_triggers")
    @classmethod
    def validate_text_list(cls, values: list[str]) -> list[str]:
        normalized = [_non_blank(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("text lists must not contain duplicates")
        return normalized

    @field_validator("evidence_links")
    @classmethod
    def validate_record_evidence_links(cls, values: list[str]) -> list[str]:
        normalized = _evidence_links(values)
        if len(set(normalized)) != len(normalized):
            raise ValueError("evidence links must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_checks_and_predecessor(self) -> CreateDeliveryAcceptanceRecordRequest:
        if self.stage is DeliveryAcceptanceStage.LOCAL_DEVELOPMENT:
            if self.predecessor_record_identity is not None:
                raise ValueError("Local Development records cannot have a predecessor")
        elif self.predecessor_record_identity is None:
            raise ValueError("later Delivery Acceptance stages require a predecessor record")
        if self.stage is DeliveryAcceptanceStage.DAILY_USE_RELEASE and self.baseline_record_identity is None:
            raise ValueError("Daily-Use records require an active Pilot baseline record")
        if self.is_pilot_entry_baseline and self.stage is not DeliveryAcceptanceStage.LIMITED_TEAM_PILOT:
            raise ValueError("only Limited-Team Pilot records can be Pilot Entry Baselines")
        if self.is_pilot_entry_baseline:
            baseline_identity_kinds = {_identity_kind(identity) for identity in self.product_identities}
            required_baseline_kinds = {"user_boundary", "data_boundary", "host", "corpus", "concurrency"}
            missing_baseline_kinds = required_baseline_kinds - baseline_identity_kinds
            if missing_baseline_kinds:
                raise ValueError(
                    "Pilot Entry Baselines require exact boundary and capacity identities: "
                    f"{', '.join(sorted(missing_baseline_kinds))}"
                )
        if self.stage in {
            DeliveryAcceptanceStage.LOCAL_DEVELOPMENT,
            DeliveryAcceptanceStage.EDITORIAL_PREVIEW,
        } and self.baseline_record_identity is not None:
            raise ValueError("only Pilot, Daily-Use, and Public Evidence records may bind a Pilot baseline")
        if self.stage is DeliveryAcceptanceStage.PUBLIC_EVIDENCE_RELEASE and not self.affected_scope.public_claim_identities:
            raise ValueError("Public Evidence records require an exact public claim or evidence bundle identity")

        check_ids = [check.check_id for check in self.checks]
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("checks must have unique check_id values")
        missing_checks = _required_check_ids(self) - set(check_ids)
        if missing_checks:
            raise ValueError(f"record does not select all applicable checks: {', '.join(sorted(missing_checks))}")
        checks_by_id = {check.check_id: check for check in self.checks}
        bypassed_checks = [
            check_id
            for check_id in _required_check_ids(self)
            if checks_by_id[check_id].result in {AcceptanceCheckResult.SKIPPED, AcceptanceCheckResult.NOT_APPLICABLE}
        ]
        if bypassed_checks:
            raise ValueError(f"mandatory checks cannot be skipped or marked not applicable: {', '.join(sorted(bypassed_checks))}")
        declared_identities = {
            *self.affected_scope.entry_identities,
            *self.affected_scope.collection_identities,
            *self.affected_scope.product_path_identities,
            *self.affected_scope.configuration_identities,
            *self.affected_scope.protected_capability_identities,
            *self.affected_scope.public_claim_identities,
            *self.content_identities,
            *self.product_identities,
        }
        if self.affected_scope.deployment_identity is not None:
            declared_identities.add(self.affected_scope.deployment_identity)
        has_candidate_publication_identities = (
            any(identity.startswith("candidate:") for identity in declared_identities)
            and any(identity.startswith("published_knowledge_version:") for identity in declared_identities)
        )
        if has_candidate_publication_identities and self.candidate_publication_binding is None:
            raise ValueError("Candidate publication evidence requires an exact Candidate publication binding")
        if self.candidate_publication_binding is not None:
            declared_identities.update(self.candidate_publication_binding.exact_identities)
            required_content_identities = {
                self.candidate_publication_binding.candidate_identity,
                self.candidate_publication_binding.inspection_record_identity,
                self.candidate_publication_binding.acceptance_record_identity,
                self.candidate_publication_binding.published_knowledge_version_identity,
                self.candidate_publication_binding.entry_identity,
            }
            if not required_content_identities.issubset(self.content_identities):
                raise ValueError("Candidate publication binding identities must be declared as content identities")
            if self.candidate_publication_binding.configuration_identity not in self.product_identities:
                raise ValueError("Candidate publication configuration must be declared as a product identity")
            if {
                self.candidate_publication_binding.entry_identity,
                self.candidate_publication_binding.published_knowledge_version_identity,
            } - set(self.affected_scope.entry_identities):
                raise ValueError("Candidate publication must bind the exact entry and published Knowledge Version scope")
            if self.candidate_publication_binding.configuration_identity not in self.affected_scope.configuration_identities:
                raise ValueError("Candidate publication must bind the exact configuration scope")
        if self.change_classification is AcceptanceChangeClassification.PROTECTED_PRODUCT_PATH:
            if not self.affected_scope.protected_capability_identities:
                raise ValueError("protected changes must bind affected protected capabilities")
        if self.affected_scope.deployment_identity is None:
            raise ValueError("Delivery Acceptance records require an exact deployment identity")
        entry_check_ids = {
            "check:entry-supported-query",
            "check:entry-boundary-query",
            "check:evidence-citation-identity",
        }
        for entry_identity in self.affected_scope.entry_identities:
            missing_entry_evidence = [
                check_id
                for check_id in entry_check_ids
                if entry_identity not in checks_by_id[check_id].identity_dependencies
            ]
            if missing_entry_evidence:
                raise ValueError(
                    f"entry checks must bind {entry_identity}: {', '.join(sorted(missing_entry_evidence))}"
                )
        if self.candidate_publication_binding is not None:
            for check_id in entry_check_ids:
                if not self.candidate_publication_binding.exact_identities.issubset(
                    checks_by_id[check_id].identity_dependencies
                ):
                    raise ValueError(
                        "Candidate publication entry checks must bind the exact Candidate, inspection, "
                        "acceptance, Published Knowledge Version, entry, and configuration"
                    )
        for check in self.checks:
            if any(self.conditions.get(key) != value for key, value in check.applicability_conditions.items()):
                raise ValueError("check applicability conditions must be bound by the record conditions")
            if not set(check.assumptions).issubset(self.assumptions):
                raise ValueError("check assumptions must be bound by the record assumptions")
            if not set(check.identity_dependencies).issubset(declared_identities):
                raise ValueError("check identity dependencies must be bound by the affected record")
            if check.blocking_scope is not None and not self.affected_scope.contains(
                check.blocking_scope.scope,
                check.blocking_scope.identity,
            ):
                raise ValueError("failed check blocking scope must be part of the affected scope")
            if check.failure_kind is AcceptanceFailureKind.ENTRY_SPECIFIC:
                if check.blocking_scope is None or check.blocking_scope.scope is not AcceptanceBlockingScope.ENTRY_VERSION:
                    raise ValueError("entry-specific failures may block only an entry or version")
            if check.failure_kind is AcceptanceFailureKind.PERFORMANCE:
                if check.blocking_scope is None or check.blocking_scope.scope is not AcceptanceBlockingScope.DEPLOYMENT:
                    raise ValueError("performance failures block only the affected deployment stage")
        return self


def _required_check_ids(payload: CreateDeliveryAcceptanceRecordRequest) -> set[str]:
    checks = {
        "check:impact-declaration",
        "check:bundle-secret-scan",
    }
    if payload.affected_scope.entry_identities:
        checks.update(
            {
                "check:entry-supported-query",
                "check:entry-boundary-query",
                "check:evidence-citation-identity",
            }
        )
    if payload.affected_scope.collection_identities:
        checks.add("check:collection-retrieval")
    checks.update(
        check_id
        for field, check_id in _IMPACT_DEPENDENCY_CHECKS.items()
        if getattr(payload.affected_scope, field)
    )
    checks.update(
        check_id
        for identity in payload.product_identities
        if (check_id := _PRODUCT_DEPENDENCY_CHECKS.get(_identity_kind(identity))) is not None
    )
    if payload.stage is not DeliveryAcceptanceStage.LOCAL_DEVELOPMENT:
        checks.update(
            {
                "check:candidate-inspection",
                "check:editorial-review-evidence",
                "check:freshness-assurance",
                "check:explicit-publication",
            }
        )
    if payload.affected_scope.protected_capability_identities:
        checks.add("check:protected-capability-activation")
    if payload.stage in {
        DeliveryAcceptanceStage.LIMITED_TEAM_PILOT,
        DeliveryAcceptanceStage.DAILY_USE_RELEASE,
        DeliveryAcceptanceStage.PUBLIC_EVIDENCE_RELEASE,
    }:
        checks.update(
            {
                "check:authorization",
                "check:evidence-identity",
                "check:privacy",
                "check:publication",
                "check:policy-isolation",
                "check:closed-outcome",
                "check:pilot-workload",
                "check:recovery-evidence",
            }
        )
    if payload.stage is DeliveryAcceptanceStage.DAILY_USE_RELEASE:
        checks.update(
            {
                "check:daily-use-adoption",
                "check:daily-use-service-objective",
            }
        )
    if payload.stage is DeliveryAcceptanceStage.PUBLIC_EVIDENCE_RELEASE:
        checks.add("check:public-claim")
    return checks


class AcceptanceStatusFailureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    reason: str
    failure_kind: AcceptanceFailureKind
    blocking_scope: FailureBlockingScopeInput
    evidence_links: list[str] = Field(min_length=1, max_length=20)
    performance_objective_identity: str | None = None

    @field_validator("check_id")
    @classmethod
    def validate_check_id(cls, value: str) -> str:
        return _check_identity(value)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("evidence_links")
    @classmethod
    def validate_evidence_links(cls, values: list[str]) -> list[str]:
        return _evidence_links(values)

    @field_validator("performance_objective_identity")
    @classmethod
    def validate_performance_objective_identity(cls, value: str | None) -> str | None:
        return _exact_identity_of(value, {"objective"}) if value is not None else None

    @model_validator(mode="after")
    def validate_failure(self) -> AcceptanceStatusFailureInput:
        if self.blocking_scope.scope not in _allowed_failure_scopes(self.failure_kind):
            raise ValueError("failure kind must use its minimum sufficient blocking scope")
        if self.failure_kind is AcceptanceFailureKind.PERFORMANCE and self.performance_objective_identity is None:
            raise ValueError("performance failures require an affected objective identity")
        if self.failure_kind is not AcceptanceFailureKind.PERFORMANCE and self.performance_objective_identity is not None:
            raise ValueError("performance objective identity is only valid for performance failures")
        return self


class AcceptanceCheckVerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    evidence_links: list[str] = Field(min_length=1, max_length=20)

    @field_validator("check_id")
    @classmethod
    def validate_check_id(cls, value: str) -> str:
        return _check_identity(value)

    @field_validator("evidence_links")
    @classmethod
    def validate_evidence_links(cls, values: list[str]) -> list[str]:
        return _evidence_links(values)


class UpdateDeliveryAcceptanceStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AcceptanceStatus
    reason_code: AcceptanceStatusReason
    superseding_record_identity: str | None = None
    verified_checks: list[AcceptanceCheckVerificationInput] = Field(default_factory=list, max_length=50)
    status_failure: AcceptanceStatusFailureInput | None = None
    reacceptance_trigger: str | None = None
    invalidated_check_ids: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("superseding_record_identity")
    @classmethod
    def validate_superseding_identity(cls, value: str | None) -> str | None:
        return _exact_identity_of(value, {"delivery_acceptance_record"}) if value is not None else None

    @field_validator("reacceptance_trigger")
    @classmethod
    def validate_reacceptance_trigger(cls, value: str | None) -> str | None:
        return _non_blank(value) if value is not None else None

    @field_validator("invalidated_check_ids")
    @classmethod
    def validate_invalidated_check_ids(cls, values: list[str]) -> list[str]:
        normalized = [_check_identity(value) for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("invalidated check IDs must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_status_reason(self) -> UpdateDeliveryAcceptanceStatusRequest:
        allowed_reasons = {
            AcceptanceStatus.ACTIVE: {AcceptanceStatusReason.CHECKS_VERIFIED},
            AcceptanceStatus.AT_RISK: {
                AcceptanceStatusReason.REACCEPTANCE_DUE,
                AcceptanceStatusReason.PERFORMANCE_OBJECTIVE_MISSED,
            },
            AcceptanceStatus.SUSPENDED: {AcceptanceStatusReason.INTEGRITY_FAILURE},
            AcceptanceStatus.SUPERSEDED: {AcceptanceStatusReason.SUPERSEDED_BY_RECORD},
        }
        if self.reason_code not in allowed_reasons[self.status]:
            raise ValueError("reason_code is not valid for the requested status")
        if self.status is AcceptanceStatus.SUPERSEDED and self.superseding_record_identity is None:
            raise ValueError("superseded status requires a superseding record identity")
        if self.status is not AcceptanceStatus.SUPERSEDED and self.superseding_record_identity is not None:
            raise ValueError("superseding record identity is only valid for superseded status")
        if self.status is AcceptanceStatus.ACTIVE:
            if not self.verified_checks:
                raise ValueError("active status requires verified check evidence")
            if self.status_failure is not None or self.reacceptance_trigger is not None:
                raise ValueError("active status cannot include a failure or reacceptance trigger")
        elif self.verified_checks:
            raise ValueError("verified checks are only valid for active status")
        if self.status is AcceptanceStatus.SUSPENDED:
            if self.status_failure is None:
                raise ValueError("suspended status requires a scoped failure")
            if self.status_failure.failure_kind is AcceptanceFailureKind.PERFORMANCE:
                raise ValueError("performance failures must remain At Risk rather than Suspended")
            if self.reacceptance_trigger is not None:
                raise ValueError("suspended status cannot include a reacceptance trigger")
            if self.invalidated_check_ids:
                raise ValueError("suspended status cannot include reacceptance invalidation")
        elif self.status is AcceptanceStatus.AT_RISK:
            if self.reason_code is AcceptanceStatusReason.PERFORMANCE_OBJECTIVE_MISSED:
                if self.status_failure is None or self.status_failure.failure_kind is not AcceptanceFailureKind.PERFORMANCE:
                    raise ValueError("performance status requires a performance failure")
            elif self.status_failure is not None:
                raise ValueError("reacceptance status cannot include a failure")
            if self.reason_code is AcceptanceStatusReason.REACCEPTANCE_DUE and (
                self.reacceptance_trigger is None or not self.invalidated_check_ids
            ):
                raise ValueError("reacceptance status requires a declared trigger and affected checks")
            if self.reason_code is AcceptanceStatusReason.PERFORMANCE_OBJECTIVE_MISSED and self.invalidated_check_ids:
                raise ValueError("performance status cannot include reacceptance invalidation")
        elif self.status_failure is not None or self.reacceptance_trigger is not None or self.invalidated_check_ids:
            raise ValueError("failure and reacceptance details are not valid for this status")
        return self
