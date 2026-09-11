from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar

_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,159}$")
_UNKNOWN = "unknown"
EnumT = TypeVar("EnumT", bound=Enum)


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class StableIdentityKind(_ValueEnum):
    EVENT = "event"
    MEMBER = "member"
    TEAM_INVITATION = "team_invitation"
    ADMISSION_ATTEMPT = "admission_attempt"
    ENTRY = "entry"
    SOURCE = "source"
    BUNDLE = "bundle"
    BUNDLE_ITEM = "bundle_item"
    BUILD_GENERATION = "build_generation"
    CANDIDATE = "candidate"
    PUBLISHED_KNOWLEDGE_VERSION = "published_knowledge_version"
    ANSWER_EXECUTION = "answer_execution"
    EVIDENCE_SET = "evidence_set"
    EVIDENCE_SNAPSHOT = "evidence_snapshot"
    MAINTENANCE_ITEM = "maintenance_item"
    DELIVERY_ACCEPTANCE_RECORD = "delivery_acceptance_record"
    COLLECTION = "collection"
    CAPABILITY = "capability"
    CONFIGURATION = "configuration"
    CONCURRENCY = "concurrency"
    CORPUS = "corpus"
    DATA_BOUNDARY = "data_boundary"
    DEPLOYMENT = "deployment"
    EDITORIAL_REVISION = "editorial_revision"
    EMBEDDING_PROFILE = "embedding_profile"
    HOST = "host"
    MIGRATION = "migration"
    OBJECTIVE = "objective"
    PRODUCT_PATH = "product_path"
    PRODUCT_REVISION = "product_revision"
    PROMPT_ENVELOPE = "prompt_envelope"
    PROVIDER_ROUTE = "provider_route"
    PUBLIC_CLAIM = "public_claim"
    RETRIEVAL_PROFILE = "retrieval_profile"
    USER_BOUNDARY = "user_boundary"


class EntryLifecycleState(_ValueEnum):
    DRAFT = "draft"
    EVIDENCE_COLLECTED = "evidence_collected"
    EDITORIAL_REVIEW = "editorial_review"
    CANDIDATE_BUILD = "candidate_build"
    PUBLISHED = "published"
    NEEDS_REVIEW = "needs_re_review"
    WITHDRAWN = "withdrawn"


class CoveragePosition(_ValueEnum):
    RAG_SOURCE_ADMISSION_AND_CHUNKING = "rag_source_admission_and_chunking"
    SPARSE_DENSE_HYBRID_AND_RERANKING_CHOICES = "sparse_dense_hybrid_and_reranking_choices"
    EVIDENCE_SUFFICIENCY_REFUSAL_AND_ACCEPTANCE = "evidence_sufficiency_refusal_and_acceptance"
    TOOLS_AND_MCP_PERMISSIONS_AND_FAILURE_BEHAVIOR = "tools_and_mcp_permissions_and_failure_behavior"
    AGENT_CONTEXT_STATE_AND_MEMORY = "agent_context_state_and_memory"
    ORCHESTRATION_RETRY_HUMAN_INTERVENTION_AND_SIDE_EFFECTS = (
        "orchestration_retry_human_intervention_and_side_effects"
    )
    PROVIDER_FAILURE_AND_OBSERVABILITY = "provider_failure_and_observability"
    PROMPT_INJECTION_ISOLATION_AND_SECURITY = "prompt_injection_isolation_and_security"


class KnowledgeAssuranceLevel(_ValueEnum):
    SOURCE_GROUNDED = "source_grounded"
    CLAIM_LINKED = "claim_linked"
    RELEASE_ASSURED = "release_assured"


class KnowledgeSourceTier(_ValueEnum):
    PRIMARY_EVIDENCE_SOURCE = "primary_evidence_source"
    REPRODUCIBLE_ENGINEERING_EVIDENCE = "reproducible_engineering_evidence"
    SECONDARY_DISCOVERY_SOURCE = "secondary_discovery_source"
    BOUNDED_INTERNAL_CASE = "bounded_internal_case"


class SourceAccessScope(_ValueEnum):
    PUBLIC = "public"
    CONTROLLED_INTERNAL = "controlled_internal"


class EditorialRevisionChangeKind(_ValueEnum):
    MATERIAL = "material"
    WORDING_ONLY = "wording_only"


class SourceAvailabilityState(_ValueEnum):
    VERIFIED_USABLE = "verified_usable"
    CHANGED_OR_UNREACHABLE_AWAITING_REVIEW = "changed_or_unreachable_awaiting_review"
    UNAVAILABLE_FOR_NEW_EVIDENCE = "unavailable_for_new_evidence"


class BundleIntakeState(_ValueEnum):
    RECEIVED = "received"
    REJECTED = "rejected"
    VALIDATING = "validating"
    VALIDATED = "validated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_REJECTIONS = "completed_with_rejections"


class BuildJobStage(_ValueEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"


class BuildJobTerminalStatus(_ValueEnum):
    CANDIDATE_READY = "candidate_ready"
    FAILED = "failed"
    CANCELED = "canceled"
    INTERRUPTED_RETRYABLE = "interrupted_retryable"
    SUPERSEDED = "superseded"


class AnswerExecutionState(_ValueEnum):
    ADMITTED = "admitted"
    QUEUED = "queued"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    THROTTLED = "throttled"
    REJECTED = "rejected"
    COMPLETED = "completed"


class AnswerOutcome(_ValueEnum):
    EVIDENCE_GATED_ANSWER = "evidence_gated_answer"
    INSUFFICIENT_EVIDENCE_REPLY = "insufficient_evidence_reply"
    NON_KNOWLEDGE_BASE_REPLY = "non_knowledge_base_reply"
    GENERATION_UNAVAILABLE = "generation_unavailable"


class EvidenceSetState(_ValueEnum):
    PLANNED = "planned"
    FROZEN = "frozen"
    WITHDRAWN = "withdrawn"


class MaintenanceState(_ValueEnum):
    OPEN = "open"
    TRIAGED = "triaged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DEFERRED = "deferred"
    CLOSED_CONFIRMATION = "closed_confirmation"


class DeliveryAcceptanceStage(_ValueEnum):
    LOCAL_DEVELOPMENT = "local_development"
    EDITORIAL_PREVIEW = "editorial_preview"
    LIMITED_TEAM_PILOT = "limited_team_pilot"
    DAILY_USE_RELEASE = "daily_use_release"
    PUBLIC_EVIDENCE_RELEASE = "public_evidence_release"


class AcceptanceStatus(_ValueEnum):
    ACTIVE = "active"
    AT_RISK = "at_risk"
    SUSPENDED = "suspended"
    SUPERSEDED = "superseded"


class CanonicalRecordClass(_ValueEnum):
    IMMUTABLE = "immutable"
    APPEND_ONLY = "append_only"
    AUTHORITATIVE = "authoritative"
    DERIVED = "derived"
    REPLACEABLE_PROJECTION = "replaceable_projection"


class CanonicalEventType(_ValueEnum):
    CREATED = "created"
    STATE_CHANGED = "state_changed"
    REPLACED = "replaced"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"
    SUPERSEDED = "superseded"
    STATUS_CHANGED = "status_changed"
    BACKFILL_PROJECTED = "backfill_projected"


_TRANSITIONS: dict[type[Enum], dict[str, frozenset[str]]] = {
    BuildJobStage: {
        BuildJobStage.QUEUED.value: frozenset({BuildJobStage.PARSING.value}),
        BuildJobStage.PARSING.value: frozenset({BuildJobStage.CHUNKING.value}),
        BuildJobStage.CHUNKING.value: frozenset({BuildJobStage.INDEXING.value}),
        BuildJobStage.INDEXING.value: frozenset(),
    },
    EntryLifecycleState: {
        EntryLifecycleState.DRAFT.value: frozenset({EntryLifecycleState.EVIDENCE_COLLECTED.value}),
        EntryLifecycleState.EVIDENCE_COLLECTED.value: frozenset({EntryLifecycleState.EDITORIAL_REVIEW.value}),
        EntryLifecycleState.EDITORIAL_REVIEW.value: frozenset({
            EntryLifecycleState.CANDIDATE_BUILD.value, EntryLifecycleState.WITHDRAWN.value,
        }),
        EntryLifecycleState.CANDIDATE_BUILD.value: frozenset({
            EntryLifecycleState.PUBLISHED.value, EntryLifecycleState.WITHDRAWN.value,
        }),
        EntryLifecycleState.PUBLISHED.value: frozenset(
            {
                EntryLifecycleState.EDITORIAL_REVIEW.value,
                EntryLifecycleState.NEEDS_REVIEW.value,
                EntryLifecycleState.WITHDRAWN.value,
            }
        ),
        EntryLifecycleState.NEEDS_REVIEW.value: frozenset(
            {EntryLifecycleState.EDITORIAL_REVIEW.value, EntryLifecycleState.WITHDRAWN.value}
        ),
        EntryLifecycleState.WITHDRAWN.value: frozenset(),
    },
    BundleIntakeState: {
        BundleIntakeState.RECEIVED.value: frozenset(
            {BundleIntakeState.VALIDATING.value, BundleIntakeState.REJECTED.value}
        ),
        BundleIntakeState.VALIDATING.value: frozenset(
            {BundleIntakeState.VALIDATED.value, BundleIntakeState.REJECTED.value}
        ),
        BundleIntakeState.VALIDATED.value: frozenset({BundleIntakeState.PROCESSING.value}),
        BundleIntakeState.PROCESSING.value: frozenset(
            {
                BundleIntakeState.COMPLETED.value,
                BundleIntakeState.COMPLETED_WITH_REJECTIONS.value,
                BundleIntakeState.REJECTED.value,
            }
        ),
        BundleIntakeState.REJECTED.value: frozenset(),
        BundleIntakeState.COMPLETED.value: frozenset(),
        BundleIntakeState.COMPLETED_WITH_REJECTIONS.value: frozenset(),
    },
    EvidenceSetState: {
        EvidenceSetState.PLANNED.value: frozenset({EvidenceSetState.FROZEN.value}),
        EvidenceSetState.FROZEN.value: frozenset({EvidenceSetState.WITHDRAWN.value}),
        EvidenceSetState.WITHDRAWN.value: frozenset(),
    },
    AnswerExecutionState: {
        AnswerExecutionState.ADMITTED.value: frozenset(
            {AnswerExecutionState.QUEUED.value, AnswerExecutionState.REJECTED.value, AnswerExecutionState.THROTTLED.value}
        ),
        AnswerExecutionState.QUEUED.value: frozenset(
            {AnswerExecutionState.RUNNING.value, AnswerExecutionState.STOPPED.value, AnswerExecutionState.FAILED.value}
        ),
        AnswerExecutionState.RUNNING.value: frozenset(
            {AnswerExecutionState.COMPLETED.value, AnswerExecutionState.STOPPED.value, AnswerExecutionState.FAILED.value}
        ),
        AnswerExecutionState.STOPPED.value: frozenset(),
        AnswerExecutionState.FAILED.value: frozenset(),
        AnswerExecutionState.THROTTLED.value: frozenset(),
        AnswerExecutionState.REJECTED.value: frozenset(),
        AnswerExecutionState.COMPLETED.value: frozenset(),
    },
    MaintenanceState: {
        MaintenanceState.OPEN.value: frozenset({MaintenanceState.TRIAGED.value}),
        MaintenanceState.TRIAGED.value: frozenset(
            {MaintenanceState.IN_PROGRESS.value, MaintenanceState.DEFERRED.value, MaintenanceState.CLOSED_CONFIRMATION.value}
        ),
        MaintenanceState.IN_PROGRESS.value: frozenset(
            {MaintenanceState.RESOLVED.value, MaintenanceState.DEFERRED.value, MaintenanceState.CLOSED_CONFIRMATION.value}
        ),
        MaintenanceState.DEFERRED.value: frozenset({MaintenanceState.OPEN.value, MaintenanceState.CLOSED_CONFIRMATION.value}),
        MaintenanceState.RESOLVED.value: frozenset({MaintenanceState.CLOSED_CONFIRMATION.value}),
        MaintenanceState.CLOSED_CONFIRMATION.value: frozenset(),
    },
    DeliveryAcceptanceStage: {
        DeliveryAcceptanceStage.LOCAL_DEVELOPMENT.value: frozenset({DeliveryAcceptanceStage.EDITORIAL_PREVIEW.value}),
        DeliveryAcceptanceStage.EDITORIAL_PREVIEW.value: frozenset(
            {DeliveryAcceptanceStage.LIMITED_TEAM_PILOT.value, DeliveryAcceptanceStage.DAILY_USE_RELEASE.value}
        ),
        DeliveryAcceptanceStage.LIMITED_TEAM_PILOT.value: frozenset({DeliveryAcceptanceStage.DAILY_USE_RELEASE.value}),
        DeliveryAcceptanceStage.DAILY_USE_RELEASE.value: frozenset({DeliveryAcceptanceStage.PUBLIC_EVIDENCE_RELEASE.value}),
        DeliveryAcceptanceStage.PUBLIC_EVIDENCE_RELEASE.value: frozenset(),
    },
    AcceptanceStatus: {
        AcceptanceStatus.ACTIVE.value: frozenset(
            {AcceptanceStatus.AT_RISK.value, AcceptanceStatus.SUSPENDED.value, AcceptanceStatus.SUPERSEDED.value}
        ),
        AcceptanceStatus.AT_RISK.value: frozenset(
            {AcceptanceStatus.ACTIVE.value, AcceptanceStatus.SUSPENDED.value, AcceptanceStatus.SUPERSEDED.value}
        ),
        AcceptanceStatus.SUSPENDED.value: frozenset(
            {AcceptanceStatus.ACTIVE.value, AcceptanceStatus.SUPERSEDED.value}
        ),
        AcceptanceStatus.SUPERSEDED.value: frozenset(),
    },
}


def _validate_stable_id(value: str) -> str:
    normalized = value.strip()
    if _STABLE_ID.fullmatch(normalized) is None:
        raise ValueError("stable identity must be a lowercase bounded identifier")
    return normalized


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, MappingProxyType):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class StableIdentity:
    kind: StableIdentityKind
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", StableIdentityKind(self.kind))
        object.__setattr__(self, "value", _validate_stable_id(self.value))

    @property
    def stable_id(self) -> str:
        return f"{self.kind.value}:{self.value}"

    @classmethod
    def new(cls, kind: StableIdentityKind, value: str | None = None) -> StableIdentity:
        generated = value or uuid.uuid4().hex
        return cls(kind=kind, value=generated)

    @classmethod
    def from_stable_id(cls, value: str) -> StableIdentity:
        kind_value, separator, identity_value = value.partition(":")
        if not separator:
            raise ValueError("stable identity must contain a kind prefix")
        return cls(kind=StableIdentityKind(kind_value), value=identity_value)

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "value": self.value, "stable_id": self.stable_id}


@dataclass(frozen=True)
class CanonicalRecord:
    identity: StableIdentity
    state: str
    record_class: CanonicalRecordClass
    payload: MappingProxyType | dict[str, Any]
    schema_version: int = 1
    legacy_id: str | None = None
    created_at: datetime = datetime.min.replace(tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        if not self.state or not self.state.strip():
            raise ValueError("state is required")
        object.__setattr__(self, "record_class", CanonicalRecordClass(self.record_class))
        payload = dict(self.payload)
        object.__setattr__(self, "payload", MappingProxyType(payload))
        if self.legacy_id is not None and not self.legacy_id.strip():
            raise ValueError("legacy_id must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "state": self.state,
            "record_class": self.record_class.value,
            "payload": _json_safe(dict(self.payload)),
            "schema_version": self.schema_version,
            "legacy_id": self.legacy_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalRecord:
        identity_data = value.get("identity")
        if not isinstance(identity_data, dict):
            raise ValueError("identity is required")
        created_at = datetime.fromisoformat(str(value["created_at"])) if value.get("created_at") else datetime.now(UTC)
        return cls(
            identity=StableIdentity(
                kind=StableIdentityKind(identity_data["kind"]),
                value=str(identity_data["value"]),
            ),
            state=str(value["state"]),
            record_class=CanonicalRecordClass(value["record_class"]),
            payload=dict(value["payload"]) if isinstance(value.get("payload"), dict) else {},
            schema_version=int(value.get("schema_version", 1)),
            legacy_id=value.get("legacy_id"),
            created_at=created_at,
        )


@dataclass(frozen=True)
class CanonicalEvent:
    event_id: StableIdentity
    aggregate: StableIdentity
    event_type: CanonicalEventType
    from_state: str | None
    to_state: str
    payload: MappingProxyType | dict[str, Any]
    occurred_at: datetime = datetime.min.replace(tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.event_id.kind is not StableIdentityKind.EVENT:
            raise ValueError("event_id must use the event identity kind")
        if not self.to_state.strip():
            raise ValueError("to_state is required")
        object.__setattr__(self, "event_type", CanonicalEventType(self.event_type))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id.to_dict(),
            "aggregate": self.aggregate.to_dict(),
            "event_type": self.event_type.value,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "payload": _json_safe(dict(self.payload)),
            "occurred_at": self.occurred_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalEvent:
        def identity(field: str) -> StableIdentity:
            data = value.get(field)
            if not isinstance(data, dict):
                raise ValueError(f"{field} is required")
            return StableIdentity(kind=StableIdentityKind(data["kind"]), value=str(data["value"]))

        occurred_at = datetime.fromisoformat(str(value["occurred_at"])) if value.get("occurred_at") else datetime.now(UTC)
        return cls(
            event_id=identity("event_id"),
            aggregate=identity("aggregate"),
            event_type=CanonicalEventType(value["event_type"]),
            from_state=value.get("from_state"),
            to_state=str(value["to_state"]),
            payload=dict(value["payload"]) if isinstance(value.get("payload"), dict) else {},
            occurred_at=occurred_at,
        )


def validate_transition(enum_type: type[EnumT], current: EnumT | str, target: EnumT | str) -> EnumT:
    """Validate a state transition without mutating a persisted record."""
    current_value = enum_type(current).value
    target_member = enum_type(target)
    allowed = _TRANSITIONS.get(enum_type, {}).get(current_value)
    if allowed is None:
        raise ValueError(f"no transition vocabulary registered for {enum_type.__name__}")
    if target_member.value not in allowed:
        raise ValueError(f"illegal {enum_type.__name__} transition: {current_value} -> {target_member.value}")
    return target_member


def legacy_identity(kind: StableIdentityKind, legacy_id: str) -> StableIdentity:
    normalized = re.sub(r"[^a-z0-9._:-]+", "-", legacy_id.strip().lower()).strip("-")
    normalized = _validate_stable_id(normalized)
    digest = hashlib.sha256(f"{kind.value}:{normalized}".encode()).hexdigest()[:24]
    return StableIdentity(kind=kind, value=f"legacy-{digest}")


def _legacy_value(record: object, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _document_state(status: object, *, published_generation: int, candidate_generation: int | None) -> EntryLifecycleState:
    if published_generation > 0:
        return EntryLifecycleState.PUBLISHED
    if candidate_generation is not None and candidate_generation > 0:
        return EntryLifecycleState.CANDIDATE_BUILD
    if status in {"failed", "deleted"}:
        return EntryLifecycleState.WITHDRAWN
    return EntryLifecycleState.DRAFT


def compatibility_read_projection(entity: StableIdentityKind | str, legacy_record: object) -> CanonicalRecord:
    """Project one legacy row deterministically without granting new authority."""
    kind = StableIdentityKind(entity)
    legacy_id = str(_legacy_value(legacy_record, "id", "")).strip()
    if not legacy_id:
        raise ValueError("legacy record id is required")
    identity = legacy_identity(kind, legacy_id)
    payload: dict[str, Any] = {"compatibility": "legacy_projection", "legacy_type": kind.value}

    if kind is StableIdentityKind.ENTRY:
        published = int(_legacy_value(legacy_record, "published_generation", 0) or 0)
        candidate = _legacy_value(legacy_record, "candidate_generation")
        state = _document_state(_legacy_value(legacy_record, "status"), published_generation=published, candidate_generation=candidate)
        payload.update(
            {
                "legacy_document_id": legacy_id,
                "published_generation": published,
                "candidate_generation": candidate,
                "answer_eligible": False,
                "source_availability": _UNKNOWN,
                "unknown_fields": ["entry_id", "editorial_review", "assurance", "source_authority"],
            }
        )
        return CanonicalRecord(identity, state.value, CanonicalRecordClass.REPLACEABLE_PROJECTION, payload, legacy_id=legacy_id)

    if kind is StableIdentityKind.BUILD_GENERATION:
        state = str(_legacy_value(legacy_record, "status", "queued"))
        payload.update(
            {
                "legacy_job_id": legacy_id,
                "document_id": _legacy_value(legacy_record, "document_id"),
                "generation": _legacy_value(legacy_record, "build_generation"),
                "stage": _legacy_value(legacy_record, "stage", _UNKNOWN),
                "unknown_fields": ["bundle_id", "input_hash", "configuration_identity"],
            }
        )
        return CanonicalRecord(identity, state, CanonicalRecordClass.REPLACEABLE_PROJECTION, payload, legacy_id=legacy_id)

    if kind is StableIdentityKind.ANSWER_EXECUTION:
        trace = _legacy_value(legacy_record, "rag_trace") or {}
        outcome = trace.get("outcome") if isinstance(trace, dict) else None
        payload.update(
            {
                "legacy_message_id": legacy_id,
                "outcome": outcome or _UNKNOWN,
                "answer_eligible": outcome == AnswerOutcome.EVIDENCE_GATED_ANSWER.value,
                "evidence_set_id": _UNKNOWN,
                "unknown_fields": ["query_condition_set", "evidence_set_id", "provider_route"],
            }
        )
        state = AnswerExecutionState.COMPLETED.value if outcome else AnswerExecutionState.FAILED.value
        return CanonicalRecord(identity, state, CanonicalRecordClass.REPLACEABLE_PROJECTION, payload, legacy_id=legacy_id)

    if kind is StableIdentityKind.MAINTENANCE_ITEM:
        payload.update(
            {
                "legacy_review_work_item_id": legacy_id,
                "classification": _legacy_value(legacy_record, "classification") or _UNKNOWN,
                "unknown_fields": ["severity", "work_owner", "validated_finding"],
            }
        )
        return CanonicalRecord(
            identity,
            _legacy_value(legacy_record, "status", MaintenanceState.OPEN.value),
            CanonicalRecordClass.REPLACEABLE_PROJECTION,
            payload,
            legacy_id=legacy_id,
        )

    payload["unknown_fields"] = ["canonical_state", "canonical_relationships"]
    return CanonicalRecord(identity, _UNKNOWN, CanonicalRecordClass.REPLACEABLE_PROJECTION, payload, legacy_id=legacy_id)


def serialized_round_trip(record: CanonicalRecord | CanonicalEvent) -> CanonicalRecord | CanonicalEvent:
    """Round-trip helper used by contract tests and adapters."""
    encoded = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    return type(record).from_dict(decoded)


MIGRATION_PLAN: tuple[dict[str, str], ...] = (
    {
        "ticket": "14",
        "legacy_surface": "User, team invitation and Redis session admission/authority paths",
        "canonical_contract": "Member/team invitation identity and content-free identity audit events",
        "remove_when": "Every pilot identity path uses one-time admission, database-derived authority and append-only audit.",
    },
    {
        "ticket": "17",
        "legacy_surface": "Document upload and batch build dispatch",
        "canonical_contract": "Reviewed Release Bundle, bundle item and build generation",
        "remove_when": "Reviewed bundle import is the only supported recurring intake and all legacy jobs have terminal projections.",
    },
    {
        "ticket": "20",
        "legacy_surface": "ChatMessage.rag_trace and ChatService outcome inference",
        "canonical_contract": "Answer execution, Query Condition Set, evidence set and snapshot",
        "remove_when": "Every answer surface persists the canonical closed outcome and transport adapters only project it.",
    },
    {
        "ticket": "21",
        "legacy_surface": "Chat HTTP, SSE and history adapters",
        "canonical_contract": "Canonical execution projection",
        "remove_when": "Normal, streaming, persistence and history read only canonical execution records.",
    },
    {
        "ticket": "24",
        "legacy_surface": "Candidate inspection and publication endpoints",
        "canonical_contract": "Candidate, published knowledge version and publication event",
        "remove_when": "Publication checks canonical Candidate identity, latest generation, bundle hash and acceptance.",
    },
    {
        "ticket": "25",
        "legacy_surface": "Document tombstone and historical redaction",
        "canonical_contract": "Withdrawal event and retained publication identity",
        "remove_when": "All withdrawal reads and writes use canonical publication identities.",
    },
    {
        "ticket": "27",
        "legacy_surface": "KnowledgeFeedbackSignal and ReviewWorkItem",
        "canonical_contract": "Maintenance item, validated finding and append-only maintenance events",
        "remove_when": "Feedback workflow persists the durable canonical maintenance record and severed raw-signal references.",
    },
    {
        "ticket": "15",
        "legacy_surface": "Acceptance scripts and release evidence",
        "canonical_contract": "Delivery Acceptance Record and status events",
        "remove_when": "Acceptance results bind exact canonical identities and no caller relies on branch/latest labels.",
    },
)
