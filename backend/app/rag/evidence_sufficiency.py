"""Deterministic sufficiency planning for authorized published evidence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from app.common.canonical_json import canonical_json_sha256
from app.contracts.candidate_claim_evidence import (
    CandidateClaimEvidenceContractError,
    parse_candidate_claim_evidence_contract,
)
from app.contracts.canonical import KnowledgeSourceTier, StableIdentity, StableIdentityKind
from app.rag.answer_evidence import AnswerEvidence
from app.rag.claim_evidence import ClaimEvidenceContractError, parse_claim_evidence_contract

_INSUFFICIENT_REASONS = frozenset(
    {
        "no_eligible_published_evidence",
        "decision_not_covered",
        "decisive_condition_missing",
        "material_evidence_conflict",
        "assurance_support_missing",
        "evidence_budget_exceeded",
        "knowledge_needs_review",
    }
)
_GOVERNING_SECTION = "recommendation_or_reviewed_branches"
_MAX_ITEMS = 3
_MAX_EXCERPT_CHARS = 1200
_MAX_TOTAL_CHARS = 3000
_MAX_QUERY_CONDITIONS = 32
_MAX_CONDITION_ID_CHARS = 160
_MAX_CONDITION_FIELD_CHARS = 160
_MAX_CONDITION_OPERATOR_CHARS = 64
_MAX_CONDITION_VALUE_CHARS = 512
_EXPLICIT_CONDITION = re.compile(r"\b([a-z][a-z0-9_.-]{0,79})\s*=\s*([a-z0-9_.:/-]{1,160})\b", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_TIERS = frozenset(item.value for item in KnowledgeSourceTier)
_SOURCE_EVIDENCE_PROJECTION_FIELDS = frozenset(
    {
        "source_identity",
        "source_id",
        "source_tier",
        "source_access_scope",
        "source_title",
        "source_authority",
        "source_url",
        "source_version",
        "source_review_date",
    }
)
_COVERAGE_TOKEN = re.compile(r"[a-z][a-z0-9_.-]{1,79}|[\u3400-\u9fff]{2,}", re.IGNORECASE)
_COVERAGE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "applies",
        "apply",
        "are",
        "be",
        "by",
        "can",
        "decision",
        "default",
        "do",
        "does",
        "for",
        "from",
        "govern",
        "governing",
        "governs",
        "how",
        "is",
        "it",
        "of",
        "operating",
        "or",
        "route",
        "the",
        "this",
        "to",
        "use",
        "what",
        "which",
    }
)


def _required_sections(normalized_question: str) -> tuple[str, ...]:
    question = normalized_question.casefold()
    requirements: list[str] = []
    if any(term in question for term in ("compare", "comparison", "versus", " vs ", "alternatives", "比较", "对比", "取舍")):
        requirements.extend(("alternatives", "trade_offs"))
    if any(term in question for term in ("diagnose", "diagnosis", "debug", "failure", "故障", "诊断", "排查")):
        requirements.extend(("failure_modes", "minimum_diagnosis_guidance"))
    if any(
        term in question
        for term in (
            "acceptance review",
            "acceptance check",
            "review the acceptance",
            "验收",
            "评审",
        )
    ):
        requirements.append("minimum_acceptance_guidance")
    if any(term in question for term in ("implementation", "checklist", "code", "实现", "清单", "代码")):
        requirements.append("minimum_implementation_guidance")
    return tuple(dict.fromkeys(requirements))


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_value(value)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True)
class QueryCondition:
    condition_id: str
    field: str
    operator: str
    value: str

    @classmethod
    def from_record(cls, value: object) -> QueryCondition:
        if not isinstance(value, Mapping):
            raise ValueError("query condition must be an object")
        fields = ("condition_id", "field", "operator", "value")
        parsed = {field: value.get(field) for field in fields}
        if any(not isinstance(item, str) or not item.strip() for item in parsed.values()):
            raise ValueError("query condition must have non-empty condition_id, field, operator, and value")
        normalized = {key: str(item).strip() for key, item in parsed.items()}
        limits = {
            "condition_id": _MAX_CONDITION_ID_CHARS,
            "field": _MAX_CONDITION_FIELD_CHARS,
            "operator": _MAX_CONDITION_OPERATOR_CHARS,
            "value": _MAX_CONDITION_VALUE_CHARS,
        }
        if any(len(normalized[field]) > limit for field, limit in limits.items()):
            raise ValueError("query condition exceeds the accepted explicit-input limits")
        return cls(**normalized)

    def to_record(self) -> dict[str, str]:
        return {
            "condition_id": self.condition_id,
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass(frozen=True)
class QueryConditionSet:
    normalized_question: str
    conditions: tuple[QueryCondition, ...]
    identity_nonce: str | None = None

    @classmethod
    def from_records(
        cls,
        *,
        normalized_question: str,
        records: Sequence[object],
        identity_nonce: str | None = None,
    ) -> QueryConditionSet:
        question = normalized_question.strip()
        if not question:
            raise ValueError("normalized question must be non-empty")
        nonce = identity_nonce.strip() if isinstance(identity_nonce, str) else identity_nonce
        if nonce is not None and (not isinstance(nonce, str) or not nonce):
            raise ValueError("query condition set identity nonce must be non-empty when present")
        if len(records) > _MAX_QUERY_CONDITIONS:
            raise ValueError("query condition set exceeds the accepted explicit-input limit")
        conditions = tuple(QueryCondition.from_record(record) for record in records)
        keys = {(condition.field, condition.operator) for condition in conditions}
        if len(keys) != len(conditions):
            raise ValueError("query condition set must not repeat a field/operator pair")
        condition_ids = {condition.condition_id for condition in conditions}
        if len(condition_ids) != len(conditions):
            raise ValueError("query condition set must not repeat a condition identity")
        return cls(normalized_question=question, conditions=conditions, identity_nonce=nonce)

    @classmethod
    def from_question(cls, normalized_question: str) -> QueryConditionSet:
        question = normalized_question.strip()
        return cls.from_records(
            normalized_question=question,
            records=[
                {
                    "condition_id": f"{field.casefold()}-{value.casefold()}",
                    "field": field.casefold(),
                    "operator": "equals",
                    "value": value.casefold(),
                }
                for field, value in _EXPLICIT_CONDITION.findall(question)
            ],
        )

    @property
    def identity(self) -> str:
        record: dict[str, object] = {
            "normalized_question": self.normalized_question,
            "conditions": [condition.to_record() for condition in self.conditions],
        }
        if self.identity_nonce is not None:
            record["identity_nonce"] = self.identity_nonce
        return canonical_json_sha256(record)

    def matches(self, condition: Mapping[str, object]) -> bool | None:
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        if (
            not isinstance(field, str)
            or not field.strip()
            or not isinstance(operator, str)
            or not operator.strip()
            or not isinstance(value, str)
            or not value.strip()
        ):
            return None
        return any(
            item.field == field.strip() and item.operator == operator.strip() and item.value == value.strip()
            for item in self.conditions
        )

    def has_field_operator(self, condition: Mapping[str, object]) -> bool:
        field = condition.get("field")
        operator = condition.get("operator")
        if not isinstance(field, str) or not field.strip() or not isinstance(operator, str) or not operator.strip():
            return False
        return any(item.field == field.strip() and item.operator == operator.strip() for item in self.conditions)

    def has_unsupported_response_assignment(self, text: str) -> bool:
        allowed_values: dict[str, set[str]] = {}
        for condition in self.conditions:
            if condition.operator == "equals":
                allowed_values.setdefault(condition.field.casefold(), set()).add(condition.value.casefold())
        for field, value in _EXPLICIT_CONDITION.findall(text):
            accepted = allowed_values.get(field.casefold())
            normalized_value = value.strip("'\"").rstrip(".,;:!?").casefold()
            if accepted is None or normalized_value not in accepted:
                return True
        return False

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "identity": self.identity,
            "normalized_question": self.normalized_question,
            "conditions": [condition.to_record() for condition in self.conditions],
        }
        if self.identity_nonce is not None:
            record["identity_nonce"] = self.identity_nonce
        return record

    def to_provider_record(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "conditions": [condition.to_record() for condition in self.conditions],
        }


@dataclass(frozen=True)
class EvidenceCitation:
    marker: str
    identity: str
    item_identity: str
    entry_id: str
    section_id: str
    snapshot_id: str

    def to_record(self) -> dict[str, str]:
        return {
            "citation_id": self.marker,
            "citation_identity": self.identity,
            "item_identity": self.item_identity,
            "entry_id": self.entry_id,
            "section_id": self.section_id,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True)
class AnswerEvidenceSet:
    identity: str
    query_condition_set_identity: str
    query_conditions: QueryConditionSet
    items: tuple[AnswerEvidence, ...]
    item_identity_bindings: tuple[Mapping[str, object], ...]
    citations: tuple[EvidenceCitation, ...]
    governing_citation: EvidenceCitation

    @classmethod
    def freeze(
        cls,
        *,
        query_conditions: QueryConditionSet,
        items: Sequence[AnswerEvidence],
        governing_item: AnswerEvidence,
        item_identities: Sequence[str] | None = None,
        item_identity_bindings: Sequence[Mapping[str, object]] | None = None,
    ) -> AnswerEvidenceSet:
        frozen_items = tuple(items)
        if not frozen_items or governing_item not in frozen_items:
            raise ValueError("a frozen Answer Evidence Set needs its governing item")
        frozen_item_identities = (
            tuple(_evidence_item_identity(item) for item in frozen_items)
            if item_identities is None
            else tuple(item_identities)
        )
        if (
            len(frozen_item_identities) != len(frozen_items)
            or any(_SHA256.fullmatch(identity) is None for identity in frozen_item_identities)
        ):
            raise ValueError("a frozen Answer Evidence Set needs one stable identity per selected item")
        frozen_item_bindings = (
            tuple(_freeze_mapping(_evidence_item_binding(item)) for item in frozen_items)
            if item_identity_bindings is None
            else tuple(_freeze_mapping(item) for item in item_identity_bindings)
        )
        if (
            len(frozen_item_bindings) != len(frozen_items)
            or any(
                canonical_json_sha256(_thaw_value(binding)) != item_identity
                for binding, item_identity in zip(frozen_item_bindings, frozen_item_identities, strict=True)
            )
        ):
            raise ValueError("a frozen Answer Evidence Set needs identity bindings for every selected item")
        governing_index = frozen_items.index(governing_item)
        identity = canonical_json_sha256(
            {
                "query_condition_set_identity": query_conditions.identity,
                "item_identities": list(frozen_item_identities),
                "governing_item_identity": frozen_item_identities[governing_index],
            }
        )
        citations = tuple(
            EvidenceCitation(
                marker=f"S{index}",
                identity=canonical_json_sha256(
                    {
                        "evidence_set_identity": identity,
                        "item_identity": item_identity,
                    }
                ),
                item_identity=item_identity,
                entry_id=str(dict(item.metadata_items).get("entry_id") or ""),
                section_id=item.section_id,
                snapshot_id=item.snapshot_id,
            )
            for index, (item, item_identity) in enumerate(zip(frozen_items, frozen_item_identities, strict=True), start=1)
        )
        return cls(
            identity=identity,
            query_condition_set_identity=query_conditions.identity,
            query_conditions=query_conditions,
            items=frozen_items,
            item_identity_bindings=frozen_item_bindings,
            citations=citations,
            governing_citation=citations[governing_index],
        )

    def to_record(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "query_condition_set_identity": self.query_condition_set_identity,
            "query_conditions": self.query_conditions.to_record(),
            "governing_citation": self.governing_citation.to_record(),
            "items": [
                {
                    "item_identity": citation.item_identity,
                    "identity_binding": _thaw_value(binding),
                    "snapshot_id": item.snapshot_id,
                    "evidence": _frozen_snapshot_record(item),
                    "citation": citation.to_record(),
                }
                for item, citation, binding in zip(
                    self.items,
                    self.citations,
                    self.item_identity_bindings,
                    strict=True,
                )
            ]
        }


@dataclass(frozen=True)
class InsufficientEvidenceReply:
    reason: str
    query_condition_set_identity: str

    def __post_init__(self) -> None:
        if self.reason not in _INSUFFICIENT_REASONS:
            raise ValueError("insufficient evidence reason is not accepted")

    def to_record(self) -> dict[str, str]:
        return {
            "outcome": "insufficient_evidence_reply",
            "reason": self.reason,
            "query_condition_set_identity": self.query_condition_set_identity,
        }


@dataclass(frozen=True)
class EvidenceSufficiencyDecision:
    query_conditions: QueryConditionSet
    evidence_set: AnswerEvidenceSet | None
    insufficient_reply: InsufficientEvidenceReply | None

    def __post_init__(self) -> None:
        if (self.evidence_set is None) == (self.insufficient_reply is None):
            raise ValueError("a sufficiency decision has exactly one closed result")

    @property
    def is_sufficient(self) -> bool:
        return self.evidence_set is not None

    @property
    def reason(self) -> str | None:
        return self.insufficient_reply.reason if self.insufficient_reply is not None else None

    def to_record(self) -> dict[str, object]:
        if self.evidence_set is not None:
            return {
                "outcome": "sufficient_evidence",
                "query_conditions": self.query_conditions.to_record(),
                "answer_evidence_set": self.evidence_set.to_record(),
            }
        assert self.insufficient_reply is not None
        return {
            "query_conditions": self.query_conditions.to_record(),
            "insufficient_evidence_reply": self.insufficient_reply.to_record(),
        }


@dataclass(frozen=True)
class _AuthorizedCandidate:
    evidence: AnswerEvidence
    entry_id: str
    section_id: str
    item_identity: str
    item_identity_binding: Mapping[str, object]
    entry_identity: str
    decision_query: str
    source_content_length: int
    applicability_conditions: tuple[Mapping[str, object], ...]
    assurance_level: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _EvidenceRequirement:
    section_id: str
    entry_id: str | None = None
    source_id: str | None = None


def _evidence_item_binding(item: AnswerEvidence) -> dict[str, object]:
    metadata = dict(item.metadata_items)
    return {
        "chunk_id": item.source_id,
        "document_id": item.document_id,
        "generation": item.generation,
        "chunk_index": item.chunk_index,
        "snapshot_id": item.snapshot_id,
        "entry_id": metadata.get("entry_id"),
        "section_id": item.section_id,
        "publication_version": item.publication_version,
        "source_content_length": len(item.excerpt),
    }


def _evidence_item_identity(item: AnswerEvidence) -> str:
    return canonical_json_sha256(_evidence_item_binding(item))


def _frozen_snapshot_record(item: AnswerEvidence) -> dict[str, Any]:
    record = item.to_record()
    record.pop("score", None)
    record.pop("retrieval_source", None)
    return record


def _stable_identity_has_kind(value: object, kind: StableIdentityKind) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return StableIdentity.from_stable_id(value).kind is kind
    except ValueError:
        return False


def _candidate_item_binding(value: Mapping[str, object], evidence: AnswerEvidence) -> dict[str, object] | None:
    entry_id = value.get("entry_id")
    entry_identity = value.get("entry_identity")
    editorial_revision_identity = value.get("editorial_revision_identity")
    publication_identity = value.get("publication_identity")
    section_id = value.get("section_id")
    section_identity = value.get("section_identity")
    chunk_identity = value.get("chunk_identity")
    content_sha256 = value.get("content_sha256")
    metadata = value.get("metadata")
    publication_version = value.get("publication_version")
    content_length = value.get("content_length")
    if (
        not isinstance(entry_id, str)
        or not isinstance(entry_identity, str)
        or not _stable_identity_has_kind(entry_identity, StableIdentityKind.ENTRY)
        or StableIdentity.from_stable_id(entry_identity).value != entry_id
        or not _stable_identity_has_kind(editorial_revision_identity, StableIdentityKind.EDITORIAL_REVISION)
        or not _stable_identity_has_kind(publication_identity, StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION)
        or not isinstance(section_id, str)
        or section_identity != f"{entry_identity}#{section_id}"
        or not isinstance(chunk_identity, Mapping)
        or not isinstance(content_sha256, str)
        or _SHA256.fullmatch(content_sha256) is None
        or not isinstance(metadata, Mapping)
        or metadata.get("entry_id") != entry_id
        or metadata.get("entry_identity") != entry_identity
        or metadata.get("editorial_revision_identity") != editorial_revision_identity
        or metadata.get("section_id") != section_id
        or metadata.get("source_tier") not in _SOURCE_TIERS
        or not isinstance(publication_version, str)
        or publication_version.strip() != evidence.publication_version
        or isinstance(content_length, bool)
        or not isinstance(content_length, int)
        or content_length < len(evidence.excerpt)
    ):
        return None
    source_evidence_identity = metadata.get("candidate_evidence_source_identity")
    if source_evidence_identity is not None:
        if not isinstance(source_evidence_identity, str):
            return None
        try:
            source_identity = StableIdentity.from_stable_id(source_evidence_identity)
        except ValueError:
            return None
        if (
            source_identity.kind is not StableIdentityKind.SOURCE
            or metadata.get("source_identity") != source_evidence_identity
            or metadata.get("source_id") != source_identity.value
        ):
            return None
    required_chunk_fields = ("document_id", "generation", "chunk_index", "content_sha256")
    if set(chunk_identity) != set(required_chunk_fields):
        return None
    if (
        not isinstance(chunk_identity.get("document_id"), str)
        or chunk_identity.get("document_id") != evidence.document_id
        or isinstance(chunk_identity.get("generation"), bool)
        or chunk_identity.get("generation") != evidence.generation
        or isinstance(chunk_identity.get("chunk_index"), bool)
        or chunk_identity.get("chunk_index") != evidence.chunk_index
        or not isinstance(chunk_identity.get("content_sha256"), str)
        or _SHA256.fullmatch(str(chunk_identity["content_sha256"])) is None
        or chunk_identity.get("content_sha256") != content_sha256
    ):
        return None
    binding = {
        "entry_identity": entry_identity,
        "editorial_revision_identity": editorial_revision_identity,
        "publication_identity": publication_identity,
        "section_identity": section_identity,
        "chunk_identity": {
            field: chunk_identity[field]
            for field in required_chunk_fields
        },
        "snapshot_id": evidence.snapshot_id,
        "source_content_length": content_length,
    }
    if source_evidence_identity is not None:
        binding["candidate_evidence_source_identity"] = source_evidence_identity
    return binding


def _has_authorized_shape(value: object) -> bool:
    if not isinstance(value, Mapping) or value.get("answer_evidence_eligible") is not True:
        return False
    if value.get("diagnostic_only") is True:
        return False
    required_identities = (
        "entry_id",
        "entry_identity",
        "editorial_revision_identity",
        "publication_identity",
        "section_id",
        "section_identity",
    )
    if any(not isinstance(value.get(field), str) or not str(value[field]).strip() for field in required_identities):
        return False
    if not isinstance(value.get("chunk_identity"), Mapping):
        return False
    if not isinstance(value.get("content_length"), int) or isinstance(value.get("content_length"), bool):
        return False
    if not isinstance(value.get("decision_query"), str) or not str(value["decision_query"]).strip():
        return False
    source_relationships = value.get("source_relationships")
    if not isinstance(source_relationships, list) or not source_relationships:
        return False
    if not isinstance(value.get("applicability_conditions"), list) or not value["applicability_conditions"]:
        return False
    return True


def _blocking_reason(value: object) -> str | None:
    if not _has_authorized_shape(value):
        return None
    assert isinstance(value, Mapping)
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    if value.get("lifecycle_state") == "needs_re_review" or metadata.get("review_status") == "needs_re_review":
        return "knowledge_needs_review"
    if value.get("known_contradiction") is True or metadata.get("evidence_conflict") == "unresolved":
        return "material_evidence_conflict"
    return None


def _insufficient(query_conditions: QueryConditionSet, reason: str) -> EvidenceSufficiencyDecision:
    return EvidenceSufficiencyDecision(
        query_conditions=query_conditions,
        evidence_set=None,
        insufficient_reply=InsufficientEvidenceReply(
            reason=reason,
            query_condition_set_identity=query_conditions.identity,
        ),
    )


def _authorized_candidate(value: object, *, max_excerpt_chars: int) -> _AuthorizedCandidate | None:
    if not _has_authorized_shape(value):
        return None
    assert isinstance(value, Mapping)
    source_relationships = value.get("source_relationships")
    assert isinstance(source_relationships, list)
    for relationship in source_relationships:
        if (
            not isinstance(relationship, Mapping)
            or not _stable_identity_has_kind(relationship.get("source_identity"), StableIdentityKind.SOURCE)
            or relationship.get("availability") != "verified_usable"
            or relationship.get("access_scope") not in {"public", "controlled_internal"}
        ):
            return None
    applicability = value.get("applicability_conditions")
    assert isinstance(applicability, list)
    assurance_level = value.get("assurance_level")
    metadata = value.get("metadata")
    if assurance_level not in {"source_grounded", "claim_linked", "release_assured"} or not isinstance(metadata, Mapping):
        return None
    decision_query = value.get("decision_query")
    if not isinstance(decision_query, str) or not decision_query.strip() or metadata.get("decision_query") != decision_query:
        return None
    evidence = AnswerEvidence.from_candidate(value, max_excerpt_chars=max_excerpt_chars)
    if evidence is None or not evidence.is_agent_entry() or not evidence.section_id:
        return None
    item_identity_binding = _candidate_item_binding(value, evidence)
    if item_identity_binding is None:
        return None
    item_identity = canonical_json_sha256(item_identity_binding)
    frozen_conditions = tuple(_freeze_mapping(item) for item in applicability if isinstance(item, Mapping))
    if len(frozen_conditions) != len(applicability):
        return None
    return _AuthorizedCandidate(
        evidence=evidence,
        entry_id=str(value["entry_id"]).strip(),
        section_id=str(value["section_id"]).strip(),
        item_identity=item_identity,
        item_identity_binding=_freeze_mapping(item_identity_binding),
        entry_identity=str(value["entry_identity"]).strip(),
        decision_query=decision_query.strip(),
        source_content_length=int(value["content_length"]),
        applicability_conditions=frozen_conditions,
        assurance_level=str(assurance_level),
        metadata=_freeze_mapping(metadata),
    )


def _authorized_candidate_variants(
    value: object,
    *,
    max_excerpt_chars: int,
) -> tuple[_AuthorizedCandidate, ...]:
    base = _authorized_candidate(value, max_excerpt_chars=max_excerpt_chars)
    if base is None:
        return ()
    if not isinstance(value, Mapping):
        return ()
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        return ()
    projections = metadata.get("source_evidence_projections")
    if projections is None:
        return (base,)
    if base.assurance_level != "claim_linked" or not isinstance(projections, list):
        return ()
    raw_relationships = value.get("source_relationships")
    if not isinstance(raw_relationships, list):
        return ()
    relationship_scopes: dict[str, str] = {}
    for relationship in raw_relationships:
        if not isinstance(relationship, Mapping):
            return ()
        source_identity = relationship.get("source_identity")
        access_scope = relationship.get("access_scope")
        if (
            not _stable_identity_has_kind(source_identity, StableIdentityKind.SOURCE)
            or access_scope not in {"public", "controlled_internal"}
            or source_identity in relationship_scopes
        ):
            return ()
        relationship_scopes[str(source_identity)] = str(access_scope)
    chunk_id = value.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        return ()
    variants: list[_AuthorizedCandidate] = []
    seen_source_identities: set[str] = set()
    for projection in projections:
        if not isinstance(projection, Mapping) or set(projection) != _SOURCE_EVIDENCE_PROJECTION_FIELDS:
            return ()
        source_identity = projection.get("source_identity")
        source_id = projection.get("source_id")
        if (
            not _stable_identity_has_kind(source_identity, StableIdentityKind.SOURCE)
            or not isinstance(source_id, str)
            or not source_id.strip()
            or source_identity != f"source:{source_id}"
            or projection.get("source_tier") not in _SOURCE_TIERS
            or projection.get("source_access_scope") != relationship_scopes.get(str(source_identity))
            or any(
                not isinstance(projection.get(field), str) or not str(projection[field]).strip()
                for field in (
                    "source_title",
                    "source_authority",
                    "source_url",
                    "source_version",
                    "source_review_date",
                )
            )
            or str(source_identity) in seen_source_identities
        ):
            return ()
        seen_source_identities.add(str(source_identity))
        projected_metadata = dict(metadata)
        projected_metadata.update(projection)
        projected_metadata["candidate_evidence_source_identity"] = source_identity
        projected_value = dict(value)
        projected_value["chunk_id"] = f"{chunk_id}@{source_identity}"
        projected_value["metadata"] = projected_metadata
        variant = _authorized_candidate(projected_value, max_excerpt_chars=max_excerpt_chars)
        if variant is None:
            return ()
        variants.append(variant)
    if seen_source_identities != set(relationship_scopes):
        return ()
    return tuple(variants)


def _claim_evidence_requirements(candidate: _AuthorizedCandidate) -> tuple[_EvidenceRequirement, ...] | str:
    raw_contract = candidate.metadata.get("claim_evidence_contract")
    raw_hash = candidate.metadata.get("claim_evidence_contract_sha256")
    source_id = candidate.metadata.get("source_id")
    if not isinstance(raw_contract, str) or not raw_contract or not isinstance(raw_hash, str) or not raw_hash:
        return "assurance_support_missing"
    if not isinstance(source_id, str) or not source_id:
        return "assurance_support_missing"
    try:
        raw_value = json.loads(raw_contract)
        candidate_contract = parse_candidate_claim_evidence_contract(raw_value)
    except (json.JSONDecodeError, CandidateClaimEvidenceContractError):
        candidate_contract = None
    if candidate_contract is not None:
        if (
            candidate_contract.sha256 != raw_hash
            or candidate_contract.entry_identity != candidate.entry_identity
            or candidate_contract.editorial_revision_identity
            != candidate.metadata.get("editorial_revision_identity")
        ):
            return "assurance_support_missing"
        matched_claims = [
            claim
            for claim in candidate_contract.claims
            if claim.section_id == candidate.section_id and source_id in claim.source_ids
        ]
        if not matched_claims:
            return "assurance_support_missing"
        return tuple(
            sorted(
                {
                    _EvidenceRequirement(
                        entry_id=candidate.entry_id,
                        section_id=claim.section_id,
                        source_id=linked_source_id,
                    )
                    for claim in matched_claims
                    for linked_source_id in claim.source_ids
                },
                key=_requirement_sort_key,
            )
        )
    try:
        contract = parse_claim_evidence_contract(raw_value)
    except ClaimEvidenceContractError:
        return "assurance_support_missing"
    if contract.sha256 != raw_hash:
        return "assurance_support_missing"
    if contract.conflict_state == "unresolved" or contract.unknown_state != "none":
        return "material_evidence_conflict"
    matched_claims = [
        claim
        for claim in contract.claims
        if any(link.section_id == candidate.section_id and link.source_id == source_id for link in claim.evidence)
    ]
    if not matched_claims:
        return "assurance_support_missing"
    return tuple(
        sorted(
            {
                _EvidenceRequirement(
                    entry_id=candidate.entry_id,
                    section_id=link.section_id,
                    source_id=link.source_id,
                )
                for claim in matched_claims
                for link in claim.evidence
            },
            key=_requirement_sort_key,
        )
    )


def _release_assurance_reason(candidate: _AuthorizedCandidate) -> str | None:
    snapshot = candidate.metadata.get("release_assurance_snapshot")
    if not isinstance(snapshot, Mapping):
        return "assurance_support_missing"
    if (
        snapshot.get("schema") != "editorial_release_assurance_snapshot/v1"
        or snapshot.get("entry_identity") != candidate.entry_identity
    ):
        return "assurance_support_missing"
    required_records = {
        "contract_identity": (
            {StableIdentityKind.PRODUCT_PATH, StableIdentityKind.PRODUCT_REVISION},
            {"authoritative", "immutable"},
        ),
        "calibration_identity": (
            {StableIdentityKind.CONFIGURATION, StableIdentityKind.CAPABILITY},
            {"authoritative", "immutable"},
        ),
        "frozen_acceptance_identity": (
            {StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD},
            {"immutable"},
        ),
        "named_gate": (
            {StableIdentityKind.CAPABILITY},
            {"authoritative", "immutable"},
        ),
    }
    records = snapshot.get("records")
    if not isinstance(records, (list, tuple)) or len(records) != len(required_records):
        return "assurance_support_missing"
    seen_fields: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            return "assurance_support_missing"
        field = record.get("field")
        identity_value = record.get("identity")
        if (
            not isinstance(field, str)
            or field in seen_fields
            or field not in required_records
            or not isinstance(identity_value, str)
        ):
            return "assurance_support_missing"
        try:
            identity = StableIdentity.from_stable_id(identity_value)
        except ValueError:
            return "assurance_support_missing"
        allowed_kinds, allowed_classes = required_records[field]
        if (
            identity.kind not in allowed_kinds
            or record.get("record_class") not in allowed_classes
            or not isinstance(record.get("payload_sha256"), str)
            or _SHA256.fullmatch(str(record["payload_sha256"])) is None
        ):
            return "assurance_support_missing"
        seen_fields.add(field)
    if seen_fields != set(required_records):
        return "assurance_support_missing"
    acceptance_status = snapshot.get("frozen_acceptance_status")
    if not isinstance(acceptance_status, Mapping):
        return "assurance_support_missing"
    if (
        not _stable_identity_has_kind(acceptance_status.get("event_id"), StableIdentityKind.EVENT)
        or not isinstance(acceptance_status.get("event_sha256"), str)
        or _SHA256.fullmatch(str(acceptance_status["event_sha256"])) is None
        or acceptance_status.get("to_state") != "active"
    ):
        return "assurance_support_missing"
    return None


def _assurance_requirements(candidate: _AuthorizedCandidate) -> tuple[_EvidenceRequirement, ...] | str:
    if candidate.assurance_level == "source_grounded":
        return ()
    if candidate.assurance_level == "claim_linked":
        return _claim_evidence_requirements(candidate)
    reason = _release_assurance_reason(candidate)
    return reason if reason is not None else ()


def _coverage_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for match in _COVERAGE_TOKEN.findall(value.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]{2,}", match) is not None:
            terms.update(match[index : index + 2] for index in range(len(match) - 1))
        else:
            normalized = match.strip("._-")
            if normalized and normalized not in _COVERAGE_STOPWORDS:
                terms.add(normalized)
    return terms


def _decision_query_covers(question: str, candidate: _AuthorizedCandidate) -> bool:
    return len(_coverage_terms(question) & _coverage_terms(candidate.decision_query)) >= 2


def _candidate_sort_key(candidate: _AuthorizedCandidate) -> tuple[str, str, str]:
    return (candidate.entry_id, candidate.section_id, candidate.item_identity)


def _requirement_sort_key(requirement: _EvidenceRequirement) -> tuple[str, str, str]:
    return (requirement.entry_id or "", requirement.section_id, requirement.source_id or "")


def _candidate_source_id(candidate: _AuthorizedCandidate) -> str | None:
    source_id = candidate.metadata.get("source_id")
    return source_id if isinstance(source_id, str) and source_id else None


def _candidate_satisfies_requirement(candidate: _AuthorizedCandidate, requirement: _EvidenceRequirement) -> bool:
    return (
        candidate.section_id == requirement.section_id
        and (requirement.entry_id is None or candidate.entry_id == requirement.entry_id)
        and (requirement.source_id is None or _candidate_source_id(candidate) == requirement.source_id)
    )


def _candidate_options(
    requirement: _EvidenceRequirement,
    *,
    applicable: Sequence[_AuthorizedCandidate],
    governing: _AuthorizedCandidate,
) -> tuple[_AuthorizedCandidate, ...]:
    matching = [
        candidate
        for candidate in applicable
        if _candidate_satisfies_requirement(candidate, requirement)
    ]
    return tuple(
        sorted(
            matching,
            key=lambda candidate: (
                0 if requirement.entry_id is None and candidate.entry_id == governing.entry_id else 1,
                _candidate_sort_key(candidate),
            ),
        )
    )


def _present_selected_candidates(
    governing: _AuthorizedCandidate,
    selected: Sequence[_AuthorizedCandidate],
    *,
    required_sections: Sequence[str],
) -> tuple[_AuthorizedCandidate, ...]:
    section_priority = {section_id: index for index, section_id in enumerate(required_sections)}
    support = [candidate for candidate in selected if candidate.item_identity != governing.item_identity]
    return (
        governing,
        *sorted(
            support,
            key=lambda candidate: (
                section_priority.get(candidate.section_id, len(section_priority)),
                candidate.section_id,
                candidate.item_identity,
            ),
        ),
    )


def _viable_plans_for_governing(
    governing: _AuthorizedCandidate,
    *,
    applicable: Sequence[_AuthorizedCandidate],
    required_sections: Sequence[str],
    item_cap: int,
    excerpt_cap: int,
    total_cap: int,
) -> tuple[tuple[tuple[_AuthorizedCandidate, ...], ...], frozenset[str]]:
    initial_requirements = tuple(_EvidenceRequirement(section_id=section_id) for section_id in required_sections)
    plans: dict[tuple[str, ...], tuple[_AuthorizedCandidate, ...]] = {}
    failures: set[str] = set()

    def add_requirement(
        requirements: tuple[_EvidenceRequirement, ...],
        requirement: _EvidenceRequirement,
    ) -> tuple[_EvidenceRequirement, ...]:
        return requirements if requirement in requirements else (*requirements, requirement)

    def visit(
        selected: tuple[_AuthorizedCandidate, ...],
        requirements: tuple[_EvidenceRequirement, ...],
    ) -> None:
        expanded = requirements
        for candidate in selected:
            assurance = _assurance_requirements(candidate)
            if isinstance(assurance, str):
                failures.add(assurance)
                return
            for requirement in assurance:
                expanded = add_requirement(expanded, requirement)

        if any(candidate.source_content_length > excerpt_cap for candidate in selected):
            failures.add("evidence_budget_exceeded")
            return
        if sum(len(candidate.evidence.excerpt) for candidate in selected) > total_cap:
            failures.add("evidence_budget_exceeded")
            return

        unsatisfied = [
            requirement
            for requirement in expanded
            if not any(_candidate_satisfies_requirement(candidate, requirement) for candidate in selected)
        ]
        if not unsatisfied:
            presented = _present_selected_candidates(
                governing,
                selected,
                required_sections=required_sections,
            )
            plans[tuple(sorted(candidate.item_identity for candidate in presented))] = presented
            return
        if len(selected) >= item_cap:
            failures.add("evidence_budget_exceeded")
            return

        requirement = min(unsatisfied, key=_requirement_sort_key)
        options = _candidate_options(requirement, applicable=applicable, governing=governing)
        if not options:
            failures.add("assurance_support_missing" if requirement.source_id is not None else "decision_not_covered")
            return
        selected_identities = {candidate.item_identity for candidate in selected}
        for candidate in options:
            if candidate.item_identity not in selected_identities:
                visit((*selected, candidate), expanded)

    visit((governing,), initial_requirements)
    return tuple(plans.values()), frozenset(failures)


def decide_answer_evidence(
    *,
    normalized_question: str,
    query_conditions: QueryConditionSet,
    candidates: Sequence[object],
    max_items: int = _MAX_ITEMS,
    max_excerpt_chars: int = _MAX_EXCERPT_CHARS,
    max_total_chars: int = _MAX_TOTAL_CHARS,
) -> EvidenceSufficiencyDecision:
    """Return an immutable set only when deterministic reviewed rules suffice."""

    if normalized_question.strip() != query_conditions.normalized_question:
        raise ValueError("the normalized question must match its Query Condition Set")
    if max_items < 1 or max_excerpt_chars < 1 or max_total_chars < 1:
        raise ValueError("evidence budgets must be positive")
    effective_item_cap = min(max_items, _MAX_ITEMS)
    effective_excerpt_cap = min(max_excerpt_chars, _MAX_EXCERPT_CHARS)
    effective_total_cap = min(max_total_chars, _MAX_TOTAL_CHARS)

    blocking_reasons = {reason for value in candidates if (reason := _blocking_reason(value)) is not None}
    if "material_evidence_conflict" in blocking_reasons:
        return _insufficient(query_conditions, "material_evidence_conflict")
    if "knowledge_needs_review" in blocking_reasons:
        return _insufficient(query_conditions, "knowledge_needs_review")

    authorized = [
        candidate
        for value in candidates
        for candidate in _authorized_candidate_variants(value, max_excerpt_chars=effective_excerpt_cap)
    ]
    if not authorized:
        return _insufficient(query_conditions, "no_eligible_published_evidence")

    canonical_authorized: dict[str, _AuthorizedCandidate] = {}
    for candidate in sorted(authorized, key=_candidate_sort_key):
        canonical_authorized.setdefault(candidate.item_identity, candidate)

    applicable: list[_AuthorizedCandidate] = []
    condition_missing = False
    for candidate in canonical_authorized.values():
        matches = [query_conditions.matches(condition) for condition in candidate.applicability_conditions]
        has_unknown_condition = any(
            match is None or not query_conditions.has_field_operator(condition)
            for match, condition in zip(matches, candidate.applicability_conditions, strict=True)
        )
        if has_unknown_condition:
            condition_missing = True
            continue
        if all(match is True for match in matches):
            applicable.append(candidate)
    if not applicable:
        return _insufficient(
            query_conditions,
            "decisive_condition_missing" if condition_missing else "decision_not_covered",
        )

    governing_candidates = [
        candidate
        for candidate in applicable
        if candidate.section_id == _GOVERNING_SECTION and _decision_query_covers(normalized_question, candidate)
    ]
    if not governing_candidates:
        return _insufficient(query_conditions, "decision_not_covered")

    required_sections = _required_sections(normalized_question)
    viable_plans: list[tuple[_AuthorizedCandidate, ...]] = []
    failure_reasons: set[str] = set()
    for governing in sorted(governing_candidates, key=_candidate_sort_key):
        plans, failures = _viable_plans_for_governing(
            governing,
            applicable=applicable,
            required_sections=required_sections,
            item_cap=effective_item_cap,
            excerpt_cap=effective_excerpt_cap,
            total_cap=effective_total_cap,
        )
        viable_plans.extend(plans)
        failure_reasons.update(failures)

    if not viable_plans:
        for reason in (
            "material_evidence_conflict",
            "assurance_support_missing",
            "evidence_budget_exceeded",
            "decision_not_covered",
        ):
            if reason in failure_reasons:
                return _insufficient(query_conditions, reason)
        return _insufficient(query_conditions, "decision_not_covered")

    selected_candidates = min(
        viable_plans,
        key=lambda plan: (
            len(plan),
            sum(len(candidate.evidence.excerpt) for candidate in plan),
            tuple(sorted(candidate.item_identity for candidate in plan)),
        ),
    )
    governing = selected_candidates[0]
    selected = tuple(candidate.evidence for candidate in selected_candidates)
    return EvidenceSufficiencyDecision(
        query_conditions=query_conditions,
        evidence_set=AnswerEvidenceSet.freeze(
            query_conditions=query_conditions,
            items=selected,
            governing_item=governing.evidence,
            item_identities=tuple(candidate.item_identity for candidate in selected_candidates),
            item_identity_bindings=tuple(candidate.item_identity_binding for candidate in selected_candidates),
        ),
        insufficient_reply=None,
    )
