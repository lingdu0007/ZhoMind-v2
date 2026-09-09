from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.canonical import (
    CoveragePosition,
    KnowledgeAssuranceLevel,
    KnowledgeSourceTier,
    SourceAccessScope,
    SourceAvailabilityState,
    StableIdentity,
    StableIdentityKind,
)
from app.contracts.claim_materiality import content_indicates_high_impact, is_high_impact_claim, is_material_claim
from app.documents.content_admission import credential_findings, recognizable_private_material, source_admission_allowed
from app.documents.source_urls import is_canonical_public_source_url
from app.rag.claim_evidence import ClaimEvidenceContractError, parse_claim_evidence_contract
from app.rag.evidence_sufficiency import QueryConditionSet

_COVERAGE_POSITIONS = frozenset(position.value for position in CoveragePosition)
_ASSURANCE_LEVELS = frozenset(level.value for level in KnowledgeAssuranceLevel)
_SOURCE_TIERS = frozenset(tier.value for tier in KnowledgeSourceTier)
_ACCESS_SCOPES = frozenset(scope.value for scope in SourceAccessScope)
_ENTRY_ID = r"^[a-z0-9][a-z0-9._:-]{2,159}$"
_BODY_SECTIONS = (
    "decision_query",
    "recommendation_or_reviewed_branches",
    "applicability",
    "non_applicability",
    "alternatives",
    "trade_offs",
    "failure_modes",
    "minimum_implementation_guidance",
    "minimum_validation_guidance",
    "minimum_diagnosis_guidance",
    "minimum_acceptance_guidance",
    "conflicts",
    "unknowns",
    "boundary_conditions",
)
_SENSITIVE_QUERY_PARTS = ("credential", "password", "secret", "signature", "signed", "token")
_REDIRECT_QUERY_KEYS = {"continue", "next", "redirect", "redirect_uri", "return_to", "target", "url"}
_CONTROLLED_LOCATOR = re.compile(r"^controlled://[a-z0-9][a-z0-9._/-]{2,159}$")
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "access_token",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)
_SECRET_FIELD_SUFFIXES = ("apikey", "accesstoken", "authorization", "credential", "credentials", "password", "secret", "token")
_AUTOMATIC_PUBLICATION_FIELD_NAMES = frozenset(
    {"automaticpublication", "autopublish", "publishautomatically", "automaticallypublish"}
)
_AUTOMATIC_PUBLICATION_TEXT = re.compile(
    r"""
    \b(?:automatic\s+publication|auto[- ]?publish(?:ing|ed)?|automatically\s+publish(?:ing|ed)?)\b
    |(?:^|[.!?]\s*)publish(?:\s+(?:this|the|an?|approved|editorial|export|artifact)){0,5}\s+
      (?:to\s+)?(?:production|prod)\b
    |(?:^|[.!?]\s*)deploy(?:\s+(?:this|the|an?|approved|editorial|export|artifact)){0,5}\s+
      (?:to\s+)?(?:production|prod)\b
    |(?:自动(?:化)?发布|自动\s*发布|发布(?:本(?:次|个)|此)?(?:导出|工件)?到生产(?:环境)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)


class CreateEditorialEntryRequest(BaseModel):
    """A Draft may be incomplete; review transitions validate the full contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    entry_id: str = Field(min_length=3, max_length=160, pattern=_ENTRY_ID)
    title: str = Field(min_length=1, max_length=240)
    coverage_position: str | None = None
    assurance_level: str | None = None
    approving_reviewer_username: str | None = None
    accountable_maintainer_username: str | None = None
    review_date: str | None = None
    applicable_versions: list[str] | None = None
    applicability_conditions: list[dict[str, Any]] | None = None
    non_applicability_conditions: list[dict[str, Any]] | None = None
    freshness_triggers: list[dict[str, Any]] | None = None
    sources: list[dict[str, Any]] | None = None
    chunk_strategy: dict[str, Any] | None = None
    acceptance_material: dict[str, Any] | None = None
    body: dict[str, Any] | None = None
    section_source_relationships: list[dict[str, Any]] | None = None
    claims: list[dict[str, Any]] | None = None
    claim_evidence_contract: dict[str, Any] | None = None
    relationship: dict[str, Any] | None = None
    release_assurance: dict[str, Any] | None = None


class ReviseEditorialEntryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: CreateEditorialEntryRequest
    change_kind: str
    lightweight_reason: str | None = None


class RecordSourceAvailabilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    availability: str = Field(min_length=1, max_length=80)


def review_validation_reasons(
    payload: CreateEditorialEntryRequest,
    *,
    allow_legacy_claim_linked_contract: bool = False,
) -> list[dict[str, str]]:
    required_values = {
        "coverage_position": payload.coverage_position,
        "assurance_level": payload.assurance_level,
        "approving_reviewer_username": payload.approving_reviewer_username,
        "accountable_maintainer_username": payload.accountable_maintainer_username,
        "review_date": payload.review_date,
        "applicable_versions": payload.applicable_versions,
        "applicability_conditions": payload.applicability_conditions,
        "non_applicability_conditions": payload.non_applicability_conditions,
        "freshness_triggers": payload.freshness_triggers,
        "sources": payload.sources,
        "chunk_strategy": payload.chunk_strategy,
        "acceptance_material": payload.acceptance_material,
        "body": payload.body,
        "section_source_relationships": payload.section_source_relationships,
        "claims": payload.claims,
    }
    reasons = [
        {"field": field, "code": "required", "message": "field is required before editorial review"}
        for field, value in required_values.items()
        if value in (None, "", [], {})
    ]
    if reasons:
        return reasons

    if payload.coverage_position not in _COVERAGE_POSITIONS:
        reasons.append(
            {
                "field": "coverage_position",
                "code": "unsupported_coverage_position",
                "message": "coverage_position must name one of the eight required operating positions",
            }
        )
    if payload.assurance_level not in _ASSURANCE_LEVELS:
        reasons.append(
            {
                "field": "assurance_level",
                "code": "unsupported_assurance_level",
                "message": "assurance_level must be source_grounded, claim_linked, or release_assured",
            }
        )
    if not _is_iso_date(payload.review_date):
        reasons.append(
            {
                "field": "review_date",
                "code": "invalid_date",
                "message": "review_date must be an ISO-8601 date",
            }
        )
    reasons.extend(_validate_versions(payload.applicable_versions))
    reasons.extend(_validate_conditions("applicability_conditions", payload.applicability_conditions))
    reasons.extend(_validate_conditions("non_applicability_conditions", payload.non_applicability_conditions))
    reasons.extend(_validate_freshness_triggers(payload.freshness_triggers))
    reasons.extend(_validate_chunk_strategy(payload.chunk_strategy))
    reasons.extend(_validate_acceptance_material(payload.acceptance_material))
    reasons.extend(_validate_body(payload.body))

    source_reasons, source_tiers = _validate_sources(payload.sources)
    reasons.extend(source_reasons)
    reasons.extend(_validate_section_sources(payload.section_source_relationships, source_tiers))
    reasons.extend(_validate_claims(payload.claims, payload.assurance_level, source_tiers, payload.body))
    reasons.extend(_validate_claim_evidence_contract(payload))
    reasons.extend(_validate_relationship(payload.relationship))
    reasons.extend(_validate_release_assurance(payload.assurance_level, payload.release_assurance))
    return reasons


def editorial_secret_scan_findings(value: object) -> list[str]:
    findings: set[str] = set()

    for path, item, field_name in _walk_values(value):
        if _is_secret_field_name(field_name) and _has_nonempty_value(item):
            findings.add(f"{path}:secret-bearing-field")
        if not isinstance(item, str):
            continue
        if recognizable_private_material(item):
            findings.add(f"{path}:private-material")
        for name in credential_findings(item):
            findings.add(f"{path}:{name}")
    return sorted(findings)


def editorial_export_safety_findings(value: object) -> list[str]:
    findings = set(editorial_secret_scan_findings(value))

    for path, item, field_name in _walk_values(value):
        normalized_key = _normalized_field_name(field_name)
        if _is_automatic_publication_field_name(normalized_key):
            findings.add(f"{path}:automatic-publication-instruction")
        if isinstance(item, str) and _AUTOMATIC_PUBLICATION_TEXT.search(item):
            findings.add(f"{path}:automatic-publication-instruction")

    return sorted(findings)


def lightweight_revision_reasons(
    previous: CreateEditorialEntryRequest,
    candidate: CreateEditorialEntryRequest,
) -> list[dict[str, str]]:
    """Allow only whitespace normalization in title and authored body text."""

    previous_data = previous.model_dump(mode="json")
    candidate_data = candidate.model_dump(mode="json")
    reasons: list[dict[str, str]] = []
    for field in sorted((set(previous_data) | set(candidate_data)) - {"title", "body"}):
        if previous_data.get(field) != candidate_data.get(field):
            reasons.append(
                {
                    "field": field,
                    "code": "material_change",
                    "message": "wording-only revisions cannot change structured editorial authority data",
                }
            )
    if not _same_lightweight_text(previous.title, candidate.title):
        reasons.append(
            {
                    "field": "title",
                    "code": "material_change",
                    "message": "wording-only revisions may only normalize title whitespace",
            }
        )
    previous_body = previous.body if isinstance(previous.body, dict) else {}
    candidate_body = candidate.body if isinstance(candidate.body, dict) else {}
    for section in sorted(set(previous_body) | set(candidate_body)):
        before = previous_body.get(section)
        after = candidate_body.get(section)
        if not isinstance(before, str) or not isinstance(after, str) or not _same_lightweight_text(before, after):
            reasons.append(
                {
                    "field": f"body.{section}",
                    "code": "material_change",
                    "message": "wording-only revisions may only normalize body whitespace",
                }
            )
    return reasons


def _validate_versions(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [{"field": "applicable_versions", "code": "invalid", "message": "applicable_versions must be a non-empty list"}]
    reasons = []
    for index, version in enumerate(value):
        if not isinstance(version, str) or not version.strip():
            reasons.append(
                {
                    "field": f"applicable_versions[{index}]",
                    "code": "invalid",
                    "message": "each applicable version must be a non-empty string",
                }
            )
    return reasons


def _validate_conditions(field: str, value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [{"field": field, "code": "invalid", "message": f"{field} must be a non-empty list"}]
    reasons: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, condition in enumerate(value):
        prefix = f"{field}[{index}]"
        if not isinstance(condition, dict):
            reasons.append({"field": prefix, "code": "invalid", "message": "condition must be an object"})
            continue
        identifier = condition.get("condition_id")
        if not _stable_identifier(identifier):
            reasons.append(
                {"field": f"{prefix}.condition_id", "code": "invalid", "message": "condition_id must be a stable identifier"}
            )
        elif isinstance(identifier, str) and identifier in seen:
            reasons.append(
                {"field": f"{prefix}.condition_id", "code": "duplicate", "message": "condition_id must be unique within its set"}
            )
        elif isinstance(identifier, str):
            seen.add(identifier)
        if not isinstance(condition.get("field"), str) or not condition["field"].strip():
            reasons.append({"field": f"{prefix}.field", "code": "required", "message": "condition field is required"})
        if condition.get("operator") not in {"equals", "not_equals", "in", "not_in", "gte", "lte", "matches", "present"}:
            reasons.append(
                {
                    "field": f"{prefix}.operator",
                    "code": "invalid",
                    "message": "condition operator is not supported",
                }
            )
        if condition.get("operator") != "present" and condition.get("value") in (None, "", []):
            reasons.append(
                {
                    "field": f"{prefix}.value",
                    "code": "required",
                    "message": "condition value is required for this operator",
                }
            )
    return reasons


def _validate_freshness_triggers(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [{"field": "freshness_triggers", "code": "invalid", "message": "freshness_triggers must be a non-empty list"}]
    reasons: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, trigger in enumerate(value):
        prefix = f"freshness_triggers[{index}]"
        if not isinstance(trigger, dict):
            reasons.append({"field": prefix, "code": "invalid", "message": "freshness trigger must be an object"})
            continue
        identifier = trigger.get("trigger_id")
        if not _stable_identifier(identifier):
            reasons.append(
                {"field": f"{prefix}.trigger_id", "code": "invalid", "message": "trigger_id must be a stable identifier"}
            )
        elif isinstance(identifier, str) and identifier in seen:
            reasons.append(
                {"field": f"{prefix}.trigger_id", "code": "duplicate", "message": "trigger_id must be unique"}
            )
        elif isinstance(identifier, str):
            seen.add(identifier)
        if not isinstance(trigger.get("trigger_type"), str) or not trigger["trigger_type"].strip():
            reasons.append({"field": f"{prefix}.trigger_type", "code": "required", "message": "trigger_type is required"})
        review_within_days = trigger.get("review_within_days")
        if not isinstance(review_within_days, int) or isinstance(review_within_days, bool) or not 1 <= review_within_days <= 3650:
            reasons.append(
                {
                    "field": f"{prefix}.review_within_days",
                    "code": "invalid",
                    "message": "review_within_days must be an integer between 1 and 3650",
                }
            )
    return reasons


def _validate_chunk_strategy(value: object) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return [{"field": "chunk_strategy", "code": "invalid", "message": "chunk_strategy must be an object"}]
    reasons = []
    if not _stable_identifier(value.get("strategy_id")):
        reasons.append(
            {"field": "chunk_strategy.strategy_id", "code": "invalid", "message": "strategy_id must be a stable identifier"}
        )
    maximum = value.get("max_characters")
    overlap = value.get("overlap_characters")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 100 <= maximum <= 20000:
        reasons.append(
            {
                "field": "chunk_strategy.max_characters",
                "code": "invalid",
                "message": "max_characters must be an integer between 100 and 20000",
            }
        )
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0 or (
        isinstance(maximum, int) and overlap >= maximum
    ):
        reasons.append(
            {
                "field": "chunk_strategy.overlap_characters",
                "code": "invalid",
                "message": "overlap_characters must be non-negative and smaller than max_characters",
            }
        )
    if value.get("preserve_section_boundaries") is not True:
        reasons.append(
            {
                "field": "chunk_strategy.preserve_section_boundaries",
                "code": "required",
                "message": "chunk strategy must preserve reviewed section boundaries",
            }
        )
    return reasons


def _validate_acceptance_material(value: object) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return [{"field": "acceptance_material", "code": "invalid", "message": "acceptance_material must be an object"}]
    reasons: list[dict[str, str]] = []
    query_ids: set[str] = set()
    _validate_acceptance_queries(value.get("supported_queries"), "supported_queries", "supported", reasons, query_ids)
    _validate_acceptance_queries(value.get("boundary_queries"), "boundary_queries", "insufficient_evidence", reasons, query_ids)
    return reasons


def _validate_acceptance_queries(
    value: object,
    field: str,
    expected_outcome: str,
    reasons: list[dict[str, str]],
    query_ids: set[str],
) -> None:
    if not isinstance(value, list) or not value:
        reasons.append(
            {
                "field": f"acceptance_material.{field}",
                "code": "required",
                "message": f"{field} must include at least one accepted query",
            }
        )
        return
    for index, query in enumerate(value):
        prefix = f"acceptance_material.{field}[{index}]"
        if not isinstance(query, dict):
            reasons.append({"field": prefix, "code": "invalid", "message": "acceptance query must be an object"})
            continue
        query_id = query.get("query_id")
        if not _stable_identifier(query_id):
            reasons.append({"field": f"{prefix}.query_id", "code": "invalid", "message": "query_id must be a stable identifier"})
        elif isinstance(query_id, str) and query_id in query_ids:
            reasons.append(
                {
                    "field": f"{prefix}.query_id",
                    "code": "duplicate",
                    "message": "query_id must be unique across accepted queries",
                }
            )
        elif isinstance(query_id, str):
            query_ids.add(query_id)
        if not isinstance(query.get("query"), str) or not query["query"].strip():
            reasons.append({"field": f"{prefix}.query", "code": "required", "message": "query is required"})
        elif "query_conditions" in query:
            raw_conditions = query.get("query_conditions")
            if not isinstance(raw_conditions, list):
                reasons.append(
                    {
                        "field": f"{prefix}.query_conditions",
                        "code": "invalid",
                        "message": "query_conditions must be a list of explicit query conditions",
                    }
                )
            else:
                try:
                    QueryConditionSet.from_records(
                        normalized_question=query["query"].strip(),
                        records=raw_conditions,
                    )
                except ValueError:
                    reasons.append(
                        {
                            "field": f"{prefix}.query_conditions",
                            "code": "invalid",
                            "message": "query_conditions must be explicit, complete, and non-duplicated",
                        }
                    )
        if query.get("expected_outcome") != expected_outcome:
            reasons.append(
                {
                    "field": f"{prefix}.expected_outcome",
                    "code": "invalid",
                    "message": f"{field} must use expected_outcome {expected_outcome}",
                }
            )


def _validate_body(value: object) -> list[dict[str, str]]:
    if not isinstance(value, dict):
        return [{"field": "body", "code": "invalid", "message": "body must be an object with every required decision section"}]
    return [
        {
            "field": f"body.{section}",
            "code": "required",
            "message": "each required authored decision section must be non-empty",
        }
        for section in _BODY_SECTIONS
        if not isinstance(value.get(section), str) or not value[section].strip()
    ]


def _validate_sources(value: object) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not isinstance(value, list):
        return [{"field": "sources", "code": "invalid", "message": "sources must be a non-empty list"}], {}
    reasons: list[dict[str, str]] = []
    tiers: dict[str, str] = {}
    for index, source in enumerate(value):
        prefix = f"sources[{index}]"
        if not isinstance(source, dict):
            reasons.append({"field": prefix, "code": "invalid", "message": "source must be an object"})
            continue
        if not source_admission_allowed(source):
            reasons.append({
                "field": f"{prefix}.content_admission", "code": "content_boundary_rejected",
                "message": "source requires reviewed restricted-sensitivity admission for all admitted members",
            })
        source_id = source.get("source_id")
        if not _stable_identifier(source_id):
            reasons.append(
                {"field": f"{prefix}.source_id", "code": "invalid", "message": "source_id must be a stable identifier"}
            )
        elif isinstance(source_id, str) and source_id in tiers:
            reasons.append(
                {"field": f"{prefix}.source_id", "code": "duplicate", "message": "source_id must be unique within an entry"}
            )
        elif isinstance(source_id, str):
            tiers[source_id] = str(source.get("source_tier") or "")
        if source.get("source_tier") not in _SOURCE_TIERS:
            reasons.append(
                {
                    "field": f"{prefix}.source_tier",
                    "code": "invalid",
                    "message": "source_tier must be a defined Knowledge Source Tier",
                }
            )
        for field in ("title", "authority", "version_or_date"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                reasons.append({"field": f"{prefix}.{field}", "code": "required", "message": f"{field} is required"})
        if source.get("availability") not in {state.value for state in SourceAvailabilityState}:
            reasons.append(
                {
                    "field": f"{prefix}.availability",
                    "code": "invalid",
                    "message": "availability must use the canonical source availability vocabulary",
                }
            )
        scope = source.get("access_scope")
        if scope not in _ACCESS_SCOPES:
            reasons.append(
                {
                    "field": f"{prefix}.access_scope",
                    "code": "invalid",
                    "message": "access_scope must be public or controlled_internal",
                }
            )
        elif scope == "public":
            if not _canonical_public_url(source.get("public_url")):
                reasons.append(
                    {
                        "field": f"{prefix}.public_url",
                        "code": "invalid",
                        "message": "public sources require a sanitized canonical public HTTPS URL",
                    }
                )
            if source.get("controlled_locator") not in (None, ""):
                reasons.append(
                    {
                        "field": f"{prefix}.controlled_locator",
                        "code": "forbidden",
                        "message": "public sources cannot carry a controlled internal locator",
                    }
                )
        elif scope == "controlled_internal":
            if not _controlled_locator(source.get("controlled_locator")):
                reasons.append(
                    {
                        "field": f"{prefix}.controlled_locator",
                        "code": "invalid",
                        "message": "controlled internal sources require a sanitized controlled locator",
                    }
                )
            if source.get("public_url") not in (None, ""):
                reasons.append(
                    {
                        "field": f"{prefix}.public_url",
                        "code": "forbidden",
                        "message": "controlled internal sources cannot falsely claim a public URL",
                    }
                )
        if source.get("source_tier") == "bounded_internal_case" and scope != "controlled_internal":
            reasons.append(
                {
                    "field": f"{prefix}.access_scope",
                    "code": "source_tier_scope_invalid",
                    "message": "Bounded Internal Cases require controlled_internal access scope",
                }
            )
    return reasons, tiers


def _validate_section_sources(value: object, source_tiers: dict[str, str]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [
            {
                "field": "section_source_relationships",
                "code": "invalid",
                "message": "section_source_relationships must be a list",
            }
        ]
    reasons: list[dict[str, str]] = []
    relationships: dict[str, list[str]] = {}
    for index, relationship in enumerate(value):
        prefix = f"section_source_relationships[{index}]"
        if not isinstance(relationship, dict):
            reasons.append({"field": prefix, "code": "invalid", "message": "section relationship must be an object"})
            continue
        section_id = relationship.get("section_id")
        source_ids = relationship.get("source_ids")
        if section_id not in _BODY_SECTIONS:
            reasons.append(
                {
                    "field": f"{prefix}.section_id",
                    "code": "invalid",
                    "message": "section_id must name a required authored decision section",
                }
            )
            continue
        if section_id in relationships:
            reasons.append(
                {
                    "field": f"{prefix}.section_id",
                    "code": "duplicate",
                    "message": "each authored decision section has one source relationship record",
                }
            )
            continue
        if not isinstance(source_ids, list) or not source_ids:
            reasons.append(
                {
                    "field": f"{prefix}.source_ids",
                    "code": "required",
                    "message": "each authored decision section requires at least one source relationship",
                }
            )
            continue
        relationships[section_id] = [item for item in source_ids if isinstance(item, str)]
        for source_id in source_ids:
            if source_id not in source_tiers:
                reasons.append(
                    {
                        "field": f"{prefix}.source_ids",
                        "code": "unknown_source",
                        "message": "section source relationship references an unknown source_id",
                    }
                )
    for section_id in _BODY_SECTIONS:
        if section_id not in relationships:
            reasons.append(
                {
                    "field": f"section_source_relationships.{section_id}",
                    "code": "required",
                    "message": "every authored decision section requires a source relationship",
                }
            )
    return reasons


def _validate_claims(
    value: object,
    assurance_level: str | None,
    source_tiers: dict[str, str],
    body: object,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [{"field": "claims", "code": "invalid", "message": "claims must be a list"}]
    reasons: list[dict[str, str]] = []
    material_claim_seen = False
    claim_ids: set[str] = set()
    linked_high_impact_claim_sections: set[str] = set()
    for index, claim in enumerate(value):
        prefix = f"claims[{index}]"
        if not isinstance(claim, dict):
            reasons.append({"field": prefix, "code": "invalid", "message": "claim must be an object"})
            continue
        claim_id = claim.get("claim_id")
        if not _stable_identifier(claim_id):
            reasons.append({"field": f"{prefix}.claim_id", "code": "invalid", "message": "claim_id must be a stable identifier"})
        elif isinstance(claim_id, str) and claim_id in claim_ids:
            reasons.append(
                {
                    "field": f"{prefix}.claim_id",
                    "code": "duplicate",
                    "message": "claim_id must be unique within an entry",
                }
            )
        elif isinstance(claim_id, str):
            claim_ids.add(claim_id)
        kind = claim.get("claim_kind")
        if not isinstance(kind, str) or not kind.strip():
            reasons.append({"field": f"{prefix}.claim_kind", "code": "required", "message": "claim_kind is required"})
            continue
        statement = claim.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            reasons.append({"field": f"{prefix}.statement", "code": "required", "message": "claim statement is required"})
        if claim.get("section_id") not in _BODY_SECTIONS:
            reasons.append(
                {
                    "field": f"{prefix}.section_id",
                    "code": "invalid",
                    "message": "claim section_id must name a required authored decision section",
                }
            )
        source_ids = claim.get("source_ids")
        if not isinstance(source_ids, list):
            source_ids = []
        is_high_impact = is_high_impact_claim(claim)
        is_material = is_material_claim(claim)
        material_claim_seen = material_claim_seen or is_material
        requires_link = is_high_impact or (assurance_level in {"claim_linked", "release_assured"} and is_material)
        if requires_link and not source_ids:
            message = (
                "material high-impact claims require at least one Claim-Evidence Link"
                if is_high_impact
                else "Claim-Linked assurance requires evidence links for material claims"
            )
            reasons.append({"field": f"{prefix}.source_ids", "code": "claim_evidence_required", "message": message})
            continue
        if source_ids:
            tiers = [source_tiers.get(source_id) for source_id in source_ids]
            section_id = claim.get("section_id")
            if (
                isinstance(section_id, str)
                and section_id in _BODY_SECTIONS
                and all(isinstance(source_id, str) and source_id in source_tiers for source_id in source_ids)
                and is_high_impact
            ):
                linked_high_impact_claim_sections.add(section_id)
            if any(tier is None for tier in tiers):
                reasons.append(
                    {
                        "field": f"{prefix}.source_ids",
                        "code": "unknown_source",
                        "message": "Claim-Evidence Link references an unknown source_id",
                    }
                )
            if is_high_impact and tiers and all(tier == "secondary_discovery_source" for tier in tiers):
                reasons.append(
                    {
                        "field": f"{prefix}.source_ids",
                        "code": "source_tier_ineligible",
                        "message": "material high-impact claims cannot rely only on Secondary Discovery Sources",
                    }
                )
            if is_high_impact and "bounded_internal_case" in tiers and claim.get("scope") != "bounded_internal":
                reasons.append(
                    {
                        "field": f"{prefix}.scope",
                        "code": "bounded_case_scope_required",
                        "message": "a Bounded Internal Case can support a high-impact claim only with bounded_internal scope",
                    }
                )
    if assurance_level in {"claim_linked", "release_assured"} and not material_claim_seen:
        reasons.append(
            {
                "field": "claims",
                "code": "material_claim_required",
                "message": "Claim-Linked and Release-Assured entries must record at least one material claim",
            }
        )
    if isinstance(body, dict):
        for section_id in _BODY_SECTIONS:
            if (
                content_indicates_high_impact(body.get(section_id))
                and section_id not in linked_high_impact_claim_sections
            ):
                reasons.append(
                    {
                        "field": f"body.{section_id}",
                        "code": "claim_evidence_required",
                        "message": "high-impact authored decision text requires a Claim-Evidence Link",
                    }
                )
    return reasons


def _validate_claim_evidence_contract(payload: CreateEditorialEntryRequest) -> list[dict[str, str]]:
    value = payload.claim_evidence_contract
    if payload.assurance_level != "claim_linked":
        if value is None:
            return []
        return [
            {
                "field": "claim_evidence_contract",
                "code": "not_applicable",
                "message": "Claim-Evidence contracts are only retained for Claim-Linked assurance",
            }
        ]
    if value is None:
        return []
    if not isinstance(value, dict):
        return [
            {
                "field": "claim_evidence_contract",
                "code": "invalid",
                "message": "when supplied, a Claim-Evidence contract must be valid",
            }
        ]
    try:
        contract = parse_claim_evidence_contract(value)
    except ClaimEvidenceContractError:
        return [
            {
                "field": "claim_evidence_contract",
                "code": "invalid",
                "message": "when supplied, a Claim-Evidence contract must be valid",
            }
        ]

    claims = payload.claims if isinstance(payload.claims, list) else []
    material_claims: dict[str, dict[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = claim.get("claim_id")
        is_material = is_material_claim(claim)
        if isinstance(claim_id, str) and is_material:
            material_claims[claim_id] = claim

    if {claim.claim_id for claim in contract.claims} != set(material_claims):
        return [
            {
                "field": "claim_evidence_contract.claims",
                "code": "claim_set_mismatch",
                "message": "Claim-Evidence contracts must bind exactly the material editorial claims",
            }
        ]

    for contract_claim in contract.claims:
        editorial_claim = material_claims[contract_claim.claim_id]
        section_id = editorial_claim.get("section_id")
        source_ids = editorial_claim.get("source_ids")
        if not isinstance(section_id, str) or not isinstance(source_ids, list):
            return [
                {
                    "field": "claim_evidence_contract.claims",
                    "code": "claim_link_mismatch",
                    "message": "Claim-Evidence contracts must match the editorial claim links",
                }
            ]
        expected_links = {
            (section_id, source_id)
            for source_id in source_ids
            if isinstance(source_id, str)
        }
        contract_links = {(link.section_id, link.source_id) for link in contract_claim.evidence}
        if not expected_links or contract_links != expected_links:
            return [
                {
                    "field": "claim_evidence_contract.claims",
                    "code": "claim_link_mismatch",
                    "message": "Claim-Evidence contracts must match the editorial claim links",
                }
            ]
    return []


def _validate_relationship(value: object) -> list[dict[str, str]]:
    if value in (None, {}):
        return []
    if not isinstance(value, dict):
        return [{"field": "relationship", "code": "invalid", "message": "relationship must be an object"}]
    reasons = []
    for field in ("replaces_entry_identity", "supersedes_entry_identity"):
        identity = value.get(field)
        if identity in (None, ""):
            continue
        try:
            parsed = StableIdentity.from_stable_id(str(identity))
        except ValueError:
            reasons.append(
                {
                    "field": f"relationship.{field}",
                    "code": "invalid",
                    "message": f"{field} must be a canonical entry identity",
                }
            )
            continue
        if parsed.kind is not StableIdentityKind.ENTRY:
            reasons.append(
                {
                    "field": f"relationship.{field}",
                    "code": "invalid",
                    "message": f"{field} must use the entry identity namespace",
                }
            )
    return reasons


def _validate_release_assurance(assurance_level: str | None, value: object) -> list[dict[str, str]]:
    if assurance_level != "release_assured":
        return []
    if not isinstance(value, dict):
        return [
            {
                "field": "release_assurance",
                "code": "required",
                "message": "Release-Assured entries require a frozen assurance contract",
            }
        ]
    reasons = []
    for field in ("contract_identity", "calibration_identity", "frozen_acceptance_identity", "named_gate"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            reasons.append(
                {
                    "field": f"release_assurance.{field}",
                    "code": "required",
                    "message": f"Release-Assured entries require {field}",
                }
            )
    references = (
        ("contract_identity", {StableIdentityKind.PRODUCT_PATH, StableIdentityKind.PRODUCT_REVISION}),
        ("calibration_identity", {StableIdentityKind.CONFIGURATION, StableIdentityKind.CAPABILITY}),
        ("frozen_acceptance_identity", {StableIdentityKind.DELIVERY_ACCEPTANCE_RECORD}),
        ("named_gate", {StableIdentityKind.CAPABILITY}),
    )
    for field, allowed_kinds in references:
        raw_identity = value.get(field)
        if not isinstance(raw_identity, str) or not raw_identity.strip():
            continue
        try:
            identity = StableIdentity.from_stable_id(raw_identity)
        except ValueError:
            identity = None
        if identity is None or identity.kind not in allowed_kinds:
            expected = (
                "delivery_acceptance_record identity namespace"
                if field == "frozen_acceptance_identity"
                else "accepted canonical identity namespace"
            )
            reasons.append(
                {
                    "field": f"release_assurance.{field}",
                    "code": "invalid",
                    "message": f"{field} must use the {expected}",
                }
            )
    return reasons


def _stable_identifier(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        StableIdentity(StableIdentityKind.ENTRY, value)
    except ValueError:
        return False
    return True


def _is_iso_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _canonical_public_url(value: object) -> bool:
    return is_canonical_public_source_url(value)


def _controlled_locator(value: object) -> bool:
    if not isinstance(value, str) or not _CONTROLLED_LOCATOR.fullmatch(value):
        return False
    return not any(part in value.lower() for part in _SENSITIVE_QUERY_PARTS)


def _same_lightweight_text(before: str, after: str) -> bool:
    return re.sub(r"\s+", " ", before).strip() == re.sub(r"\s+", " ", after).strip()


def _normalized_field_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _is_secret_field_name(value: str | None) -> bool:
    normalized = _normalized_field_name(value)
    return normalized in _SECRET_FIELD_NAMES or normalized.endswith(_SECRET_FIELD_SUFFIXES)


def _is_automatic_publication_field_name(normalized: str) -> bool:
    return normalized in _AUTOMATIC_PUBLICATION_FIELD_NAMES or normalized.startswith(
        ("autopublish", "automaticpublication", "publishautomatically", "automaticallypublish")
    )


def _has_nonempty_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return value is not None


def _walk_values(value: object, path: str = "", field_name: str | None = None) -> Iterator[tuple[str, object, str | None]]:
    yield path, value, field_name
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            yield from _walk_values(nested, nested_path, str(key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_values(nested, f"{path}[{index}]", None)
