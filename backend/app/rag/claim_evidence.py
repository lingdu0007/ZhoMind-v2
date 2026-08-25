from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from app.rag.answer_evidence import AnswerEvidence

_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_CONTRACT_KEYS = {
    "schema_version",
    "review_id",
    "review_revision",
    "conflict_state",
    "unknown_state",
    "resolver",
    "claims",
}
_RESOLVER_KEYS = {"resolver_id", "calibration_id", "calibration_version", "minimum_confidence"}
_CLAIM_KEYS = {"claim_id", "scope", "evidence"}
_LINK_KEYS = {"section_id", "source_id"}


class ClaimEvidenceContractError(ValueError):
    pass


@dataclass(frozen=True)
class ClaimEvidenceLink:
    section_id: str
    source_id: str


@dataclass(frozen=True)
class ClaimDefinition:
    claim_id: str
    scope: str
    evidence: tuple[ClaimEvidenceLink, ...]


@dataclass(frozen=True)
class ResolverContract:
    resolver_id: str
    calibration_id: str
    calibration_version: str
    minimum_confidence: float


@dataclass(frozen=True)
class ClaimEvidenceContract:
    schema_version: int
    review_id: str
    review_revision: str
    conflict_state: str
    unknown_state: str
    resolver: ResolverContract
    claims: tuple[ClaimDefinition, ...]

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "review_id": self.review_id,
            "review_revision": self.review_revision,
            "conflict_state": self.conflict_state,
            "unknown_state": self.unknown_state,
            "resolver": {
                "resolver_id": self.resolver.resolver_id,
                "calibration_id": self.resolver.calibration_id,
                "calibration_version": self.resolver.calibration_version,
                "minimum_confidence": self.resolver.minimum_confidence,
            },
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "scope": claim.scope,
                    "evidence": [
                        {"section_id": link.section_id, "source_id": link.source_id} for link in claim.evidence
                    ],
                }
                for claim in self.claims
            ],
        }


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value.strip()) is None:
        raise ClaimEvidenceContractError(f"{field} must be a stable lowercase identifier")
    return value.strip()


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 2000:
        raise ClaimEvidenceContractError(f"{field} must be a non-empty bounded string")
    return value.strip()


def _mapping(value: object, *, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ClaimEvidenceContractError(f"{field} must contain exactly {sorted(keys)}")
    return value


def parse_claim_evidence_contract(value: object) -> ClaimEvidenceContract:
    raw = _mapping(value, field="claim_evidence_contract", keys=_CONTRACT_KEYS)
    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise ClaimEvidenceContractError("claim_evidence_contract.schema_version must be 1")
    conflict_state = raw.get("conflict_state")
    if conflict_state not in {"none", "resolved", "unresolved"}:
        raise ClaimEvidenceContractError("claim_evidence_contract.conflict_state is invalid")
    unknown_state = raw.get("unknown_state")
    if unknown_state not in {"none", "present"}:
        raise ClaimEvidenceContractError("claim_evidence_contract.unknown_state is invalid")

    raw_resolver = _mapping(raw.get("resolver"), field="claim_evidence_contract.resolver", keys=_RESOLVER_KEYS)
    raw_minimum_confidence = raw_resolver.get("minimum_confidence")
    if isinstance(raw_minimum_confidence, bool) or not isinstance(raw_minimum_confidence, (int, float)):
        raise ClaimEvidenceContractError("claim_evidence_contract.resolver.minimum_confidence is invalid")
    minimum_confidence = float(raw_minimum_confidence)
    if not 0 < minimum_confidence < 1:
        raise ClaimEvidenceContractError("claim_evidence_contract.resolver.minimum_confidence must be between zero and one")
    resolver = ResolverContract(
        resolver_id=_safe_id(raw_resolver.get("resolver_id"), field="claim_evidence_contract.resolver.resolver_id"),
        calibration_id=_safe_id(raw_resolver.get("calibration_id"), field="claim_evidence_contract.resolver.calibration_id"),
        calibration_version=_text(
            raw_resolver.get("calibration_version"), field="claim_evidence_contract.resolver.calibration_version"
        ),
        minimum_confidence=minimum_confidence,
    )

    raw_claims = raw.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ClaimEvidenceContractError("claim_evidence_contract.claims must be a non-empty list")
    claims: list[ClaimDefinition] = []
    seen_claim_ids: set[str] = set()
    for index, raw_claim in enumerate(raw_claims):
        claim = _mapping(raw_claim, field=f"claim_evidence_contract.claims[{index}]", keys=_CLAIM_KEYS)
        claim_id = _safe_id(claim.get("claim_id"), field=f"claim_evidence_contract.claims[{index}].claim_id")
        if claim_id in seen_claim_ids:
            raise ClaimEvidenceContractError("claim_evidence_contract.claims contains duplicate claim_id")
        seen_claim_ids.add(claim_id)
        raw_links = claim.get("evidence")
        if not isinstance(raw_links, list) or not raw_links:
            raise ClaimEvidenceContractError(f"claim_evidence_contract.claims[{index}].evidence must be non-empty")
        links: list[ClaimEvidenceLink] = []
        seen_links: set[tuple[str, str]] = set()
        for link_index, raw_link in enumerate(raw_links):
            link = _mapping(
                raw_link,
                field=f"claim_evidence_contract.claims[{index}].evidence[{link_index}]",
                keys=_LINK_KEYS,
            )
            parsed_link = ClaimEvidenceLink(
                section_id=_safe_id(
                    link.get("section_id"),
                    field=f"claim_evidence_contract.claims[{index}].evidence[{link_index}].section_id",
                ),
                source_id=_safe_id(
                    link.get("source_id"),
                    field=f"claim_evidence_contract.claims[{index}].evidence[{link_index}].source_id",
                ),
            )
            key = (parsed_link.section_id, parsed_link.source_id)
            if key in seen_links:
                raise ClaimEvidenceContractError("claim_evidence_contract evidence links must not repeat")
            seen_links.add(key)
            links.append(parsed_link)
        claims.append(
            ClaimDefinition(
                claim_id=claim_id,
                scope=_text(claim.get("scope"), field=f"claim_evidence_contract.claims[{index}].scope"),
                evidence=tuple(links),
            )
        )
    return ClaimEvidenceContract(
        schema_version=schema_version,
        review_id=_safe_id(raw.get("review_id"), field="claim_evidence_contract.review_id"),
        review_revision=_text(raw.get("review_revision"), field="claim_evidence_contract.review_revision"),
        conflict_state=str(conflict_state),
        unknown_state=str(unknown_state),
        resolver=resolver,
        claims=tuple(claims),
    )


def validate_contract_sources(contract: ClaimEvidenceContract, sources: object) -> None:
    if not isinstance(sources, list):
        raise ClaimEvidenceContractError("Agent entry sources must be a list")
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ClaimEvidenceContractError(f"sources[{index}] must be an object")
        source_id = _safe_id(source.get("source_id"), field=f"sources[{index}].source_id")
        if source_id in source_ids:
            raise ClaimEvidenceContractError("Agent entry sources must not reuse source_id")
        source_ids.add(source_id)
        if source.get("availability") != "verified":
            raise ClaimEvidenceContractError(f"sources[{index}].availability must be verified")
        reviewed = source.get("review_date")
        try:
            date.fromisoformat(reviewed.isoformat() if isinstance(reviewed, date) else str(reviewed))
        except ValueError as exc:
            raise ClaimEvidenceContractError(f"sources[{index}].review_date must be an ISO date") from exc
        freshness_days = source.get("freshness_days")
        if isinstance(freshness_days, bool) or not isinstance(freshness_days, int) or not 1 <= freshness_days <= 365:
            raise ClaimEvidenceContractError(f"sources[{index}].freshness_days must be an integer between 1 and 365")
    linked_source_ids = {link.source_id for claim in contract.claims for link in claim.evidence}
    unknown_source_ids = linked_source_ids - source_ids
    if unknown_source_ids:
        raise ClaimEvidenceContractError(f"claim_evidence_contract references unknown source_id {sorted(unknown_source_ids)}")
    section_sources: dict[str, str] = {}
    for claim in contract.claims:
        for link in claim.evidence:
            existing = section_sources.setdefault(link.section_id, link.source_id)
            if existing != link.source_id:
                raise ClaimEvidenceContractError(
                    "claim_evidence_contract maps one section to more than one source_id"
                )


def validate_contract_sections(contract: ClaimEvidenceContract, body: str) -> None:
    """Ensure every reviewed link points to a real Markdown section in this entry."""

    section_ids = {"entry"}
    for line in body.splitlines():
        if line.startswith("#") and re.match(r"^#{1,6}\s+\S", line):
            label = line.lstrip("#").strip().lower()
            section_id = re.sub(r"[^a-z0-9]+", "-", label).strip("-") or "section"
            section_ids.add(section_id)
    missing = sorted(
        {
            link.section_id
            for claim in contract.claims
            for link in claim.evidence
            if link.section_id not in section_ids
        }
    )
    if missing:
        raise ClaimEvidenceContractError(f"claim_evidence_contract references unknown section_id {missing}")


def validate_contract_entry_state(metadata: Mapping[str, object]) -> None:
    if metadata.get("review_status") != "approved":
        raise ClaimEvidenceContractError("claim_evidence_contract requires review_status approved")
    if metadata.get("evidence_conflict") not in {"none", "resolved"}:
        raise ClaimEvidenceContractError(
            "claim_evidence_contract requires evidence_conflict none or resolved"
        )


@dataclass(frozen=True)
class ResolvedClaim:
    entry_id: str
    claim_id: str
    confidence: float


@dataclass(frozen=True)
class ClaimResolution:
    required_claims: tuple[ResolvedClaim, ...]
    out_of_scope: bool
    reason: str


class ClaimResolver(Protocol):
    """Independently deployed, calibrated claim-resolution authority.

    It must be registered by trusted process bootstrap, never derived from a
    request, retrieved metadata, Acceptance Set, or Gold Evidence.
    """

    resolver_id: str
    calibration_id: str
    calibration_version: str

    async def resolve(
        self,
        question: str,
        contracts: Mapping[str, ClaimEvidenceContract],
    ) -> ClaimResolution: ...


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    required_claims: tuple[ResolvedClaim, ...]
    audit: Mapping[str, object]
    protected_evidence: tuple[AnswerEvidence, ...] = ()


def is_unlinked_agent_evidence(metadata: Mapping[str, object]) -> bool:
    """Return True only for agent-entry evidence outside its claim-evidence links.

    Non-agent evidence is never affected. Agent-entry evidence counts as
    unlinked when it carries no verifiable contract or when its
    (section_id, source_id) pair is outside the reviewed links of that
    contract; such passages can never become protected answer evidence.
    """
    if not isinstance(metadata, Mapping):
        return False
    entry_id = metadata.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        return False
    raw_contract = metadata.get("claim_evidence_contract")
    raw_hash = metadata.get("claim_evidence_contract_sha256")
    if not isinstance(raw_contract, str) or not raw_contract or not isinstance(raw_hash, str) or not raw_hash:
        return True
    try:
        contract = parse_claim_evidence_contract(json.loads(raw_contract))
    except (json.JSONDecodeError, ClaimEvidenceContractError):
        return True
    if contract.sha256 != raw_hash:
        return True
    linked = {
        (link.section_id, link.source_id)
        for claim in contract.claims
        for link in claim.evidence
    }
    return (metadata.get("section_id"), metadata.get("source_id")) not in linked


class ClaimEvidenceGate:
    def __init__(self, *, resolver: ClaimResolver) -> None:
        self._resolver = resolver

    @staticmethod
    def _metadata(evidence: AnswerEvidence) -> dict[str, str]:
        return {key: value for key, value in evidence.metadata_items if isinstance(value, str)}

    def _contract_for_evidence(self, evidence: AnswerEvidence) -> tuple[ClaimEvidenceContract, dict[str, str]] | None:
        metadata = self._metadata(evidence)
        raw_contract = metadata.get("claim_evidence_contract")
        raw_hash = metadata.get("claim_evidence_contract_sha256")
        if not raw_contract or not raw_hash:
            return None
        try:
            contract = parse_claim_evidence_contract(json.loads(raw_contract))
        except (json.JSONDecodeError, ClaimEvidenceContractError):
            return None
        if contract.sha256 != raw_hash:
            return None
        return contract, metadata

    @staticmethod
    def _source_is_fresh(metadata: Mapping[str, str]) -> bool:
        if metadata.get("source_availability") != "verified":
            return False
        try:
            reviewed = date.fromisoformat(str(metadata.get("source_review_date") or ""))
            freshness_days = int(str(metadata.get("source_freshness_days") or ""))
        except ValueError:
            return False
        if freshness_days < 1 or freshness_days > 365:
            return False
        age_days = (date.today() - reviewed).days
        return 0 <= age_days <= freshness_days

    @staticmethod
    def _matching_evidence(
        evidence: Iterable[tuple[AnswerEvidence, dict[str, str]]],
        *,
        entry_id: str,
        link: ClaimEvidenceLink,
    ) -> list[tuple[AnswerEvidence, dict[str, str]]]:
        return [
            item
            for item in evidence
            if item[1].get("entry_id") == entry_id
            and item[1].get("section_id") == link.section_id
            and item[1].get("source_id") == link.source_id
        ]

    async def evaluate(self, question: str, evidence: tuple[AnswerEvidence, ...]) -> GateDecision:
        protected_evidence: list[tuple[AnswerEvidence, dict[str, str], ClaimEvidenceContract]] = []
        contracts: dict[str, ClaimEvidenceContract] = {}
        excluded_unlinked = 0
        excluded_unverified = 0
        for item in evidence:
            parsed = self._contract_for_evidence(item)
            if parsed is None:
                # Evidence without a reviewable contract can never enter the
                # protected answer evidence set; it is excluded instead of
                # poisoning the whole request.
                excluded_unverified += 1
                continue
            contract, metadata = parsed
            entry_id = metadata.get("entry_id")
            if not entry_id or contract.conflict_state == "unresolved" or contract.unknown_state != "none":
                return GateDecision(False, "reject_claim_contract_conflict", (), {"contract_count": 0})
            existing = contracts.get(entry_id)
            if existing is not None and existing.sha256 != contract.sha256:
                return GateDecision(False, "reject_claim_contract_ambiguous", (), {"contract_count": 0})
            contracts[entry_id] = contract
            expected_links = {
                (link.section_id, link.source_id)
                for claim in contract.claims
                for link in claim.evidence
            }
            if (metadata.get("section_id"), metadata.get("source_id")) not in expected_links:
                # Sections outside the reviewed claim-evidence links stay out
                # of the protected answer evidence set; the claim coverage
                # checks below still fail closed when support is missing.
                excluded_unlinked += 1
                continue
            protected_evidence.append((item, metadata, contract))

        if not protected_evidence:
            if excluded_unlinked:
                return GateDecision(False, "reject_claim_evidence_unlinked", (), {"contract_count": len(contracts)})
            if excluded_unverified:
                return GateDecision(False, "reject_claim_contract_invalid", (), {"contract_count": 0})
            return GateDecision(False, "reject_claim_contract_missing", (), {"contract_count": 0})
        contract_audit = [
            {
                "entry_id": entry_id,
                "review_id": contract.review_id,
                "review_revision": contract.review_revision,
                "sha256": contract.sha256,
            }
            for entry_id, contract in sorted(contracts.items())
        ]
        expected_resolvers = {
            (contract.resolver.resolver_id, contract.resolver.calibration_id, contract.resolver.calibration_version)
            for contract in contracts.values()
        }
        actual_resolver = (self._resolver.resolver_id, self._resolver.calibration_id, self._resolver.calibration_version)
        if expected_resolvers != {actual_resolver}:
            return GateDecision(
                False,
                "reject_claim_resolver_unavailable",
                (),
                {"contract_count": len(contracts), "contracts": contract_audit},
            )

        audit: dict[str, object] = {
            "resolver_id": self._resolver.resolver_id,
            "calibration_id": self._resolver.calibration_id,
            "calibration_version": self._resolver.calibration_version,
            "contract_hashes": sorted(contract.sha256 for contract in contracts.values()),
            "contracts": contract_audit,
            "required_claims": [],
            "required_evidence_links": [],
            "covered_snapshot_ids": [],
            "excluded_evidence_counts": {
                "unlinked": excluded_unlinked,
                "without_contract": excluded_unverified,
            },
            "protected_evidence_count": len(protected_evidence),
        }
        artifact_identity = {
            key: value
            for key, value in {
                "resolver_profile_sha256": getattr(self._resolver, "profile_sha256", None),
                "calibration_set_sha256": getattr(self._resolver, "calibration_set_sha256", None),
                "calibration_report_sha256": getattr(self._resolver, "calibration_report_sha256", None),
                "embedding_contract_fingerprint": getattr(
                    self._resolver,
                    "embedding_contract_fingerprint",
                    None,
                ),
            }.items()
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        }
        audit.update(artifact_identity)
        try:
            resolution = await self._resolver.resolve(question, contracts)
        except Exception:
            # Resolver failures are deliberately indistinguishable at this
            # boundary: preserving the provider error could expose internals.
            return GateDecision(False, "reject_claim_resolution_unavailable", (), audit)
        if not isinstance(resolution, ClaimResolution):
            return GateDecision(False, "reject_claim_resolution_invalid", (), audit)
        if (
            not isinstance(resolution.out_of_scope, bool)
            or not isinstance(resolution.reason, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{2,95}", resolution.reason) is None
            or not isinstance(resolution.required_claims, tuple)
            or any(not isinstance(item, ResolvedClaim) for item in resolution.required_claims)
        ):
            return GateDecision(False, "reject_claim_resolution_invalid", (), audit)
        if any(
            not isinstance(item.entry_id, str) or not isinstance(item.claim_id, str)
            for item in resolution.required_claims
        ):
            return GateDecision(False, "reject_claim_resolution_invalid", (), audit)
        resolved_keys = {(item.entry_id, item.claim_id) for item in resolution.required_claims}
        contract_claim_keys = {
            (entry_id, claim.claim_id)
            for entry_id, contract in contracts.items()
            for claim in contract.claims
        }
        if not resolved_keys <= contract_claim_keys or len(resolved_keys) != len(resolution.required_claims):
            return GateDecision(False, "reject_claim_resolution_invalid", (), audit)
        if any(
            isinstance(item.confidence, bool)
            or not isinstance(item.confidence, (int, float))
            or not math.isfinite(item.confidence)
            or item.confidence < contracts[item.entry_id].resolver.minimum_confidence
            or item.confidence > 1
            for item in resolution.required_claims
        ):
            return GateDecision(False, "reject_claim_resolution_untrusted", (), audit)
        required_evidence_links = [
            {
                "entry_id": resolved_claim.entry_id,
                "claim_id": resolved_claim.claim_id,
                "section_id": link.section_id,
                "source_id": link.source_id,
            }
            for resolved_claim in resolution.required_claims
            for claim in contracts[resolved_claim.entry_id].claims
            if claim.claim_id == resolved_claim.claim_id
            for link in claim.evidence
        ]
        audit["required_claims"] = [
            {
                "entry_id": claim.entry_id,
                "claim_id": claim.claim_id,
                "confidence": claim.confidence,
            }
            for claim in resolution.required_claims
        ]
        audit["required_evidence_links"] = required_evidence_links
        if resolution.out_of_scope or not resolution.required_claims:
            return GateDecision(False, resolution.reason, resolution.required_claims, audit)

        auditable_evidence = [(item, metadata) for item, metadata, _contract in protected_evidence]
        covered_snapshot_ids: list[str] = []
        for resolved_claim in resolution.required_claims:
            contract = contracts[resolved_claim.entry_id]
            claim = next(item for item in contract.claims if item.claim_id == resolved_claim.claim_id)
            for link in claim.evidence:
                matches = self._matching_evidence(
                    auditable_evidence,
                    entry_id=resolved_claim.entry_id,
                    link=link,
                )
                if not matches:
                    return GateDecision(False, "reject_claim_evidence_missing", resolution.required_claims, audit)
                fresh_matches = [
                    item
                    for item in matches
                    if item[1].get("review_status") == "approved"
                    and item[1].get("evidence_conflict") in {"none", "resolved"}
                    and self._source_is_fresh(item[1])
                ]
                if not fresh_matches:
                    return GateDecision(False, "reject_claim_evidence_stale", resolution.required_claims, audit)
                covered_snapshot_ids.extend(item[0].snapshot_id for item in fresh_matches)
        audit["covered_snapshot_ids"] = list(dict.fromkeys(covered_snapshot_ids))
        return GateDecision(
            True,
            "sufficient_claim_evidence",
            resolution.required_claims,
            audit,
            tuple(item for item, _metadata, _contract in protected_evidence),
        )
