from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from app.contracts.canonical import StableIdentity, StableIdentityKind

_SCHEMA = "candidate_claim_evidence_contract/v1"
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,159}")
_HIGH_IMPACT_CLAIM_KINDS = frozenset({"prescriptive", "numeric", "version", "security", "privacy", "high_impact"})


class CandidateClaimEvidenceContractError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateClaimEvidenceDefinition:
    claim_id: str
    section_id: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateClaimEvidenceContract:
    entry_identity: str
    editorial_revision_identity: str
    claims: tuple[CandidateClaimEvidenceDefinition, ...]

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "entry_identity": self.entry_identity,
            "editorial_revision_identity": self.editorial_revision_identity,
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "section_id": claim.section_id,
                    "source_ids": list(claim.source_ids),
                }
                for claim in self.claims
            ],
        }


def build_candidate_claim_evidence_contract(
    *,
    entry_identity: object,
    editorial_revision_identity: object,
    claims: object,
) -> CandidateClaimEvidenceContract:
    entry = _stable_identity(entry_identity, StableIdentityKind.ENTRY, "entry_identity")
    revision = _stable_identity(
        editorial_revision_identity,
        StableIdentityKind.EDITORIAL_REVISION,
        "editorial_revision_identity",
    )
    if not isinstance(claims, list):
        raise CandidateClaimEvidenceContractError("claims must be a list")

    definitions: list[CandidateClaimEvidenceDefinition] = []
    seen_claim_ids: set[str] = set()
    for raw_claim in claims:
        if not isinstance(raw_claim, dict) or not _is_material_claim(raw_claim):
            continue
        claim_id = _safe_id(raw_claim.get("claim_id"), "claim_id")
        if claim_id in seen_claim_ids:
            raise CandidateClaimEvidenceContractError("claims contains duplicate claim_id")
        seen_claim_ids.add(claim_id)
        section_id = _safe_id(raw_claim.get("section_id"), "section_id")
        raw_source_ids = raw_claim.get("source_ids")
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise CandidateClaimEvidenceContractError("material claim source_ids must be non-empty")
        source_ids = tuple(sorted({_safe_id(source_id, "source_ids") for source_id in raw_source_ids}))
        if len(source_ids) != len(raw_source_ids):
            raise CandidateClaimEvidenceContractError("material claim source_ids must be unique")
        definitions.append(
            CandidateClaimEvidenceDefinition(
                claim_id=claim_id,
                section_id=section_id,
                source_ids=source_ids,
            )
        )
    if not definitions:
        raise CandidateClaimEvidenceContractError("Claim-Linked entries require material Claim-Evidence Links")
    return CandidateClaimEvidenceContract(
        entry_identity=entry,
        editorial_revision_identity=revision,
        claims=tuple(sorted(definitions, key=lambda item: item.claim_id)),
    )


def parse_candidate_claim_evidence_contract(value: object) -> CandidateClaimEvidenceContract:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "entry_identity",
        "editorial_revision_identity",
        "claims",
    }:
        raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract has an invalid shape")
    if value.get("schema") != _SCHEMA:
        raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract schema is invalid")
    entry = _stable_identity(value.get("entry_identity"), StableIdentityKind.ENTRY, "entry_identity")
    revision = _stable_identity(
        value.get("editorial_revision_identity"),
        StableIdentityKind.EDITORIAL_REVISION,
        "editorial_revision_identity",
    )
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract claims are invalid")
    definitions: list[CandidateClaimEvidenceDefinition] = []
    seen_claim_ids: set[str] = set()
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict) or set(raw_claim) != {"claim_id", "section_id", "source_ids"}:
            raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract claim is invalid")
        claim_id = _safe_id(raw_claim.get("claim_id"), "claim_id")
        if claim_id in seen_claim_ids:
            raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract claims duplicate claim_id")
        seen_claim_ids.add(claim_id)
        section_id = _safe_id(raw_claim.get("section_id"), "section_id")
        raw_source_ids = raw_claim.get("source_ids")
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract source_ids are invalid")
        source_ids = tuple(sorted({_safe_id(source_id, "source_ids") for source_id in raw_source_ids}))
        if len(source_ids) != len(raw_source_ids):
            raise CandidateClaimEvidenceContractError("Candidate Claim-Evidence contract source_ids are invalid")
        definitions.append(
            CandidateClaimEvidenceDefinition(
                claim_id=claim_id,
                section_id=section_id,
                source_ids=source_ids,
            )
        )
    return CandidateClaimEvidenceContract(
        entry_identity=entry,
        editorial_revision_identity=revision,
        claims=tuple(sorted(definitions, key=lambda item: item.claim_id)),
    )


def _stable_identity(value: object, kind: StableIdentityKind, field: str) -> str:
    if not isinstance(value, str):
        raise CandidateClaimEvidenceContractError(f"{field} must be a stable identity")
    try:
        identity = StableIdentity.from_stable_id(value)
    except ValueError as exc:
        raise CandidateClaimEvidenceContractError(f"{field} must be a stable identity") from exc
    if identity.kind is not kind:
        raise CandidateClaimEvidenceContractError(f"{field} has an invalid identity kind")
    return identity.stable_id


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value.strip()) is None:
        raise CandidateClaimEvidenceContractError(f"{field} must be a stable identifier")
    return value.strip()


def _is_material_claim(value: dict[str, Any]) -> bool:
    return value.get("material") is True or value.get("claim_kind") in _HIGH_IMPACT_CLAIM_KINDS
