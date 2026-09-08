from __future__ import annotations

import json

from app.contracts.candidate_claim_evidence import (
    CandidateClaimEvidenceContract,
    CandidateClaimEvidenceContractError,
    build_candidate_claim_evidence_contract,
    parse_candidate_claim_evidence_contract,
)
from app.rag.claim_evidence import ClaimEvidenceContractError, parse_claim_evidence_contract


def candidate_assurance_metadata(
    *,
    artifact: dict[str, object],
    entry: dict[str, object],
    entry_identity: str,
) -> dict[str, object]:
    assurance_level = entry.get("assurance_level")
    if assurance_level == "source_grounded":
        return {}
    if assurance_level == "claim_linked":
        raw_contract = artifact.get("claim_evidence_contract")
        raw_hash = artifact.get("claim_evidence_contract_sha256")
        if not isinstance(raw_contract, str) or not raw_contract or not isinstance(raw_hash, str) or not raw_hash:
            raise ValueError("Candidate artifact Claim-Evidence contract is invalid")
        try:
            raw_value = json.loads(raw_contract)
            expected = build_candidate_claim_evidence_contract(
                entry_identity=entry_identity,
                editorial_revision_identity=artifact.get("editorial_revision_identity"),
                claims=entry.get("claims"),
            )
        except (json.JSONDecodeError, CandidateClaimEvidenceContractError) as exc:
            raise ValueError("Candidate artifact Claim-Evidence contract is invalid") from exc
        try:
            candidate_contract = parse_candidate_claim_evidence_contract(raw_value)
        except CandidateClaimEvidenceContractError:
            try:
                full_contract = parse_claim_evidence_contract(raw_value)
            except ClaimEvidenceContractError as exc:
                raise ValueError("Candidate artifact Claim-Evidence contract is invalid") from exc
            if full_contract.sha256 != raw_hash or not _full_contract_matches_candidate_links(full_contract, expected):
                raise ValueError("Candidate artifact Claim-Evidence contract is invalid") from None
            return {
                "claim_evidence_contract": full_contract.canonical_json,
                "claim_evidence_contract_sha256": full_contract.sha256,
            }
        if candidate_contract.sha256 != raw_hash or candidate_contract != expected:
            raise ValueError("Candidate artifact Claim-Evidence contract is invalid")
        return {
            "claim_evidence_contract": candidate_contract.canonical_json,
            "claim_evidence_contract_sha256": candidate_contract.sha256,
        }
    if assurance_level == "release_assured":
        snapshot = artifact.get("release_assurance_snapshot")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != "editorial_release_assurance_snapshot/v1"
            or snapshot.get("entry_identity") != entry_identity
        ):
            raise ValueError("Candidate artifact release assurance snapshot is invalid")
        return {"release_assurance_snapshot": snapshot}
    raise ValueError("Candidate artifact assurance level is invalid")


def _full_contract_matches_candidate_links(
    contract,
    expected: CandidateClaimEvidenceContract,
) -> bool:
    actual = {
        claim.claim_id: (
            {
                link.section_id
                for link in claim.evidence
            },
            {
                (link.section_id, link.source_id)
                for link in claim.evidence
            },
        )
        for claim in contract.claims
    }
    expected_links = {
        claim.claim_id: (
            {claim.section_id},
            {(claim.section_id, source_id) for source_id in claim.source_ids},
        )
        for claim in expected.claims
    }
    return actual == expected_links
