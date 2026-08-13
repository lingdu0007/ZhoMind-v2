import asyncio
from collections.abc import Mapping
from typing import cast

from app.rag.answer_evidence import AnswerEvidence
from app.rag.claim_evidence import (
    ClaimEvidenceContract,
    ClaimEvidenceGate,
    ClaimResolution,
    ResolvedClaim,
    parse_claim_evidence_contract,
)


def _contract() -> dict:
    return {
        "schema_version": 1,
        "review_id": "editorial-review-20260813-claim-gate",
        "review_revision": "2026-08-13.1",
        "conflict_state": "none",
        "unknown_state": "none",
        "resolver": {
            "resolver_id": "fixture-calibrated-claim-resolver-v1",
            "calibration_id": "fixture-calibration-20260813",
            "calibration_version": "2026-08-13",
            "minimum_confidence": 0.80,
        },
        "claims": [
            {
                "claim_id": "claim-control-topology",
                "scope": "Known execution paths with explicit termination conditions.",
                "evidence": [{"section_id": "stable-principle", "source_id": "source-workflow"}],
            },
            {
                "claim_id": "claim-agent-budgets",
                "scope": "Local Agent subtasks that depend on runtime tool observations.",
                "evidence": [{"section_id": "recommendation", "source_id": "source-agent"}],
            },
        ],
    }


def _candidate(*, section_id: str, source_id: str, content: str) -> dict:
    contract = parse_claim_evidence_contract(_contract())
    return {
        "chunk_id": f"chunk-{section_id}",
        "document_id": "published-agent-entry",
        "generation": 4,
        "chunk_index": 0 if section_id == "stable-principle" else 1,
        "content_preview": content,
        "metadata": {
            "entry_id": "pae-workflow-gate-001",
            "entry_title": "Choose deterministic control for known execution paths",
            "domain": "workflow-vs-agent",
            "section_id": section_id,
            "review_status": "approved",
            "review_date": "2026-08-13",
            "evidence_conflict": "none",
            "source_id": source_id,
            "source_title": "Public source",
            "source_authority": "Example authority",
            "source_url": "https://example.com/agent-gate",
            "source_version": "v2026-08-13",
            "source_availability": "verified",
            "source_review_date": "2026-08-13",
            "source_freshness_days": 90,
            "claim_evidence_contract": contract.canonical_json,
            "claim_evidence_contract_sha256": contract.sha256,
            "title": "Choose deterministic control for known execution paths",
            "publication_version": "v4",
        },
    }


def _evidence(*, section_id: str, source_id: str, content: str) -> AnswerEvidence:
    value = AnswerEvidence.from_candidate(
        _candidate(section_id=section_id, source_id=source_id, content=content),
        max_excerpt_chars=400,
    )
    assert value is not None
    return value


class _FixtureCalibratedClaimResolver:
    resolver_id = "fixture-calibrated-claim-resolver-v1"
    calibration_id = "fixture-calibration-20260813"
    calibration_version = "2026-08-13"

    async def resolve(self, question: str, _contracts: Mapping[str, ClaimEvidenceContract]) -> ClaimResolution:
        claims_by_question = {
            "Should known execution paths use a deterministic workflow?": ("claim-control-topology",),
            "My task has fixed steps and an explicit termination condition; do I need an autonomous Agent loop?": (
                "claim-control-topology",
            ),
            (
                "Code owns known branches, but one subtask needs runtime tool observations. "
                "How should control and step/tool budgets be split?"
            ): (
                "claim-control-topology",
                "claim-agent-budgets",
            ),
        }
        claim_ids = claims_by_question.get(question)
        if claim_ids is None:
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_scope")
        return ClaimResolution(
            required_claims=tuple(
                ResolvedClaim(entry_id="pae-workflow-gate-001", claim_id=claim_id, confidence=0.95)
                for claim_id in claim_ids
            ),
            out_of_scope=False,
            reason="resolved_claims",
        )


def _gate() -> ClaimEvidenceGate:
    return ClaimEvidenceGate(resolver=_FixtureCalibratedClaimResolver())


def test_claim_evidence_gate_allows_direct_paraphrase_and_combined_claims_only_with_full_snapshot_coverage() -> None:
    evidence = (
        _evidence(
            section_id="stable-principle",
            source_id="source-workflow",
            content="Known execution paths with fixed steps should use deterministic workflow control.",
        ),
        _evidence(
            section_id="recommendation",
            source_id="source-agent",
            content="A local Agent subtask needs explicit step, tool-call, latency, and cost budgets.",
        ),
    )

    direct = asyncio.run(_gate().evaluate("Should known execution paths use a deterministic workflow?", evidence))
    paraphrase = asyncio.run(
        _gate().evaluate("My task has fixed steps and an explicit termination condition; do I need an autonomous Agent loop?", evidence)
    )
    combined = asyncio.run(
        _gate().evaluate(
            "Code owns known branches, but one subtask needs runtime tool observations. How should control and step/tool budgets be split?",
            evidence,
        )
    )

    assert direct.passed is True
    assert paraphrase.passed is True
    assert combined.passed is True
    assert {item.claim_id for item in combined.required_claims} == {"claim-control-topology", "claim-agent-budgets"}
    assert combined.audit["resolver_id"] == "fixture-calibrated-claim-resolver-v1"
    assert combined.audit["required_claims"] == [
        {
            "entry_id": "pae-workflow-gate-001",
            "claim_id": "claim-control-topology",
            "confidence": 0.95,
        },
        {
            "entry_id": "pae-workflow-gate-001",
            "claim_id": "claim-agent-budgets",
            "confidence": 0.95,
        },
    ]
    assert combined.audit["covered_snapshot_ids"] == [item.snapshot_id for item in evidence]


def test_claim_evidence_gate_rejects_broad_boundary_and_missing_or_stale_support() -> None:
    complete = (
        _evidence(
            section_id="stable-principle",
            source_id="source-workflow",
            content="Known execution paths with fixed steps should use deterministic workflow control.",
        ),
        _evidence(
            section_id="recommendation",
            source_id="source-agent",
            content="A local Agent subtask needs explicit step, tool-call, latency, and cost budgets.",
        ),
    )
    missing = complete[:1]
    stale_candidate = _candidate(
        section_id="recommendation",
        source_id="source-agent",
        content="A local Agent subtask needs explicit step, tool-call, latency, and cost budgets.",
    )
    stale_candidate["metadata"]["source_review_date"] = "2025-01-01"
    stale = AnswerEvidence.from_candidate(stale_candidate, max_excerpt_chars=400)
    assert stale is not None

    boundary = asyncio.run(
        _gate().evaluate("At exactly how many branches must every production system switch to an Agent?", complete)
    )
    partial = asyncio.run(
        _gate().evaluate(
            "Code owns known branches, but one subtask needs runtime tool observations. How should control and step/tool budgets be split?",
            missing,
        )
    )
    stale_result = asyncio.run(
        _gate().evaluate(
            "Code owns known branches, but one subtask needs runtime tool observations. How should control and step/tool budgets be split?",
            (complete[0], stale),
        )
    )

    assert (boundary.passed, boundary.reason) == (False, "reject_claim_scope")
    assert (partial.passed, partial.reason) == (False, "reject_claim_evidence_missing")
    assert (stale_result.passed, stale_result.reason) == (False, "reject_claim_evidence_stale")


def test_claim_evidence_gate_rejects_tampered_conflicting_unlinked_and_resolver_mismatched_contracts() -> None:
    valid_candidate = _candidate(
        section_id="stable-principle",
        source_id="source-workflow",
        content="Known execution paths use deterministic workflow control.",
    )
    tampered_candidate = _candidate(
        section_id="stable-principle",
        source_id="source-workflow",
        content="Known execution paths use deterministic workflow control.",
    )
    tampered_candidate["metadata"]["claim_evidence_contract_sha256"] = "0" * 64
    tampered = AnswerEvidence.from_candidate(tampered_candidate, max_excerpt_chars=400)
    assert tampered is not None

    conflicting_contract = _contract()
    conflicting_contract["conflict_state"] = "unresolved"
    parsed_conflict = parse_claim_evidence_contract(conflicting_contract)
    conflict_candidate = _candidate(
        section_id="stable-principle",
        source_id="source-workflow",
        content="Known execution paths use deterministic workflow control.",
    )
    conflict_candidate["metadata"]["claim_evidence_contract"] = parsed_conflict.canonical_json
    conflict_candidate["metadata"]["claim_evidence_contract_sha256"] = parsed_conflict.sha256
    conflict = AnswerEvidence.from_candidate(conflict_candidate, max_excerpt_chars=400)
    assert conflict is not None

    unlinked = _evidence(
        section_id="unlinked-section",
        source_id="source-workflow",
        content="Known execution paths use deterministic workflow control.",
    )
    valid = AnswerEvidence.from_candidate(valid_candidate, max_excerpt_chars=400)
    assert valid is not None
    mismatched_resolver = ClaimEvidenceGate(
        resolver=type(
            "MismatchedFixtureCalibratedClaimResolver",
            (_FixtureCalibratedClaimResolver,),
            {"calibration_version": "2026-08-14"},
        )()
    )

    question = "Should known execution paths use a deterministic workflow?"
    tampered_result = asyncio.run(_gate().evaluate(question, (tampered,)))
    conflict_result = asyncio.run(_gate().evaluate(question, (conflict,)))
    unlinked_result = asyncio.run(_gate().evaluate(question, (unlinked,)))
    resolver_result = asyncio.run(mismatched_resolver.evaluate(question, (valid,)))

    assert (tampered_result.passed, tampered_result.reason) == (False, "reject_claim_contract_invalid")
    assert (conflict_result.passed, conflict_result.reason) == (False, "reject_claim_contract_conflict")
    assert (unlinked_result.passed, unlinked_result.reason) == (False, "reject_claim_evidence_unlinked")
    assert (resolver_result.passed, resolver_result.reason) == (False, "reject_claim_resolver_unavailable")


def test_claim_evidence_gate_rejects_unreviewed_universal_variant_before_evidence_coverage() -> None:
    evidence = (
        _evidence(
            section_id="stable-principle",
            source_id="source-workflow",
            content="Known execution paths with fixed steps should use deterministic workflow control.",
        ),
        _evidence(
            section_id="recommendation",
            source_id="source-agent",
            content="A local Agent subtask needs explicit step, tool-call, latency, and cost budgets.",
        ),
    )

    result = asyncio.run(
        _gate().evaluate("Is a deterministic workflow always better than an Agent?", evidence)
    )

    assert (result.passed, result.reason) == (False, "reject_claim_scope")
    assert result.audit["required_claims"] == []


def test_claim_evidence_gate_rejects_forged_resolver_claim_even_when_its_identity_matches_contract() -> None:
    class _ForgedClaimResolver(_FixtureCalibratedClaimResolver):
        async def resolve(self, _question: str, _contracts: Mapping[str, ClaimEvidenceContract]) -> ClaimResolution:
            return ClaimResolution(
                required_claims=(
                    ResolvedClaim(
                        entry_id="pae-workflow-gate-001",
                        claim_id="claim-not-in-reviewed-contract",
                        confidence=0.99,
                    ),
                ),
                out_of_scope=False,
                reason="resolved_claims",
            )

    evidence = (
        _evidence(
            section_id="stable-principle",
            source_id="source-workflow",
            content="Known execution paths with fixed steps should use deterministic workflow control.",
        ),
    )
    result = asyncio.run(
        ClaimEvidenceGate(resolver=_ForgedClaimResolver()).evaluate(
            "Should known execution paths use a deterministic workflow?",
            evidence,
        )
    )

    assert (result.passed, result.reason) == (False, "reject_claim_resolution_invalid")
    assert result.audit["required_claims"] == []
    assert result.audit["contracts"] == [
        {
            "entry_id": "pae-workflow-gate-001",
            "review_id": "editorial-review-20260813-claim-gate",
            "review_revision": "2026-08-13.1",
            "sha256": parse_claim_evidence_contract(_contract()).sha256,
        }
    ]


def test_claim_evidence_gate_rejects_non_string_resolver_claim_identity_before_hashing() -> None:
    class _MalformedClaimResolver(_FixtureCalibratedClaimResolver):
        async def resolve(self, _question: str, _contracts: Mapping[str, ClaimEvidenceContract]) -> ClaimResolution:
            return ClaimResolution(
                required_claims=(
                    ResolvedClaim(
                        entry_id=cast(str, []),
                        claim_id=cast(str, []),
                        confidence=0.95,
                    ),
                ),
                out_of_scope=False,
                reason="resolved_claims",
            )

    evidence = (
        _evidence(
            section_id="stable-principle",
            source_id="source-workflow",
            content="Known execution paths with fixed steps should use deterministic workflow control.",
        ),
    )
    result = asyncio.run(
        ClaimEvidenceGate(resolver=_MalformedClaimResolver()).evaluate(
            "Should known execution paths use a deterministic workflow?",
            evidence,
        )
    )

    assert (result.passed, result.reason) == (False, "reject_claim_resolution_invalid")
    assert result.audit["required_claims"] == []
