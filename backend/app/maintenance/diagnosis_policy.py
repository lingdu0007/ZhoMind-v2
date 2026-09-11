from collections.abc import Mapping

from app.common.exceptions import AppError


def diagnosis_route(observation: str, facts: Mapping[str, object]) -> tuple[str, str]:
    """Route a verified observation; callers must derive facts from retained authority."""
    outcome = facts.get("outcome")
    if observation == "coverage_gap" and (
        outcome == "insufficient_evidence_reply"
        and facts.get("reason") in {"no_eligible_published_evidence", "decision_not_covered", "decisive_condition_missing"}
        and facts.get("reference_supported") is not True
    ):
        return "coverage-gap", "coverage-work"
    if observation == "citation_drift" and (
        facts.get("reference_supported") is True
        and facts.get("verified_difference") is True
        and facts.get("frozen_evidence") is True
        and outcome == "generation_unavailable"
    ):
        return "retrieval-answer-behavior", "retrieval-experiment"
    if observation == "retrieval_miss" and (
        facts.get("reference_supported") is True
        and outcome == "insufficient_evidence_reply"
        and facts.get("frozen_evidence") is False
        and facts.get("reason") == "no_eligible_published_evidence"
    ):
        return "retrieval-answer-behavior", "retrieval-experiment"
    if observation == "condition_loss" and (
        facts.get("reference_supported") is True
        and facts.get("verified_difference") is True
        and facts.get("state") == "failed"
        and outcome is None
    ):
        return "retrieval-answer-behavior", "retrieval-experiment"
    if observation == "provider_failure" and outcome == "generation_unavailable" and facts.get("frozen_evidence") is True:
        return "product-privacy-operations", "provider-work"
    if observation == "product_failure" and facts.get("state") == "failed" and facts.get("reference_supported") is True:
        return "product-privacy-operations", "product-repair"
    if observation == "stale_source" and facts.get("source_state") in {
        "changed_or_unreachable_awaiting_review",
        "unavailable_for_new_evidence",
    }:
        return "source-freshness", "source-change"
    if observation == "wrong_content" and (
        facts.get("verified_publication") is True and facts.get("verified_integrity_review") is True
    ):
        return "content-integrity", "entry-revision"
    if observation == "confirmation" and outcome == facts.get("expected_outcome") and outcome == "evidence_gated_answer":
        return "confirmation", "confirmation"
    raise AppError(status_code=409, code="MAINTENANCE_EVIDENCE_REQUIRED", message="independent current authenticated evidence required")
