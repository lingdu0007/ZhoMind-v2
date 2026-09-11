import pytest

from app.common.exceptions import AppError


@pytest.mark.parametrize(
    ("observation", "facts", "expected"),
    [
        ("coverage_gap", {"outcome": "insufficient_evidence_reply", "reason": "decision_not_covered"}, ("coverage-gap", "coverage-work")),
        (
            "retrieval_miss",
            {
                "outcome": "insufficient_evidence_reply", "reference_supported": True,
                "frozen_evidence": False, "reason": "no_eligible_published_evidence",
            },
            ("retrieval-answer-behavior", "retrieval-experiment"),
        ),
        (
            "condition_loss",
            {"outcome": None, "state": "failed", "reference_supported": True, "verified_difference": True},
            ("retrieval-answer-behavior", "retrieval-experiment"),
        ),
        (
            "citation_drift",
            {"outcome": "generation_unavailable", "reference_supported": True, "verified_difference": True, "frozen_evidence": True},
            ("retrieval-answer-behavior", "retrieval-experiment"),
        ),
        (
            "provider_failure",
            {"outcome": "generation_unavailable", "frozen_evidence": True},
            ("product-privacy-operations", "provider-work"),
        ),
        ("product_failure", {"state": "failed", "reference_supported": True}, ("product-privacy-operations", "product-repair")),
        ("stale_source", {"source_state": "changed_or_unreachable_awaiting_review"}, ("source-freshness", "source-change")),
        (
            "wrong_content", {"verified_publication": True, "verified_integrity_review": True},
            ("content-integrity", "entry-revision"),
        ),
        (
            "confirmation",
            {"outcome": "evidence_gated_answer", "expected_outcome": "evidence_gated_answer"},
            ("confirmation", "confirmation"),
        ),
    ],
)
def test_evidence_first_diagnosis_routes_only_verified_facts(observation, facts, expected):
    from app.maintenance.diagnosis_policy import diagnosis_route

    assert diagnosis_route(observation, facts) == expected


@pytest.mark.parametrize(
    "observation", ["retrieval_miss", "condition_loss", "citation_drift", "provider_failure", "wrong_content", "stale_source"]
)
def test_insufficiency_alone_does_not_prove_a_retrieval_or_content_defect(observation):
    from app.maintenance.diagnosis_policy import diagnosis_route

    with pytest.raises(AppError) as rejected:
        diagnosis_route(observation, {"outcome": "insufficient_evidence_reply", "reason": "no_eligible_published_evidence"})
    assert rejected.value.code == "MAINTENANCE_EVIDENCE_REQUIRED"
