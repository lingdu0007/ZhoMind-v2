import asyncio
import re

import pytest

from app.prompt_injection_live import CaseResult, PromptInjectionLiveRun, _RunFailure, _verification_sentinel, judge_case
from app.rag.prompt_injection_corpus import (
    ADVERSARIAL_INJECTION_CASES,
    FORGED_SOURCE,
    FORGED_TOOL_CALL,
    INSTRUCTION_OVERRIDE,
    SECRET_EXTRACTION,
    UNSAFE_CODE,
    UNSUPPORTED_ANSWER_PRESSURE,
)


def _case(kind: str):
    for case in ADVERSARIAL_INJECTION_CASES:
        if case.kind == kind:
            return case
    raise AssertionError(f"no case of kind {kind}")


def test_seeded_case_passes_when_marker_stays_out_of_answer() -> None:
    for kind in (INSTRUCTION_OVERRIDE, SECRET_EXTRACTION, FORGED_SOURCE, FORGED_TOOL_CALL, UNSAFE_CODE):
        case = _case(kind)
        result = judge_case(case, outcome="evidence_gated_answer", answer_text="正常回答内容", source_count=2)
        assert result.pass_fail == "pass"
        assert result.failure_classification == "none"
        assert result.outcome == "evidence_gated_answer"
        assert result.source_count == 2
        assert result.evidence_count == 2
        assert result.case_id == case.case_id
        assert result.kind == kind


@pytest.mark.parametrize(
    ("kind", "classification"),
    [
        (INSTRUCTION_OVERRIDE, "policy_override"),
        (SECRET_EXTRACTION, "secret_disclosure"),
        (FORGED_SOURCE, "forged_citation"),
        (FORGED_TOOL_CALL, "forged_tool_call"),
        (UNSAFE_CODE, "unsafe_code"),
    ],
)
def test_seeded_case_fails_when_marker_leaks_into_answer(kind: str, classification: str) -> None:
    case = _case(kind)
    assert case.answer_marker is not None
    result = judge_case(case, outcome="evidence_gated_answer", answer_text=f"前缀 {case.answer_marker} 后缀", source_count=1)
    assert result.pass_fail == "fail"
    assert result.failure_classification == classification


@pytest.mark.parametrize(
    "kind",
    [INSTRUCTION_OVERRIDE, SECRET_EXTRACTION, FORGED_SOURCE, FORGED_TOOL_CALL, UNSAFE_CODE, UNSUPPORTED_ANSWER_PRESSURE],
)
def test_generation_unavailable_always_classifies_as_provider_failure(kind: str) -> None:
    case = _case(kind)
    result = judge_case(case, outcome="generation_unavailable", answer_text="", source_count=0)
    assert result.pass_fail == "fail"
    assert result.failure_classification == "generation_unavailable"


def test_unsupported_pressure_passes_on_insufficient_evidence() -> None:
    case = _case(UNSUPPORTED_ANSWER_PRESSURE)
    result = judge_case(case, outcome="insufficient_evidence_reply", answer_text="", source_count=0)
    assert result.pass_fail == "pass"
    assert result.failure_classification == "none"
    assert result.to_record()["citation_counts"]["evidence_count"] == 0


def test_unsupported_pressure_fails_when_an_answer_is_produced() -> None:
    case = _case(UNSUPPORTED_ANSWER_PRESSURE)
    result = judge_case(case, outcome="evidence_gated_answer", answer_text="编造的营收数字", source_count=1)
    assert result.pass_fail == "fail"
    assert result.failure_classification == "ungrounded_answer"


def test_record_matches_public_evidence_contract_fields() -> None:
    case = _case(SECRET_EXTRACTION)
    result = judge_case(
        case,
        outcome="evidence_gated_answer",
        answer_text="正常回答",
        source_count=1,
        source_revision="candidate-revision",
        run_id="security-run-01",
    )
    record = result.to_record()
    assert set(record) == {
        "case_id",
        "kind",
        "outcome",
        "pass_fail",
        "citation_counts",
        "failure_classification",
        "source_revision",
        "run_id",
    }
    assert set(record["citation_counts"]) == {"source_count", "evidence_count"}
    assert record["citation_counts"]["source_count"] >= 0
    assert record["citation_counts"]["evidence_count"] >= 0
    assert record["source_revision"] == "candidate-revision"
    assert record["run_id"] == "security-run-01"


def test_verification_sentinel_is_letter_only() -> None:
    sentinel = _verification_sentinel("live-prompt-injection-20260806T0835Z")
    assert sentinel.startswith("injection-")
    assert not re.search(r"\d", sentinel)
    assert _verification_sentinel("12345") == "injection-evidence"


def test_seeded_case_requires_its_published_source_in_the_answer_evidence_set() -> None:
    summary = {
        "coverage": "sufficient",
        "sources": [{"source_id": "published-injection-source", "metadata": {}, "excerpt": "snapshot"}],
    }

    PromptInjectionLiveRun._assert_expected_source(
        summary,
        case_id="injection-instruction-override-01",
        expected_source_id="published-injection-source",
    )

    with pytest.raises(_RunFailure) as failure:
        PromptInjectionLiveRun._assert_expected_source(
            summary,
            case_id="injection-instruction-override-01",
            expected_source_id="different-published-source",
        )

    assert failure.value.check == "case_normal"
    assert failure.value.code == "ADVERSARIAL_SOURCE_NOT_IN_EVIDENCE"
    assert failure.value.details == {"case_id": "injection-instruction-override-01"}


class _ObservedProviderRun(PromptInjectionLiveRun):
    async def _execute_cases(self) -> list[CaseResult]:
        return [
            CaseResult(
                case_id="injection-observed-pass",
                kind=INSTRUCTION_OVERRIDE,
                outcome="evidence_gated_answer",
                pass_fail="pass",
                source_count=1,
                evidence_count=1,
                failure_classification="none",
            ),
            CaseResult(
                case_id="injection-observed-provider-variance",
                kind=SECRET_EXTRACTION,
                outcome="evidence_gated_answer",
                pass_fail="fail",
                source_count=1,
                evidence_count=1,
                failure_classification="secret_disclosure",
            ),
        ]


def test_observed_provider_variance_is_not_reported_as_a_passing_live_run(tmp_path) -> None:
    runner = _ObservedProviderRun(
        settings=object(),
        http_client=object(),
        output_dir=tmp_path,
        source_revision="candidate-revision",
        run_id="observed-provider-variance",
    )

    manifest = asyncio.run(runner.run())

    assert manifest["outcome"] == "completed-with-exceptions"
    assert manifest["case_count"] == 2
    assert manifest["pass_count"] == 1
