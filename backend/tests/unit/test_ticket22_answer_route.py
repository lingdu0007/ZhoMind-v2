import pytest

from app.extensions.provider_router import ProviderRouter
from app.rag.answer_execution import AnswerOutcomeKind, EvidenceGatedAnswerExecutor
from tests.unit.test_approved_generation_route import approved_route
from tests.unit.test_ticket19_evidence_execution import _SufficientPilotRetriever


@pytest.mark.asyncio
async def test_route_timeout_exhaustion_is_closed_generation_unavailable_with_frozen_evidence():
    class TimeoutProvider:
        async def complete(self, prompt, *, system_prompt=None):
            raise TimeoutError("private upstream details")

    executor = EvidenceGatedAnswerExecutor(
        retriever=_SufficientPilotRetriever(), reranker=None, judge=None,
        provider_router=ProviderRouter(
            providers={"primary": TimeoutProvider()}, approved_route=approved_route("primary"),
        ),
        primary_provider="primary", retriever_name="fixture", reranker_name="disabled",
        judge_name="disabled", retrieval_top_k=20, max_evidence_items=3, max_excerpt_chars=1200,
    )
    result = await executor.execute(
        request_id="ticket22-exhaustion", user_id="member", session_id="session",
        question="Which reviewed operating decision applies for environment=production?",
    )
    assert result.kind is AnswerOutcomeKind.GENERATION_UNAVAILABLE
    assert result.evidence_set is not None
    assert "private upstream" not in str(result.to_rag_trace())
    assert result.runtime["provider_attempts"][0]["error_code"] == "timeout"


@pytest.mark.asyncio
async def test_executor_validates_each_answer_before_advancing_the_same_frozen_set():
    from tests.unit.test_ticket19_evidence_execution import _valid_answer

    calls = []

    class Provider:
        def __init__(self, answer):
            self.answer = answer

        async def complete(self, prompt, *, system_prompt=None):
            calls.append((prompt, system_prompt))
            return self.answer

    executor = EvidenceGatedAnswerExecutor(
        retriever=_SufficientPilotRetriever(), reranker=None, judge=None,
        provider_router=ProviderRouter(
            providers={"primary": Provider("invented answer [S9]"), "fallback": Provider(_valid_answer())},
            approved_route=approved_route("primary", "fallback"),
        ),
        primary_provider="ignored", retriever_name="fixture", reranker_name="disabled",
        judge_name="disabled", retrieval_top_k=20, max_evidence_items=3, max_excerpt_chars=1200,
    )
    result = await executor.execute(
        request_id="ticket22-fallback", user_id="member", session_id="session",
        question="Which reviewed operating decision applies for environment=production?",
    )
    assert result.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result.runtime["provider_attempts"][0]["error_code"] == "citation_invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("private_text,reason", [
    ("api_key=fixture_value", "privacy_refusal"),
    ("from my knowledge", "policy_refusal"),
])
async def test_output_protection_rejection_never_advances(private_text, reason):
    from tests.unit.test_ticket19_evidence_execution import _valid_answer

    calls = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append(prompt)
            return _valid_answer().replace("[S1]", f"{private_text} [S1]", 1)

    executor = EvidenceGatedAnswerExecutor(
        retriever=_SufficientPilotRetriever(), reranker=None, judge=None,
        provider_router=ProviderRouter(
            providers={"primary": Provider(), "fallback": Provider()},
            approved_route=approved_route("primary", "fallback"),
        ),
        primary_provider="ignored", retriever_name="fixture", reranker_name="disabled",
        judge_name="disabled", retrieval_top_k=20, max_evidence_items=3, max_excerpt_chars=1200,
    )
    result = await executor.execute(
        request_id="ticket22-protection", user_id="member", session_id="session",
        question="Which reviewed operating decision applies for environment=production?",
    )
    assert len(calls) == 1
    assert result.kind is AnswerOutcomeKind.GENERATION_UNAVAILABLE
    assert result.runtime["provider_attempts"][0]["error_code"] == reason
    assert private_text not in result.text
