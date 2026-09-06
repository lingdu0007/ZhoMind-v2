import asyncio
import json

import pytest

from app.common.config import get_settings
from app.extensions.provider_router import ProviderRouter
from app.rag.answer_execution import (
    AnswerOutcomeKind,
    EvidenceGatedAnswerExecutor,
)
from app.rag.prompt_injection_corpus import (
    ADVERSARIAL_INJECTION_CASES,
    FORGED_SOURCE,
    FORGED_TOOL_CALL,
    INSTRUCTION_OVERRIDE,
    SECRET_EXTRACTION,
    UNSAFE_CODE,
    UNSUPPORTED_ANSWER_PRESSURE,
)
from app.rag.prompt_regions import (
    EVIDENCE_SOURCES_REGION,
    SYSTEM_POLICY,
    USER_QUESTION_REGION,
)
from app.retrieval.policy import LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID
from app.settings.runtime import get_system_settings_runtime


@pytest.fixture(autouse=True)
def _run_legacy_injection_contract_fixtures_under_the_explicit_migration_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID)
    get_system_settings_runtime().reset()
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_system_settings_runtime().reset()
        get_settings.cache_clear()


def _candidate(case, *, index: int = 1) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": f"document-{index}",
        "generation": index,
        "chunk_index": 0,
        "score": float(10 - index),
        "content_preview": case.document_source or "无关候选内容",
        "metadata": {
            "title": case.document_title or f"资料 {index}.md",
            "publication_version": f"v{index}",
        },
        "retrieval_source": "dense",
    }


class _RecordingRetriever:
    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        self.calls.append((query, top_k))
        return self.items


class _IdentityReranker:
    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        return items


class _EvidenceJudge:
    async def judge(self, query: str, context: list[dict]) -> bool:
        return bool(context)


class _RecordingProvider:
    def __init__(self, *, answer: str = "基于证据生成的回答", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        if self.error is not None:
            raise self.error
        return self.answer


def _executor(
    *,
    retriever: _RecordingRetriever,
    provider: _RecordingProvider,
    secondary: _RecordingProvider | None = None,
) -> EvidenceGatedAnswerExecutor:
    providers: dict = {"approved": provider}
    if secondary is not None:
        providers["secondary"] = secondary
    return EvidenceGatedAnswerExecutor(
        retriever=retriever,
        reranker=_IdentityReranker(),
        judge=_EvidenceJudge(),
        provider_router=ProviderRouter(providers=providers),
        primary_provider="approved",
        retriever_name="test-retriever",
        reranker_name="test-reranker",
        judge_name="test-judge",
        retrieval_top_k=5,
        max_evidence_items=3,
        max_excerpt_chars=400,
    )


def _execute(executor: EvidenceGatedAnswerExecutor, question: str):
    return asyncio.run(
        executor.execute(
            request_id="injection-request-1",
            user_id="knowledge-user",
            session_id="injection-session-1",
            question=question,
        )
    )


def _seeded_case(kind: str):
    for case in ADVERSARIAL_INJECTION_CASES:
        if case.kind == kind and case.document_source is not None:
            return case
    raise AssertionError(f"no seeded case of kind {kind}")


def test_injected_override_instruction_stays_inside_evidence_region() -> None:
    case = _seeded_case(INSTRUCTION_OVERRIDE)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()
    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    assert len(provider.prompts) == 1
    user_prompt = provider.prompts[0]
    system_prompt = provider.system_prompts[0]

    # The injection marker travels only inside the untrusted evidence region.
    envelope = json.loads(user_prompt)
    assert case.answer_marker in envelope[EVIDENCE_SOURCES_REGION][0]["excerpt"]
    assert case.answer_marker not in envelope[USER_QUESTION_REGION]
    assert case.answer_marker not in (system_prompt or "")
    assert SYSTEM_POLICY in (system_prompt or "")


def test_secret_extraction_exposes_no_credentials_or_admin_diagnostics() -> None:
    case = _seeded_case(SECRET_EXTRACTION)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()
    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    user_prompt = provider.prompts[0]
    system_prompt = provider.system_prompts[0] or ""
    envelope = json.loads(user_prompt)

    # The fixture marker is untrusted snapshot content: it may appear only
    # inside the evidence region, never as policy or in the question region.
    assert case.answer_marker in envelope[EVIDENCE_SOURCES_REGION][0]["excerpt"]
    assert case.answer_marker not in envelope[USER_QUESTION_REGION]
    assert case.answer_marker not in system_prompt

    # Minimal exposure: no credentials, raw settings, private history, or
    # administrator diagnostics reach the provider.
    for forbidden in ("chunk-1", "document-1", "retrieval_source", "score", "10.0", "chunk_index", "generation"):
        assert forbidden not in user_prompt
    assert "DATABASE_URL" not in user_prompt
    assert "JWT_SECRET" not in user_prompt


def test_forged_source_instruction_cannot_inject_citation_identity() -> None:
    case = _seeded_case(FORGED_SOURCE)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()
    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    user_prompt = provider.prompts[0]
    system_prompt = provider.system_prompts[0] or ""
    envelope = json.loads(user_prompt)

    # The forged-source instruction is evidence data; the system policy still
    # forbids fabricating citations and the question region stays clean.
    assert "不要编造或虚构来源" in system_prompt
    assert case.answer_marker in envelope[EVIDENCE_SOURCES_REGION][0]["excerpt"]
    assert case.answer_marker not in envelope[USER_QUESTION_REGION]
    assert case.answer_marker not in system_prompt


@pytest.mark.parametrize("kind", [FORGED_TOOL_CALL, UNSAFE_CODE])
def test_agent_specific_effect_instructions_remain_untrusted_evidence(kind: str) -> None:
    case = _seeded_case(kind)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()

    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    envelope = json.loads(provider.prompts[0])
    system_prompt = provider.system_prompts[0] or ""
    assert case.answer_marker in envelope[EVIDENCE_SOURCES_REGION][0]["excerpt"]
    assert case.answer_marker not in envelope[USER_QUESTION_REGION]
    assert case.answer_marker not in system_prompt
    assert "不可信数据" in system_prompt


def test_no_evidence_pressure_returns_insufficient_without_generation() -> None:
    case = next(case for case in ADVERSARIAL_INJECTION_CASES if case.kind == UNSUPPORTED_ANSWER_PRESSURE)
    retriever = _RecordingRetriever([])
    provider = _RecordingProvider()

    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.evidence == ()
    assert provider.prompts == []
    assert provider.system_prompts == []
    assert outcome.to_rag_trace()["runtime"]["final_provider"] is None


def test_generation_unavailable_invokes_no_secondary_provider() -> None:
    case = _seeded_case(FORGED_SOURCE)
    retriever = _RecordingRetriever([_candidate(case)])
    primary = _RecordingProvider(error=TimeoutError("upstream timeout"))
    secondary = _RecordingProvider()

    outcome = _execute(
        _executor(retriever=retriever, provider=primary, secondary=secondary),
        case.question,
    )

    assert outcome.kind is AnswerOutcomeKind.GENERATION_UNAVAILABLE
    assert outcome.text.startswith("【生成不可用】")
    assert [item.source_id for item in outcome.evidence] == ["chunk-1"]
    assert outcome.evidence_summary()["coverage"] == "sufficient"
    # Fail-closed: the approved provider failed and no other provider is
    # consulted; retrieved evidence is never sent to a fallback provider.
    assert secondary.prompts == []
    assert secondary.system_prompts == []
    assert outcome.to_rag_trace()["runtime"]["fallback_hops"] == 0


def test_injection_cases_preserve_closed_outcome_and_snapshot_identity() -> None:
    seeded = [case for case in ADVERSARIAL_INJECTION_CASES if case.document_source is not None]
    for case in seeded:
        retriever = _RecordingRetriever([_candidate(case)])
        provider = _RecordingProvider()
        outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

        assert outcome.kind in {
            AnswerOutcomeKind.EVIDENCE_GATED_ANSWER,
            AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY,
            AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY,
            AnswerOutcomeKind.GENERATION_UNAVAILABLE,
        }
        trace = outcome.to_rag_trace()
        assert trace["outcome"] == outcome.kind.value
        assert [item["content_preview"] for item in trace["evidence"]] == [item.excerpt for item in outcome.evidence]
        assert [source["excerpt"] for source in outcome.evidence_summary()["sources"]] == [
            item.excerpt for item in outcome.evidence
        ]
        assert [item.source_id for item in outcome.evidence] == ["chunk-1"]
        with pytest.raises(AttributeError):
            outcome.evidence[0].excerpt = "rewritten"  # type: ignore[misc]
