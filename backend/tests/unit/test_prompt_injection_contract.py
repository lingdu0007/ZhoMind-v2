import asyncio

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


def test_legacy_migration_override_instruction_cannot_open_generation() -> None:
    case = _seeded_case(INSTRUCTION_OVERRIDE)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()
    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert provider.prompts == []
    assert provider.system_prompts == []


def test_legacy_migration_secret_fixture_never_reaches_a_provider() -> None:
    case = _seeded_case(SECRET_EXTRACTION)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()
    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert provider.prompts == []
    assert provider.system_prompts == []


def test_legacy_migration_forged_source_cannot_open_generation() -> None:
    case = _seeded_case(FORGED_SOURCE)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()
    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert provider.prompts == []
    assert provider.system_prompts == []


@pytest.mark.parametrize("kind", [FORGED_TOOL_CALL, UNSAFE_CODE])
def test_legacy_migration_effect_instructions_cannot_open_generation(kind: str) -> None:
    case = _seeded_case(kind)
    assert case.answer_marker is not None
    retriever = _RecordingRetriever([_candidate(case)])
    provider = _RecordingProvider()

    outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert provider.prompts == []
    assert provider.system_prompts == []


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


def test_legacy_migration_never_reports_generation_unavailable() -> None:
    case = _seeded_case(FORGED_SOURCE)
    retriever = _RecordingRetriever([_candidate(case)])
    primary = _RecordingProvider(error=TimeoutError("upstream timeout"))
    secondary = _RecordingProvider()

    outcome = _execute(
        _executor(retriever=retriever, provider=primary, secondary=secondary),
        case.question,
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "no_eligible_published_evidence"
    assert primary.prompts == []
    assert primary.system_prompts == []
    assert secondary.prompts == []
    assert secondary.system_prompts == []


def test_legacy_migration_injection_cases_freeze_explicit_insufficiency() -> None:
    seeded = [case for case in ADVERSARIAL_INJECTION_CASES if case.document_source is not None]
    for case in seeded:
        retriever = _RecordingRetriever([_candidate(case)])
        provider = _RecordingProvider()
        outcome = _execute(_executor(retriever=retriever, provider=provider), case.question)

        assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
        assert outcome.gate_reason == "no_eligible_published_evidence"
        assert outcome.evidence == ()
        assert outcome.evidence_set is None
        assert provider.prompts == []
        assert provider.system_prompts == []
        trace = outcome.to_rag_trace()
        assert trace["outcome"] == outcome.kind.value
        assert outcome.evidence_summary() == {"coverage": "insufficient", "source_count": 0, "sources": []}
