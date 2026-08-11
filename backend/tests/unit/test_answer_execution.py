import asyncio

import pytest

from app.extensions.provider_router import ProviderRouter
from app.rag.answer_execution import (
    AnswerOutcomeKind,
    EvidenceGatedAnswerExecutor,
)


def _candidate(index: int, *, content: str | None = None) -> dict:
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": f"document-{index}",
        "generation": index,
        "chunk_index": 0,
        "score": float(10 - index),
        "content_preview": content or f"第 {index} 条已发布证据",
        "metadata": {
            "title": f"已发布资料 {index}.md",
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
    def __init__(self) -> None:
        self.contexts: list[list[dict]] = []

    async def judge(self, query: str, context: list[dict]) -> bool:
        self.contexts.append(context)
        return bool(context)


class _RecordingProvider:
    def __init__(self, *, answer: str = "基于证据生成的回答", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer


def _executor(
    *,
    retriever: _RecordingRetriever,
    provider: _RecordingProvider,
    judge: _EvidenceJudge | None = None,
) -> EvidenceGatedAnswerExecutor:
    return EvidenceGatedAnswerExecutor(
        retriever=retriever,
        reranker=_IdentityReranker(),
        judge=judge or _EvidenceJudge(),
        provider_router=ProviderRouter(providers={"approved": provider}),
        primary_provider="approved",
        retriever_name="test-retriever",
        reranker_name="test-reranker",
        judge_name="test-judge",
        retrieval_top_k=5,
        max_evidence_items=3,
        max_excerpt_chars=160,
    )


def _execute(executor: EvidenceGatedAnswerExecutor, question: str = "发布事实是什么？"):
    return asyncio.run(
        executor.execute(
            request_id="answer-request-1",
            user_id="knowledge-user",
            session_id="answer-session-1",
            question=question,
        )
    )


def test_smalltalk_returns_closed_non_knowledge_outcome_before_retrieval() -> None:
    retriever = _RecordingRetriever([_candidate(1)])
    provider = _RecordingProvider()

    outcome = _execute(_executor(retriever=retriever, provider=provider), question="你好")

    assert outcome.kind is AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY
    assert outcome.evidence == ()
    assert outcome.text.startswith("【非知识库回复】")
    assert retriever.calls == []
    assert provider.prompts == []


def test_evidence_gated_answer_uses_one_bounded_snapshot_everywhere() -> None:
    long_content = "  第一条证据\n包含   多余空白。" + ("甲" * 200)
    candidates = [
        _candidate(1, content=long_content),
        {"chunk_id": "diagnostic-only", "content_preview": "缺少发布来源身份"},
        _candidate(2),
        _candidate(3),
        _candidate(4),
    ]
    retriever = _RecordingRetriever(candidates)
    provider = _RecordingProvider()
    judge = _EvidenceJudge()

    outcome = _execute(_executor(retriever=retriever, provider=provider, judge=judge))

    assert outcome.kind is AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
    assert [item.source_id for item in outcome.evidence] == ["chunk-1", "chunk-2", "chunk-3"]
    snapshots = [item.excerpt for item in outcome.evidence]
    assert snapshots[0].startswith("第一条证据 包含 多余空白。")
    assert len(snapshots[0]) == 160
    assert [item["content_preview"] for item in judge.contexts[0]] == snapshots
    assert len(provider.prompts) == 1
    assert all(snapshot in provider.prompts[0] for snapshot in snapshots)
    assert "第 4 条已发布证据" not in provider.prompts[0]

    trace = outcome.to_rag_trace()
    assert trace["outcome"] == "evidence_gated_answer"
    assert set(trace["runtime"]["timing_ms"]) == {
        "retrieval_ms",
        "generation_provider_ms",
        "embedding_provider_ms",
    }
    assert [item["content_preview"] for item in trace["evidence"]] == snapshots
    assert [source["excerpt"] for source in outcome.evidence_summary()["sources"]] == snapshots
    with pytest.raises(TypeError):
        outcome.runtime["final_provider"] = "other-provider"


def test_malformed_candidates_cannot_open_the_evidence_gate() -> None:
    candidates = [
        {"document_id": "missing-source", "generation": 1, "content_preview": "没有 source id"},
        {
            "chunk_id": "missing-title",
            "document_id": "document-2",
            "generation": 1,
            "content_preview": "没有标题",
            "metadata": {"publication_version": "v1"},
        },
        {
            "chunk_id": "missing-excerpt",
            "document_id": "document-3",
            "generation": 1,
            "content_preview": "   ",
            "metadata": {"title": "空证据.md", "publication_version": "v1"},
        },
        {
            "chunk_id": 42,
            "document_id": "numeric-source-id",
            "generation": 1,
            "content_preview": "数字 ID 不能成为稳定来源身份",
            "metadata": {"title": "错误资料.md", "publication_version": "v1"},
        },
    ]
    retriever = _RecordingRetriever(candidates)
    provider = _RecordingProvider()
    judge = _EvidenceJudge()

    outcome = _execute(_executor(retriever=retriever, provider=provider, judge=judge))

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.evidence == ()
    assert outcome.evidence_summary() == {"coverage": "insufficient", "source_count": 0, "sources": []}
    assert judge.contexts == []
    assert provider.prompts == []


def test_generation_unavailable_preserves_the_exact_answer_evidence_set() -> None:
    retriever = _RecordingRetriever([_candidate(1), _candidate(2)])
    provider = _RecordingProvider(error=TimeoutError("upstream timeout"))

    outcome = _execute(_executor(retriever=retriever, provider=provider))

    assert outcome.kind is AnswerOutcomeKind.GENERATION_UNAVAILABLE
    assert outcome.text.startswith("【生成不可用】")
    assert [item.source_id for item in outcome.evidence] == ["chunk-1", "chunk-2"]
    assert outcome.evidence_summary()["coverage"] == "sufficient"
    assert outcome.to_rag_trace()["runtime"]["provider_attempts"][0]["error_code"] == "TimeoutError"
