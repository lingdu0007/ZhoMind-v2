import asyncio

import pytest

from app.common.config import get_settings
from app.extensions.provider_router import ProviderRouter
from app.rag.answer_evidence import evidence_snapshot_id, evidence_summary_from_trace
from app.rag.answer_execution import (
    AnswerOutcomeKind,
    EvidenceGatedAnswerExecutor,
)
from app.retrieval.policy import LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID
from app.settings.runtime import get_system_settings_runtime


@pytest.fixture(autouse=True)
def _run_legacy_execution_fixtures_under_the_explicit_migration_profile(
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


def _agent_candidate(
    *,
    availability: str = "verified",
    section_id: str = "stable-principle",
    review_date: str = "2026-08-12",
    evidence_conflict: str | None = None,
) -> dict:
    return {
        "chunk_id": "internal-chunk-1",
        "document_id": "internal-document-1",
        "generation": 2,
        "chunk_index": 0,
        "score": 9.5,
        "content_preview": "已知路径应由 deterministic workflow 控制。",
        "metadata": {
            "title": "Prefer deterministic workflows",
            "publication_version": "v2",
            "entry_id": "pae-workflow-001",
            "entry_title": "Prefer deterministic workflows",
            "domain": "workflow-vs-agent",
            "section_id": section_id,
            "source_title": "Building effective agents",
            "source_authority": "Anthropic",
            "source_url": "https://www.anthropic.com/engineering/building-effective-agents",
            "source_version": "2024-12-19",
            "review_date": review_date,
            "review_status": "approved",
            "source_availability": availability,
            "evidence_conflict": evidence_conflict,
        },
        "retrieval_source": "lexical",
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
    assert all(len(item["snapshot_id"]) == 64 for item in trace["evidence"])
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


def test_agent_answer_fails_closed_until_a_real_evidence_gate_exists() -> None:
    answer = """## 建议
使用 deterministic workflow。[S1]

## 适用边界
仅适用于执行路径已知的任务。[S1]

## 备选方案
运行时路径未知时使用 bounded Agent。[S1]

## 最小实现或验收检查
固定输入应可重放并有 step budget。[S1]"""
    outcome = _execute(
        _executor(retriever=_RecordingRetriever([_agent_candidate()]), provider=_RecordingProvider(answer=answer)),
        question="什么时候使用 deterministic workflow？",
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "reject_evidence_gate_unavailable"
    assert outcome.evidence == ()


def test_agent_answer_fails_closed_for_invalid_summary_or_unavailable_source() -> None:
    invalid_summary = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate()]),
            provider=_RecordingProvider(answer="没有结构或 citation marker 的回答"),
        )
    )
    unavailable = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate(availability="unavailable")]),
            provider=_RecordingProvider(),
        )
    )

    assert invalid_summary.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert unavailable.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY


def test_agent_answer_rejects_stale_version_mapping_and_unresolved_evidence_conflict() -> None:
    stale = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate(section_id="version-mapping", review_date="2020-01-01")]),
            provider=_RecordingProvider(),
        )
    )
    conflict = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate(evidence_conflict="unresolved")]),
            provider=_RecordingProvider(),
        )
    )

    assert stale.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert conflict.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY


def test_agent_answer_rejects_unknowns_section_as_answer_authority() -> None:
    outcome = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate(section_id="evidence-conflicts-and-unknowns")]),
            provider=_RecordingProvider(),
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY


def test_agent_implementation_request_fails_closed_without_verified_evidence_gate() -> None:
    answer = """【Evidence-Bounded Implementation Aid】

## 建议
使用显式 workflow boundary。[S1]

## 适用边界
只覆盖证据中的已知执行路径。[S1]

## 备选方案
动态路径可改用 bounded Agent。[S1]

## 最小实现或验收检查
```python
mode = "workflow"
```
[S1]

## 缺失条件与版本范围
缺少实际 workload 与 provider version，不能视为 production-ready。[S1]"""
    outcome = _execute(
        _executor(retriever=_RecordingRetriever([_agent_candidate()]), provider=_RecordingProvider(answer=answer)),
        question="请给我 implementation checklist 和 Python 代码",
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "reject_evidence_gate_unavailable"


def test_snapshot_identity_binds_public_source_identity() -> None:
    base = {
        "title": "同一标题",
        "publication_version": "v1",
        "excerpt": "同一摘录",
    }
    first = evidence_snapshot_id(**base, citation_metadata={"entry_id": "entry-1", "source_url": "https://a.example"})
    second = evidence_snapshot_id(**base, citation_metadata={"entry_id": "entry-2", "source_url": "https://b.example"})

    assert first != second


def test_summary_recomputes_snapshot_identity_instead_of_trusting_trace() -> None:
    candidate = _agent_candidate()
    evidence = {
        "chunk_id": candidate["chunk_id"],
        "generation": candidate["generation"],
        "metadata": candidate["metadata"],
        "content_preview": candidate["content_preview"],
        "snapshot_id": "forged-snapshot-id",
    }

    summary = evidence_summary_from_trace(
        {"outcome": "evidence_gated_answer", "gate": {"passed": True}, "evidence": [evidence]}
    )

    source = summary["sources"][0]
    assert source["snapshot_id"] != "forged-snapshot-id"
    assert "provider_prompt_snapshot_ids" not in summary
    assert "provider_generation_envelope" not in summary


def test_agent_evidence_rejects_an_uncalibrated_gate_before_generation() -> None:
    class _UncalibratedJudge:
        async def judge(self, query: str, context: list[dict]) -> bool:
            return True

    provider = _RecordingProvider()
    outcome = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate()]),
            provider=provider,
            judge=_UncalibratedJudge(),  # type: ignore[arg-type]
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "reject_evidence_gate_unavailable"
    assert provider.prompts == []


def test_agent_evidence_rejects_any_unverified_gate_before_generation() -> None:
    class _SelfDeclaredGate:
        async def judge(self, query: str, context: list[dict]) -> bool:
            return True

    provider = _RecordingProvider()
    outcome = _execute(
        _executor(
            retriever=_RecordingRetriever([_agent_candidate()]),
            provider=provider,
            judge=_SelfDeclaredGate(),  # type: ignore[arg-type]
        )
    )

    assert outcome.kind is AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
    assert outcome.gate_reason == "reject_evidence_gate_unavailable"
    assert provider.prompts == []


def test_summary_hides_observed_envelope_when_citations_do_not_match() -> None:
    candidate = _agent_candidate()
    evidence = {
        "chunk_id": candidate["chunk_id"],
        "generation": candidate["generation"],
        "metadata": candidate["metadata"],
        "content_preview": candidate["content_preview"],
    }

    summary = evidence_summary_from_trace(
        {
            "outcome": "evidence_gated_answer",
            "gate": {"passed": True},
            "evidence": [evidence],
            "runtime": {
                "provider_generation_envelope": {
                    "identity": "a" * 64,
                    "snapshot_ids": ["b" * 64],
                    "source_count": 1,
                }
            },
        }
    )

    assert "provider_prompt_snapshot_ids" not in summary
    assert "provider_generation_envelope" not in summary


def test_summary_hides_malformed_generation_envelope_identity() -> None:
    candidate = _agent_candidate()
    evidence = {
        "chunk_id": candidate["chunk_id"],
        "generation": candidate["generation"],
        "metadata": candidate["metadata"],
        "content_preview": candidate["content_preview"],
    }

    summary = evidence_summary_from_trace(
        {
            "outcome": "evidence_gated_answer",
            "gate": {"passed": True},
            "evidence": [evidence],
            "runtime": {
                "provider_generation_envelope": {
                    "identity": "g" * 64,
                    "snapshot_ids": [evidence_snapshot_id(
                        title=candidate["metadata"]["title"],
                        publication_version=candidate["metadata"]["publication_version"],
                        excerpt=candidate["content_preview"],
                        citation_metadata=candidate["metadata"],
                    )],
                    "source_count": 1,
                }
            },
        }
    )

    assert "provider_generation_envelope" not in summary
