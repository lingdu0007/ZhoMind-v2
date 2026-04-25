from app.rag.runtime.state import RagState


def test_rag_state_has_required_keys() -> None:
    state = RagState.new(
        request_id="rid-1",
        user_id="u1",
        session_id="s1",
        query_raw="你好",
    )
    assert state["request_id"] == "rid-1"
    assert state["query_norm"] == "你好"
    assert state["trace_steps"] == []
    assert state["tool_calls"] == []


import asyncio

import pytest

from app.rag.interfaces import RetrieveResult
from app.rag.runtime.graph_runner import RagGraphRunner


class _RetrieverBoom:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        raise RuntimeError("retriever exploded")


class _RetrieverDenseFallback:
    async def retrieve(self, query: str, top_k: int) -> RetrieveResult:
        return RetrieveResult(
            items=[
                {
                    "chunk_id": "lexical-fallback-1",
                    "document_id": "doc-lexical",
                    "chunk_index": 0,
                    "score": 0.7,
                    "content_preview": "lexical fallback content",
                    "metadata": {},
                    "retrieval_source": "lexical",
                }
            ],
            strategy="dense_plus_lexical_migration",
            dense_candidate_count=0,
            dense_hydrated_count=0,
            lexical_candidate_count=1,
            merged_count=1,
            dense_query_failed=True,
            lexical_scope="full_published_live",
            fallback_used=True,
            provider_error={
                "code": "PROVIDER_EXEC_FAILED",
                "message": "milvus unavailable",
                "type": "RuntimeError",
            },
        )


class _RetrieverMixedOrder:
    async def retrieve(self, query: str, top_k: int) -> RetrieveResult:
        return RetrieveResult(
            items=[
                {
                    "chunk_id": "lexical-first",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "score": 0.9,
                    "content_preview": "lexical first",
                    "metadata": {},
                    "retrieval_source": "lexical",
                },
                {
                    "chunk_id": "dense-second",
                    "document_id": "doc-2",
                    "chunk_index": 0,
                    "score": 0.8,
                    "content_preview": "dense second",
                    "metadata": {},
                    "retrieval_source": "dense",
                },
                {
                    "chunk_id": "lexical-third",
                    "document_id": "doc-3",
                    "chunk_index": 0,
                    "score": 0.7,
                    "content_preview": "lexical third",
                    "metadata": {},
                    "retrieval_source": "lexical",
                },
            ],
            strategy="dense_plus_lexical_migration",
            dense_candidate_count=1,
            dense_hydrated_count=1,
            lexical_candidate_count=2,
            merged_count=3,
            dense_query_failed=False,
            lexical_scope="not_dense_ready_published",
        )


def test_graph_runner_returns_gate_and_answer() -> None:
    runner = RagGraphRunner()
    result = asyncio.run(
        runner.run(
            request_id="rid-2",
            user_id="u2",
            session_id="s2",
            question="测试系统状态",
        )
    )
    assert "request_id" in result
    assert "session_id" in result
    assert result["request_id"] == "rid-2"
    assert result["session_id"] == "s2"
    assert "gate" in result
    assert "steps" in result
    assert "answer" in result
    assert "graph_alias" in result
    assert result["graph_alias"] == "default_v1"
    assert isinstance(result["steps"], list)

    steps = {item["step"]: item for item in result["steps"]}
    assert "memory_read" in steps
    assert "memory_write_gate" in steps
    assert steps["memory_write_gate"]["detail"]["allow"] is False
    assert steps["memory_write_gate"]["detail"]["reason"] == "unsafe"


def test_graph_runner_retriever_failure_uses_fallback_trace_and_rejects() -> None:
    runner = RagGraphRunner(retriever=_RetrieverBoom())
    result = asyncio.run(
        runner.run(
            request_id="rid-3",
            user_id="u3",
            session_id="s3",
            question="测试检索失败",
        )
    )

    assert result["request_id"] == "rid-3"
    assert result["session_id"] == "s3"
    assert result["graph_alias"] == "default_v1"

    step_order = [step["step"] for step in result["steps"]]
    assert step_order == [
        "normalize",
        "memory_read",
        "query_understand",
        "plan",
        "tool_plan",
        "tool_execute",
        "tool_verify",
        "retrieve",
        "fusion",
        "rerank",
        "verify",
        "context_pack",
        "generate",
        "memory_write_gate",
        "finalize",
    ]

    retrieve_step = next(step for step in result["steps"] if step["step"] == "retrieve")
    assert retrieve_step["detail"]["fallback_used"] is True
    assert result["gate"]["reason"] == "reject_insufficient_evidence"


def test_graph_runner_langgraph_path_when_available() -> None:
    runner = RagGraphRunner()

    if runner._compiled_graph is None:
        pytest.skip("LangGraph is optional; sequential fallback is expected when dependency is unavailable")

    assert runner._compiled_graph is not None

    result = asyncio.run(
        runner.run(
            request_id="rid-4",
            user_id="u4",
            session_id="s4",
            question="验证 LangGraph 路径",
        )
    )

    assert result["request_id"] == "rid-4"
    assert result["session_id"] == "s4"
    assert result["graph_alias"] == "default_v1"


def test_graph_runner_retrieve_step_exposes_mixed_mode_diagnostics() -> None:
    runner = RagGraphRunner()

    result = asyncio.run(
        runner.run(
            request_id="rid-5",
            user_id="u5",
            session_id="s5",
            question="无知识库命中时也要输出诊断",
        )
    )

    retrieve_step = next(step for step in result["steps"] if step["step"] == "retrieve")
    assert retrieve_step["detail"]["strategy"] == "sparse_only"
    assert retrieve_step["detail"]["dense_candidate_count"] == 0
    assert retrieve_step["detail"]["dense_hydrated_count"] == 0
    assert retrieve_step["detail"]["lexical_candidate_count"] == 0
    assert retrieve_step["detail"]["merged_count"] == 0
    assert retrieve_step["detail"]["dense_query_failed"] is False
    assert retrieve_step["detail"]["lexical_scope"] == "full_published_live"

    provider_trace = result["provider_trace"]["retrieve"]
    assert provider_trace["strategy"] == "sparse_only"
    assert provider_trace["dense_candidate_count"] == 0
    assert provider_trace["dense_hydrated_count"] == 0
    assert provider_trace["lexical_candidate_count"] == 0
    assert provider_trace["merged_count"] == 0
    assert provider_trace["dense_query_failed"] is False
    assert provider_trace["lexical_scope"] == "full_published_live"


def test_graph_runner_dense_query_failure_marks_runtime_fallback_and_provider_error() -> None:
    runner = RagGraphRunner(retriever=_RetrieverDenseFallback())

    result = asyncio.run(
        runner.run(
            request_id="rid-6",
            user_id="u6",
            session_id="s6",
            question="dense failure fallback",
        )
    )

    retrieve_step = next(step for step in result["steps"] if step["step"] == "retrieve")
    assert retrieve_step["detail"]["dense_query_failed"] is True
    assert retrieve_step["detail"]["fallback_used"] is True
    assert retrieve_step["detail"]["provider_error"] == {
        "code": "PROVIDER_EXEC_FAILED",
        "message": "milvus unavailable",
        "type": "RuntimeError",
    }

    provider_trace = result["provider_trace"]["retrieve"]
    assert provider_trace["dense_query_failed"] is True
    assert provider_trace["fallback_used"] is True
    assert provider_trace["provider_error"] == {
        "code": "PROVIDER_EXEC_FAILED",
        "message": "milvus unavailable",
        "type": "RuntimeError",
    }


def test_graph_runner_preserves_retriever_merged_order() -> None:
    runner = RagGraphRunner(retriever=_RetrieverMixedOrder())

    result = asyncio.run(
        runner.run(
            request_id="rid-7",
            user_id="u7",
            session_id="s7",
            question="preserve order",
        )
    )

    assert [item["chunk_id"] for item in result["retrieved"]] == [
        "lexical-first",
        "dense-second",
        "lexical-third",
    ]
