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

from app.rag.runtime.graph_runner import RagGraphRunner


class _RetrieverBoom:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        raise RuntimeError("retriever exploded")


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
