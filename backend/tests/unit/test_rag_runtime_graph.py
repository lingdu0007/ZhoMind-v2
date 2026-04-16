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
    assert "gate" in result
    assert "steps" in result
    assert "answer" in result
    assert isinstance(result["steps"], list)
