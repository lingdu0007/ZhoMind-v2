import asyncio

from app.rag.runtime.graph_runner import RagGraphRunner


def test_tool_steps_are_reserved_even_when_disabled() -> None:
    runner = RagGraphRunner(enable_tools=False)
    result = asyncio.run(
        runner.run(request_id="rid-tool", user_id="u-tool", session_id="s-tool", question="查询外部信息")
    )
    steps = [item["step"] for item in result["steps"]]
    assert "tool_plan" in steps
    assert "tool_execute" in steps
    assert "tool_verify" in steps
