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


def test_tool_runtime_uses_budget_when_enabled() -> None:
    runner = RagGraphRunner(enable_tools=True, tool_max_calls=2, tool_max_parallel=2, tool_timeout_ms=5000)
    result = asyncio.run(
        runner.run(request_id="rid-tool-2", user_id="u-tool", session_id="s-tool", question="查询外部信息")
    )

    tool_plan_step = next(item for item in result["steps"] if item["step"] == "tool_plan")
    tool_execute_step = next(item for item in result["steps"] if item["step"] == "tool_execute")

    assert tool_plan_step["detail"]["enabled"] is True
    assert tool_plan_step["detail"]["items"] == 1
    assert tool_execute_step["detail"]["calls"] == 1


def test_tool_runtime_skips_when_budget_zero() -> None:
    runner = RagGraphRunner(enable_tools=True, tool_max_calls=0, tool_max_parallel=2, tool_timeout_ms=5000)
    result = asyncio.run(
        runner.run(request_id="rid-tool-3", user_id="u-tool", session_id="s-tool", question="查询外部信息")
    )

    tool_plan_step = next(item for item in result["steps"] if item["step"] == "tool_plan")
    tool_execute_step = next(item for item in result["steps"] if item["step"] == "tool_execute")

    assert tool_plan_step["detail"]["items"] == 0
    assert tool_execute_step["detail"]["calls"] == 0
