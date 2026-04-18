import asyncio

from app.rag.runtime.graph_runner import RagGraphRunner
from app.rag.tools.runtime import ToolExecutionRuntime


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
    tool_verify_step = next(item for item in result["steps"] if item["step"] == "tool_verify")

    assert tool_plan_step["detail"]["enabled"] is True
    assert tool_plan_step["detail"]["items"] == 1
    assert tool_execute_step["detail"]["calls"] == 1
    assert tool_plan_step["detail"]["max_calls"] == 2
    assert tool_plan_step["detail"]["max_parallel"] == 2
    assert tool_plan_step["detail"]["max_latency_ms"] == 5000
    assert tool_execute_step["detail"]["max_calls"] == 2
    assert tool_execute_step["detail"]["max_parallel"] == 2
    assert tool_execute_step["detail"]["max_latency_ms"] == 5000
    assert tool_verify_step["detail"]["tool_errors"] == []


def test_tool_runtime_skips_when_budget_zero() -> None:
    runner = RagGraphRunner(enable_tools=True, tool_max_calls=0, tool_max_parallel=2, tool_timeout_ms=5000)
    result = asyncio.run(
        runner.run(request_id="rid-tool-3", user_id="u-tool", session_id="s-tool", question="查询外部信息")
    )

    tool_plan_step = next(item for item in result["steps"] if item["step"] == "tool_plan")
    tool_execute_step = next(item for item in result["steps"] if item["step"] == "tool_execute")

    assert tool_plan_step["detail"]["items"] == 0
    assert tool_execute_step["detail"]["calls"] == 0
    assert tool_plan_step["detail"]["max_calls"] == 0
    assert tool_plan_step["detail"]["max_parallel"] == 2
    assert tool_plan_step["detail"]["max_latency_ms"] == 5000
    assert tool_execute_step["detail"]["max_calls"] == 0
    assert tool_execute_step["detail"]["max_parallel"] == 2
    assert tool_execute_step["detail"]["max_latency_ms"] == 5000


def test_tool_runtime_normalizes_execution_errors() -> None:
    runtime = ToolExecutionRuntime(enabled=True, max_calls=2, max_parallel=1, timeout_ms=1000)

    async def _boom(_: dict) -> dict:
        raise RuntimeError("planned call blew up")

    runtime._execute_item = _boom  # type: ignore[method-assign]

    calls, errors = asyncio.run(runtime.execute([{"tool_name": "search"}]))

    assert calls == []
    assert errors == [
        {
            "tool_name": "search",
            "code": "TOOL_EXEC_FAILED",
            "message": "planned call blew up",
            "type": "RuntimeError",
            "retryable": False,
        }
    ]

