class ToolExecutionRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        max_calls: int,
        max_parallel: int,
        timeout_ms: int,
    ) -> None:
        self.enabled = enabled
        self.max_calls = max(0, int(max_calls))
        self.max_parallel = max(1, int(max_parallel))
        self.timeout_ms = max(1, int(timeout_ms))

    async def plan(self, query: str) -> list[dict]:
        if not self.enabled or self.max_calls <= 0:
            return []
        return [
            {
                "tool_name": "reserved",
                "reason": "future_mcp_integration",
                "query": query,
                "max_parallel": self.max_parallel,
                "timeout_ms": self.timeout_ms,
            }
        ]

    async def execute(self, plan: list[dict]) -> tuple[list[dict], list[dict]]:
        if not self.enabled or self.max_calls <= 0:
            return [], []

        bounded_plan = plan[: self.max_calls]
        calls = [
            {
                "tool_name": item.get("tool_name", "unknown"),
                "status": "reserved_noop",
            }
            for item in bounded_plan
        ]
        return calls, []
