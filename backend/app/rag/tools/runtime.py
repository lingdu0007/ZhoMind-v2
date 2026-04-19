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

    def _normalize_execution_error(self, *, tool_name: str, exc: Exception) -> dict:
        message = str(exc).strip() or "tool execution failed"
        return {
            "tool_name": tool_name,
            "code": "TOOL_EXEC_FAILED",
            "message": message,
            "type": exc.__class__.__name__,
            "retryable": False,
        }

    async def _execute_item(self, item: dict) -> dict:
        if not isinstance(item, dict):
            raise TypeError("tool plan item must be a mapping")

        return {
            "tool_name": item.get("tool_name", "unknown"),
            "status": "reserved_noop",
        }

    async def execute(self, plan: list[dict]) -> tuple[list[dict], list[dict]]:
        if not self.enabled or self.max_calls <= 0:
            return [], []

        bounded_plan = plan[: self.max_calls]
        calls: list[dict] = []
        errors: list[dict] = []

        for item in bounded_plan:
            tool_name = item.get("tool_name", "unknown") if isinstance(item, dict) else "unknown"
            try:
                call = await self._execute_item(item)
                calls.append(call)
            except Exception as exc:
                errors.append(self._normalize_execution_error(tool_name=tool_name, exc=exc))

        return calls, errors

