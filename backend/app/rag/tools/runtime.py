class ToolExecutionRuntime:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    async def plan(self, query: str) -> list[dict]:
        if not self.enabled:
            return []
        return [{"tool_name": "reserved", "reason": "future_mcp_integration"}]

    async def execute(self, plan: list[dict]) -> tuple[list[dict], list[dict]]:
        if not self.enabled:
            return [], []
        return [], []
