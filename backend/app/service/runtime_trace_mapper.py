from typing import Any


class RuntimeTraceMapper:
    @staticmethod
    def map_runtime(runtime_result: dict) -> dict[str, Any]:
        runtime_steps = list(runtime_result.get("steps") or [])
        trace: dict[str, Any] = {
            "request_id": runtime_result.get("request_id"),
            "session_id": runtime_result.get("session_id"),
            "graph_alias": runtime_result.get("graph_alias"),
            "gate": runtime_result.get("gate") or {},
            "steps": runtime_steps,
            "step_names": [str(item.get("step") or "") for item in runtime_steps],
        }
        if runtime_result.get("tool_budget") is not None:
            trace["tool_budget"] = runtime_result.get("tool_budget")
        if runtime_result.get("tool_errors") is not None:
            trace["tool_errors"] = list(runtime_result.get("tool_errors") or [])
        if runtime_result.get("provider_trace") is not None:
            trace["provider_trace"] = runtime_result.get("provider_trace")
        trace["final_provider"] = runtime_result.get("final_provider")
        trace["provider_attempts"] = list(runtime_result.get("provider_attempts") or [])
        trace["fallback_hops"] = int(runtime_result.get("fallback_hops") or 0)
        return trace
