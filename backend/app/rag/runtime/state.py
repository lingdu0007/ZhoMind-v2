from typing import Any, TypedDict


class RagStateDict(TypedDict):
    request_id: str
    user_id: str
    session_id: str
    query_raw: str
    query_norm: str
    retrieval_plan: dict[str, Any]
    candidates_sparse: list[dict]
    candidates_dense: list[dict]
    candidates_fused: list[dict]
    candidates_reranked: list[dict]
    gate_result: dict[str, Any]
    evidence_pack: list[dict]
    answer: str
    memory_read_set: dict[str, Any]
    memory_write_decision: dict[str, Any]
    trace_steps: list[dict]
    latency_ms: int
    token_usage: dict[str, int]
    tool_plan: list[dict]
    tool_calls: list[dict]
    tool_observations: list[dict]
    tool_errors: list[dict]
    external_evidence: list[dict]
    tool_budget: dict[str, int]


class RagState:
    @staticmethod
    def new(*, request_id: str, user_id: str, session_id: str, query_raw: str) -> RagStateDict:
        query_norm = query_raw.strip()
        return {
            "request_id": request_id,
            "user_id": user_id,
            "session_id": session_id,
            "query_raw": query_raw,
            "query_norm": query_norm,
            "retrieval_plan": {},
            "candidates_sparse": [],
            "candidates_dense": [],
            "candidates_fused": [],
            "candidates_reranked": [],
            "gate_result": {"passed": False, "reason": "not_checked"},
            "evidence_pack": [],
            "answer": "",
            "memory_read_set": {},
            "memory_write_decision": {},
            "trace_steps": [],
            "latency_ms": 0,
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "tool_plan": [],
            "tool_calls": [],
            "tool_observations": [],
            "tool_errors": [],
            "external_evidence": [],
            "tool_budget": {"max_calls": 3, "max_parallel": 2, "max_latency_ms": 8000},
        }
