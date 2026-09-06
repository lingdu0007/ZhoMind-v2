from typing import Any, NotRequired, TypedDict

from app.rag.evidence_sufficiency import EvidenceSufficiencyDecision, QueryConditionSet
from app.rag.interfaces import ProviderExecError


class ProviderTraceDetail(TypedDict):
    provider: str
    fallback_used: bool
    provider_error: ProviderExecError | None
    # Additional detail fields are retrieval-stage specific; keep them optional
    # so every stage can share the same TypedDict without a runtime schema.
    strategy: NotRequired[str]
    dense_candidate_count: NotRequired[int]
    dense_hydrated_count: NotRequired[int]
    lexical_candidate_count: NotRequired[int]
    merged_count: NotRequired[int]
    dense_query_failed: NotRequired[bool]
    lexical_scope: NotRequired[str]
    sparse_count: NotRequired[int]
    dense_count: NotRequired[int]
    embedding_provider_ms: NotRequired[float]
    profile_identity: NotRequired[str | None]
    candidate_pool_scope: NotRequired[str | None]
    candidate_exclusions: NotRequired[list[str]]


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
    query_condition_set: QueryConditionSet
    evidence_sufficiency_decision: EvidenceSufficiencyDecision | None
    retrieval_profile_identity: str | None
    candidate_pool_scope: str | None
    gate_result: dict[str, Any]
    claim_evidence_audit: dict[str, Any]
    evidence_pack: list[dict]
    answer: str
    memory_read_set: dict[str, Any]
    memory_write_decision: dict[str, Any]
    trace_steps: list[dict[str, Any]]
    provider_trace: dict[str, ProviderTraceDetail]
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
            "query_condition_set": QueryConditionSet.from_question(query_norm),
            "evidence_sufficiency_decision": None,
            "retrieval_profile_identity": None,
            "candidate_pool_scope": None,
            "gate_result": {"passed": False, "reason": "not_checked"},
            "claim_evidence_audit": {},
            "evidence_pack": [],
            "answer": "",
            "memory_read_set": {},
            "memory_write_decision": {},
            "trace_steps": [],
            "provider_trace": {},
            "latency_ms": 0,
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "tool_plan": [],
            "tool_calls": [],
            "tool_observations": [],
            "tool_errors": [],
            "external_evidence": [],
            "tool_budget": {"max_calls": 3, "max_parallel": 2, "max_latency_ms": 8000},
        }
