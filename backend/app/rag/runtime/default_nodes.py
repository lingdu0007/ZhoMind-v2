from app.rag.answer_evidence import select_answer_evidence
from app.rag.dense_contract import dense_mode_active
from app.rag.runtime.provider_adapters import JudgeAdapter, RerankerAdapter, RetrieverAdapter
from app.rag.runtime.state import ProviderTraceDetail, RagStateDict
from app.settings.runtime import get_runtime_settings


class NormalizeNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        state["query_norm"] = state["query_raw"].strip()
        state["trace_steps"].append({"step": "normalize", "detail": {"ok": True}})
        return state


class QueryUnderstandNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        query = state["query_norm"]
        state["retrieval_plan"] = {
            "intent": "qa",
            "query": query,
            "language": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in query) else "unknown",
        }
        state["trace_steps"].append(
            {
                "step": "query_understand",
                "detail": {
                    "intent": state["retrieval_plan"]["intent"],
                    "language": state["retrieval_plan"]["language"],
                },
            }
        )
        return state


class RetrievalPlanNode:
    def __init__(self, *, default_top_k: int = 3) -> None:
        self.default_top_k = default_top_k

    async def run(self, state: RagStateDict) -> RagStateDict:
        plan = state.get("retrieval_plan") or {}
        plan["strategy"] = "dense_plus_lexical_migration" if dense_mode_active(get_runtime_settings()) else "sparse_only"
        plan["top_k"] = int(plan.get("top_k") or self.default_top_k)
        state["retrieval_plan"] = plan
        state["trace_steps"].append(
            {
                "step": "plan",
                "detail": {
                    "strategy": plan["strategy"],
                    "top_k": plan["top_k"],
                },
            }
        )
        return state


class RetrieveNode:
    def __init__(self, retriever: RetrieverAdapter, *, top_k: int = 3) -> None:
        self.retriever = retriever
        self.top_k = top_k

    async def run(self, state: RagStateDict) -> RagStateDict:
        plan = state.get("retrieval_plan") or {}
        top_k = int(plan.get("top_k") or self.top_k)

        retrieved, exec_detail = await self.retriever.retrieve(state["query_norm"], top_k=top_k)
        ordered_items = []
        for index, item in enumerate(retrieved.items):
            ordered_item = dict(item)
            ordered_item["__retrieval_order"] = int(item.get("__retrieval_order", index))
            ordered_items.append(ordered_item)

        dense_items = [item for item in ordered_items if item.get("retrieval_source") == "dense"]
        sparse_items = [item for item in ordered_items if item.get("retrieval_source") != "dense"]

        state["candidates_sparse"] = sparse_items
        state["candidates_dense"] = dense_items
        retrieve_detail: ProviderTraceDetail = {
            "strategy": retrieved.strategy,
            "dense_candidate_count": retrieved.dense_candidate_count,
            "dense_hydrated_count": retrieved.dense_hydrated_count,
            "lexical_candidate_count": retrieved.lexical_candidate_count,
            "merged_count": retrieved.merged_count,
            "dense_query_failed": retrieved.dense_query_failed,
            "lexical_scope": retrieved.lexical_scope,
            "sparse_count": len(state["candidates_sparse"]),
            "dense_count": len(state["candidates_dense"]),
            "provider": exec_detail["provider"],
            "fallback_used": exec_detail["fallback_used"],
            "provider_error": exec_detail["error"],
            "embedding_provider_ms": retrieved.embedding_provider_ms,
        }
        state["provider_trace"]["retrieve"] = retrieve_detail
        state["trace_steps"].append(
            {
                "step": "retrieve",
                "detail": retrieve_detail,
            }
        )
        return state



class FusionNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        merged = [*state["candidates_dense"], *state["candidates_sparse"]]
        merged.sort(key=lambda item: int(item.get("__retrieval_order", 0)))
        deduped: list[dict] = []
        seen: set[str] = set()

        for item in merged:
            key = str(item.get("chunk_id") or item.get("id") or "")
            if not key:
                key = f"anon-{len(deduped)}"
            if key in seen:
                continue
            seen.add(key)
            normalized_item = dict(item)
            normalized_item.pop("__retrieval_order", None)
            deduped.append(normalized_item)

        state["candidates_fused"] = deduped
        state["trace_steps"].append(
            {
                "step": "fusion",
                "detail": {
                    "merged": len(merged),
                    "deduped": len(deduped),
                },
            }
        )
        return state


class RerankNode:
    def __init__(self, reranker: RerankerAdapter) -> None:
        self.reranker = reranker

    async def run(self, state: RagStateDict) -> RagStateDict:
        items = state["candidates_fused"]
        reranked, exec_detail = await self.reranker.rerank(state["query_norm"], items)
        state["candidates_reranked"] = reranked
        rerank_detail: ProviderTraceDetail = {
            "provider": exec_detail["provider"],
            "fallback_used": exec_detail["fallback_used"],
            "provider_error": exec_detail["error"],
        }
        state["provider_trace"]["rerank"] = rerank_detail
        state["trace_steps"].append(
            {
                "step": "rerank",
                "detail": {
                    "reranked_count": len(reranked),
                    "provider": exec_detail["provider"],
                    "fallback_used": exec_detail["fallback_used"],
                    "provider_error": exec_detail["error"],
                },
            }
        )
        return state



class VerifyNode:
    def __init__(self, judge: JudgeAdapter) -> None:
        self.judge = judge

    async def run(self, state: RagStateDict) -> RagStateDict:
        if state["evidence_pack"]:
            passed, exec_detail = await self.judge.judge(state["query_norm"], state["evidence_pack"])
        else:
            passed = False
            exec_detail = {
                "provider": self.judge.provider_name,
                "fallback_used": False,
                "error": None,
            }
        state["gate_result"] = {
            "passed": passed,
            "reason": "sufficient_evidence" if passed else "reject_insufficient_evidence",
        }
        verify_detail: ProviderTraceDetail = {
            "provider": exec_detail["provider"],
            "fallback_used": exec_detail["fallback_used"],
            "provider_error": exec_detail["error"],
        }
        state["provider_trace"]["verify"] = verify_detail
        state["trace_steps"].append(
            {
                "step": "verify",
                "detail": {
                    **state["gate_result"],
                    "provider": exec_detail["provider"],
                    "fallback_used": exec_detail["fallback_used"],
                    "provider_error": exec_detail["error"],
                },
            }
        )
        return state


class ContextPackNode:
    def __init__(self, *, top_k: int = 3, max_excerpt_chars: int = 160) -> None:
        self.top_k = top_k
        self.max_excerpt_chars = max_excerpt_chars

    async def run(self, state: RagStateDict) -> RagStateDict:
        eligible_candidates = [
            candidate
            for candidate in state["candidates_reranked"]
            if candidate.get("answer_evidence_eligible") is not False
        ]
        evidence = select_answer_evidence(
            eligible_candidates,
            max_items=self.top_k,
            max_excerpt_chars=self.max_excerpt_chars,
        )
        state["evidence_pack"] = [item.to_record() for item in evidence]
        state["trace_steps"].append(
            {
                "step": "context_pack",
                "detail": {
                    "evidence_count": len(state["evidence_pack"]),
                },
            }
        )
        return state


class GenerateNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        state["trace_steps"].append(
            {
                "step": "generate",
                "detail": {
                    "used_evidence": len(state["evidence_pack"]),
                },
            }
        )
        return state


async def add_tool_reservation_steps(state: RagStateDict, *, enabled: bool) -> RagStateDict:
    budget = state.get("tool_budget") or {}
    budget_meta = {
        "max_calls": int(budget.get("max_calls", 0)),
        "max_parallel": int(budget.get("max_parallel", 0)),
        "max_latency_ms": int(budget.get("max_latency_ms", 0)),
    }

    state["trace_steps"].append(
        {
            "step": "tool_plan",
            "detail": {
                "enabled": enabled,
                "items": len(state["tool_plan"]),
                **budget_meta,
            },
        }
    )
    state["trace_steps"].append(
        {
            "step": "tool_execute",
            "detail": {
                "enabled": enabled,
                "calls": len(state["tool_calls"]),
                **budget_meta,
            },
        }
    )
    state["trace_steps"].append(
        {
            "step": "tool_verify",
            "detail": {
                "enabled": enabled,
                "errors": len(state["tool_errors"]),
                "tool_errors": list(state["tool_errors"]),
            },
        }
    )
    return state


class MemoryWriteNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        state["trace_steps"].append(
            {
                "step": "memory_write_gate",
                "detail": state.get("memory_write_decision") or {"allow": False, "reason": "not_evaluated"},
            }
        )
        return state


class FinalizeNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        state["trace_steps"].append({"step": "finalize", "detail": {"ok": True}})
        return state
