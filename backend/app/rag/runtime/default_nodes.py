from app.rag.runtime.provider_adapters import JudgeAdapter, RerankerAdapter, RetrieverAdapter
from app.rag.runtime.state import RagStateDict


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
        plan["strategy"] = "sparse_only"
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

        state["candidates_sparse"] = retrieved
        state["candidates_dense"] = []
        state["provider_trace"]["retrieve"] = {
            "provider": exec_detail["provider"],
            "fallback_used": exec_detail["fallback_used"],
            "provider_error": exec_detail["error"],
        }
        state["trace_steps"].append(
            {
                "step": "retrieve",
                "detail": {
                    "strategy": plan.get("strategy", "sparse_only"),
                    "sparse_count": len(state["candidates_sparse"]),
                    "dense_count": len(state["candidates_dense"]),
                    "provider": exec_detail["provider"],
                    "fallback_used": exec_detail["fallback_used"],
                    "provider_error": exec_detail["error"],
                },
            }
        )
        return state



class FusionNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        merged = [*state["candidates_sparse"], *state["candidates_dense"]]
        deduped: list[dict] = []
        seen: set[str] = set()

        for item in merged:
            key = str(item.get("chunk_id") or item.get("id") or "")
            if not key:
                key = f"anon-{len(deduped)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

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
        state["provider_trace"]["rerank"] = {
            "provider": exec_detail["provider"],
            "fallback_used": exec_detail["fallback_used"],
            "provider_error": exec_detail["error"],
        }
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
        passed, exec_detail = await self.judge.judge(state["query_norm"], state["candidates_reranked"])
        state["gate_result"] = {
            "passed": passed,
            "reason": "sufficient_evidence" if passed else "reject_insufficient_evidence",
        }
        state["provider_trace"]["verify"] = {
            "provider": exec_detail["provider"],
            "fallback_used": exec_detail["fallback_used"],
            "provider_error": exec_detail["error"],
        }
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
    def __init__(self, *, top_k: int = 3) -> None:
        self.top_k = top_k

    async def run(self, state: RagStateDict) -> RagStateDict:
        state["evidence_pack"] = list(state["candidates_reranked"][: self.top_k])
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
        if state["gate_result"].get("passed") and state["evidence_pack"]:
            lines = ["根据检索到的知识片段，先给你一个最小可用回答："]
            for idx, item in enumerate(state["evidence_pack"][:3], start=1):
                content = str(item.get("content_preview") or item.get("content") or "")
                lines.append(f"{idx}. {content}")
            state["answer"] = "\n".join(lines)

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
        if not state["gate_result"].get("passed"):
            state["answer"] = "未检索到足够相关的知识片段，请补充更具体的问题或关键词。"
        elif not state["answer"]:
            state["answer"] = "根据检索到的知识片段，先给你一个最小可用回答："

        state["trace_steps"].append({"step": "finalize", "detail": {"ok": True}})
        return state
