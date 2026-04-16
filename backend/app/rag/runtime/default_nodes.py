from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.runtime.state import RagStateDict


class NormalizeNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        state["query_norm"] = state["query_raw"].strip()
        state["trace_steps"].append({"step": "normalize", "detail": {"ok": True}})
        return state


class RetrieveNode:
    def __init__(self, retriever: Retriever | None, *, top_k: int = 3) -> None:
        self.retriever = retriever
        self.top_k = top_k

    async def run(self, state: RagStateDict) -> RagStateDict:
        if self.retriever is None:
            retrieved: list[dict] = []
        else:
            retrieved = await self.retriever.retrieve(state["query_norm"], top_k=self.top_k)
        state["candidates_sparse"] = retrieved
        state["candidates_fused"] = retrieved
        state["trace_steps"].append({"step": "retrieve", "detail": {"retrieved_count": len(retrieved)}})
        return state


class RerankNode:
    def __init__(self, reranker: Reranker | None) -> None:
        self.reranker = reranker

    async def run(self, state: RagStateDict) -> RagStateDict:
        items = state["candidates_fused"]
        if self.reranker is None:
            reranked = items
        else:
            reranked = await self.reranker.rerank(state["query_norm"], items)
        state["candidates_reranked"] = reranked
        state["trace_steps"].append({"step": "rerank", "detail": {"reranked_count": len(reranked)}})
        return state


class VerifyNode:
    def __init__(self, judge: RelevanceJudge | None) -> None:
        self.judge = judge

    async def run(self, state: RagStateDict) -> RagStateDict:
        if self.judge is None:
            passed = len(state["candidates_reranked"]) > 0
        else:
            passed = await self.judge.judge(state["query_norm"], state["candidates_reranked"])
        state["gate_result"] = {
            "passed": passed,
            "reason": "sufficient_evidence" if passed else "reject_insufficient_evidence",
        }
        state["trace_steps"].append({"step": "verify", "detail": state["gate_result"]})
        return state


async def add_tool_reservation_steps(state: RagStateDict, *, enabled: bool) -> RagStateDict:
    state["trace_steps"].append({"step": "tool_plan", "detail": {"enabled": enabled, "items": len(state["tool_plan"])}})
    state["trace_steps"].append({"step": "tool_execute", "detail": {"enabled": enabled, "calls": len(state["tool_calls"])}})
    state["trace_steps"].append({"step": "tool_verify", "detail": {"enabled": enabled, "errors": len(state["tool_errors"])}})
    return state


class FinalizeNode:
    async def run(self, state: RagStateDict) -> RagStateDict:
        if not state["gate_result"].get("passed"):
            state["answer"] = "未检索到足够相关的知识片段，请补充更具体的问题或关键词。"
        elif not state["answer"]:
            state["answer"] = "根据检索到的知识片段，先给你一个最小可用回答："
        state["trace_steps"].append(
            {
                "step": "memory_write_gate",
                "detail": state.get("memory_write_decision") or {"allow": False, "reason": "not_evaluated"},
            }
        )
        state["trace_steps"].append({"step": "finalize", "detail": {"ok": True}})
        return state


