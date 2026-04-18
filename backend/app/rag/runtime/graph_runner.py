from typing import Any

from app.common.config import get_settings
from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.memory.inmemory_store import InMemorySessionStore, InMemoryUserStore
from app.rag.memory.policies import ConservativeMemoryWritePolicy
from app.rag.runtime.default_nodes import (
    ContextPackNode,
    FinalizeNode,
    FusionNode,
    GenerateNode,
    MemoryWriteNode,
    NormalizeNode,
    QueryUnderstandNode,
    RerankNode,
    RetrievalPlanNode,
    RetrieveNode,
    VerifyNode,
    add_tool_reservation_steps,
)
from app.rag.runtime.state import RagState, RagStateDict
from app.rag.tools.runtime import ToolExecutionRuntime

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:
    END = None
    START = None
    StateGraph = None


class RagGraphRunner:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        judge: RelevanceJudge | None = None,
        enable_tools: bool | None = None,
        tool_max_calls: int | None = None,
        tool_max_parallel: int | None = None,
        tool_timeout_ms: int | None = None,
    ) -> None:
        settings = get_settings()

        self.retriever = retriever
        self.reranker = reranker
        self.judge = judge
        self.enable_tools = settings.rag_enable_tools if enable_tools is None else enable_tools
        self.tool_max_calls = settings.rag_tool_max_calls if tool_max_calls is None else tool_max_calls
        self.tool_max_parallel = settings.rag_tool_max_parallel if tool_max_parallel is None else tool_max_parallel
        self.tool_timeout_ms = settings.rag_tool_timeout_ms if tool_timeout_ms is None else tool_timeout_ms
        self.graph_alias = settings.rag_graph_alias
        self.session_memory_store = InMemorySessionStore()
        self.user_memory_store = InMemoryUserStore()
        self.memory_write_policy = ConservativeMemoryWritePolicy(
            min_importance=0.7,
            min_novelty=0.6,
            min_stability=0.8,
        )

        self._normalize_node = NormalizeNode()
        self._query_understand_node = QueryUnderstandNode()
        self._retrieval_plan_node = RetrievalPlanNode(default_top_k=3)
        self._retrieve_node = RetrieveNode(self.retriever)
        self._fusion_node = FusionNode()
        self._rerank_node = RerankNode(self.reranker)
        self._verify_node = VerifyNode(self.judge)
        self._context_pack_node = ContextPackNode(top_k=3)
        self._generate_node = GenerateNode()
        self._memory_write_node = MemoryWriteNode()
        self._finalize_node = FinalizeNode()

        self._compiled_graph = self._build_graph()

    def _build_graph(self) -> Any:
        if StateGraph is None or START is None or END is None:
            return None

        graph = StateGraph(RagStateDict)
        graph.add_node("normalize", self._run_normalize)
        graph.add_node("memory_read", self._run_memory)
        graph.add_node("query_understand", self._run_query_understand)
        graph.add_node("plan", self._run_plan)
        graph.add_node("tools", self._run_tools)
        graph.add_node("retrieve", self._run_retrieve)
        graph.add_node("fusion", self._run_fusion)
        graph.add_node("rerank", self._run_rerank)
        graph.add_node("verify", self._run_verify)
        graph.add_node("context_pack", self._run_context_pack)
        graph.add_node("generate", self._run_generate)
        graph.add_node("memory_write", self._run_memory_write)
        graph.add_node("finalize", self._run_finalize)

        graph.add_edge(START, "normalize")
        graph.add_edge("normalize", "memory_read")
        graph.add_edge("memory_read", "query_understand")
        graph.add_edge("query_understand", "plan")
        graph.add_edge("plan", "tools")
        graph.add_edge("tools", "retrieve")
        graph.add_edge("retrieve", "fusion")
        graph.add_edge("fusion", "rerank")
        graph.add_edge("rerank", "verify")
        graph.add_edge("verify", "context_pack")
        graph.add_edge("context_pack", "generate")
        graph.add_edge("generate", "memory_write")
        graph.add_edge("memory_write", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def _run_normalize(self, state: RagStateDict) -> RagStateDict:
        return await self._normalize_node.run(state)

    async def _run_query_understand(self, state: RagStateDict) -> RagStateDict:
        return await self._query_understand_node.run(state)

    async def _run_plan(self, state: RagStateDict) -> RagStateDict:
        return await self._retrieval_plan_node.run(state)

    async def _run_tools(self, state: RagStateDict) -> RagStateDict:
        tool_runtime = ToolExecutionRuntime(
            enabled=self.enable_tools,
            max_calls=self.tool_max_calls,
            max_parallel=self.tool_max_parallel,
            timeout_ms=self.tool_timeout_ms,
        )
        tool_plan = await tool_runtime.plan(state["query_norm"])
        tool_calls, tool_errors = await tool_runtime.execute(tool_plan)
        state["tool_plan"] = tool_plan
        state["tool_calls"] = tool_calls
        state["tool_errors"] = tool_errors
        state["tool_budget"] = {
            "max_calls": self.tool_max_calls,
            "max_parallel": self.tool_max_parallel,
            "max_latency_ms": self.tool_timeout_ms,
        }
        return await add_tool_reservation_steps(state, enabled=self.enable_tools)

    async def _run_memory(self, state: RagStateDict) -> RagStateDict:
        session_payload = await self.session_memory_store.read(state["session_id"])
        user_facts = await self.user_memory_store.read(state["user_id"])

        state["memory_read_set"] = {
            "session": session_payload,
            "user_facts": user_facts,
        }

        fact_candidate = {
            "safe": False,
            "importance": 0.0,
            "novelty": 0.0,
            "stability": 0.0,
            "source": "runtime_reserved",
        }
        decision = await self.memory_write_policy.should_write(fact_candidate, user_facts)
        state["memory_write_decision"] = decision

        state["trace_steps"].append(
            {
                "step": "memory_read",
                "detail": {
                    "session_keys": len(session_payload.keys()) if isinstance(session_payload, dict) else 0,
                    "user_fact_count": len(user_facts),
                },
            }
        )
        return state

    async def _run_retrieve(self, state: RagStateDict) -> RagStateDict:
        return await self._retrieve_node.run(state)

    async def _run_fusion(self, state: RagStateDict) -> RagStateDict:
        return await self._fusion_node.run(state)

    async def _run_rerank(self, state: RagStateDict) -> RagStateDict:
        return await self._rerank_node.run(state)

    async def _run_verify(self, state: RagStateDict) -> RagStateDict:
        return await self._verify_node.run(state)

    async def _run_context_pack(self, state: RagStateDict) -> RagStateDict:
        return await self._context_pack_node.run(state)

    async def _run_generate(self, state: RagStateDict) -> RagStateDict:
        return await self._generate_node.run(state)

    async def _run_memory_write(self, state: RagStateDict) -> RagStateDict:
        return await self._memory_write_node.run(state)

    async def _run_finalize(self, state: RagStateDict) -> RagStateDict:
        return await self._finalize_node.run(state)

    async def _run_sequential(self, state: RagStateDict) -> RagStateDict:
        state = await self._run_normalize(state)
        state = await self._run_memory(state)
        state = await self._run_query_understand(state)
        state = await self._run_plan(state)
        state = await self._run_tools(state)
        state = await self._run_retrieve(state)
        state = await self._run_fusion(state)
        state = await self._run_rerank(state)
        state = await self._run_verify(state)
        state = await self._run_context_pack(state)
        state = await self._run_generate(state)
        state = await self._run_memory_write(state)
        state = await self._run_finalize(state)
        return state

    def _to_output(self, state: RagStateDict) -> dict:
        return {
            "request_id": state["request_id"],
            "session_id": state["session_id"],
            "answer": state["answer"],
            "steps": state["trace_steps"],
            "gate": state["gate_result"],
            "evidence": state["candidates_reranked"],
            "retrieved": state["candidates_fused"],
            "graph_alias": self.graph_alias,
        }

    async def run(self, *, request_id: str, user_id: str, session_id: str, question: str) -> dict:
        state = RagState.new(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            query_raw=question,
        )

        if self._compiled_graph is None:
            state = await self._run_sequential(state)
        else:
            state = await self._compiled_graph.ainvoke(state)

        return self._to_output(state)
