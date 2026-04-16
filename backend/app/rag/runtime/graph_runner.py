from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.runtime.default_nodes import FinalizeNode, NormalizeNode, RerankNode, RetrieveNode, VerifyNode, add_tool_reservation_steps
from app.rag.runtime.state import RagState
from app.rag.tools.runtime import ToolExecutionRuntime


class RagGraphRunner:
    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        judge: RelevanceJudge | None = None,
        enable_tools: bool = False,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.judge = judge
        self.enable_tools = enable_tools

    async def run(self, *, request_id: str, user_id: str, session_id: str, question: str) -> dict:
        state = RagState.new(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            query_raw=question,
        )
        state = await NormalizeNode().run(state)

        tool_runtime = ToolExecutionRuntime(enabled=self.enable_tools)
        tool_plan = await tool_runtime.plan(state["query_norm"])
        tool_calls, tool_errors = await tool_runtime.execute(tool_plan)
        state["tool_plan"] = tool_plan
        state["tool_calls"] = tool_calls
        state["tool_errors"] = tool_errors
        state = await add_tool_reservation_steps(state, enabled=self.enable_tools)

        state = await RetrieveNode(self.retriever).run(state)
        state = await RerankNode(self.reranker).run(state)
        state = await VerifyNode(self.judge).run(state)
        state = await FinalizeNode().run(state)
        return {
            "answer": state["answer"],
            "steps": state["trace_steps"],
            "gate": state["gate_result"],
            "evidence": state["candidates_reranked"],
            "retrieved": state["candidates_fused"],
        }
