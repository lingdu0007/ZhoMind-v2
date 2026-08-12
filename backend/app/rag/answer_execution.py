from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from types import MappingProxyType
from typing import Any

from app.extensions.provider_router import ProviderRouter
from app.rag.answer_evidence import AnswerEvidence, evidence_summary_from_trace
from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.prompt_regions import build_generation_prompt, validate_agent_response
from app.rag.runtime.graph_runner import RagGraphRunner


class AnswerOutcomeKind(str, Enum):
    EVIDENCE_GATED_ANSWER = "evidence_gated_answer"
    INSUFFICIENT_EVIDENCE_REPLY = "insufficient_evidence_reply"
    NON_KNOWLEDGE_BASE_REPLY = "non_knowledge_base_reply"
    GENERATION_UNAVAILABLE = "generation_unavailable"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class AnswerExecutionOutcome:
    kind: AnswerOutcomeKind
    text: str
    evidence: tuple[AnswerEvidence, ...]
    question: str
    request_id: str
    session_id: str
    gate_passed: bool | None
    gate_reason: str
    steps: tuple[Mapping[str, Any], ...]
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "steps", tuple(_freeze(step) for step in self.steps))
        object.__setattr__(self, "runtime", _freeze(self.runtime))

    def to_rag_trace(self) -> dict[str, Any]:
        return {
            "outcome": self.kind.value,
            "query": self.question,
            "steps": _thaw(self.steps),
            "gate": {"passed": self.gate_passed, "reason": self.gate_reason},
            "evidence": [item.to_record() for item in self.evidence],
            "answer_preview": self.text[:120],
            "runtime": _thaw(self.runtime),
        }

    def evidence_summary(self) -> dict[str, Any]:
        return evidence_summary_from_trace(self.to_rag_trace())


class EvidenceGatedAnswerExecutor:
    _SMALLTALK_PATTERNS = frozenset(
        {
            "你是谁",
            "你叫什麼",
            "你叫什么",
            "介绍你自己",
            "自我介绍",
            "whoareyou",
            "whatyourname",
            "whatareyou",
            "你好",
            "您好",
            "hello",
            "hi",
            "hey",
        }
    )
    _SMALLTALK_REPLY = "【非知识库回复】我是 ZhoMind 智能助手，可以帮你基于知识库问答、梳理文档与会话内容。"
    _INSUFFICIENT_REPLY = "未检索到足够相关的知识片段，请补充更具体的问题或关键词。"
    _GENERATION_UNAVAILABLE_REPLY = "【生成不可用】生成服务暂不可用，请稍后重试。"

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        judge: RelevanceJudge | None,
        provider_router: ProviderRouter,
        primary_provider: str,
        retriever_name: str,
        reranker_name: str,
        judge_name: str,
        retrieval_top_k: int,
        max_evidence_items: int,
        max_excerpt_chars: int,
    ) -> None:
        self._runner = RagGraphRunner(
            retriever=retriever,
            reranker=reranker,
            judge=judge,
            retrieval_top_k=retrieval_top_k,
            evidence_top_k=max_evidence_items,
            evidence_excerpt_chars=max_excerpt_chars,
        )
        self._provider_router = provider_router
        self._primary_provider = primary_provider
        self._retriever_name = retriever_name
        self._reranker_name = reranker_name
        self._judge_name = judge_name
        self._max_excerpt_chars = max_excerpt_chars

    @staticmethod
    def _compact(text: str) -> str:
        return "".join(character for character in text.lower() if character.isalnum())

    def _is_smalltalk(self, question: str) -> bool:
        compact = self._compact(question.strip())
        return bool(compact) and compact in self._SMALLTALK_PATTERNS

    @staticmethod
    def _runtime_trace(runtime_result: dict[str, Any]) -> dict[str, Any]:
        runtime_steps = list(runtime_result.get("steps") or [])
        trace: dict[str, Any] = {
            "request_id": runtime_result.get("request_id"),
            "session_id": runtime_result.get("session_id"),
            "graph_alias": runtime_result.get("graph_alias"),
            "gate": runtime_result.get("gate") or {},
            "steps": runtime_steps,
            "step_names": [str(item.get("step") or "") for item in runtime_steps],
            "tool_budget": runtime_result.get("tool_budget") or {},
            "tool_errors": list(runtime_result.get("tool_errors") or []),
            "provider_trace": runtime_result.get("provider_trace") or {},
            "final_provider": runtime_result.get("final_provider"),
            "provider_attempts": list(runtime_result.get("provider_attempts") or []),
            "fallback_hops": int(runtime_result.get("fallback_hops") or 0),
            "timing_ms": runtime_result.get("timing_ms") or {},
            "provider_prompt_snapshot_ids": list(runtime_result.get("provider_prompt_snapshot_ids") or []),
            "provider_generation_envelope": runtime_result.get("provider_generation_envelope"),
        }
        return trace

    def _steps(
        self,
        *,
        question: str,
        runtime_result: Mapping[str, Any],
        gate_passed: bool | None,
        gate_reason: str,
        final_provider: str | None,
        kind: AnswerOutcomeKind,
    ) -> tuple[dict[str, Any], ...]:
        retrieved = runtime_result.get("retrieved")
        reranked = runtime_result.get("candidates_reranked")
        return (
            {
                "step": "retrieve",
                "detail": {
                    "query": question,
                    "retriever": self._retriever_name,
                    "retrieved_count": len(retrieved) if isinstance(retrieved, list) else 0,
                    "gate_passed": gate_passed,
                    "gate_reason": gate_reason,
                },
            },
            {
                "step": "rerank",
                "detail": {
                    "model": self._reranker_name,
                    "reranked_count": len(reranked) if isinstance(reranked, list) else 0,
                },
            },
            {"step": "verify", "detail": {"judge": self._judge_name}},
            {
                "step": "generate",
                "detail": {
                    "llm": final_provider,
                    "outcome": kind.value,
                },
            },
        )

    def _smalltalk_outcome(
        self,
        *,
        request_id: str,
        session_id: str,
        question: str,
    ) -> AnswerExecutionOutcome:
        runtime_steps = [{"step": "classify", "detail": {"outcome": AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY.value}}]
        runtime = {
            "request_id": request_id,
            "session_id": session_id,
            "graph_alias": self._runner.graph_alias,
            "gate": {"passed": None, "reason": "not_applicable_non_knowledge_base"},
            "steps": runtime_steps,
            "step_names": ["classify"],
            "tool_budget": {},
            "tool_errors": [],
            "provider_trace": {},
            "final_provider": None,
            "provider_attempts": [],
            "fallback_hops": 0,
        }
        return AnswerExecutionOutcome(
            kind=AnswerOutcomeKind.NON_KNOWLEDGE_BASE_REPLY,
            text=self._SMALLTALK_REPLY,
            evidence=(),
            question=question,
            request_id=request_id,
            session_id=session_id,
            gate_passed=None,
            gate_reason="not_applicable_non_knowledge_base",
            steps=tuple(runtime_steps),
            runtime=runtime,
        )

    async def execute(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        question: str,
    ) -> AnswerExecutionOutcome:
        normalized_question = question.strip()
        if self._is_smalltalk(normalized_question):
            return self._smalltalk_outcome(
                request_id=request_id,
                session_id=session_id,
                question=normalized_question,
            )
        retrieval_started = perf_counter()
        runtime_result = await self._runner.run(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            question=normalized_question,
        )
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000)
        evidence = tuple(
            item
            for candidate in runtime_result.get("answer_evidence") or []
            if (item := AnswerEvidence.from_candidate(candidate, max_excerpt_chars=self._max_excerpt_chars)) is not None
        )
        _gate_value = runtime_result.get("gate")
        gate = _gate_value if isinstance(_gate_value, Mapping) else {}
        gate_passed = bool(gate.get("passed")) and bool(evidence)
        gate_reason = str(gate.get("reason") or "reject_insufficient_evidence")
        provider_result: dict[str, Any] = {
            "text": "",
            "final_provider": None,
            "provider_attempts": [],
            "fallback_hops": 0,
        }
        generation_provider_ms = 0
        if not gate_passed:
            kind = AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
            text = self._INSUFFICIENT_REPLY
            evidence = ()
            if not gate_reason.startswith("reject_"):
                gate_reason = "reject_insufficient_evidence"
        else:
            generation_prompt = build_generation_prompt(normalized_question, evidence)
            generation_started = perf_counter()
            provider_result = await self._provider_router.complete(
                primary=self._primary_provider,
                fallbacks=[],
                prompt=generation_prompt.user_prompt,
                system_prompt=generation_prompt.system_prompt,
            )
            generation_provider_ms = round((perf_counter() - generation_started) * 1000)
            completion = str(provider_result.get("text") or "").strip()
            if completion and validate_agent_response(completion, question=normalized_question, evidence=evidence):
                kind = AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
                text = completion
            else:
                kind = AnswerOutcomeKind.GENERATION_UNAVAILABLE
                text = self._GENERATION_UNAVAILABLE_REPLY

        runtime_result["gate"] = {"passed": gate_passed, "reason": gate_reason}
        runtime_result["provider_prompt_snapshot_ids"] = [item.snapshot_id for item in evidence]
        runtime_result["provider_generation_envelope"] = provider_result.get("generation_envelope")
        runtime_result["timing_ms"] = {
            "retrieval_ms": retrieval_ms,
            "generation_provider_ms": generation_provider_ms,
            "embedding_provider_ms": float(
                (runtime_result.get("provider_trace") or {}).get("retrieve", {}).get("embedding_provider_ms") or 0
            ),
        }
        runtime_result["final_provider"] = provider_result.get("final_provider")
        runtime_result["provider_attempts"] = list(provider_result.get("provider_attempts") or [])
        runtime_result["fallback_hops"] = int(provider_result.get("fallback_hops") or 0)
        for step in runtime_result.get("steps") or []:
            if step.get("step") == "generate" and isinstance(step.get("detail"), dict):
                step["detail"].update(
                    {
                        "llm": provider_result.get("final_provider"),
                        "outcome": kind.value,
                        "used_evidence": len(evidence),
                    }
                )
            elif step.get("step") == "finalize" and isinstance(step.get("detail"), dict):
                step["detail"].update({"outcome": kind.value})

        final_provider = provider_result.get("final_provider")
        return AnswerExecutionOutcome(
            kind=kind,
            text=text,
            evidence=evidence,
            question=normalized_question,
            request_id=request_id,
            session_id=session_id,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            steps=self._steps(
                question=normalized_question,
                runtime_result=runtime_result,
                gate_passed=gate_passed,
                gate_reason=gate_reason,
                final_provider=str(final_provider) if final_provider else None,
                kind=kind,
            ),
            runtime=self._runtime_trace(runtime_result),
        )
