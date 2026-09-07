from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from types import MappingProxyType
from typing import Any

from app.extensions.provider_router import ProviderRouter
from app.rag.answer_evidence import AnswerEvidence, evidence_summary_from_trace
from app.rag.claim_evidence import ClaimResolver
from app.rag.evidence_sufficiency import (
    AnswerEvidenceSet,
    EvidenceSufficiencyDecision,
    QueryConditionSet,
    decide_answer_evidence,
)
from app.rag.generation_observation import provider_visible_generation_input
from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.prompt_regions import (
    build_generation_prompt,
    frozen_generation_input_record,
    validate_agent_response,
)
from app.rag.runtime.graph_runner import RagGraphRunner
from app.retrieval.policy import LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID, get_retrieval_policy
from app.settings.runtime import get_runtime_settings

_PERSISTED_DIAGNOSTIC_DETAIL_FIELDS = frozenset(
    {
        "allow",
        "calls",
        "candidate_pool_scope",
        "deduped",
        "dense_candidate_count",
        "dense_count",
        "dense_hydrated_count",
        "dense_query_failed",
        "embedding_provider_ms",
        "enabled",
        "errors",
        "evidence_count",
        "fallback_used",
        "gate_passed",
        "gate_reason",
        "intent",
        "items",
        "judge",
        "language",
        "lexical_candidate_count",
        "lexical_scope",
        "llm",
        "max_calls",
        "max_latency_ms",
        "max_parallel",
        "merged",
        "merged_count",
        "model",
        "ok",
        "outcome",
        "passed",
        "profile_identity",
        "provider",
        "reason",
        "reranked_count",
        "retriever",
        "retrieved_count",
        "session_keys",
        "sparse_count",
        "strategy",
        "top_k",
        "used_evidence",
        "user_fact_count",
    }
)
_PERSISTED_DIAGNOSTIC_PROVIDER_ATTEMPT_FIELDS = frozenset(
    {"attempt", "error_code", "fallback", "latency_ms", "provider", "success"}
)


class AnswerOutcomeKind(str, Enum):
    EVIDENCE_GATED_ANSWER = "evidence_gated_answer"
    INSUFFICIENT_EVIDENCE_REPLY = "insufficient_evidence_reply"
    NON_KNOWLEDGE_BASE_REPLY = "non_knowledge_base_reply"
    GENERATION_UNAVAILABLE = "generation_unavailable"


class AnswerExecutionContractError(RuntimeError):
    """The deterministic execution contract could not form a closed result."""


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


def _persisted_provider_error(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    error: dict[str, object] = {}
    for key in ("code", "type"):
        item = value.get(key)
        if isinstance(item, str) and item:
            error[key] = item
    return error or None


def _persisted_diagnostic_detail(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    detail: dict[str, object] = {}
    for key in _PERSISTED_DIAGNOSTIC_DETAIL_FIELDS:
        item = value.get(key)
        if isinstance(item, (str, bool, int, float)):
            detail[key] = item
    provider_error = _persisted_provider_error(value.get("provider_error"))
    if provider_error is not None:
        detail["provider_error"] = provider_error
    return detail


def _persisted_diagnostic_steps(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    steps: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        step = item.get("step")
        if not isinstance(step, str) or not step:
            continue
        record: dict[str, object] = {"step": step}
        detail = _persisted_diagnostic_detail(item.get("detail"))
        if detail:
            record["detail"] = detail
        steps.append(record)
    return steps


def _persisted_provider_trace(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    trace: dict[str, object] = {}
    for stage, item in value.items():
        if not isinstance(stage, str) or not stage:
            continue
        detail = _persisted_diagnostic_detail(item)
        if detail:
            trace[stage] = detail
    return trace


def _persisted_provider_attempts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    attempts: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        attempt: dict[str, object] = {}
        for key in _PERSISTED_DIAGNOSTIC_PROVIDER_ATTEMPT_FIELDS:
            candidate = item.get(key)
            if isinstance(candidate, (str, bool, int, float)):
                attempt[key] = candidate
        if attempt:
            attempts.append(attempt)
    return attempts


@dataclass(frozen=True)
class AnswerExecutionOutcome:
    kind: AnswerOutcomeKind
    text: str
    evidence: tuple[AnswerEvidence, ...]
    question: str
    query_conditions: QueryConditionSet
    request_id: str
    session_id: str
    gate_passed: bool | None
    gate_reason: str
    steps: tuple[Mapping[str, Any], ...]
    runtime: Mapping[str, Any]
    evidence_set: AnswerEvidenceSet | None = None
    sufficiency_decision: EvidenceSufficiencyDecision | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "steps", tuple(_freeze(step) for step in self.steps))
        object.__setattr__(self, "runtime", _freeze(self.runtime))

    def to_rag_trace(self) -> dict[str, Any]:
        trace = {
            "outcome": self.kind.value,
            "query": self.question,
            "query_condition_set": self.query_conditions.to_record(),
            "steps": _thaw(self.steps),
            "gate": {"passed": self.gate_passed, "reason": self.gate_reason},
            "evidence": [item.to_record() for item in self.evidence],
            "answer_preview": self.text[:120],
            "runtime": _thaw(self.runtime),
        }
        if self.evidence_set is not None:
            frozen_set = self.evidence_set.to_record()
            trace["answer_evidence_set"] = frozen_set
            frozen_items = frozen_set.get("items")
            if isinstance(frozen_items, list):
                trace["evidence"] = [
                    item["evidence"]
                    for item in frozen_items
                    if isinstance(item, Mapping) and isinstance(item.get("evidence"), Mapping)
                ]
        if self.sufficiency_decision is not None:
            trace["evidence_sufficiency_decision"] = self.sufficiency_decision.to_record()
            if self.sufficiency_decision.insufficient_reply is not None:
                trace["insufficient_evidence_reply"] = self.sufficiency_decision.insufficient_reply.to_record()
        return trace

    def to_persisted_rag_trace(self) -> dict[str, Any]:
        """Return compatibility diagnostics without retaining private turn content."""

        runtime_value = _thaw(self.runtime)
        runtime = runtime_value if isinstance(runtime_value, Mapping) else {}
        runtime_steps = _persisted_diagnostic_steps(runtime.get("steps"))
        timing_value = runtime.get("timing_ms")
        timing = (
            {
                str(key): item
                for key, item in timing_value.items()
                if isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(item, bool)
            }
            if isinstance(timing_value, Mapping)
            else {}
        )
        runtime_gate = _persisted_diagnostic_detail(runtime.get("gate"))
        persisted_runtime: dict[str, object] = {
            key: item
            for key in ("request_id", "session_id", "graph_alias", "final_provider", "fallback_hops")
            if isinstance((item := runtime.get(key)), (str, int)) and not isinstance(item, bool)
        }
        persisted_runtime["steps"] = runtime_steps
        persisted_runtime["step_names"] = [step["step"] for step in runtime_steps]
        persisted_runtime["provider_trace"] = _persisted_provider_trace(runtime.get("provider_trace"))
        persisted_runtime["provider_attempts"] = _persisted_provider_attempts(runtime.get("provider_attempts"))
        persisted_runtime["timing_ms"] = timing
        if runtime_gate:
            persisted_runtime["gate"] = runtime_gate

        return {
            "outcome": self.kind.value,
            "steps": _persisted_diagnostic_steps(self.steps),
            "gate": {"passed": self.gate_passed, "reason": self.gate_reason},
            "runtime": persisted_runtime,
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
        reranker: Reranker | None,
        judge: RelevanceJudge | None,
        provider_router: ProviderRouter,
        primary_provider: str,
        retriever_name: str,
        reranker_name: str,
        judge_name: str,
        retrieval_top_k: int,
        max_evidence_items: int,
        max_excerpt_chars: int,
        claim_resolver: ClaimResolver | None = None,
    ) -> None:
        self._runner = RagGraphRunner(
            retriever=retriever,
            reranker=reranker,
            judge=judge,
            claim_resolver=claim_resolver,
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
            "claim_evidence_audit": runtime_result.get("claim_evidence_audit") or {},
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
        query_conditions: QueryConditionSet,
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
            query_conditions=query_conditions,
            request_id=request_id,
            session_id=session_id,
            gate_passed=None,
            gate_reason="not_applicable_non_knowledge_base",
            steps=tuple(runtime_steps),
            runtime=runtime,
        )

    @staticmethod
    def _expected_provider_visible_input(
        *,
        question: str,
        query_conditions: QueryConditionSet,
        evidence_set: AnswerEvidenceSet,
    ) -> dict[str, object]:
        if evidence_set.query_conditions != query_conditions:
            raise AnswerExecutionContractError("frozen Answer Evidence Set changed the admitted query condition set")
        try:
            return frozen_generation_input_record(question, evidence_set)
        except ValueError as exc:
            raise AnswerExecutionContractError(
                "frozen Answer Evidence Set has no valid provider-visible input"
            ) from exc

    async def execute(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        question: str,
        query_conditions: QueryConditionSet | None = None,
        progress: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> AnswerExecutionOutcome:
        normalized_question = question.strip()
        resolved_conditions = query_conditions or QueryConditionSet.from_question(normalized_question)
        if resolved_conditions.normalized_question != normalized_question:
            raise AnswerExecutionContractError("query condition set does not match the normalized question")
        if self._is_smalltalk(normalized_question):
            return self._smalltalk_outcome(
                request_id=request_id,
                session_id=session_id,
                question=normalized_question,
                query_conditions=resolved_conditions,
            )
        if progress is not None:
            await progress("retrieval", "正在检索知识库并核验证据…")
        retrieval_started = perf_counter()
        runtime_result = await self._runner.run(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            question=normalized_question,
            query_conditions=resolved_conditions,
            require_closed_evidence_decision=query_conditions is not None,
        )
        retrieval_ms = round((perf_counter() - retrieval_started) * 1000)
        decision = runtime_result.get("evidence_sufficiency_decision")
        if not isinstance(decision, EvidenceSufficiencyDecision):
            policy = get_retrieval_policy(get_runtime_settings())
            if query_conditions is not None or policy.identity != LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID:
                raise AnswerExecutionContractError("closed answer execution requires an evidence sufficiency decision")
            decision = decide_answer_evidence(
                normalized_question=normalized_question,
                query_conditions=resolved_conditions,
                candidates=(),
            )
        if decision.query_conditions != resolved_conditions:
            raise AnswerExecutionContractError("evidence decision changed the admitted query condition set")
        evidence_set = decision.evidence_set
        evidence = evidence_set.items if evidence_set is not None else ()
        gate_passed = decision.is_sufficient
        if gate_passed:
            gate_reason = "sufficient_evidence"
        else:
            if decision.reason is None:
                raise AnswerExecutionContractError("insufficient evidence decision has no reason")
            gate_reason = decision.reason
        provider_result: dict[str, Any] = {
            "text": "",
            "final_provider": None,
            "provider_attempts": [],
            "fallback_hops": 0,
        }
        provider_prompt_snapshot_ids: list[str] = []
        generation_provider_ms = 0
        if not gate_passed:
            kind = AnswerOutcomeKind.INSUFFICIENT_EVIDENCE_REPLY
            text = self._INSUFFICIENT_REPLY
            evidence = ()
        else:
            if progress is not None:
                await progress("generating", "证据核验通过，正在生成回答（深度生成约需 1~5 分钟）…")
            if evidence_set is None:
                raise AnswerExecutionContractError(
                    "evidence-gated generation requires a frozen Answer Evidence Set"
                )
            generation_input = evidence_set
            generation_prompt = build_generation_prompt(normalized_question, generation_input)
            expected_provider_input = self._expected_provider_visible_input(
                question=normalized_question,
                query_conditions=resolved_conditions,
                evidence_set=evidence_set,
            )
            observed_provider_input = provider_visible_generation_input(generation_prompt.user_prompt)
            if observed_provider_input is None or observed_provider_input != expected_provider_input:
                raise AnswerExecutionContractError(
                    "provider-visible prompt input contradicts the frozen answer execution"
                )
            provider_prompt_snapshot_ids = [item.snapshot_id for item in evidence_set.items]
            generation_started = perf_counter()
            provider_result = await self._provider_router.complete(
                primary=self._primary_provider,
                fallbacks=[],
                prompt=generation_prompt.user_prompt,
                system_prompt=generation_prompt.system_prompt,
            )
            generation_provider_ms = round((perf_counter() - generation_started) * 1000)
            if provider_result.get("generation_envelope_invalid") is True:
                raise AnswerExecutionContractError(
                    "provider generation envelope is malformed"
                )
            if provider_result.get("provider_failure") is True:
                raise AnswerExecutionContractError(
                    "provider generation failed without a completed provider result"
                )
            completion = str(provider_result.get("text") or "").strip()
            observed_envelope = provider_result.get("generation_envelope")
            observed_snapshot_ids = (
                observed_envelope.get("snapshot_ids")
                if isinstance(observed_envelope, Mapping)
                else None
            )
            if observed_snapshot_ids is not None and (
                not isinstance(observed_snapshot_ids, list)
                or observed_snapshot_ids != provider_prompt_snapshot_ids
            ):
                raise AnswerExecutionContractError(
                    "provider generation envelope snapshots contradict the frozen Answer Evidence Set"
                )
            if completion and validate_agent_response(
                completion,
                question=normalized_question,
                evidence=generation_input,
            ):
                kind = AnswerOutcomeKind.EVIDENCE_GATED_ANSWER
                text = completion
            else:
                kind = AnswerOutcomeKind.GENERATION_UNAVAILABLE
                text = self._GENERATION_UNAVAILABLE_REPLY

        runtime_result["gate"] = {"passed": gate_passed, "reason": gate_reason}
        runtime_result["query_condition_set"] = resolved_conditions.to_record()
        runtime_result["evidence_sufficiency_decision"] = decision.to_record()
        runtime_result["answer_evidence_set"] = evidence_set.to_record() if evidence_set is not None else None
        runtime_result["provider_prompt_snapshot_ids"] = provider_prompt_snapshot_ids
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
            query_conditions=resolved_conditions,
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
            evidence_set=evidence_set,
            sufficiency_decision=decision,
        )
