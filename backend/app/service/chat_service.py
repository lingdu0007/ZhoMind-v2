import asyncio
import json
import math
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.canonical import AnswerExecutionState, AnswerOutcome
from app.extensions.provider_router import ProviderRouter
from app.extensions.registry import get_extension_registry
from app.rag.answer_evidence import evidence_summary_from_trace
from app.rag.answer_execution import EvidenceGatedAnswerExecutor
from app.rag.claim_evidence import ClaimResolver
from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.repository.chat_repository import ChatRepository
from app.retrieval.policy import get_retrieval_policy
from app.service.answer_execution_store import AnswerExecutionHandle, AnswerExecutionStore
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService
from app.settings.runtime import get_runtime_settings

CHAT_RETRIEVER_PROVIDER = "chat-default-retriever"
CHAT_RERANK_PROVIDER = "chat-default-reranker"
CHAT_JUDGE_PROVIDER = "chat-default-judge"
CHAT_CLAIM_RESOLVER_PROVIDER = "chat-default-claim-resolver"
DIAGNOSTIC_MAX_TIMELINE_STEPS = 16
DIAGNOSTIC_MAX_PROVIDER_ERRORS = 5
DIAGNOSTIC_MAX_TRACE_PREVIEW_CHARS = 1600
_DIAGNOSTIC_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_DIAGNOSTIC_PREVIEW_COUNTS = frozenset(
    {
        "attempt",
        "calls",
        "dense_candidate_count",
        "dense_count",
        "dense_hydrated_count",
        "deduped",
        "errors",
        "evidence_count",
        "fallback_hops",
        "items",
        "latency_ms",
        "lexical_candidate_count",
        "max_calls",
        "max_latency_ms",
        "max_parallel",
        "merged",
        "merged_count",
        "reranked_count",
        "retrieved_count",
        "sparse_count",
        "top_k",
        "used_evidence",
        "user_fact_count",
        "session_keys",
    }
)
_DIAGNOSTIC_PREVIEW_FLAGS = frozenset(
    {"allow", "dense_query_failed", "enabled", "fallback_used", "gate_passed", "ok", "passed"}
)
_DIAGNOSTIC_PREVIEW_STEP_FIELDS = {
    "normalize": ("ok",),
    "memory_read": ("session_keys", "user_fact_count"),
    "query_understand": ("intent", "language"),
    "plan": ("strategy", "top_k"),
    "tool_plan": ("enabled", "items", "max_calls", "max_parallel", "max_latency_ms"),
    "tool_execute": ("enabled", "calls", "max_calls", "max_parallel", "max_latency_ms"),
    "tool_verify": ("enabled", "errors"),
    "retrieve": (
        "strategy",
        "dense_candidate_count",
        "dense_hydrated_count",
        "lexical_candidate_count",
        "merged_count",
        "dense_query_failed",
        "lexical_scope",
        "sparse_count",
        "dense_count",
        "retriever",
        "retrieved_count",
        "gate_passed",
        "gate_reason",
        "provider",
        "fallback_used",
    ),
    "fusion": ("merged", "deduped"),
    "rerank": ("reranked_count", "model", "provider", "fallback_used"),
    "verify": ("passed", "reason", "judge", "provider", "fallback_used"),
    "context_pack": ("evidence_count",),
    "generate": ("used_evidence", "llm"),
    "memory_write_gate": ("allow", "reason"),
    "finalize": ("ok",),
}


class _IdentityReranker:
    name = "inmemory-identity-reranker"

    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        return items


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ChatRepository(session)
        self.answer_execution_store = AnswerExecutionStore(session, self.repo)

    def _resolve_retriever(self) -> tuple[Retriever, str]:
        policy = get_retrieval_policy(get_runtime_settings())
        fallback = MixedModeDocumentRetrieverService(self.session)
        if not policy.diagnostic_or_migration_only:
            # The active product profile owns the ordinary-user retrieval boundary.
            return fallback, fallback.name

        provider = get_extension_registry().get_retriever(CHAT_RETRIEVER_PROVIDER)
        if provider is not None:
            return provider, CHAT_RETRIEVER_PROVIDER

        return fallback, fallback.name

    def _resolve_reranker(self) -> tuple[Reranker | None, str]:
        policy = get_retrieval_policy(get_runtime_settings())
        if not policy.reranker_enabled:
            return None, f"disabled-by-{policy.identity}"
        provider = get_extension_registry().get_rerank(CHAT_RERANK_PROVIDER)
        if provider is not None:
            return provider, CHAT_RERANK_PROVIDER

        fallback = _IdentityReranker()
        return fallback, fallback.name

    def _resolve_judge(self) -> tuple[RelevanceJudge | None, str]:
        policy = get_retrieval_policy(get_runtime_settings())
        if not policy.online_llm_sufficiency_judge_enabled:
            return None, f"disabled-by-{policy.identity}"
        provider = get_extension_registry().get_judge(CHAT_JUDGE_PROVIDER)
        if provider is not None:
            return provider, CHAT_JUDGE_PROVIDER

        return None, "unconfigured-agent-evidence-gate"

    def _resolve_claim_resolver(self) -> ClaimResolver | None:
        return get_extension_registry().get_claim_resolver(CHAT_CLAIM_RESOLVER_PROVIDER)

    def _provider_router(self) -> ProviderRouter:
        return ProviderRouter(providers=get_extension_registry().llm_providers)

    def _evidence_summary(self, rag_trace: dict | None) -> dict:
        return evidence_summary_from_trace(rag_trace)

    @staticmethod
    def _diagnostic_code(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text if _DIAGNOSTIC_CODE.fullmatch(text) else None

    @staticmethod
    def _diagnostic_count(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _diagnostic_duration(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        duration = float(value)
        if not math.isfinite(duration) or duration < 0:
            return None
        return duration

    def _diagnostic_preview_value(self, key: str, value: object) -> bool | int | str | None:
        if key in _DIAGNOSTIC_PREVIEW_FLAGS:
            return value if isinstance(value, bool) else None
        if key in _DIAGNOSTIC_PREVIEW_COUNTS:
            return self._diagnostic_count(value)
        return self._diagnostic_code(value)

    def _diagnostic_preview_steps(self, steps: object) -> list[dict]:
        if not isinstance(steps, list):
            return []
        preview: list[dict] = []
        for item in steps[:DIAGNOSTIC_MAX_TIMELINE_STEPS]:
            if not isinstance(item, dict):
                continue
            step = self._diagnostic_code(item.get("step"))
            if step is None:
                continue
            preview_item: dict[str, object] = {"step": step}
            _detail_value = item.get("detail")
            detail = _detail_value if isinstance(_detail_value, dict) else {}
            allowed_fields = _DIAGNOSTIC_PREVIEW_STEP_FIELDS.get(step, ())
            preview_detail = {
                key: self._diagnostic_preview_value(key, detail.get(key))
                for key in allowed_fields
                if key in detail
            }
            if preview_detail:
                preview_item["detail"] = preview_detail
            preview.append(preview_item)
        return preview

    def _diagnostic_trace_preview(self, rag_trace: dict) -> str:
        _runtime_value = rag_trace.get("runtime")
        runtime = _runtime_value if isinstance(_runtime_value, dict) else {}
        _gate_value = rag_trace.get("gate")
        gate = _gate_value if isinstance(_gate_value, dict) else {}
        _steps_value = rag_trace.get("steps")
        steps = _steps_value if isinstance(_steps_value, list) else []
        if not runtime and not gate and not steps:
            return ""
        _attempts_value = runtime.get("provider_attempts")
        attempts = _attempts_value if isinstance(_attempts_value, list) else []
        preview_attempts: list[dict] = []
        for attempt in attempts[:DIAGNOSTIC_MAX_PROVIDER_ERRORS]:
            if not isinstance(attempt, dict):
                continue
            preview_attempt = {
                key: self._diagnostic_preview_value(key, attempt.get(key))
                for key in ("provider", "attempt", "latency_ms", "error_code")
                if key in attempt
            }
            if preview_attempt:
                preview_attempts.append(preview_attempt)

        preview = {
            "gate": {
                "passed": self._diagnostic_preview_value("passed", gate.get("passed")),
                "reason": self._diagnostic_code(gate.get("reason")),
            },
            "steps": self._diagnostic_preview_steps(rag_trace.get("steps")),
            "runtime": {
                "request_id": self._diagnostic_code(runtime.get("request_id")),
                "session_id": self._diagnostic_code(runtime.get("session_id")),
                "graph_alias": self._diagnostic_code(runtime.get("graph_alias")),
                "steps": self._diagnostic_preview_steps(runtime.get("steps")),
                "final_provider": self._diagnostic_code(runtime.get("final_provider")),
                "fallback_hops": self._diagnostic_count(runtime.get("fallback_hops")),
                "provider_attempts": preview_attempts,
            },
        }
        serialized = json.dumps(preview, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) <= DIAGNOSTIC_MAX_TRACE_PREVIEW_CHARS:
            return serialized
        return f"{serialized[: DIAGNOSTIC_MAX_TRACE_PREVIEW_CHARS - 3]}..."

    def _diagnostic_timeline(self, rag_trace: dict) -> list[dict]:
        _runtime_value = rag_trace.get("runtime")
        runtime = _runtime_value if isinstance(_runtime_value, dict) else {}
        _runtime_steps_value = runtime.get("steps")
        runtime_steps = _runtime_steps_value if isinstance(_runtime_steps_value, list) else []
        timeline: list[dict] = []
        for item in runtime_steps[:DIAGNOSTIC_MAX_TIMELINE_STEPS]:
            if not isinstance(item, dict):
                continue
            step = self._diagnostic_code(item.get("step"))
            if step is not None:
                timeline.append({"step": step})
        return timeline

    def _diagnostic_candidate_counts(self, rag_trace: dict) -> dict:
        retrieved = None
        reranked = None
        _steps_value = rag_trace.get("steps")
        steps = _steps_value if isinstance(_steps_value, list) else []
        for item in steps:
            if not isinstance(item, dict) or not isinstance(item.get("detail"), dict):
                continue
            detail = item["detail"]
            if item.get("step") == "retrieve":
                retrieved = self._diagnostic_count(detail.get("retrieved_count"))
            elif item.get("step") == "rerank":
                reranked = self._diagnostic_count(detail.get("reranked_count"))

        _runtime_value = rag_trace.get("runtime")
        runtime = _runtime_value if isinstance(_runtime_value, dict) else {}
        _runtime_steps_value = runtime.get("steps")
        runtime_steps = _runtime_steps_value if isinstance(_runtime_steps_value, list) else []
        for item in runtime_steps:
            if not isinstance(item, dict) or not isinstance(item.get("detail"), dict):
                continue
            detail = item["detail"]
            if item.get("step") == "retrieve" and retrieved is None:
                retrieved = self._diagnostic_count(detail.get("merged_count"))
            elif item.get("step") == "rerank" and reranked is None:
                reranked = self._diagnostic_count(detail.get("reranked_count"))

        return {"retrieved": retrieved, "reranked": reranked}

    def _diagnostic_provider_errors(self, rag_trace: dict) -> list[dict]:
        _runtime_value = rag_trace.get("runtime")
        runtime = _runtime_value if isinstance(_runtime_value, dict) else {}
        _provider_trace_value = runtime.get("provider_trace")
        provider_trace = _provider_trace_value if isinstance(_provider_trace_value, dict) else {}
        errors: list[dict] = []
        for stage, detail in provider_trace.items():
            if len(errors) >= DIAGNOSTIC_MAX_PROVIDER_ERRORS:
                break
            if not isinstance(detail, dict):
                continue
            error = detail.get("provider_error") or detail.get("error")
            if not isinstance(error, dict):
                continue
            errors.append(
                {
                    "stage": self._diagnostic_code(stage),
                    "code": self._diagnostic_code(error.get("code")),
                    "type": self._diagnostic_code(error.get("type")),
                }
            )

        _attempts_value = runtime.get("provider_attempts")
        attempts = _attempts_value if isinstance(_attempts_value, list) else []
        for attempt in attempts:
            if len(errors) >= DIAGNOSTIC_MAX_PROVIDER_ERRORS:
                break
            if not isinstance(attempt, dict) or not attempt.get("error_code"):
                continue
            errors.append(
                {
                    "stage": "generate",
                    "code": self._diagnostic_code(attempt.get("error_code")),
                    "type": None,
                }
            )
        return errors

    def _retrieval_diagnostics(self, rag_trace: dict | None) -> dict:
        trace = rag_trace if isinstance(rag_trace, dict) else {}
        _gate_value = trace.get("gate")
        gate = _gate_value if isinstance(_gate_value, dict) else {}
        gate_passed = gate.get("passed")
        gate_outcome = "passed" if gate_passed is True else "rejected" if gate_passed is False else "unavailable"

        _runtime_value = trace.get("runtime")
        runtime = _runtime_value if isinstance(_runtime_value, dict) else {}
        fallback_hops = self._diagnostic_count(runtime.get("fallback_hops"))
        fallback_state = "unavailable"
        if fallback_hops is not None:
            fallback_state = "used" if fallback_hops > 0 else "not_used"

        diagnostics = {
            "timeline": self._diagnostic_timeline(trace),
            "candidate_counts": self._diagnostic_candidate_counts(trace),
            "evidence_gate": {
                "outcome": gate_outcome,
                "reason": self._diagnostic_code(gate.get("reason")),
            },
            "fallback": {
                "state": fallback_state,
                "hops": fallback_hops,
                "final_provider": self._diagnostic_code(runtime.get("final_provider")),
            },
            "provider_errors": self._diagnostic_provider_errors(trace),
        }
        timing = runtime.get("timing_ms")
        if isinstance(timing, dict):
            diagnostics["timing_ms"] = {
                key: self._diagnostic_duration(timing.get(key))
                for key in ("retrieval_ms", "generation_provider_ms", "embedding_provider_ms", "persistence_ms")
                if self._diagnostic_duration(timing.get(key)) is not None
            }
        diagnostics["trace_preview"] = self._diagnostic_trace_preview(trace)
        return diagnostics

    def project_message(self, message: dict, role: str) -> dict:
        projection = {
            "id": message.get("id"),
            "type": message.get("type"),
            "content": message.get("content") or "",
            "timestamp": message.get("timestamp"),
        }
        execution = message.get("answer_execution")
        if projection["type"] == "user" and isinstance(execution, dict):
            admitted_question = execution.get("question")
            if not isinstance(admitted_question, str) or projection["content"] != admitted_question:
                raise ValueError("user message text contradicts the admitted answer execution question")
            projection["answer_execution"] = execution
        if projection["type"] == "assistant":
            execution_result = message.get("answer_execution_result")
            if isinstance(execution, dict):
                state = AnswerExecutionState(str(execution.get("state") or ""))
                projection["answer_execution"] = execution
                if state is AnswerExecutionState.COMPLETED:
                    if not isinstance(execution_result, dict):
                        raise ValueError("completed answer execution has no closed result")
                    text = execution_result.get("text")
                    frozen_text = execution.get("answer_text")
                    if (
                        not isinstance(text, str)
                        or not isinstance(frozen_text, str)
                        or projection["content"] != text
                        or frozen_text != text
                    ):
                        raise ValueError("assistant message text contradicts the closed answer execution")
                    outcome = execution.get("outcome")
                    if not isinstance(outcome, str):
                        raise ValueError("completed answer execution has no closed outcome")
                    completed_outcome = AnswerOutcome(outcome)
                    frozen_summary = execution.get("evidence_summary")
                    if not isinstance(frozen_summary, dict):
                        raise ValueError("completed answer execution has no frozen evidence summary")
                    projection["content"] = text
                    projection["outcome"] = completed_outcome.value
                    projection["evidence_summary"] = frozen_summary
                    if completed_outcome is AnswerOutcome.INSUFFICIENT_EVIDENCE_REPLY:
                        insufficient_reply = execution.get("insufficient_evidence_reply")
                        if not isinstance(insufficient_reply, dict):
                            raise ValueError("insufficient answer execution has no structured insufficiency reply")
                        projection["insufficient_evidence_reply"] = dict(insufficient_reply)
                elif (
                    state
                    not in {
                        AnswerExecutionState.STOPPED,
                        AnswerExecutionState.FAILED,
                        AnswerExecutionState.THROTTLED,
                        AnswerExecutionState.REJECTED,
                    }
                    or not isinstance(execution_result, dict)
                    or execution_result.get("state") != state.value
                    or projection["content"]
                ):
                    raise ValueError("assistant message has no closed terminal answer execution")
                if role == "admin":
                    projection["retrieval_diagnostics"] = self._retrieval_diagnostics(message.get("rag_trace"))
                return projection

            rag_trace = message.get("rag_trace")
            projection["evidence_summary"] = self._evidence_summary(rag_trace)
            if isinstance(rag_trace, dict) and isinstance(rag_trace.get("outcome"), str):
                projection["outcome"] = rag_trace["outcome"]
            if role == "admin":
                projection["retrieval_diagnostics"] = self._retrieval_diagnostics(rag_trace)
        return projection

    def project_chat_result(self, result: dict, role: str) -> dict:
        message = self.project_message(result["message"], role)
        projection = {
            "session_id": result["session_id"],
            "answer": message["content"],
            "message": message,
        }
        if message.get("answer_execution") is not None:
            projection["answer_execution"] = message["answer_execution"]
        if message.get("outcome") is not None:
            projection["outcome"] = message["outcome"]
        if message.get("insufficient_evidence_reply") is not None:
            projection["insufficient_evidence_reply"] = message["insufficient_evidence_reply"]
        if role == "admin":
            projection["retrieval_diagnostics"] = message["retrieval_diagnostics"]
        return projection

    async def ensure_session_id(self, session_id: str | None) -> str:
        if session_id and session_id.strip():
            return session_id.strip()
        return f"session_{uuid.uuid4().hex[:16]}"

    async def list_sessions(self, user_id: str) -> list[dict]:
        if await self.repo.purge_expired_sessions():
            await self.session.commit()
        sessions = await self.repo.list_sessions(user_id=user_id, limit=20)
        messages_by_session = {
            session.id: await self.repo.list_messages(session_id=session.id, user_id=user_id)
            for session in sessions
        }
        execution_ids, invalid_message_ids = await self.answer_execution_store.execution_ids_for_messages(
            messages=[message for messages in messages_by_session.values() for message in messages]
        )
        items: list[dict] = []
        for session in sessions:
            messages = messages_by_session[session.id]
            latest_execution_state = None
            bindings_valid = not any(message.id in invalid_message_ids for message in messages)
            if not bindings_valid:
                latest_execution_state = "unavailable"
            title = (
                next(
                    (
                        " ".join(message.content.split())[:80]
                        for message in messages
                        if message.type == "user" and message.content.strip()
                    ),
                    "未命名会话",
                )
                if bindings_valid
                else "未命名会话"
            )
            for message in reversed(messages):
                if not bindings_valid:
                    break
                execution_id = execution_ids.get(message.id)
                if execution_id is None:
                    continue
                try:
                    loaded = await self.answer_execution_store.load_for_message(
                        user_id=user_id,
                        session_id=session.id,
                        message_id=message.id,
                        message_type=message.type,
                        indexed_execution_id=execution_id,
                    )
                except ValueError:
                    latest_execution_state = "unavailable"
                    break
                if loaded is not None:
                    state = loaded.projection.get("state")
                    latest_execution_state = state if isinstance(state, str) else None
                    break
            item = {
                "session_id": session.id,
                "title": title,
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(messages),
            }
            if latest_execution_state is not None:
                item["latest_execution_state"] = latest_execution_state
            items.append(item)
        return items

    async def get_session_messages(self, session_id: str, user_id: str, role: str) -> list[dict]:
        if await self.repo.purge_expired_sessions():
            await self.session.commit()
        session = await self.repo.get_session(session_id=session_id, user_id=user_id)
        if session is None:
            return []
        messages = await self.repo.list_messages(session_id=session_id, user_id=user_id)
        projections: list[dict] = []
        for item in messages:
            message = {
                "id": item.id,
                "type": item.type,
                "content": item.content,
                "timestamp": item.created_at.isoformat(),
                "answer_execution_id": item.answer_execution_id,
                "rag_trace": item.rag_trace,
            }
            loaded = await self.answer_execution_store.load_for_message(
                user_id=user_id,
                session_id=item.session_id,
                message_id=item.id,
                message_type=item.type,
                indexed_execution_id=item.answer_execution_id,
            )
            if loaded is not None:
                message["answer_execution"] = loaded.projection
                message["answer_execution_result"] = loaded.result
            projections.append(self.project_message(message, role))
        return projections

    async def get_answer_execution_projection(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> dict:
        loaded = await self.answer_execution_store.load(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )
        if loaded is None:
            raise ValueError("private answer execution is missing after admission")
        return loaded.projection

    async def refresh_pending_stream_message(
        self,
        message: Mapping[str, Any],
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
        role: str,
    ) -> dict:
        if message.get("answer_execution_id") != execution_id:
            raise ValueError("completed stream message has no frozen execution binding")
        loaded = await self.answer_execution_store.load_pending_stream_result(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
        )
        refreshed = dict(message)
        refreshed["answer_execution"] = loaded.projection
        refreshed["answer_execution_result"] = loaded.result
        return self.project_message(refreshed, role)

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        deleted = await self.repo.delete_session(session_id=session_id, user_id=user_id)
        if deleted:
            await self.session.commit()
        return deleted

    async def record_stream_delivery_interruption(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        try:
            recorded = await self.answer_execution_store.record_stream_delivery_interruption(
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
            )
            await self.session.commit()
            return recorded
        except BaseException:
            await self.session.rollback()
            raise

    async def record_stream_delivery_pending(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        try:
            recorded = await self.answer_execution_store.record_stream_delivery_pending(
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
            )
            await self.session.commit()
            return recorded
        except BaseException:
            await self.session.rollback()
            raise

    async def record_stream_delivery_completion(
        self,
        *,
        execution_id: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        try:
            recorded = await self.answer_execution_store.record_stream_delivery_completion(
                execution_id=execution_id,
                user_id=user_id,
                session_id=session_id,
            )
            await self.session.commit()
            return recorded
        except BaseException:
            await self.session.rollback()
            raise

    async def _persist_noncompleted_terminal(
        self,
        *,
        execution: AnswerExecutionHandle,
        state: AnswerExecutionState,
        failure_code: str,
    ) -> None:
        """Roll back an interrupted completion before retaining its terminal state."""

        await self.session.rollback()
        loaded = await self.answer_execution_store.load(
            execution_id=execution.execution_id,
            user_id=execution.user_id,
            session_id=execution.session_id,
        )
        if loaded is None or loaded.projection["state"] in {
            AnswerExecutionState.COMPLETED.value,
            AnswerExecutionState.STOPPED.value,
            AnswerExecutionState.FAILED.value,
            AnswerExecutionState.THROTTLED.value,
            AnswerExecutionState.REJECTED.value,
        }:
            return
        if loaded.projection["state"] != AnswerExecutionState.RUNNING.value:
            raise ValueError("answer execution cannot be terminalized from its retained state")
        try:
            if state is AnswerExecutionState.STOPPED:
                await self.answer_execution_store.stop(handle=execution)
            elif state is AnswerExecutionState.FAILED:
                await self.answer_execution_store.fail(
                    handle=execution,
                    failure_code=failure_code,
                )
            else:
                raise ValueError("unsupported answer execution terminal state")
        except Exception as terminal_persistence_error:
            await self.session.rollback()
            loaded = await self.answer_execution_store.load(
                execution_id=execution.execution_id,
                user_id=execution.user_id,
                session_id=execution.session_id,
            )
            if loaded is None:
                raise ValueError(
                    "private answer execution is missing after persistence rollback"
                ) from terminal_persistence_error
            if loaded.projection["state"] in {
                AnswerExecutionState.COMPLETED.value,
                AnswerExecutionState.STOPPED.value,
                AnswerExecutionState.FAILED.value,
                AnswerExecutionState.THROTTLED.value,
                AnswerExecutionState.REJECTED.value,
            }:
                return
            if loaded.projection["state"] != AnswerExecutionState.RUNNING.value:
                raise ValueError("answer execution cannot recover from its retained state") from terminal_persistence_error
            try:
                await self.answer_execution_store.fail_persistence_without_assistant_message(handle=execution)
                session = await self.repo.get_session(
                    session_id=execution.session_id,
                    user_id=execution.user_id,
                )
                if session is None:
                    raise ValueError("private conversation is missing for its answer execution")
                session.updated_at = datetime.now(UTC)
                await self.session.commit()
            except BaseException:
                await self.session.rollback()
                raise
            return
        try:
            session = await self.repo.get_session(
                session_id=execution.session_id,
                user_id=execution.user_id,
            )
            if session is None:
                raise ValueError("private conversation is missing for its answer execution")
            session.updated_at = datetime.now(UTC)
            await self.session.commit()
        except BaseException:
            await self.session.rollback()
            raise

    async def run_chat(
        self,
        user_id: str,
        question: str,
        session_id: str | None,
        query_conditions: list[dict] | None = None,
        inherit_conditions: bool = False,
        progress: Callable[[str, str], Awaitable[None]] | None = None,
        on_admitted: Callable[[AnswerExecutionHandle], Awaitable[None]] | None = None,
    ) -> dict:
        # Capture generation dependencies at request admission. A later provider
        # replacement only affects requests admitted after its atomic cutover.
        generation_settings = get_runtime_settings()
        provider_router = self._provider_router()
        await self.repo.acquire_private_conversation_write_fence()
        await self.repo.purge_expired_sessions()
        sid = await self.ensure_session_id(session_id)
        session = await self.repo.get_or_create_session_for_admission(session_id=sid, user_id=user_id)

        normalized_question = question.strip()
        resolution = await self.answer_execution_store.resolve_query_conditions(
            user_id=user_id,
            session_id=session.id,
            question=normalized_question,
            explicit_conditions=query_conditions,
            inherit_conditions=inherit_conditions,
        )
        request_id = f"chat-{uuid.uuid4().hex[:8]}"
        execution = await self.answer_execution_store.admit(
            request_id=request_id,
            user_id=user_id,
            session_id=session.id,
            question=normalized_question,
            resolution=resolution,
        )

        try:
            # A cancellation during final persistence can only become a durable
            # stopped execution if this admitted running record already exists.
            await self.session.commit()
            if on_admitted is not None:
                await on_admitted(execution)
            retriever, retriever_name = self._resolve_retriever()
            reranker, reranker_name = self._resolve_reranker()
            judge, judge_name = self._resolve_judge()
            claim_resolver = self._resolve_claim_resolver()
            executor = EvidenceGatedAnswerExecutor(
                retriever=retriever,
                reranker=reranker,
                judge=judge,
                claim_resolver=claim_resolver,
                provider_router=provider_router,
                primary_provider=generation_settings.rag_primary_llm_provider,
                retriever_name=retriever_name,
                reranker_name=reranker_name,
                judge_name=judge_name,
                retrieval_top_k=generation_settings.runtime_retrieval_top_k,
                max_evidence_items=generation_settings.runtime_answer_evidence_max_items,
                max_excerpt_chars=generation_settings.runtime_answer_evidence_max_chars_per_source,
            )
            outcome = await executor.execute(
                request_id=request_id,
                user_id=user_id,
                session_id=session.id,
                question=normalized_question,
                query_conditions=resolution.query_conditions,
                progress=progress,
            )
            assistant_message, _loaded_execution = await self.answer_execution_store.complete(
                handle=execution,
                outcome=outcome,
            )
            session.updated_at = datetime.now(UTC)
            await self.session.commit()
        except asyncio.CancelledError:
            await self._persist_noncompleted_terminal(
                execution=execution,
                state=AnswerExecutionState.STOPPED,
                failure_code="ANSWER_EXECUTION_STOPPED",
            )
            raise
        except Exception:
            await self._persist_noncompleted_terminal(
                execution=execution,
                state=AnswerExecutionState.FAILED,
                failure_code="ANSWER_EXECUTION_FAILED",
            )
            raise

        return {
            "session_id": session.id,
            "message": assistant_message,
            "rag_steps": list(outcome.to_rag_trace()["steps"]),
        }
