from datetime import datetime, timezone
import json
import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.extensions.provider_router import ProviderRouter
from app.extensions.registry import get_extension_registry
from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.runtime.graph_runner import RagGraphRunner
from app.repository.chat_repository import ChatRepository
from app.service.document_retrieval_service import MixedModeDocumentRetrieverService
from app.service.runtime_trace_mapper import RuntimeTraceMapper
from app.settings.runtime import get_runtime_settings

CHAT_RETRIEVER_PROVIDER = "chat-default-retriever"
CHAT_RERANK_PROVIDER = "chat-default-reranker"
CHAT_JUDGE_PROVIDER = "chat-default-judge"
CHAT_LLM_PROVIDER = "chat-default-llm"
DIAGNOSTIC_MAX_TIMELINE_STEPS = 16
DIAGNOSTIC_MAX_PROVIDER_ERRORS = 5
DIAGNOSTIC_MAX_TRACE_PREVIEW_CHARS = 1600
GENERATION_CONTEXT_MAX_ITEMS = 3
GENERATION_CONTEXT_MAX_CHARS_PER_SOURCE = 160
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


class _EvidenceJudge:
    name = "inmemory-evidence-judge"

    async def judge(self, query: str, context: list[dict]) -> bool:
        return len(context) > 0


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ChatRepository(session)

    def _compact(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _resolve_retriever(self) -> tuple[Retriever, str]:
        provider = get_extension_registry().get_retriever(CHAT_RETRIEVER_PROVIDER)
        if provider is not None:
            return provider, CHAT_RETRIEVER_PROVIDER

        fallback = MixedModeDocumentRetrieverService(self.session)
        return fallback, fallback.name

    def _resolve_reranker(self) -> tuple[Reranker, str]:
        provider = get_extension_registry().get_rerank(CHAT_RERANK_PROVIDER)
        if provider is not None:
            return provider, CHAT_RERANK_PROVIDER

        fallback = _IdentityReranker()
        return fallback, fallback.name

    def _resolve_judge(self) -> tuple[RelevanceJudge, str]:
        provider = get_extension_registry().get_judge(CHAT_JUDGE_PROVIDER)
        if provider is not None:
            return provider, CHAT_JUDGE_PROVIDER

        fallback = _EvidenceJudge()
        return fallback, fallback.name

    def _provider_router(self) -> ProviderRouter:
        return ProviderRouter(providers=get_extension_registry().llm_providers)

    def _compose_llm_prompt(self, question: str, retrieved: list[dict]) -> str:
        lines = ["请基于以下证据回答用户问题。", f"问题：{question}"]
        for idx, item in enumerate(retrieved[:GENERATION_CONTEXT_MAX_ITEMS], start=1):
            content = str(item.get("content_preview") or item.get("content") or "")[:GENERATION_CONTEXT_MAX_CHARS_PER_SOURCE]
            lines.append(f"证据{idx}：{content}")
        lines.append("请给出简洁中文回答。")
        return "\n".join(lines)

    def _is_smalltalk_question(self, question: str) -> bool:
        compact = self._compact(question.strip().lower())
        if not compact:
            return False

        patterns = {
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
        return compact in patterns

    def _smalltalk_reply(self) -> str:
        return "【非知识库回复】我是 ZhoMind 智能助手，可以帮你基于知识库问答、梳理文档与会话内容。"

    async def _assistant_reply(
        self,
        question: str,
        retrieved: list[dict],
        gate_passed: bool,
        *,
        gate_reason: str,
    ) -> tuple[str, dict]:
        if gate_reason == "smalltalk_fallback":
            text = self._smalltalk_reply()
            return text, {
                "text": text,
                "final_provider": "smalltalk",
                "provider_attempts": [],
                "fallback_hops": 0,
            }

        if not gate_passed:
            text = "未检索到足够相关的知识片段，请补充更具体的问题或关键词。"
            return text, {
                "text": text,
                "final_provider": None,
                "provider_attempts": [],
                "fallback_hops": 0,
            }

        prompt = self._compose_llm_prompt(question=question, retrieved=retrieved)
        settings = get_runtime_settings()
        llm_result = await self._provider_router().complete(
            primary=settings.rag_primary_llm_provider,
            fallbacks=[],
            prompt=prompt,
        )
        completion = str(llm_result.get("text") or "").strip()
        if completion:
            return completion, llm_result

        text = "【生成不可用】生成服务暂不可用，请稍后重试。"
        llm_result["text"] = text
        return text, llm_result

    def _rag_steps(
        self,
        *,
        question: str,
        retriever_name: str,
        reranker_name: str,
        judge_name: str,
        llm_name: str,
        retrieved_count: int,
        reranked_count: int,
        gate_passed: bool,
        gate_reason: str,
    ) -> list[dict]:
        return [
            {
                "step": "retrieve",
                "detail": {
                    "query": question,
                    "retriever": retriever_name,
                    "retrieved_count": retrieved_count,
                    "gate_passed": gate_passed,
                    "gate_reason": gate_reason,
                },
            },
            {
                "step": "rerank",
                "detail": {
                    "model": reranker_name,
                    "reranked_count": reranked_count,
                },
            },
            {
                "step": "verify",
                "detail": {
                    "judge": judge_name,
                },
            },
            {
                "step": "generate",
                "detail": {
                    "llm": llm_name,
                },
            },
        ]

    def _runtime_trace(self, runtime_result: dict) -> dict:
        return RuntimeTraceMapper.map_runtime(runtime_result)

    def _evidence_summary(self, rag_trace: dict | None) -> dict:
        trace = rag_trace if isinstance(rag_trace, dict) else {}
        evidence = trace.get("evidence")
        evidence_items = evidence if isinstance(evidence, list) else []
        gate = trace.get("gate") if isinstance(trace.get("gate"), dict) else {}
        gate_passed = gate.get("passed")

        sources: list[dict] = []
        for index, item in enumerate(evidence_items, start=1):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            source_metadata = (
                {
                    key: metadata[key]
                    for key in ("title", "publication_version", "filename", "source_file", "source", "document_name", "path")
                    if isinstance(metadata, dict) and isinstance(metadata.get(key), str) and metadata[key].strip()
                }
                if isinstance(metadata, dict)
                else {}
            )
            source_id = item.get("chunk_id") or item.get("source_id") or item.get("document_id") or f"source-{index}"
            if item.get("withdrawn") is True:
                sources.append(
                    {
                        "source_id": str(source_id),
                        "metadata": source_metadata,
                        "withdrawal_notice": "This source has been withdrawn.",
                    }
                )
                continue
            excerpt = item.get("content_preview") or item.get("content") or ""
            sources.append(
                {
                    "source_id": str(source_id),
                    "metadata": source_metadata,
                    "excerpt": str(excerpt),
                }
            )

        if gate_passed is False:
            coverage = "insufficient"
        elif sources:
            coverage = "sufficient"
        else:
            coverage = "unavailable"

        return {
            "coverage": coverage,
            "source_count": len(sources),
            "sources": sources,
        }

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
            preview_item = {"step": step}
            detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
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
        runtime = rag_trace.get("runtime") if isinstance(rag_trace.get("runtime"), dict) else {}
        gate = rag_trace.get("gate") if isinstance(rag_trace.get("gate"), dict) else {}
        attempts = runtime.get("provider_attempts") if isinstance(runtime.get("provider_attempts"), list) else []
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
        runtime = rag_trace.get("runtime") if isinstance(rag_trace.get("runtime"), dict) else {}
        runtime_steps = runtime.get("steps") if isinstance(runtime.get("steps"), list) else []
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
        steps = rag_trace.get("steps") if isinstance(rag_trace.get("steps"), list) else []
        for item in steps:
            if not isinstance(item, dict) or not isinstance(item.get("detail"), dict):
                continue
            detail = item["detail"]
            if item.get("step") == "retrieve":
                retrieved = self._diagnostic_count(detail.get("retrieved_count"))
            elif item.get("step") == "rerank":
                reranked = self._diagnostic_count(detail.get("reranked_count"))

        runtime = rag_trace.get("runtime") if isinstance(rag_trace.get("runtime"), dict) else {}
        runtime_steps = runtime.get("steps") if isinstance(runtime.get("steps"), list) else []
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
        runtime = rag_trace.get("runtime") if isinstance(rag_trace.get("runtime"), dict) else {}
        provider_trace = runtime.get("provider_trace") if isinstance(runtime.get("provider_trace"), dict) else {}
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

        attempts = runtime.get("provider_attempts") if isinstance(runtime.get("provider_attempts"), list) else []
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
        gate = trace.get("gate") if isinstance(trace.get("gate"), dict) else {}
        gate_passed = gate.get("passed")
        gate_outcome = "passed" if gate_passed is True else "rejected" if gate_passed is False else "unavailable"

        runtime = trace.get("runtime") if isinstance(trace.get("runtime"), dict) else {}
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
        diagnostics["trace_preview"] = self._diagnostic_trace_preview(trace)
        return diagnostics

    def project_message(self, message: dict, role: str) -> dict:
        projection = {
            "id": message.get("id"),
            "type": message.get("type"),
            "content": message.get("content") or "",
            "timestamp": message.get("timestamp"),
        }
        if projection["type"] == "assistant":
            projection["evidence_summary"] = self._evidence_summary(message.get("rag_trace"))
            if role == "admin":
                projection["retrieval_diagnostics"] = self._retrieval_diagnostics(message.get("rag_trace"))
        return projection

    def project_chat_result(self, result: dict, role: str) -> dict:
        message = self.project_message(result["message"], role)
        projection = {
            "session_id": result["session_id"],
            "answer": message["content"],
            "message": message,
        }
        if role == "admin":
            projection["retrieval_diagnostics"] = message["retrieval_diagnostics"]
        return projection

    async def ensure_session_id(self, session_id: str | None) -> str:
        if session_id and session_id.strip():
            return session_id.strip()
        return f"session_{uuid.uuid4().hex[:16]}"

    async def list_sessions(self, user_id: str) -> list[dict]:
        sessions = await self.repo.list_sessions(user_id=user_id, limit=20)
        items: list[dict] = []
        for session in sessions:
            messages = await self.repo.list_messages(session_id=session.id, user_id=user_id)
            items.append(
                {
                    "session_id": session.id,
                    "updated_at": session.updated_at.isoformat(),
                    "message_count": len(messages),
                }
            )
        return items

    async def get_session_messages(self, session_id: str, user_id: str, role: str) -> list[dict]:
        session = await self.repo.get_session(session_id=session_id, user_id=user_id)
        if session is None:
            return []
        messages = await self.repo.list_messages(session_id=session_id, user_id=user_id)
        return [
            self.project_message(
                {
                "id": item.id,
                "type": item.type,
                "content": item.content,
                "timestamp": item.created_at.isoformat(),
                "rag_trace": item.rag_trace,
                },
                role,
            )
            for item in messages
        ]

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        deleted = await self.repo.delete_session(session_id=session_id, user_id=user_id)
        if deleted:
            await self.session.commit()
        return deleted

    async def run_chat(self, user_id: str, question: str, session_id: str | None) -> dict:
        sid = await self.ensure_session_id(session_id)
        session = await self.repo.get_or_create_session(session_id=sid, user_id=user_id)

        normalized_question = question.strip()

        await self.repo.add_message(
            session_id=session.id,
            user_id=user_id,
            message_type="user",
            content=normalized_question,
        )

        retriever, retriever_name = self._resolve_retriever()
        reranker, reranker_name = self._resolve_reranker()
        judge, judge_name = self._resolve_judge()

        runner = RagGraphRunner(
            retriever=retriever,
            reranker=reranker,
            judge=judge,
        )
        runtime_result = await runner.run(
            request_id=f"chat-{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            session_id=session.id,
            question=normalized_question,
        )

        gate = runtime_result["gate"]
        gate_passed = bool(gate.get("passed"))
        gate_reason = str(gate.get("reason") or "reject_insufficient_evidence")
        reranked = runtime_result["evidence"]
        retrieved = runtime_result.get("retrieved") or reranked

        if not reranked and gate_passed:
            gate_passed = False
            gate_reason = "reject_insufficient_evidence"

        if not gate_passed and self._is_smalltalk_question(normalized_question):
            gate_passed = True
            gate_reason = "smalltalk_fallback"

        reply, llm_result = await self._assistant_reply(
            question=normalized_question,
            retrieved=reranked,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
        )
        llm_name = str(llm_result.get("final_provider") or CHAT_LLM_PROVIDER)
        rag_steps = self._rag_steps(
            question=normalized_question,
            retriever_name=retriever_name,
            reranker_name=reranker_name,
            judge_name=judge_name,
            llm_name=llm_name,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            gate_passed=gate_passed,
            gate_reason=gate_reason,
        )

        runtime_result["final_provider"] = llm_result.get("final_provider")
        runtime_result["provider_attempts"] = list(llm_result.get("provider_attempts") or [])
        runtime_result["fallback_hops"] = int(llm_result.get("fallback_hops") or 0)

        runtime_trace = self._runtime_trace(runtime_result)
        runtime_trace["gate"] = {
            "passed": gate_passed,
            "reason": gate_reason,
        }

        rag_trace = {
            "query": normalized_question,
            "steps": rag_steps,
            "gate": {
                "passed": gate_passed,
                "reason": gate_reason,
            },
            "evidence": reranked,
            "answer_preview": reply[:120],
            "runtime": runtime_trace,
        }

        assistant_message = await self.repo.add_message(
            session_id=session.id,
            user_id=user_id,
            message_type="assistant",
            content=reply,
            rag_trace=rag_trace,
        )

        session.updated_at = datetime.now(timezone.utc)
        await self.session.commit()

        return {
            "session_id": session.id,
            "message": {
                "id": assistant_message.id,
                "type": assistant_message.type,
                "content": assistant_message.content,
                "timestamp": assistant_message.created_at.isoformat(),
                "rag_trace": rag_trace,
            },
            "rag_steps": rag_steps,
        }
