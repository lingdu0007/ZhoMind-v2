from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.extensions.provider_router import ProviderRouter
from app.extensions.registry import get_extension_registry
from app.model.document import DocumentChunk
from app.rag.interfaces import RelevanceJudge, Reranker, Retriever
from app.rag.runtime.graph_runner import RagGraphRunner
from app.repository.chat_repository import ChatRepository
from app.service.runtime_trace_mapper import RuntimeTraceMapper

CHAT_RETRIEVER_PROVIDER = "chat-default-retriever"
CHAT_RERANK_PROVIDER = "chat-default-reranker"
CHAT_JUDGE_PROVIDER = "chat-default-judge"
CHAT_LLM_PROVIDER = "chat-default-llm"


class _SessionLexicalRetriever:
    name = "inmemory-lexical-retriever"

    def __init__(self, loader: Callable[[str, int], Awaitable[list[dict]]]) -> None:
        self._loader = loader

    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        return await self._loader(query, top_k)


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

    def _tokenize(self, text: str) -> list[str]:
        return [item for item in re.split(r"[\s\W_]+", text.lower()) if len(item) >= 2]

    def _compact(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _bigram_overlap(self, a: str, b: str) -> int:
        if len(a) < 2 or len(b) < 2:
            return 0
        a_set = {a[i : i + 2] for i in range(len(a) - 1)}
        b_set = {b[i : i + 2] for i in range(len(b) - 1)}
        return len(a_set & b_set)

    def _score_chunk(self, query: str, content: str) -> float:
        query_norm = query.strip().lower()
        content_norm = (content or "").strip().lower()
        if not query_norm or not content_norm:
            return 0.0

        score = 0.0
        query_compact = self._compact(query_norm)
        content_compact = self._compact(content_norm)

        if query_compact and query_compact in content_compact:
            score += 5.0

        query_tokens = self._tokenize(query_norm)
        if query_tokens:
            content_tokens = set(self._tokenize(content_norm))
            overlap = sum(1 for token in query_tokens if token in content_tokens)
            score += float(overlap)

        score += min(self._bigram_overlap(query_compact, content_compact), 6) * 0.8
        return score

    async def _default_retrieve_chunks(self, question: str, top_k: int = 3) -> list[dict]:
        query = question.strip()
        if not query:
            return []

        result = await self.session.execute(select(DocumentChunk).limit(200))
        candidates = list(result.scalars().all())

        ranked: list[dict] = []
        for chunk in candidates:
            score = self._score_chunk(query=query, content=chunk.content)
            if score <= 0:
                continue
            ranked.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "score": round(score, 4),
                    "content_preview": chunk.content[:160],
                    "metadata": chunk.chunk_metadata,
                }
            )

        ranked.sort(key=lambda item: (-item["score"], item["chunk_index"], item["chunk_id"]))
        return ranked[:top_k]

    def _resolve_retriever(self) -> tuple[Retriever, str]:
        provider = get_extension_registry().get_retriever(CHAT_RETRIEVER_PROVIDER)
        if provider is not None:
            return provider, CHAT_RETRIEVER_PROVIDER

        fallback = _SessionLexicalRetriever(loader=self._default_retrieve_chunks)
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
        for idx, item in enumerate(retrieved[:3], start=1):
            content = str(item.get("content_preview") or item.get("content") or "")
            lines.append(f"证据{idx}：{content}")
        lines.append("请给出简洁中文回答。")
        return "\n".join(lines)

    def _is_smalltalk_question(self, question: str) -> bool:
        compact = self._compact(question.strip().lower())
        if not compact:
            return False

        patterns = (
            "你是谁",
            "你叫什麼",
            "你叫什么",
            "介绍你自己",
            "自我介绍",
            "whoareyou",
            "whatyourname",
            "whatareyou",
        )
        if any(pattern in compact for pattern in patterns):
            return True

        return compact in {"你好", "您好", "hello", "hi", "hey"}

    def _smalltalk_reply(self) -> str:
        return "我是 ZhoMind 智能助手，可以帮你基于知识库问答、梳理文档与会话内容。"

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
        settings = get_settings()
        llm_result = await self._provider_router().complete(
            primary=settings.rag_primary_llm_provider,
            fallbacks=settings.rag_llm_fallback_providers,
            prompt=prompt,
        )
        completion = str(llm_result.get("text") or "").strip()
        if completion:
            return completion, llm_result

        lines = ["根据检索到的知识片段，先给你一个最小可用回答："]
        for idx, item in enumerate(retrieved[:3], start=1):
            content = str(item.get("content_preview") or item.get("content") or "")
            lines.append(f"{idx}. {content}")
        text = "\n".join(lines)
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

    async def get_session_messages(self, session_id: str, user_id: str) -> list[dict]:
        session = await self.repo.get_session(session_id=session_id, user_id=user_id)
        if session is None:
            return []
        messages = await self.repo.list_messages(session_id=session_id, user_id=user_id)
        return [
            {
                "id": item.id,
                "type": item.type,
                "content": item.content,
                "timestamp": item.created_at.isoformat(),
                "rag_trace": item.rag_trace,
            }
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

        if get_settings().rag_disable_gate:
            gate_passed = True
            gate_reason = "gate_disabled"

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
