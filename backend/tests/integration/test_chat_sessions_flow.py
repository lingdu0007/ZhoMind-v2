import asyncio
from collections.abc import Generator
import hashlib

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import Settings, get_settings
from app.extensions.registry import get_extension_registry
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.document import Document, DocumentChunk
from app.repository.chat_repository import ChatRepository
from app.service.chat_service import CHAT_JUDGE_PROVIDER, CHAT_RERANK_PROVIDER, CHAT_RETRIEVER_PROVIDER
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._store[key] = dict(mapping)
        return len(mapping)

    async def expire(self, key: str, ttl: int) -> bool:
        return key in self._store

    async def exists(self, key: str) -> int:
        return 1 if key in self._store else 0


def _extract_data(payload: dict) -> dict:
    return payload.get("data") or payload


def _auth_headers(client: TestClient, username: str = "chat-user", role: str = "user") -> dict[str, str]:
    token = asyncio.run(
        create_authenticated_test_token(
            client.app.state.test_auth_session_factory,
            client.app.state.test_auth_redis,
            username=username,
            role=role,
        )
    )
    return {"Authorization": f"Bearer {token}"}


async def _load_assistant_trace(session_factory, session_id: str, user_id: str) -> dict:
    async with session_factory() as session:
        messages = await ChatRepository(session).list_messages(session_id=session_id, user_id=user_id)
    trace = next((item.rag_trace for item in messages if item.type == "assistant"), None)
    assert isinstance(trace, dict)
    return trace


class _CustomRetriever:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        return [
            {
                "chunk_id": "custom-chunk-1",
                "document_id": "doc-custom",
                "generation": 1,
                "chunk_index": 0,
                "score": 0.4,
                "content_preview": f"来自自定义检索器：{query}",
                "metadata": {"title": "自定义资料.md", "publication_version": "v1"},
            },
            {
                "chunk_id": "custom-chunk-2",
                "document_id": "doc-custom",
                "generation": 1,
                "chunk_index": 1,
                "score": 0.9,
                "content_preview": "高优先级证据",
                "metadata": {"title": "自定义资料.md", "publication_version": "v1"},
            },
        ][:top_k]


class _CustomReranker:
    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        if not items:
            return []
        return sorted(items, key=lambda item: item.get("score", 0), reverse=True)[:1]


class _CustomJudge:
    async def judge(self, query: str, context: list[dict]) -> bool:
        return len(context) > 0


class _RecordingLlmProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        return "已基于发布资料生成回答。"


class _StubEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.9, 0.1] for _ in texts]


class _FakeDenseDocumentIndex:
    def __init__(self, rows: list[dict] | None = None, *, error: Exception | None = None) -> None:
        self._rows = list(rows or [])
        self._error = error

    async def search(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        filter: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        if self._error is not None:
            raise self._error
        return self._rows[:limit]

    async def search_batches(
        self,
        *,
        collection_name: str,
        vector: list[float],
        batch_size: int,
        filter: str = "",
        output_fields: list[str] | None = None,
    ):
        if self._error is not None:
            raise self._error
        for offset in range(0, len(self._rows), batch_size):
            yield self._rows[offset : offset + batch_size]


async def _seed_chat_retrieval_docs(
    session_factory,
    *,
    fingerprint: str,
) -> None:
    async with session_factory() as session:
        dense_chunk = DocumentChunk(
            id="chunk-chat-dense",
            document_id="doc-chat-dense",
            generation=1,
            chunk_index=0,
            content="alpha evidence from dense corpus",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "dense"},
        )
        lexical_chunk = DocumentChunk(
            id="chunk-chat-lexical",
            document_id="doc-chat-lexical",
            generation=1,
            chunk_index=0,
            content="beta evidence from lexical fallback",
            keywords=[],
            generated_questions=[],
            chunk_metadata={"source": "lexical"},
        )

        session.add_all(
            [
                Document(
                    id="doc-chat-dense",
                    filename="chat-dense.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=fingerprint,
                    next_generation=2,
                    latest_requested_generation=1,
                ),
                Document(
                    id="doc-chat-lexical",
                    filename="chat-lexical.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=0,
                    dense_ready_fingerprint=None,
                    next_generation=2,
                    latest_requested_generation=1,
                ),
                dense_chunk,
                lexical_chunk,
            ]
        )
        await session.commit()


async def _seed_chat_large_published_corpus_with_tail_match(
    session_factory,
    *,
    fingerprint: str,
    filler_count: int = 205,
) -> None:
    async with session_factory() as session:
        records: list[Document | DocumentChunk] = []
        for idx in range(filler_count):
            doc_id = f"doc-chat-filler-{idx:03d}"
            records.append(
                Document(
                    id=doc_id,
                    filename=f"chat-filler-{idx:03d}.txt",
                    file_type="txt",
                    file_size=10,
                    status="ready",
                    chunk_strategy="general",
                    chunk_count=1,
                    published_generation=1,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=fingerprint,
                    next_generation=2,
                    latest_requested_generation=1,
                )
            )
            records.append(
                DocumentChunk(
                    id=f"chunk-chat-filler-{idx:03d}",
                    document_id=doc_id,
                    generation=1,
                    chunk_index=0,
                    content=f"aaaaa bbbbb ccccc {idx:03d}",
                    keywords=[],
                    generated_questions=[],
                    chunk_metadata={"source": "filler"},
                )
            )

        records.append(
            Document(
                id="doc-chat-tail-match",
                filename="chat-tail-match.txt",
                file_type="txt",
                file_size=10,
                status="ready",
                chunk_strategy="general",
                chunk_count=1,
                published_generation=1,
                dense_ready_generation=1,
                dense_ready_fingerprint=fingerprint,
                next_generation=2,
                latest_requested_generation=1,
            )
        )
        records.append(
            DocumentChunk(
                id="chunk-chat-tail-match",
                document_id="doc-chat-tail-match",
                generation=1,
                chunk_index=0,
                content="xqvzjk chat tail evidence",
                keywords=[],
                generated_questions=[],
                chunk_metadata={"source": "tail-match"},
            )
        )
        session.add_all(records)
        await session.commit()


def test_chat_and_sessions_flow(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "missing-test-llm")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "")
    get_settings.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()

    async def override_get_redis_client() -> _InMemoryRedis:
        return fake_redis

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.register_retriever(CHAT_RETRIEVER_PROVIDER, _CustomRetriever())
    registry.register_rerank(CHAT_RERANK_PROVIDER, _CustomReranker())
    registry.register_judge(CHAT_JUDGE_PROVIDER, _CustomJudge())

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, role="admin")

            chat_response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "请介绍系统当前状态", "session_id": "session_test_1"},
            )
            assert chat_response.status_code == 200
            chat_body = chat_response.json()
            chat_data = _extract_data(chat_body)
            assert chat_data["session_id"] == "session_test_1"
            assert isinstance(chat_data["answer"], str)
            assert chat_data["answer"] == "【生成不可用】生成服务暂不可用，请稍后重试。"
            assert chat_data["message"]["type"] == "assistant"
            diagnostics = chat_data["retrieval_diagnostics"]
            assert diagnostics["candidate_counts"] == {"retrieved": 2, "reranked": 1}
            assert diagnostics["evidence_gate"] == {"outcome": "passed", "reason": "sufficient_evidence"}
            assert "rag_steps" not in chat_data
            assert "rag_trace" not in chat_data

            saved_trace = asyncio.run(_load_assistant_trace(session_factory, "session_test_1", "chat-user"))
            assert saved_trace["query"] == "请介绍系统当前状态"
            runtime_trace = saved_trace["runtime"]
            assert runtime_trace["request_id"].startswith("chat-")
            assert runtime_trace["session_id"] == "session_test_1"
            assert runtime_trace["graph_alias"] == "default_v1"
            assert isinstance(runtime_trace.get("tool_errors", []), list)
            assert runtime_trace["gate"]["passed"] is True
            assert runtime_trace["gate"]["reason"] == "sufficient_evidence"
            assert runtime_trace["step_names"] == [
                "normalize",
                "memory_read",
                "query_understand",
                "plan",
                "tool_plan",
                "tool_execute",
                "tool_verify",
                "retrieve",
                "fusion",
                "rerank",
                "context_pack",
                "verify",
                "generate",
                "memory_write_gate",
                "finalize",
            ]
            assert runtime_trace["steps"][0]["step"] == "normalize"
            assert saved_trace["steps"][0]["step"] == "retrieve"
            assert saved_trace["steps"][0]["detail"]["retriever"] == CHAT_RETRIEVER_PROVIDER
            assert saved_trace["steps"][0]["detail"]["gate_passed"] is True
            assert saved_trace["steps"][0]["detail"]["gate_reason"] == "sufficient_evidence"
            assert saved_trace["steps"][1]["detail"]["model"] == CHAT_RERANK_PROVIDER
            assert saved_trace["steps"][2]["detail"]["judge"] == CHAT_JUDGE_PROVIDER
            assert len(saved_trace["evidence"]) == 1
            assert "request_id" in chat_body

            stream_response = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "证据不足时系统会如何提示", "session_id": "session_test_1"},
            )
            assert stream_response.status_code == 200
            assert stream_response.headers["content-type"].startswith("text/event-stream")
            text = stream_response.text
            assert "event: content" in text
            assert "event: retrieval_diagnostics" in text
            assert "event: rag_step" not in text
            assert "event: trace" not in text
            assert "event: done" in text
            assert "data: [DONE]" in text

            list_response = client.get("/api/v1/sessions", headers=headers)
            assert list_response.status_code == 200
            list_body = list_response.json()
            list_data = _extract_data(list_body)
            assert isinstance(list_data["sessions"], list)
            assert any(item["session_id"] == "session_test_1" for item in list_data["sessions"])
            assert "request_id" in list_body

            detail_response = client.get("/api/v1/sessions/session_test_1", headers=headers)
            assert detail_response.status_code == 200
            detail_data = _extract_data(detail_response.json())
            assert detail_data["session_id"] == "session_test_1"
            assert len(detail_data["messages"]) == 4
            assert detail_data["messages"][0]["type"] == "user"
            assert detail_data["messages"][1]["type"] == "assistant"
            assert "retrieval_diagnostics" in detail_data["messages"][1]

            delete_response = client.delete("/api/v1/sessions/session_test_1", headers=headers)
            assert delete_response.status_code == 200
            delete_data = _extract_data(delete_response.json())
            assert delete_data["session_id"] == "session_test_1"
            assert delete_data["deleted"] is True

            post_delete_detail = client.get("/api/v1/sessions/session_test_1", headers=headers)
            assert post_delete_detail.status_code == 200
            post_delete_data = _extract_data(post_delete_detail.json())
            assert post_delete_data["messages"] == []

            post_delete_list = client.get("/api/v1/sessions", headers=headers)
            assert post_delete_list.status_code == 200
            post_delete_list_data = _extract_data(post_delete_list.json())
            assert not any(item["session_id"] == "session_test_1" for item in post_delete_list_data["sessions"])

            missing_delete = client.delete("/api/v1/sessions/does-not-exist", headers=headers)
            assert missing_delete.status_code == 200
            missing_delete_data = _extract_data(missing_delete.json())
            assert missing_delete_data["deleted"] is False

            unauth = client.get("/api/v1/sessions")
            assert unauth.status_code == 401
            unauth_body = unauth.json()
            assert unauth_body["code"] == "AUTH_INVALID_TOKEN"
            assert "request_id" in unauth_body
    finally:
        if prev_retriever is None:
            registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
        else:
            registry.register_retriever(CHAT_RETRIEVER_PROVIDER, prev_retriever)

        if prev_reranker is None:
            registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
        else:
            registry.register_rerank(CHAT_RERANK_PROVIDER, prev_reranker)

        if prev_judge is None:
            registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
        else:
            registry.register_judge(CHAT_JUDGE_PROVIDER, prev_judge)

        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_chat_reject_gate_when_no_evidence(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "missing-test-llm")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "")
    get_settings.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()

    async def override_get_redis_client() -> _InMemoryRedis:
        return fake_redis

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="reject-user", role="admin")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "完全随机且无知识库证据的问题", "session_id": "session_reject_1"},
            )
            assert response.status_code == 200
            body = response.json()
            data = _extract_data(body)
            assert data["retrieval_diagnostics"]["evidence_gate"] == {
                "outcome": "rejected",
                "reason": "reject_insufficient_evidence",
            }
            assert "rag_trace" not in data
            assert "未检索到足够相关的知识片段" in data["answer"]
            assert "request_id" in body
    finally:
        if prev_retriever is not None:
            registry.register_retriever(CHAT_RETRIEVER_PROVIDER, prev_retriever)
        if prev_reranker is not None:
            registry.register_rerank(CHAT_RERANK_PROVIDER, prev_reranker)
        if prev_judge is not None:
            registry.register_judge(CHAT_JUDGE_PROVIDER, prev_judge)
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_chat_returns_non_knowledge_base_reply_without_retrieval_evidence(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "missing-test-llm")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "")
    get_settings.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()

    async def override_get_redis_client() -> _InMemoryRedis:
        return fake_redis

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="smalltalk-user", role="admin")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "你是谁", "session_id": "session_smalltalk_1"},
            )
            assert response.status_code == 200
            body = response.json()
            data = _extract_data(body)
            assert data["retrieval_diagnostics"]["evidence_gate"] == {
                "outcome": "unavailable",
                "reason": "not_applicable_non_knowledge_base",
            }
            assert data["outcome"] == "non_knowledge_base_reply"
            assert "rag_trace" not in data
            assert "我是 ZhoMind 智能助手" in data["answer"]
            assert "request_id" in body
    finally:
        if prev_retriever is not None:
            registry.register_retriever(CHAT_RETRIEVER_PROVIDER, prev_retriever)
        if prev_reranker is not None:
            registry.register_rerank(CHAT_RERANK_PROVIDER, prev_reranker)
        if prev_judge is not None:
            registry.register_judge(CHAT_JUDGE_PROVIDER, prev_judge)
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_chat_dense_trace_uses_default_mixed_mode_retriever(monkeypatch) -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "missing-test-llm")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)
    dense_row = {
        "document_id": "doc-chat-dense",
        "generation": 1,
        "chunk_index": 0,
        "content_sha256": hashlib.sha256("alpha evidence from dense corpus".encode("utf-8")).hexdigest(),
        "distance": 0.97,
    }

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())
    asyncio.run(_seed_chat_retrieval_docs(session_factory, fingerprint=fingerprint))

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()

    async def override_get_redis_client() -> _InMemoryRedis:
        return fake_redis

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_embedding = registry.get_embedding("embedding-default")
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
    registry.register_embedding("embedding-default", _StubEmbeddingProvider())

    class _InjectedMixedModeRetriever(MixedModeDocumentRetrieverService):
        def __init__(self, session: AsyncSession) -> None:
            super().__init__(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(rows=[dense_row]),
            )

    monkeypatch.setattr("app.service.chat_service.MixedModeDocumentRetrieverService", _InjectedMixedModeRetriever)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="dense-default-admin", role="admin")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "alpha beta", "session_id": "session_dense_default_1"},
            )
            assert response.status_code == 200
            data = _extract_data(response.json())
            assert data["retrieval_diagnostics"]["evidence_gate"]["outcome"] == "passed"
            saved_trace = asyncio.run(_load_assistant_trace(session_factory, "session_dense_default_1", "dense-default-admin"))
            assert saved_trace["steps"][0]["detail"]["retriever"] == "inmemory-mixed-mode-retriever"

            retrieve_trace = saved_trace["runtime"]["provider_trace"]["retrieve"]
            assert retrieve_trace["strategy"] == "dense_plus_lexical_migration"
            assert retrieve_trace["dense_candidate_count"] == 1
            assert retrieve_trace["dense_hydrated_count"] == 1
            assert retrieve_trace["lexical_candidate_count"] == 1
            assert retrieve_trace["merged_count"] == 2
            assert retrieve_trace["dense_query_failed"] is False
            assert retrieve_trace["lexical_scope"] == "not_dense_ready_published"

            evidence = saved_trace["evidence"]
            assert [item["retrieval_source"] for item in evidence] == ["dense", "lexical"]
            assert data["answer"] == "【生成不可用】生成服务暂不可用，请稍后重试。"
    finally:
        if prev_retriever is not None:
            registry.register_retriever(CHAT_RETRIEVER_PROVIDER, prev_retriever)
        else:
            registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
        if prev_embedding is not None:
            registry.register_embedding("embedding-default", prev_embedding)
        else:
            registry.embedding_providers.pop("embedding-default", None)
        if prev_reranker is not None:
            registry.register_rerank(CHAT_RERANK_PROVIDER, prev_reranker)
        else:
            registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
        if prev_judge is not None:
            registry.register_judge(CHAT_JUDGE_PROVIDER, prev_judge)
        else:
            registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        get_extension_registry.cache_clear()
        asyncio.run(db_engine.dispose())


def test_knowledge_user_chat_hydrates_published_evidence_beyond_stale_dense_candidates(monkeypatch) -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()
    provider = _RecordingLlmProvider()

    monkeypatch.delenv("RUNTIME_RETRIEVAL_TOP_K", raising=False)
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)
    published_content = "发布验收锚点：当前发布版本必须经过产品对话路径返回引用。"

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            session.add_all(
                [
                    Document(
                        id="doc-chat-current-published",
                        filename="当前发布验收.md",
                        file_type="md",
                        file_size=100,
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        dense_ready_generation=1,
                        dense_ready_fingerprint=fingerprint,
                        next_generation=2,
                        latest_requested_generation=1,
                    ),
                    DocumentChunk(
                        id="chunk-chat-current-published",
                        document_id="doc-chat-current-published",
                        generation=1,
                        chunk_index=0,
                        content=published_content,
                        keywords=[],
                        generated_questions=[],
                        chunk_metadata={},
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_init_db())

    stale_rows = [
        {
            "entity": {
                "document_id": f"withdrawn-document-{index}",
                "generation": 1,
                "chunk_index": 0,
                "content_sha256": hashlib.sha256(f"stale-{index}".encode()).hexdigest(),
            },
            "distance": 1.0 - (index / 100),
        }
        for index in range(224)
    ]
    current_published_row = {
        "entity": {
            "document_id": "doc-chat-current-published",
            "generation": 1,
            "chunk_index": 0,
            "content_sha256": hashlib.sha256(published_content.encode()).hexdigest(),
        },
        "distance": 0.8,
    }

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    registry.register_embedding("embedding-default", _StubEmbeddingProvider())
    registry.register_llm("ark", provider)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)

    class _InjectedMixedModeRetriever(MixedModeDocumentRetrieverService):
        def __init__(self, session: AsyncSession) -> None:
            super().__init__(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(rows=[*stale_rows, current_published_row]),
            )

    monkeypatch.setattr("app.service.chat_service.MixedModeDocumentRetrieverService", _InjectedMixedModeRetriever)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="published-window-user", role="user")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "发布验收锚点", "session_id": "published-window"},
            )

            assert response.status_code == 200
            data = _extract_data(response.json())
            assert data["answer"] == "已基于发布资料生成回答。"
            assert data["message"]["evidence_summary"] == {
                "coverage": "sufficient",
                "source_count": 1,
                "sources": [
                    {
                        "source_id": "chunk-chat-current-published",
                        "metadata": {"title": "当前发布验收.md", "publication_version": "v1"},
                        "excerpt": published_content,
                    }
                ],
            }
            assert len(provider.prompts) == 1
            assert published_content in provider.prompts[0]
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        get_extension_registry.cache_clear()
        asyncio.run(db_engine.dispose())


def test_chat_dense_failure_trace_marks_runtime_fallback_and_error(monkeypatch) -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "missing-test-llm")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())
    asyncio.run(_seed_chat_retrieval_docs(session_factory, fingerprint=fingerprint))

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()

    async def override_get_redis_client() -> _InMemoryRedis:
        return fake_redis

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_embedding = registry.get_embedding("embedding-default")
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
    registry.register_embedding("embedding-default", _StubEmbeddingProvider())

    class _InjectedMixedModeRetriever(MixedModeDocumentRetrieverService):
        def __init__(self, session: AsyncSession) -> None:
            super().__init__(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(error=RuntimeError("milvus unavailable")),
            )

    monkeypatch.setattr("app.service.chat_service.MixedModeDocumentRetrieverService", _InjectedMixedModeRetriever)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="dense-failure-admin", role="admin")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "alpha beta", "session_id": "session_dense_failure_1"},
            )
            assert response.status_code == 200
            data = _extract_data(response.json())
            assert {"stage": "retrieve", "code": "PROVIDER_EXEC_FAILED", "type": "RuntimeError"} in data[
                "retrieval_diagnostics"
            ]["provider_errors"]

            saved_trace = asyncio.run(_load_assistant_trace(session_factory, "session_dense_failure_1", "dense-failure-admin"))
            retrieve_trace = saved_trace["runtime"]["provider_trace"]["retrieve"]
            assert retrieve_trace["dense_query_failed"] is True
            assert retrieve_trace["fallback_used"] is True
            assert retrieve_trace["provider_error"] == {
                "code": "PROVIDER_EXEC_FAILED",
                "message": "milvus unavailable",
                "type": "RuntimeError",
            }

            retrieve_step = next(
                step for step in saved_trace["runtime"]["steps"] if step["step"] == "retrieve"
            )
            assert retrieve_step["detail"]["dense_query_failed"] is True
            assert retrieve_step["detail"]["fallback_used"] is True
            assert retrieve_step["detail"]["provider_error"] == {
                "code": "PROVIDER_EXEC_FAILED",
                "message": "milvus unavailable",
                "type": "RuntimeError",
            }
    finally:
        if prev_retriever is not None:
            registry.register_retriever(CHAT_RETRIEVER_PROVIDER, prev_retriever)
        else:
            registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
        if prev_embedding is not None:
            registry.register_embedding("embedding-default", prev_embedding)
        else:
            registry.embedding_providers.pop("embedding-default", None)
        if prev_reranker is not None:
            registry.register_rerank(CHAT_RERANK_PROVIDER, prev_reranker)
        else:
            registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
        if prev_judge is not None:
            registry.register_judge(CHAT_JUDGE_PROVIDER, prev_judge)
        else:
            registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        get_extension_registry.cache_clear()
        asyncio.run(db_engine.dispose())


def test_chat_dense_failure_full_lexical_fallback_reads_tail_of_published_live_corpus(monkeypatch) -> None:
    from app.rag.dense_contract import build_embedding_contract_fingerprint
    from app.service.document_retrieval_service import MixedModeDocumentRetrieverService

    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "missing-test-llm")
    monkeypatch.setenv("RAG_LLM_FALLBACK_PROVIDERS", "")
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    settings = Settings(
        EMBEDDING_API_KEY="emb-key",
        EMBEDDING_BASE_URL="https://emb.example.com/v1",
        EMBEDDING_MODEL="emb-model",
        DENSE_EMBEDDING_DIM=2,
        MILVUS_URI="http://milvus.example.com:19530",
    )
    fingerprint = build_embedding_contract_fingerprint(settings)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())
    asyncio.run(_seed_chat_large_published_corpus_with_tail_match(session_factory, fingerprint=fingerprint))

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    fake_redis = _InMemoryRedis()

    async def override_get_redis_client() -> _InMemoryRedis:
        return fake_redis

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = override_get_redis_client
    app.state.test_auth_redis = fake_redis

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_embedding = registry.get_embedding("embedding-default")
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
    registry.register_embedding("embedding-default", _StubEmbeddingProvider())

    class _InjectedMixedModeRetriever(MixedModeDocumentRetrieverService):
        def __init__(self, session: AsyncSession) -> None:
            super().__init__(
                session,
                settings=settings,
                embedding_provider=_StubEmbeddingProvider(),
                document_index=_FakeDenseDocumentIndex(error=RuntimeError("milvus unavailable")),
            )

    monkeypatch.setattr("app.service.chat_service.MixedModeDocumentRetrieverService", _InjectedMixedModeRetriever)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="dense-tail-failure-admin", role="admin")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "xqvzjk", "session_id": "session_dense_tail_failure_1"},
            )
            assert response.status_code == 200
            data = _extract_data(response.json())
            assert {"stage": "retrieve", "code": "PROVIDER_EXEC_FAILED", "type": "RuntimeError"} in data[
                "retrieval_diagnostics"
            ]["provider_errors"]

            saved_trace = asyncio.run(
                _load_assistant_trace(session_factory, "session_dense_tail_failure_1", "dense-tail-failure-admin")
            )
            retrieve_trace = saved_trace["runtime"]["provider_trace"]["retrieve"]
            assert retrieve_trace["dense_query_failed"] is True
            assert retrieve_trace["fallback_used"] is True
            assert retrieve_trace["lexical_scope"] == "full_published_live"
            assert retrieve_trace["lexical_candidate_count"] == 1

            evidence = saved_trace["evidence"]
            assert [item["document_id"] for item in evidence] == ["doc-chat-tail-match"]
            assert data["answer"] == "【生成不可用】生成服务暂不可用，请稍后重试。"
    finally:
        if prev_retriever is not None:
            registry.register_retriever(CHAT_RETRIEVER_PROVIDER, prev_retriever)
        else:
            registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
        if prev_embedding is not None:
            registry.register_embedding("embedding-default", prev_embedding)
        else:
            registry.embedding_providers.pop("embedding-default", None)
        if prev_reranker is not None:
            registry.register_rerank(CHAT_RERANK_PROVIDER, prev_reranker)
        else:
            registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
        if prev_judge is not None:
            registry.register_judge(CHAT_JUDGE_PROVIDER, prev_judge)
        else:
            registry.judges.pop(CHAT_JUDGE_PROVIDER, None)
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        get_extension_registry.cache_clear()
        asyncio.run(db_engine.dispose())
