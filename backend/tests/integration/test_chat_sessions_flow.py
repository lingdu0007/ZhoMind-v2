import asyncio
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.extensions.registry import get_extension_registry
from app.infra.db import get_db_session
from app.main import app
from app.model.base import Base
from app.service.chat_service import CHAT_JUDGE_PROVIDER, CHAT_RERANK_PROVIDER, CHAT_RETRIEVER_PROVIDER


def _extract_data(payload: dict) -> dict:
    return payload.get("data") or payload


def _auth_headers(client: TestClient, username: str = "chat-user") -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "secret-123", "role": "user"},
    )
    assert response.status_code == 200
    token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


class _CustomRetriever:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        return [
            {
                "chunk_id": "custom-chunk-1",
                "document_id": "doc-custom",
                "chunk_index": 0,
                "score": 0.4,
                "content_preview": f"来自自定义检索器：{query}",
                "metadata": {"source": "custom-retriever"},
            },
            {
                "chunk_id": "custom-chunk-2",
                "document_id": "doc-custom",
                "chunk_index": 1,
                "score": 0.9,
                "content_preview": "高优先级证据",
                "metadata": {"source": "custom-retriever"},
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


def test_chat_and_sessions_flow() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.register_retriever(CHAT_RETRIEVER_PROVIDER, _CustomRetriever())
    registry.register_rerank(CHAT_RERANK_PROVIDER, _CustomReranker())
    registry.register_judge(CHAT_JUDGE_PROVIDER, _CustomJudge())

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client)

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
            assert "高优先级证据" in chat_data["answer"]
            assert chat_data["message"]["type"] == "assistant"
            assert isinstance(chat_data["rag_steps"], list)
            assert chat_data["rag_trace"]["query"] == "请介绍系统当前状态"
            runtime_trace = chat_data["rag_trace"]["runtime"]
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
                "verify",
                "context_pack",
                "generate",
                "memory_write_gate",
                "finalize",
            ]
            assert runtime_trace["steps"][0]["step"] == "normalize"
            assert chat_data["rag_steps"][0]["step"] == "retrieve"
            assert chat_data["rag_steps"][0]["detail"]["retriever"] == CHAT_RETRIEVER_PROVIDER
            assert chat_data["rag_steps"][0]["detail"]["gate_passed"] is True
            assert chat_data["rag_steps"][0]["detail"]["gate_reason"] == "sufficient_evidence"
            assert chat_data["rag_steps"][1]["detail"]["model"] == CHAT_RERANK_PROVIDER
            assert chat_data["rag_steps"][2]["detail"]["judge"] == CHAT_JUDGE_PROVIDER
            assert len(chat_data["rag_trace"]["evidence"]) == 1
            assert "request_id" in chat_body

            stream_response = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "证据不足时系统会如何提示", "session_id": "session_test_1"},
            )
            assert stream_response.status_code == 200
            assert stream_response.headers["content-type"].startswith("text/event-stream")
            text = stream_response.text
            assert "event: rag_step" in text
            assert "event: content" in text
            assert "event: trace" in text
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
        asyncio.run(db_engine.dispose())


def test_chat_reject_gate_when_no_evidence() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session

    registry = get_extension_registry()
    prev_retriever = registry.get_retriever(CHAT_RETRIEVER_PROVIDER)
    prev_reranker = registry.get_rerank(CHAT_RERANK_PROVIDER)
    prev_judge = registry.get_judge(CHAT_JUDGE_PROVIDER)
    registry.retrievers.pop(CHAT_RETRIEVER_PROVIDER, None)
    registry.rerank_providers.pop(CHAT_RERANK_PROVIDER, None)
    registry.judges.pop(CHAT_JUDGE_PROVIDER, None)

    try:
        with TestClient(app) as client:
            headers = _auth_headers(client, username="reject-user")
            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "完全随机且无知识库证据的问题", "session_id": "session_reject_1"},
            )
            assert response.status_code == 200
            body = response.json()
            data = _extract_data(body)
            assert data["rag_steps"][0]["step"] == "retrieve"
            runtime_trace = data["rag_trace"]["runtime"]
            assert runtime_trace["request_id"].startswith("chat-")
            assert runtime_trace["session_id"] == "session_reject_1"
            assert runtime_trace["graph_alias"] == "default_v1"
            assert runtime_trace["gate"]["passed"] is False
            assert runtime_trace["gate"]["reason"] == "reject_insufficient_evidence"
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
                "verify",
                "context_pack",
                "generate",
                "memory_write_gate",
                "finalize",
            ]
            assert data["rag_steps"][0]["detail"]["retriever"] == "inmemory-lexical-retriever"
            assert data["rag_steps"][0]["detail"]["gate_passed"] is False
            assert data["rag_steps"][0]["detail"]["gate_reason"] == "reject_insufficient_evidence"
            assert data["rag_steps"][1]["detail"]["model"] == "inmemory-identity-reranker"
            assert data["rag_steps"][2]["detail"]["judge"] == "inmemory-evidence-judge"
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
        asyncio.run(db_engine.dispose())
