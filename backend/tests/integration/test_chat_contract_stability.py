import asyncio
from collections.abc import Generator
import json

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.repository.chat_repository import ChatRepository
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0


def _extract_sse_event_data(payload: str, event: str) -> str | None:
    for block in payload.split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        if lines[0] != f"event: {event}":
            continue
        if not lines[1].startswith("data: "):
            continue
        return lines[1][6:]
    return None


def test_administrator_chat_response_contract_projects_bounded_retrieval_diagnostics(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    get_settings.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_redis = fake_redis

    try:
        with TestClient(app) as client:
            token = asyncio.run(
                create_authenticated_test_token(
                    session_factory, fake_redis, username="contract-admin", role="admin"
                )
            )
            headers = {"Authorization": f"Bearer {token}"}

            resp = client.post("/api/v1/chat", headers=headers, json={"message": "合同稳定性随机无证据问题", "session_id": "contract_s1"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == "OK"
            assert "request_id" in body
            data = body["data"]
            assert set(["session_id", "answer", "message", "retrieval_diagnostics"]).issubset(data.keys())
            assert data["message"]["retrieval_diagnostics"] == data["retrieval_diagnostics"]
            assert "rag_steps" not in data
            assert "rag_trace" not in data
            assert "rag_trace" not in data["message"]

            diagnostics = data["retrieval_diagnostics"]
            assert diagnostics["candidate_counts"] == {"retrieved": 0, "reranked": 0}
            assert diagnostics["evidence_gate"] == {
                "outcome": "rejected",
                "reason": "reject_insufficient_evidence",
            }
            assert diagnostics["fallback"] == {
                "state": "not_used",
                "hops": 0,
                "final_provider": None,
            }
            assert diagnostics["provider_errors"] == []
            assert [item["step"] for item in diagnostics["timeline"]] == [
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
            assert diagnostics["trace_preview"].startswith("{")
            assert len(diagnostics["trace_preview"]) <= 1600
            assert '"runtime":' in diagnostics["trace_preview"]
            assert '"request_id":"chat-' in diagnostics["trace_preview"]

            stream_resp = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "合同稳定性流式随机无证据问题", "session_id": "contract_s1"},
            )
            assert stream_resp.status_code == 200
            assert stream_resp.headers["content-type"].startswith("text/event-stream")

            diagnostics_data = _extract_sse_event_data(stream_resp.text, "retrieval_diagnostics")
            assert diagnostics_data is not None
            streamed = json.loads(diagnostics_data)["retrieval_diagnostics"]
            assert streamed["evidence_gate"]["reason"] == "reject_insufficient_evidence"
            assert [item["step"] for item in streamed["timeline"]] == [item["step"] for item in diagnostics["timeline"]]
            assert "event: rag_step" not in stream_resp.text
            assert "event: trace" not in stream_resp.text
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_knowledge_user_chat_projection_excludes_retrieval_diagnostics(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    monkeypatch.setenv("RAG_DISABLE_GATE", "false")
    get_settings.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_redis = fake_redis

    try:
        with TestClient(app) as client:
            token = asyncio.run(create_authenticated_test_token(session_factory, fake_redis, username="evidence-user"))
            headers = {"Authorization": f"Bearer {token}"}

            response = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "没有检索证据的问题", "session_id": "evidence_projection_s1"},
            )
            assert response.status_code == 200
            data = response.json()["data"]
            assert data["message"]["evidence_summary"] == {
                "coverage": "insufficient",
                "source_count": 0,
                "sources": [],
            }
            assert "rag_steps" not in data
            assert "rag_trace" not in data
            assert "rag_trace" not in data["message"]
            assert "retrieval_diagnostics" not in data
            assert "retrieval_diagnostics" not in data["message"]

            stream_response = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "流式没有检索证据的问题", "session_id": "evidence_projection_s2"},
            )
            assert stream_response.status_code == 200
            assert "event: evidence_summary" in stream_response.text
            assert "event: rag_step" not in stream_response.text
            assert "event: trace" not in stream_response.text
            assert "event: retrieval_diagnostics" not in stream_response.text
            assert "event: done" in stream_response.text
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())


def test_knowledge_user_session_history_projects_source_excerpts_without_trace_data() -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    async def _seed_history() -> None:
        async with session_factory() as session:
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session("evidence_history_s1", "history-user")
            await repository.add_message(
                session_id=chat_session.id,
                user_id="history-user",
                message_type="assistant",
                content="历史证据回答",
                rag_trace={
                    "gate": {"passed": True, "reason": "sufficient_evidence"},
                    "evidence": [
                        {
                            "chunk_id": "chunk-deploy-7",
                            "content_preview": "发布前由值班负责人完成变更审批。",
                            "metadata": {
                                "source_file": "deploy-runbook.md",
                                "provider_error": "must not reach a Knowledge User",
                            },
                        }
                    ],
                    "runtime": {"provider_attempts": [{"error": "must not reach a Knowledge User"}]},
                },
            )
            await session.commit()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_redis = fake_redis

    try:
        with TestClient(app) as client:
            token = asyncio.run(create_authenticated_test_token(session_factory, fake_redis, username="history-user"))
            asyncio.run(_seed_history())

            response = client.get(
                "/api/v1/sessions/evidence_history_s1",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            message = response.json()["data"]["messages"][0]
            assert message["evidence_summary"] == {
                "coverage": "sufficient",
                "source_count": 1,
                "sources": [
                    {
                        "source_id": "chunk-deploy-7",
                        "metadata": {"source_file": "deploy-runbook.md"},
                        "excerpt": "发布前由值班负责人完成变更审批。",
                    }
                ],
            }
            assert "rag_trace" not in message
            assert "provider_attempts" not in response.text
            assert "provider_error" not in response.text
    finally:
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())


def test_administrator_session_history_projects_bounded_diagnostics_without_sensitive_trace_values(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    get_settings.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    async def _seed_history() -> None:
        async with session_factory() as session:
            repository = ChatRepository(session)
            chat_session = await repository.get_or_create_session("admin_history_s1", "diagnostic-admin")
            await repository.add_message(
                session_id=chat_session.id,
                user_id="diagnostic-admin",
                message_type="assistant",
                content="历史管理员回答",
                rag_trace={
                    "steps": [
                        {"step": "retrieve", "detail": {"retrieved_count": 7}},
                        {"step": "rerank", "detail": {"reranked_count": 3}},
                    ],
                    "gate": {"passed": True, "reason": "sufficient_evidence"},
                    "runtime": {
                        "steps": [{"step": "retrieve", "detail": {"provider_error": {"message": "token=admin-secret"}}}],
                        "fallback_hops": 2,
                        "final_provider": "fallback-llm",
                        "provider_trace": {
                            "retrieve": {
                                "provider_error": {
                                    "code": "PROVIDER_EXEC_FAILED",
                                    "type": "TimeoutError",
                                    "message": "api_key=admin-secret in backend/.env",
                                }
                            }
                        },
                        "provider_attempts": [{"error_code": "PROVIDER_NOT_CONFIGURED"}],
                    },
                    "environment": {"DATABASE_URL": "postgres://admin-secret"},
                },
            )
            await session.commit()

    asyncio.run(_init_db())
    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.test_auth_redis = fake_redis

    try:
        with TestClient(app) as client:
            token = asyncio.run(
                create_authenticated_test_token(
                    session_factory, fake_redis, username="diagnostic-admin", role="admin"
                )
            )
            asyncio.run(_seed_history())

            response = client.get(
                "/api/v1/sessions/admin_history_s1",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            message = response.json()["data"]["messages"][0]
            diagnostics = message["retrieval_diagnostics"]
            assert diagnostics["candidate_counts"] == {"retrieved": 7, "reranked": 3}
            assert diagnostics["evidence_gate"] == {"outcome": "passed", "reason": "sufficient_evidence"}
            assert diagnostics["fallback"] == {"state": "used", "hops": 2, "final_provider": "fallback-llm"}
            assert diagnostics["provider_errors"] == [
                {"stage": "retrieve", "code": "PROVIDER_EXEC_FAILED", "type": "TimeoutError"},
                {"stage": "generate", "code": "PROVIDER_NOT_CONFIGURED", "type": None},
            ]
            assert diagnostics["timeline"] == [{"step": "retrieve"}]
            assert len(diagnostics["trace_preview"]) <= 1600
            assert "rag_trace" not in message
            assert "admin-secret" not in response.text
            assert "DATABASE_URL" not in response.text
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())
