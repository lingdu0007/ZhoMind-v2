import asyncio
from collections.abc import Generator
import json

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base


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


def test_chat_response_contract_stable() -> None:
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

    try:
        with TestClient(app) as client:
            reg = client.post(
                "/api/v1/auth/register",
                json={"username": "contract-u", "password": "secret-123", "role": "user"},
            )
            token = reg.json()["data"]["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            resp = client.post("/api/v1/chat", headers=headers, json={"message": "合同稳定性随机无证据问题", "session_id": "contract_s1"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == "OK"
            assert "request_id" in body
            data = body["data"]
            assert set(["session_id", "answer", "message", "rag_steps", "rag_trace"]).issubset(data.keys())
            assert data["message"]["rag_trace"] == data["rag_trace"]

            runtime = data["rag_trace"]["runtime"]
            assert runtime["request_id"].startswith("chat-")
            assert runtime["session_id"] == "contract_s1"
            assert runtime["graph_alias"] == "default_v1"
            assert set(["gate", "steps", "step_names"]).issubset(runtime.keys())
            assert runtime["gate"]["passed"] is False
            assert runtime["gate"]["reason"] == "reject_insufficient_evidence"
            assert runtime["step_names"] == [
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
            assert runtime["steps"][0]["step"] == "normalize"
            assert "tool_budget" in runtime
            assert "max_calls" in runtime["tool_budget"]
            assert "tool_errors" in runtime

            stream_resp = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "合同稳定性流式随机无证据问题", "session_id": "contract_s1"},
            )
            assert stream_resp.status_code == 200
            assert stream_resp.headers["content-type"].startswith("text/event-stream")

            trace_data = _extract_sse_event_data(stream_resp.text, "trace")
            assert trace_data is not None
            trace_event = json.loads(trace_data)
            trace = trace_event["trace"]
            assert trace["runtime"]["request_id"].startswith("chat-")
            assert trace["runtime"]["session_id"] == "contract_s1"
            assert trace["runtime"]["graph_alias"] == "default_v1"
            assert trace["runtime"]["step_names"] == runtime["step_names"]
            assert trace["runtime"]["gate"]["reason"] == "reject_insufficient_evidence"
    finally:
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
