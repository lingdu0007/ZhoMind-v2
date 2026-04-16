import asyncio
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.main import app
from app.model.base import Base


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

            resp = client.post("/api/v1/chat", headers=headers, json={"message": "你好", "session_id": "contract_s1"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["code"] == "OK"
            assert "request_id" in body
            data = body["data"]
            assert set(["session_id", "answer", "message", "rag_steps", "rag_trace"]).issubset(data.keys())
    finally:
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
