import asyncio
from collections.abc import Generator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.main import app
from app.model.base import Base


def _auth_headers(client: TestClient, username: str = "bob") -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "secret-123", "role": "user"},
    )
    assert response.status_code == 200
    token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_sessions_route_exists_and_returns_envelope() -> None:
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
            headers = _auth_headers(client)
            response = client.get("/api/v1/sessions", headers=headers)
            assert response.status_code == 200
            body = response.json()
            assert body["code"] == "OK"
            assert "request_id" in body
            assert "sessions" in body
            assert body["sessions"] == []
            assert body["data"]["sessions"] == []
    finally:
        app.dependency_overrides.clear()
        asyncio.run(db_engine.dispose())
