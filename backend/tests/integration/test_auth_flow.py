import asyncio
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.security import build_auth_session_key, decode_access_token
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0


@pytest.fixture
def client(tmp_path) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "auth-flow.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    async def _init_db() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    with TestClient(app) as test_client:
        test_client.app.state.test_redis = fake_redis
        yield test_client
    app.dependency_overrides.clear()
    asyncio.run(engine.dispose())


def test_register_login_me_flow(client: TestClient) -> None:
    register_response = client.post(
        "/api/v1/auth/register",
        json={"username": "alice", "password": "secret-123", "role": "user"},
    )
    assert register_response.status_code == 200
    token = register_response.json()["data"]["access_token"]
    payload = decode_access_token(token)
    session_key = build_auth_session_key(subject="alice", jti=payload["jti"])
    assert session_key in client.app.state.test_redis.hashes

    login_response = client.post("/api/v1/auth/login", json={"username": "alice", "password": "secret-123"})
    assert login_response.status_code == 200

    me_response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    assert me_response.json()["data"]["username"] == "alice"


def test_auth_routes_registered(client: TestClient) -> None:
    assert client.post("/api/v1/auth/register", json={"username": "u", "password": "p"}).status_code != 404
    assert client.post("/api/v1/auth/login", json={"username": "u", "password": "p"}).status_code != 404
    assert client.get("/api/v1/auth/me").status_code != 404


def test_me_rejects_when_redis_session_missing(client: TestClient) -> None:
    register_response = client.post(
        "/api/v1/auth/register",
        json={"username": "bob", "password": "secret-123", "role": "user"},
    )
    assert register_response.status_code == 200
    token = register_response.json()["data"]["access_token"]
    payload = decode_access_token(token)
    session_key = build_auth_session_key(subject="bob", jti=payload["jti"])

    client.app.state.test_redis.hashes.pop(session_key, None)

    me_response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 401
    body = me_response.json()
    assert body["code"] == "AUTH_INVALID_TOKEN"
    assert "request_id" in body
