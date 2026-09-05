import asyncio
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.common.security import build_auth_session_key, decode_access_token
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.delete_error: Exception | None = None

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return 1 if key in self.hashes else 0

    async def delete(self, *keys: str) -> int:
        if self.delete_error is not None:
            raise self.delete_error
        removed = 0
        for key in keys:
            if key in self.hashes:
                del self.hashes[key]
                removed += 1
        return removed


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
    original_session_factory = app.state.settings_session_factory
    settings = get_settings()
    original_bootstrap_username = settings.bootstrap_admin_username
    original_bootstrap_password = settings.bootstrap_admin_password
    settings.bootstrap_admin_username = "bootstrap-admin"
    settings.bootstrap_admin_password = "bootstrap-password"
    app.state.settings_session_factory = session_factory
    with TestClient(app) as test_client:
        test_client.app.state.test_redis = fake_redis
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = original_session_factory
    settings.bootstrap_admin_username = original_bootstrap_username
    settings.bootstrap_admin_password = original_bootstrap_password
    asyncio.run(engine.dispose())


def _bootstrap_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['data']['access_token']}"}


def _register_knowledge_user(client: TestClient, username: str) -> dict:
    invitation = client.post("/api/v1/members/invitations", headers=_bootstrap_headers(client), json={})
    assert invitation.status_code == 200
    response = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "secret-123", "invitation_code": invitation.json()["data"]["invitation_code"]},
    )
    assert response.status_code == 200
    return response


def test_register_login_me_flow(client: TestClient) -> None:
    register_response = _register_knowledge_user(client, "alice")
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


def test_me_projects_the_enabled_system_settings_capability_for_administrators(client: TestClient, monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "system_settings_draft_enabled", True)
    monkeypatch.setattr(settings, "system_settings_application_enabled", True)

    user_response = _register_knowledge_user(client, "knowledge-user")
    admin_login = client.post("/api/v1/auth/login", json={"username": "bootstrap-admin", "password": "bootstrap-password"})

    user_me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {user_response.json()['data']['access_token']}"})
    admin_me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {admin_login.json()['data']['access_token']}"})

    assert user_me.json()["data"]["capabilities"] == {"system_settings": False}
    assert admin_me.json()["data"]["capabilities"] == {"system_settings": True}


@pytest.mark.parametrize(
    ("draft_enabled", "application_enabled"),
    [(False, False), (False, True), (True, False)],
)
def test_me_hides_system_settings_when_its_lifecycle_is_incomplete(
    client: TestClient,
    monkeypatch,
    draft_enabled: bool,
    application_enabled: bool,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "system_settings_draft_enabled", draft_enabled)
    monkeypatch.setattr(settings, "system_settings_application_enabled", application_enabled)

    registration = client.post("/api/v1/auth/login", json={"username": "bootstrap-admin", "password": "bootstrap-password"})
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {registration.json()['data']['access_token']}"},
    )

    assert response.json()["data"]["capabilities"] == {"system_settings": False}


def test_auth_routes_registered(client: TestClient) -> None:
    assert client.post("/api/v1/auth/register", json={"username": "u", "password": "p", "invitation_code": "invalid"}).status_code != 404
    assert client.post("/api/v1/auth/login", json={"username": "u", "password": "p"}).status_code != 404
    assert client.get("/api/v1/auth/me").status_code != 404


def test_me_rejects_when_redis_session_missing(client: TestClient) -> None:
    register_response = _register_knowledge_user(client, "bob")
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


def test_logout_revokes_only_the_current_bearer_session(client: TestClient) -> None:
    registration = _register_knowledge_user(client, "logout-member")
    first_token = registration.json()["data"]["access_token"]
    second_login = client.post(
        "/api/v1/auth/login",
        json={"username": "logout-member", "password": "secret-123"},
    )
    assert second_login.status_code == 200
    second_token = second_login.json()["data"]["access_token"]

    logout = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {first_token}"})

    assert logout.status_code == 200
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {first_token}"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {second_token}"}).status_code == 200


def test_logout_persists_a_pending_audit_event_before_a_session_store_failure(client: TestClient) -> None:
    registration = _register_knowledge_user(client, "logout-audit-member")
    token = registration.json()["data"]["access_token"]
    client.app.state.test_redis.delete_error = ConnectionError("redis unavailable")

    with pytest.raises(ConnectionError, match="redis unavailable"):
        client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})

    client.app.state.test_redis.delete_error = None
    audit = client.get("/api/v1/members/identity-audit", headers=_bootstrap_headers(client))

    assert audit.status_code == 200
    assert ("logout", "pending", "session_revocation_pending") in {
        (event["action"], event["outcome"], event["reason"])
        for event in audit.json()["data"]
    }
