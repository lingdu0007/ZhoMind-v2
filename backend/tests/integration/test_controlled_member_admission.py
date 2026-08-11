import asyncio
from collections.abc import Generator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.common.security import hash_password
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.service import member_admission_service


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self.hashes)

    async def scan_iter(self, match: str):
        prefix = match.removesuffix("*")
        for key in list(self.hashes):
            if key.startswith(prefix):
                yield key

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.hashes:
                del self.hashes[key]
                removed += 1
        return removed


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "controlled-member-admission.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()
    settings = get_settings()
    monkeypatch.setattr(settings, "bootstrap_admin_username", "bootstrap-admin", raising=False)
    monkeypatch.setattr(settings, "bootstrap_admin_password", "bootstrap-password", raising=False)

    async def _init_db() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def _seed_legacy_administrator() -> None:
        async with session_factory() as session:
            from app.model.user import User

            session.add(User(username="legacy-admin", password_hash=hash_password("legacy-password"), role="admin"))
            await session.commit()

    asyncio.run(_init_db())
    if getattr(request, "param", False):
        asyncio.run(_seed_legacy_administrator())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.settings_session_factory = session_factory
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = SessionLocal
    asyncio.run(engine.dispose())


def test_bootstrap_administrator_issues_a_reusable_team_invitation_that_admits_knowledge_users(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    assert administrator_login.status_code == 200
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}

    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
    assert invitation.status_code == 200
    invitation_code = invitation.json()["data"]["invitation_code"]

    first_registration = client.post(
        "/api/v1/auth/register",
        json={"username": "first-member", "password": "safe-password", "invitation_code": invitation_code},
    )
    second_registration = client.post(
        "/api/v1/auth/register",
        json={"username": "second-member", "password": "safe-password", "invitation_code": invitation_code},
    )

    assert first_registration.status_code == 200
    assert second_registration.status_code == 200
    assert first_registration.json()["data"]["role"] == "user"
    assert second_registration.json()["data"]["role"] == "user"


@pytest.mark.parametrize("client", [True], indirect=True)
def test_deployment_adds_a_bootstrap_administrator_when_a_legacy_administrator_already_exists(client: TestClient) -> None:
    bootstrap_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    assert bootstrap_login.status_code == 200
    headers = {"Authorization": f"Bearer {bootstrap_login.json()['data']['access_token']}"}
    members = client.get("/api/v1/members", headers=headers)
    assert members.status_code == 200
    assert {member["username"] for member in members.json()["data"]} == {"bootstrap-admin", "legacy-admin"}


def test_only_administrators_manage_members_and_deactivation_revokes_access(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}
    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
    invitation_code = invitation.json()["data"]["invitation_code"]
    registration = client.post(
        "/api/v1/auth/register",
        json={"username": "admitted-member", "password": "safe-password", "invitation_code": invitation_code},
    )
    member_headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}

    assert client.get("/api/v1/members", headers=member_headers).status_code == 403
    promotion = client.post("/api/v1/members/admitted-member/promote", headers=admin_headers)
    assert promotion.status_code == 200
    assert promotion.json()["data"]["role"] == "admin"

    deactivation = client.post("/api/v1/members/admitted-member/deactivate", headers=admin_headers)
    assert deactivation.status_code == 200
    assert deactivation.json()["data"]["is_active"] is False
    assert client.get("/api/v1/auth/me", headers=member_headers).status_code == 401

    future_login = client.post(
        "/api/v1/auth/login",
        json={"username": "admitted-member", "password": "safe-password"},
    )
    assert future_login.status_code == 403
    assert future_login.json()["code"] == "AUTH_INACTIVE"


def test_public_registration_rejects_invalid_expired_and_revoked_invitations_without_an_admin_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}
    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
    invitation_data = invitation.json()["data"]
    invitation_code = invitation_data["invitation_code"]

    listed = client.get("/api/v1/members/invitations", headers=admin_headers)
    assert listed.status_code == 200
    assert "invitation_code" not in listed.json()["data"][0]
    assert client.post(
        "/api/v1/auth/register",
        json={"username": "invalid", "password": "safe-password", "invitation_code": "not-a-team-invitation"},
    ).status_code == 403
    assert client.post(
        "/api/v1/auth/register",
        json={"username": "attempted-admin", "password": "safe-password", "role": "admin", "invitation_code": invitation_code},
    ).status_code == 422

    revocation = client.post(f"/api/v1/members/invitations/{invitation_data['id']}/revoke", headers=admin_headers)
    assert revocation.status_code == 200
    assert client.post(
        "/api/v1/auth/register",
        json={"username": "revoked", "password": "safe-password", "invitation_code": invitation_code},
    ).status_code == 403

    expiring_invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
    expiring_code = expiring_invitation.json()["data"]["invitation_code"]
    expiry_time = member_admission_service._now() + member_admission_service.DEFAULT_TEAM_INVITATION_LIFETIME + timedelta(days=1)
    monkeypatch.setattr(member_admission_service, "_now", lambda: expiry_time)
    expired = client.post(
        "/api/v1/auth/register",
        json={"username": "expired", "password": "safe-password", "invitation_code": expiring_code},
    )
    assert expired.status_code == 403
    assert expired.json()["code"] == "INVITATION_INVALID"
