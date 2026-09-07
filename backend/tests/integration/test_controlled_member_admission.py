import asyncio
import fnmatch
from collections.abc import Generator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.common.security import build_auth_session_key, create_access_token, decode_access_token, hash_password
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.user import User
from app.repository.team_invitation_repository import TeamInvitationRepository
from app.service import member_admission_service


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.values: dict[str, str] = {}
        self.delete_error: Exception | None = None

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self.hashes or key in self.values)

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str) -> bool:
        self.values[key] = str(value)
        return True

    async def scan_iter(self, match: str):
        for key in list(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key

    async def delete(self, *keys: str) -> int:
        if self.delete_error is not None:
            raise self.delete_error
        removed = 0
        for key in keys:
            if key in self.hashes:
                del self.hashes[key]
                removed += 1
            if key in self.values:
                del self.values[key]
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


def test_bootstrap_administrator_issues_a_one_time_team_invitation_that_admits_one_knowledge_user(client: TestClient) -> None:
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
    assert first_registration.json()["data"]["role"] == "user"
    assert second_registration.status_code == 403
    assert second_registration.json()["code"] == "INVITATION_REPLAYED"


def test_consumed_invitation_rejects_a_stale_second_claim(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}
    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    assert client.post(
        "/api/v1/auth/register",
        json={
            "username": "first-claimant",
            "password": "safe-password",
            "invitation_code": invitation["invitation_code"],
        },
    ).status_code == 200

    async def _attempt_stale_claim() -> bool:
        async with client.app.state.settings_session_factory() as session:
            candidate = User(username="stale-claimant", password_hash=hash_password("safe-password"), role="user")
            session.add(candidate)
            await session.flush()
            invitations = TeamInvitationRepository(session)
            persisted = await invitations.get_by_code_hash(
                member_admission_service._invitation_code_hash(invitation["invitation_code"])
            )
            assert persisted is not None
            claimed = await invitations.consume_once(
                invitation_id=persisted.id,
                user_id=candidate.id,
                consumed_at=member_admission_service._now(),
            )
            await session.commit()
            return claimed

    assert asyncio.run(_attempt_stale_claim()) is False


def test_invitation_replay_precedes_username_conflict_and_later_revocation(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}
    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    registration_payload = {
        "username": "replayed-member",
        "password": "safe-password",
        "invitation_code": invitation["invitation_code"],
    }
    assert client.post("/api/v1/auth/register", json=registration_payload).status_code == 200

    same_username = client.post("/api/v1/auth/register", json=registration_payload)
    assert same_username.status_code == 403
    assert same_username.json()["code"] == "INVITATION_REPLAYED"
    assert client.post(
        f"/api/v1/members/invitations/{invitation['id']}/revoke",
        headers=admin_headers,
    ).status_code == 200
    replay_after_revocation = client.post(
        "/api/v1/auth/register",
        json={
            "username": "second-replayed-member",
            "password": "safe-password",
            "invitation_code": invitation["invitation_code"],
        },
    )
    assert replay_after_revocation.status_code == 403
    assert replay_after_revocation.json()["code"] == "INVITATION_REPLAYED"


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


def test_bootstrap_replay_preserves_the_existing_identity_and_password_and_records_a_replay(client: TestClient) -> None:
    async def _repeat_bootstrap() -> None:
        async with client.app.state.settings_session_factory() as session:
            await member_admission_service.MemberAdmissionService(session, redis=None).create_bootstrap_administrator(
                username="bootstrap-admin",
                password="replacement-password",
            )

    asyncio.run(_repeat_bootstrap())

    original_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    replacement_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "replacement-password"},
    )
    assert original_login.status_code == 200
    assert replacement_login.status_code == 401
    admin_headers = {"Authorization": f"Bearer {original_login.json()['data']['access_token']}"}
    members = client.get("/api/v1/members", headers=admin_headers)
    assert [member["username"] for member in members.json()["data"]] == ["bootstrap-admin"]

    audit = client.get("/api/v1/members/identity-audit", headers=admin_headers)
    assert ("bootstrap_administrator", "replayed", "bootstrap_administrator_exists") in {
        (event["action"], event["outcome"], event["reason"])
        for event in audit.json()["data"]
    }


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


def test_deactivation_revokes_all_member_sessions_without_deleting_private_history(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}
    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "username": "multi-session-member",
            "password": "safe-password",
            "invitation_code": invitation.json()["data"]["invitation_code"],
        },
    )
    first_headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}
    second_login = client.post(
        "/api/v1/auth/login",
        json={"username": "multi-session-member", "password": "safe-password"},
    )
    second_headers = {"Authorization": f"Bearer {second_login.json()['data']['access_token']}"}

    assert client.post(
        "/api/v1/chat",
        headers=first_headers,
        json={"message": "history must remain private and retained", "session_id": "retained-history"},
    ).status_code == 200
    assert client.post(
        "/api/v1/members/multi-session-member/deactivate",
        headers=admin_headers,
    ).status_code == 200

    assert client.get("/api/v1/auth/me", headers=first_headers).status_code == 401
    assert client.get("/api/v1/auth/me", headers=second_headers).status_code == 401
    future_login = client.post(
        "/api/v1/auth/login",
        json={"username": "multi-session-member", "password": "safe-password"},
    )
    assert future_login.status_code == 403
    assert future_login.json()["code"] == "AUTH_INACTIVE"

    async def _load_retained_state() -> tuple[User, list[ChatSession], list[ChatMessage]]:
        async with client.app.state.settings_session_factory() as session:
            member = await session.scalar(select(User).where(User.username == "multi-session-member"))
            sessions = list(
                await session.scalars(select(ChatSession).where(ChatSession.user_id == "multi-session-member"))
            )
            messages = list(
                await session.scalars(select(ChatMessage).where(ChatMessage.user_id == "multi-session-member"))
            )
            assert member is not None
            return member, sessions, messages

    member, sessions, messages = asyncio.run(_load_retained_state())
    assert member.is_active is False
    assert [session.id for session in sessions] == ["retained-history"]
    assert messages


def test_deactivation_does_not_treat_redis_glob_characters_in_a_username_as_a_session_wildcard(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}

    def _register(username: str) -> dict[str, str]:
        invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "password": "safe-password",
                "invitation_code": invitation.json()["data"]["invitation_code"],
            },
        )
        assert response.status_code == 200
        return {"Authorization": f"Bearer {response.json()['data']['access_token']}"}

    wildcard_member_headers = _register("member*with-glob")
    unaffected_member_headers = _register("unaffected-member")

    assert client.post(
        "/api/v1/members/member*with-glob/deactivate",
        headers=admin_headers,
    ).status_code == 200
    assert client.get("/api/v1/auth/me", headers=wildcard_member_headers).status_code == 401
    assert client.get("/api/v1/auth/me", headers=unaffected_member_headers).status_code == 200
    assert client.get("/api/v1/auth/me", headers=admin_headers).status_code == 200


def test_deactivation_retry_appends_completion_after_a_session_store_failure(client: TestClient) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}
    invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    assert client.post(
        "/api/v1/auth/register",
        json={
            "username": "retry-deactivation-member",
            "password": "safe-password",
            "invitation_code": invitation["invitation_code"],
        },
    ).status_code == 200
    redis = client.app.dependency_overrides[get_redis_client]()
    redis.delete_error = ConnectionError("redis unavailable")

    with pytest.raises(ConnectionError, match="redis unavailable"):
        client.post("/api/v1/members/retry-deactivation-member/deactivate", headers=admin_headers)

    redis.delete_error = None
    assert client.post(
        "/api/v1/members/retry-deactivation-member/deactivate",
        headers=admin_headers,
    ).status_code == 200
    audit = client.get("/api/v1/members/identity-audit", headers=admin_headers)
    outcomes = {
        (event["outcome"], event["reason"])
        for event in audit.json()["data"]
        if event["action"] == "deactivation"
    }
    assert {
        ("pending", "session_revocation_pending"),
        ("failed", "session_store_unavailable"),
        ("deactivated", "administrator_authorized"),
    }.issubset(outcomes)


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
    invalid = client.post(
        "/api/v1/auth/register",
        json={"username": "invalid", "password": "safe-password", "invitation_code": "not-a-team-invitation"},
    )
    assert invalid.status_code == 403
    assert invalid.json()["code"] == "INVITATION_INVALID"
    assert client.post(
        "/api/v1/auth/register",
        json={"username": "attempted-admin", "password": "safe-password", "role": "admin", "invitation_code": invitation_code},
    ).status_code == 422

    revocation = client.post(f"/api/v1/members/invitations/{invitation_data['id']}/revoke", headers=admin_headers)
    assert revocation.status_code == 200
    revoked = client.post(
        "/api/v1/auth/register",
        json={"username": "revoked", "password": "safe-password", "invitation_code": invitation_code},
    )
    assert revoked.status_code == 403
    assert revoked.json()["code"] == "INVITATION_REVOKED"

    expiring_invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
    expiring_code = expiring_invitation.json()["data"]["invitation_code"]
    expiry_time = member_admission_service._now() + member_admission_service.DEFAULT_TEAM_INVITATION_LIFETIME + timedelta(days=1)
    monkeypatch.setattr(member_admission_service, "_now", lambda: expiry_time)
    expired = client.post(
        "/api/v1/auth/register",
        json={"username": "expired", "password": "safe-password", "invitation_code": expiring_code},
    )
    assert expired.status_code == 403
    assert expired.json()["code"] == "INVITATION_EXPIRED"


def test_post_claim_state_resolution_preserves_expired_and_revoked_outcomes(client: TestClient) -> None:
    async def _denial_code(*, revoked: bool) -> str:
        async with client.app.state.settings_session_factory() as session:
            admission = member_admission_service.MemberAdmissionService(session, redis=None)
            invitation, code = await admission.issue_invitation(
                administrator=await session.scalar(select(User).where(User.username == "bootstrap-admin")),
                expires_at=None,
            )
            if revoked:
                invitation.revoked_at = member_admission_service._now()
            else:
                invitation.expires_at = member_admission_service._now() - timedelta(seconds=1)
            await session.commit()
            error = await admission.registration_denial_error(code)
            return error.code

    assert asyncio.run(_denial_code(revoked=False)) == "INVITATION_EXPIRED"
    assert asyncio.run(_denial_code(revoked=True)) == "INVITATION_REVOKED"


def test_identity_audit_is_durable_queryable_and_excludes_credentials_and_private_content(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    administrator_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {administrator_login.json()['data']['access_token']}"}

    admitted_invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    admitted = client.post(
        "/api/v1/auth/register",
        json={
            "username": "audit-admitted-member",
            "password": "safe-password",
            "invitation_code": admitted_invitation["invitation_code"],
        },
    )
    assert admitted.status_code == 200
    replayed = client.post(
        "/api/v1/auth/register",
        json={
            "username": "audit-replayed-member",
            "password": "safe-password",
            "invitation_code": admitted_invitation["invitation_code"],
        },
    )
    assert replayed.status_code == 403

    revoked_invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    assert client.post(
        f"/api/v1/members/invitations/{revoked_invitation['id']}/revoke",
        headers=admin_headers,
    ).status_code == 200
    revoked = client.post(
        "/api/v1/auth/register",
        json={
            "username": "audit-revoked-member",
            "password": "safe-password",
            "invitation_code": revoked_invitation["invitation_code"],
        },
    )
    assert revoked.status_code == 403

    expiring_invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    logout_invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={}).json()["data"]
    logout_registration = client.post(
        "/api/v1/auth/register",
        json={
            "username": "audit-logout-member",
            "password": "safe-password",
            "invitation_code": logout_invitation["invitation_code"],
        },
    )
    assert logout_registration.status_code == 200
    logout_token = logout_registration.json()["data"]["access_token"]

    expiry_time = member_admission_service._now() + member_admission_service.DEFAULT_TEAM_INVITATION_LIFETIME + timedelta(days=1)
    monkeypatch.setattr(member_admission_service, "_now", lambda: expiry_time)
    expired = client.post(
        "/api/v1/auth/register",
        json={
            "username": "audit-expired-member",
            "password": "safe-password",
            "invitation_code": expiring_invitation["invitation_code"],
        },
    )
    assert expired.status_code == 403

    assert client.post("/api/v1/members/audit-admitted-member/promote", headers=admin_headers).status_code == 200
    assert client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {logout_token}"}).status_code == 200
    assert client.post("/api/v1/members/audit-admitted-member/deactivate", headers=admin_headers).status_code == 200

    audit = client.get("/api/v1/members/identity-audit", headers=admin_headers)

    assert audit.status_code == 200
    events = {
        (event["action"], event["outcome"], event["reason"])
        for event in audit.json()["data"]
    }
    assert {
        ("bootstrap_administrator", "created", "server_configuration"),
        ("issue_invitation", "issued", "administrator_authorized"),
        ("registration_invitation", "admitted", "valid"),
        ("registration_invitation", "denied", "replayed"),
        ("registration_invitation", "denied", "revoked"),
        ("registration_invitation", "denied", "expired"),
        ("promotion", "promoted", "administrator_authorized"),
        ("logout", "revoked", "current_bearer"),
        ("deactivation", "deactivated", "administrator_authorized"),
    }.issubset(events)
    event_data = audit.json()["data"]
    for invitation_data, denial_reason in (
        (admitted_invitation, "replayed"),
        (revoked_invitation, "revoked"),
        (expiring_invitation, "expired"),
    ):
        reference = f"sha256:{member_admission_service._invitation_code_hash(invitation_data['invitation_code'])}"
        issued = next(
            event
            for event in event_data
            if event["action"] == "issue_invitation" and event["reference_identity"] == reference
        )
        denied = next(
            event
            for event in event_data
            if event["action"] == "registration_invitation"
            and event["outcome"] == "denied"
            and event["reason"] == denial_reason
            and event["reference_identity"] == reference
        )
        assert denied["target_identity"] == issued["target_identity"]
    audit_body = audit.text
    for forbidden in (
        "bootstrap-password",
        "safe-password",
        logout_token,
        admitted_invitation["invitation_code"],
        revoked_invitation["invitation_code"],
        expiring_invitation["invitation_code"],
        "private conversation content",
    ):
        assert forbidden not in audit_body


def test_protected_route_matrix_uses_database_roles_and_rejects_stale_sessions(client: TestClient) -> None:
    admin_login = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    admin_headers = {"Authorization": f"Bearer {admin_login.json()['data']['access_token']}"}

    def _register(username: str) -> str:
        invitation = client.post("/api/v1/members/invitations", headers=admin_headers, json={})
        assert invitation.status_code == 200
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "password": "safe-password",
                "invitation_code": invitation.json()["data"]["invitation_code"],
            },
        )
        assert response.status_code == 200
        return response.json()["data"]["access_token"]

    member_token = _register("matrix-member")
    stale_token = _register("matrix-stale-member")
    member_headers = {"Authorization": f"Bearer {member_token}"}
    stale_headers = {"Authorization": f"Bearer {stale_token}"}

    assert client.post(
        "/api/v1/chat",
        headers=member_headers,
        json={"message": "matrix private question", "session_id": "matrix-session"},
    ).status_code == 200
    assert client.post("/api/v1/auth/logout", headers=stale_headers).status_code == 200

    member_routes = [
        ("GET", "/api/v1/auth/me", None),
        ("GET", "/api/v1/knowledge-map", None),
        ("GET", "/api/v1/knowledge-map/missing-entry", None),
        ("POST", "/api/v1/chat", {"message": "matrix question", "session_id": "matrix-session"}),
        ("POST", "/api/v1/chat/stream", {"message": "matrix question", "session_id": "matrix-session"}),
        ("GET", "/api/v1/sessions", None),
        ("GET", "/api/v1/sessions/matrix-session", None),
        ("DELETE", "/api/v1/sessions/missing-session", None),
        (
            "POST",
            "/api/v1/knowledge-feedback",
            {"answer_id": "missing-answer", "entry_id": "matrix-entry", "label": "helpful"},
        ),
        ("GET", "/api/v1/knowledge-feedback?answer_id=missing-answer", None),
        ("DELETE", "/api/v1/knowledge-feedback/missing-signal", None),
    ]
    for method, path, payload in member_routes:
        unauthenticated = client.request(method, path, json=payload)
        stale = client.request(method, path, headers=stale_headers, json=payload)
        member = client.request(method, path, headers=member_headers, json=payload)
        assert unauthenticated.status_code == 401, path
        assert stale.status_code == 401, path
        assert member.status_code not in {401, 403}, path

    admin_only_routes = [
        ("GET", "/api/v1/members", None),
        ("POST", "/api/v1/members/invitations", {}),
        ("GET", "/api/v1/members/invitations", None),
        ("GET", "/api/v1/members/identity-audit", None),
        ("POST", "/api/v1/members/invitations/00000000-0000-0000-0000-000000000001/revoke", None),
        ("POST", "/api/v1/members/missing-member/promote", None),
        ("POST", "/api/v1/members/missing-member/deactivate", None),
        ("GET", "/api/v1/operations", None),
        ("GET", "/api/v1/knowledge-review-queue", None),
        (
            "PATCH",
            "/api/v1/knowledge-review-queue/missing-item",
            {"classification": "p3", "status": "reviewed"},
        ),
        ("GET", "/api/v1/documents", None),
        ("POST", "/api/v1/documents/ops/migration-drain", None),
        ("GET", "/api/v1/documents/ops/migration-status", None),
        ("GET", "/api/v1/documents/ops/dense-status", None),
        ("POST", "/api/v1/documents/ops/migration-reconcile", None),
        ("POST", "/api/v1/documents/ops/dense-backfill", {"limit": 1}),
        ("POST", "/api/v1/documents/ops/dense-reconcile", {"limit": 1}),
        ("POST", "/api/v1/documents/ops/migration-resume", None),
        ("POST", "/api/v1/documents/missing-document/publish", None),
        ("POST", "/api/v1/documents/missing-document/build", {"chunk_strategy": "general"}),
        ("POST", "/api/v1/documents/batch-build", {"document_ids": []}),
        ("POST", "/api/v1/documents/batch-delete", {"document_ids": []}),
        ("GET", "/api/v1/documents/missing-document/chunks", None),
        ("DELETE", "/api/v1/documents/missing.md", None),
        ("GET", "/api/v1/documents/jobs", None),
        ("GET", "/api/v1/documents/jobs/missing-job", None),
        ("POST", "/api/v1/documents/jobs/missing-job/cancel", None),
        ("GET", "/api/v1/settings/draft", None),
        ("PUT", "/api/v1/settings/draft", {}),
        ("POST", "/api/v1/settings/apply", {"version": 1}),
    ]
    for method, path, payload in admin_only_routes:
        unauthenticated = client.request(method, path, json=payload)
        stale = client.request(method, path, headers=stale_headers, json=payload)
        member = client.request(method, path, headers=member_headers, json=payload)
        administrator = client.request(method, path, headers=admin_headers, json=payload)
        assert unauthenticated.status_code == 401, path
        assert stale.status_code == 401, path
        assert member.status_code == 403, path
        assert administrator.status_code not in {401, 403}, path

    def _upload(headers: dict[str, str] | None):
        return client.post(
            "/api/v1/documents/upload",
            headers=headers,
            data={"chunk_strategy": "general"},
            files={"file": ("matrix.md", b"matrix document", "text/markdown")},
        )

    assert _upload(None).status_code == 401
    assert _upload(stale_headers).status_code == 401
    assert _upload(member_headers).status_code == 403
    assert _upload(admin_headers).status_code not in {401, 403}

    forged_role_token = create_access_token(subject="matrix-member", role="admin")
    forged_payload = decode_access_token(forged_role_token)
    forged_key = build_auth_session_key(subject="matrix-member", jti=forged_payload["jti"])
    redis = client.app.dependency_overrides[get_redis_client]()
    asyncio.run(redis.hset(forged_key, mapping={"username": "matrix-member", "role": "admin", "issued_at": "0"}))
    assert client.get("/api/v1/members", headers={"Authorization": f"Bearer {forged_role_token}"}).status_code == 403
