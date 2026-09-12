import asyncio
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from fnmatch import fnmatch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.security import hash_password
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.document import Document, DocumentJob
from app.model.operational_event import OperationalEvent
from app.model.user import User
from app.operations.chat_capacity import get_chat_admission_gate
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.values: dict[str, str] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self.hashes)

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str) -> bool:
        self.values[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.values:
                del self.values[key]
                removed += 1
            if key in self.hashes:
                del self.hashes[key]
                removed += 1
        return removed

    async def scan_iter(self, match: str):
        for key in list(self.hashes):
            if fnmatch(key, match):
                yield key


@pytest.fixture
def client(tmp_path) -> Generator[TestClient, None, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'privacy-operations.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()

    async def _init_db() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.settings_session_factory = session_factory
    app.state.operational_event_session_factory = session_factory
    app.state.test_auth_session_factory = session_factory
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = SessionLocal
    app.state.operational_event_session_factory = SessionLocal
    delattr(app.state, "test_auth_session_factory")
    asyncio.run(engine.dispose())


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _register(client: TestClient, *, username: str, role: str = "user") -> str:
    return asyncio.run(
        create_authenticated_test_token(
            client.app.state.test_auth_session_factory,
            client.app.dependency_overrides[get_redis_client](),
            username=username,
            role=role,
        )
    )


def test_administrator_operations_surface_is_admin_only_and_never_projects_private_conversation_content(client: TestClient) -> None:
    admin_token = _register(client, username="operations-admin", role="admin")
    member_token = _register(client, username="knowledge-member")
    private_question = "private question that must not reach operations"
    private_answer = "private answer that must not reach operations"

    async def _seed() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                Document(
                    filename="published.md",
                    file_type="md",
                    file_size=1,
                    status="ready",
                    published_generation=1,
                )
            )
            session.add(ChatSession(id="private-session", user_id="knowledge-member"))
            session.add_all(
                [
                    ChatMessage(
                        session_id="private-session",
                        user_id="knowledge-member",
                        type="user",
                        content=private_question,
                    ),
                    ChatMessage(
                        session_id="private-session",
                        user_id="knowledge-member",
                        type="assistant",
                        content=private_answer,
                        rag_trace={"evidence": [{"content": "private source excerpt"}]},
                    ),
                ]
            )
            await session.commit()

    asyncio.run(_seed())

    forbidden = client.get("/api/v1/operations", headers=_headers(member_token))
    assert forbidden.status_code == 403

    response = client.get("/api/v1/operations", headers=_headers(admin_token))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["documents"]["published_sources"] == 1
    assert set(data) == {"documents", "generation", "failures", "retry_actions", "limits", "admission", "events"}
    assert private_question not in response.text
    assert private_answer not in response.text
    assert "private source excerpt" not in response.text


def test_listing_sessions_removes_private_conversation_records_after_thirty_days(client: TestClient) -> None:
    member_token = _register(client, username="retention-member")
    expired_at = datetime.now(UTC) - timedelta(days=31)

    async def _seed() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                ChatSession(
                    id="expired-session",
                    user_id="retention-member",
                    created_at=expired_at,
                    updated_at=expired_at,
                )
            )
            session.add(
                ChatMessage(
                    session_id="expired-session",
                    user_id="retention-member",
                    type="user",
                    content="expired private question",
                    created_at=expired_at,
                )
            )
            await session.commit()

    asyncio.run(_seed())

    response = client.get("/api/v1/sessions", headers=_headers(member_token))

    assert response.status_code == 200
    assert response.json()["sessions"] == []

    async def _remaining_records() -> tuple[list[ChatSession], list[ChatMessage]]:
        async with client.app.state.test_auth_session_factory() as session:
            return list((await session.scalars(select(ChatSession))).all()), list((await session.scalars(select(ChatMessage))).all())

    sessions, messages = asyncio.run(_remaining_records())
    assert sessions == []
    assert messages == []


def test_chat_records_a_content_free_operational_event(client: TestClient) -> None:
    member_token = _register(client, username="event-member")
    private_question = "private question excluded from operational events"

    response = client.post(
        "/api/v1/chat",
        headers={**_headers(member_token), "x-request-id": "event-request-id"},
        json={"message": private_question, "session_id": "event-session"},
    )

    assert response.status_code == 200

    async def _events() -> list[OperationalEvent]:
        async with client.app.state.test_auth_session_factory() as session:
            return list((await session.scalars(select(OperationalEvent))).all())

    events = asyncio.run(_events())
    event = next(item for item in events if item.request_id == response.headers["x-request-id"])
    assert event.request_id != "event-request-id"
    assert event.route_outcome == "POST /api/v1/chat:success"
    assert event.gate_outcome == "rejected"
    assert event.candidate_count == 0
    assert event.provider_identity is None
    assert event.normalized_error is None
    assert event.generation_route is None
    assert private_question not in str({column.name: getattr(event, column.name) for column in event.__table__.columns})
    assert {column.name for column in event.__table__.columns} == {
        "id",
        "request_id",
        "route_outcome",
        "duration_ms",
        "gate_outcome",
        "provider_identity",
        "normalized_error",
        "candidate_count",
        "generation_route",
        "dimensions",
        "created_at",
    }


def test_chat_stream_records_a_deferred_operational_event(client: TestClient) -> None:
    member_token = _register(client, username="stream-event-member")

    stream_response = client.post(
        "/api/v1/chat/stream",
        headers={**_headers(member_token), "x-request-id": "stream-event-request-id"},
        json={"message": "stream question excluded from operational events", "session_id": "stream-event-session"},
    )

    assert stream_response.status_code == 200
    assert stream_response.headers["content-type"].startswith("text/event-stream")
    assert "event: stage" in stream_response.text
    assert "event: done" in stream_response.text

    async def _events() -> list[OperationalEvent]:
        async with client.app.state.test_auth_session_factory() as session:
            return list((await session.scalars(select(OperationalEvent))).all())

    events = asyncio.run(_events())
    event = next(item for item in events if item.request_id == stream_response.headers["x-request-id"])
    assert event.request_id != "stream-event-request-id"
    assert event.route_outcome == "POST /api/v1/chat/stream:success"
    assert event.gate_outcome == "rejected"
    assert event.candidate_count == 0


def test_any_request_purges_operational_events_after_thirty_days(client: TestClient) -> None:
    expired_at = datetime.now(UTC) - timedelta(days=31)

    async def _seed() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                OperationalEvent(
                    id="expired-event",
                    request_id="expired-request",
                    route_outcome="GET /api/v1/health:success",
                    duration_ms=1,
                    gate_outcome="unavailable",
                    candidate_count=0,
                    created_at=expired_at,
                )
            )
            await session.commit()

    asyncio.run(_seed())

    response = client.get("/api/v1/health")

    assert response.status_code == 200

    async def _event_ids() -> list[str]:
        async with client.app.state.test_auth_session_factory() as session:
            return list((await session.scalars(select(OperationalEvent.id))).all())

    assert "expired-event" not in asyncio.run(_event_ids())


def test_private_conversation_records_are_visible_and_deletable_only_by_their_owner(client: TestClient) -> None:
    owner_token = _register(client, username="conversation-owner")
    other_token = _register(client, username="conversation-other")

    async def _seed() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add(ChatSession(id="owner-session", user_id="conversation-owner"))
            session.add(
                ChatMessage(
                    session_id="owner-session",
                    user_id="conversation-owner",
                    type="user",
                    content="owner-only conversation content",
                )
            )
            await session.commit()

    asyncio.run(_seed())

    other_read = client.get("/api/v1/sessions/owner-session", headers=_headers(other_token))
    other_delete = client.delete("/api/v1/sessions/owner-session", headers=_headers(other_token))
    owner_read = client.get("/api/v1/sessions/owner-session", headers=_headers(owner_token))
    owner_delete = client.delete("/api/v1/sessions/owner-session", headers=_headers(owner_token))

    assert other_read.status_code == 200
    assert other_read.json()["messages"] == []
    assert other_delete.json()["deleted"] is False
    assert owner_read.json()["messages"][0]["content"] == "owner-only conversation content"
    assert owner_delete.json()["deleted"] is True


def test_administrator_operations_surface_projects_normalized_failures_with_supported_retry_actions(client: TestClient) -> None:
    admin_token = _register(client, username="failure-admin", role="admin")
    private_failure_message = "private source content must not be exposed as a failure"

    async def _seed() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add(Document(id="failed-document", filename="failed.md", file_type="md", file_size=1, status="failed"))
            session.add(
                DocumentJob(
                    id="failed-job",
                    document_id="failed-document",
                    status="failed",
                    stage="failed",
                    progress=50,
                    message=private_failure_message,
                )
            )
            session.add(
                OperationalEvent(
                    id="provider-failure-event",
                    request_id="provider-failure-request",
                    route_outcome="POST /api/v1/chat:server_error",
                    duration_ms=12,
                    gate_outcome="passed",
                    normalized_error="PROVIDER_TIMEOUT",
                    candidate_count=1,
                )
            )
            await session.commit()

    asyncio.run(_seed())

    response = client.get("/api/v1/operations", headers=_headers(admin_token))

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["failures"] == [
        {
            "kind": "document_build",
            "code": "DOCUMENT_BUILD_FAILED",
            "document_id": "failed-document",
            "job_id": "failed-job",
        },
        {
            "kind": "generation_provider",
            "code": "PROVIDER_TIMEOUT",
            "request_id": "unknown-request",
        },
    ]
    assert data["retry_actions"] == [
        {
            "action": "retry_document_build",
            "document_id": "failed-document",
            "method": "POST",
            "path": "/api/v1/documents/failed-document/build",
        }
    ]
    assert private_failure_message not in response.text


def test_registration_rejects_the_twenty_sixth_active_member(client: TestClient) -> None:
    admin_token = _register(client, username="capacity-admin", role="admin")

    async def _seed_active_members() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add_all(
                [
                    User(username=f"active-member-{index}", password_hash=hash_password("test-password"), role="user")
                    for index in range(24)
                ]
            )
            await session.commit()

    asyncio.run(_seed_active_members())
    invitation = client.post("/api/v1/members/invitations", headers=_headers(admin_token), json={})
    assert invitation.status_code == 200

    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "over-capacity-member",
            "password": "safe-password",
            "invitation_code": invitation.json()["data"]["invitation_code"],
        },
    )

    assert response.status_code == 409
    assert response.json()["code"] == "ACTIVE_MEMBER_LIMIT_REACHED"


def test_legacy_publication_bypass_rejects_before_source_capacity_evaluation(client: TestClient) -> None:
    admin_token = _register(client, username="publication-capacity-admin", role="admin")

    async def _seed_published_sources() -> None:
        async with client.app.state.test_auth_session_factory() as session:
            session.add_all(
                [
                    Document(
                        id=f"published-{index}",
                        filename=f"published-{index}.md",
                        file_type="md",
                        file_size=1,
                        status="ready",
                        published_generation=1,
                    )
                    for index in range(500)
                ]
            )
            session.add(
                Document(
                    id="candidate-over-capacity",
                    filename="candidate-over-capacity.md",
                    file_type="md",
                    file_size=1,
                    status="candidate",
                    candidate_generation=1,
                    candidate_chunk_strategy="general",
                    candidate_chunk_count=1,
                )
            )
            await session.commit()

    asyncio.run(_seed_published_sources())

    response = client.post("/api/v1/documents/candidate-over-capacity/publish", headers=_headers(admin_token))

    assert response.status_code == 410
    assert response.json()["code"] == "LEGACY_PUBLICATION_BYPASS_REJECTED"


def test_chat_rejects_fifth_request_over_the_pilot_admission_envelope(client: TestClient) -> None:
    member_token = _register(client, username="chat-capacity-member")
    gate = get_chat_admission_gate()
    reservations = [gate.reserve(member_id=f"capacity-{index}") for index in range(4)]
    assert [gate.observe(item)["state"] for item in reservations] == ["running", "running", "queued", "queued"]
    try:
        response = client.post(
            "/api/v1/chat",
            headers=_headers(member_token),
            json={"message": "capacity test", "session_id": "capacity-session"},
        )
    finally:
        for reservation in reservations:
            gate.finish(reservation)

    assert response.status_code == 429
    assert response.json()["code"] == "CHAT_QUEUE_FULL"
