import asyncio
import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.document import Document, DocumentChunk
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, ReviewWorkItem
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._values[key] = mapping
        return len(mapping)

    async def expire(self, key: str, _ttl: int) -> bool:
        return key in self._values

    async def exists(self, key: str) -> int:
        return int(key in self._values)


def _answer_trace(entry_id: str = "pae-tools-001", publication_version: str = "v3") -> dict:
    return {
        "outcome": "decision_summary",
        "gate": {"passed": True},
        "evidence": [
            {
                "source_id": "private-chunk-identity",
                "generation": 3,
                "content_preview": "private evidence excerpt that must not enter the review queue",
                "metadata": {
                    "entry_id": entry_id,
                    "entry_title": "Make tool effects idempotent",
                    "domain": "tools-and-mcp",
                    "section_id": "stable-principle",
                    "source_title": "HTTP Semantics",
                    "source_authority": "IETF",
                    "source_url": "https://www.rfc-editor.org/rfc/rfc9110.html",
                    "source_version": "RFC 9110",
                    "publication_version": publication_version,
                    "review_date": "2026-08-12",
                    "review_status": "approved",
                    "source_availability": "verified",
                },
            }
        ],
    }


@pytest.fixture
def feedback_client(tmp_path) -> Generator[tuple[TestClient, async_sessionmaker[AsyncSession], _InMemoryRedis], None, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'knowledge-feedback.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()

    async def init() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    asyncio.run(init())
    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    try:
        with TestClient(app) as client:
            yield client, session_factory, redis
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())


async def _seed_answers(session_factory: async_sessionmaker[AsyncSession]) -> None:
    async with session_factory() as session:
        session.add(ChatSession(id="private-session", user_id="feedback-user"))
        session.add_all(
            [
                ChatMessage(
                    id=f"answer-{index}",
                    session_id="private-session",
                    user_id="feedback-user",
                    type="assistant",
                    content=f"private answer body {index}",
                    rag_trace=_answer_trace(),
                )
                for index in range(1, 6)
            ]
        )
        session.add(
            ChatMessage(
                id="other-answer",
                session_id="other-private-session",
                user_id="other-user",
                type="assistant",
                content="another member's private answer",
                rag_trace=_answer_trace(),
            )
        )
        await session.commit()


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_user_submits_all_feedback_labels_with_idempotent_duplicate_handling(feedback_client) -> None:
    client, session_factory, redis = feedback_client
    asyncio.run(_seed_answers(session_factory))
    token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="feedback-user"))
    labels = ("helpful", "insufficient_evidence", "outdated", "out_of_scope")

    responses = []
    for index, label in enumerate(labels, start=1):
        response = client.post(
            "/api/v1/knowledge-feedback",
            headers=_headers(token),
            json={
                "answer_id": f"answer-{index}",
                "entry_id": "pae-tools-001",
                "label": label,
                "note": "The retry condition needs a clearer version boundary." if index == 2 else None,
            },
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["answer_id"] == f"answer-{index}"
        assert data["entry_id"] == "pae-tools-001"
        assert data["knowledge_edition"] == "publication:v3"
        assert data["label"] == label
        assert data["duplicate"] is False
        assert data["retention_days"] == 180
        responses.append(data)

    duplicate = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        json={
            "answer_id": "answer-1",
            "entry_id": "pae-tools-001",
            "label": "helpful",
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["data"]["id"] == responses[0]["id"]
    assert duplicate.json()["data"]["duplicate"] is True

    conflict = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        json={"answer_id": "answer-1", "entry_id": "pae-tools-001", "label": "outdated"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "KNOWLEDGE_FEEDBACK_ALREADY_SUBMITTED"

    secret_note = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        json={
            "answer_id": "answer-5",
            "entry_id": "pae-tools-001",
            "label": "outdated",
            "note": "api_key = should-not-be-stored",
        },
    )
    assert secret_note.status_code == 422

    async def records() -> list[KnowledgeFeedbackSignal]:
        async with session_factory() as session:
            return list((await session.scalars(select(KnowledgeFeedbackSignal))).all())

    signals = asyncio.run(records())
    assert len(signals) == 4
    serialized = json.dumps([signal.normalized_metadata for signal in signals])
    assert "private answer body" not in serialized
    assert "private evidence excerpt" not in serialized
    assert {signal.label for signal in signals} == set(labels)


def test_admin_review_queue_exposes_only_normalized_feedback_and_classifies_without_publication(feedback_client) -> None:
    client, session_factory, redis = feedback_client
    asyncio.run(_seed_answers(session_factory))
    user_token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="feedback-user"))
    other_token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="other-user"))
    admin_token = asyncio.run(
        create_authenticated_test_token(session_factory, redis, username="feedback-admin", role="admin")
    )

    forbidden_answer = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(other_token),
        json={"answer_id": "answer-1", "entry_id": "pae-tools-001", "label": "helpful"},
    )
    assert forbidden_answer.status_code == 404

    submitted = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(user_token),
        json={
            "answer_id": "answer-2",
            "entry_id": "pae-tools-001",
            "label": "insufficient_evidence",
            "note": "Please clarify the supported retry status codes.",
        },
    ).json()["data"]

    assert client.get("/api/v1/knowledge-review-queue", headers=_headers(user_token)).status_code == 403
    queue_response = client.get("/api/v1/knowledge-review-queue", headers=_headers(admin_token))
    assert queue_response.status_code == 200
    item = queue_response.json()["data"]["items"][0]
    assert item == {
        "id": item["id"],
        "kind": "feedback_signal",
        "subject_id": "pae-tools-001",
        "status": "pending",
        "classification": None,
        "created_at": item["created_at"],
        "metadata": {
            "answer_id": "answer-2",
            "entry_id": "pae-tools-001",
            "knowledge_edition": "publication:v3",
            "label": "insufficient_evidence",
            "note": "Please clarify the supported retry status codes.",
            "evidence_coverage": "sufficient",
            "source_count": 1,
        },
    }
    assert "private-session" not in queue_response.text
    assert "private answer body" not in queue_response.text
    assert "private evidence excerpt" not in queue_response.text
    assert "feedback-user" not in queue_response.text

    classified = client.patch(
        f"/api/v1/knowledge-review-queue/{item['id']}",
        headers=_headers(admin_token),
        json={"classification": "p1", "status": "reviewed"},
    )
    assert classified.status_code == 200
    assert classified.json()["data"]["classification"] == "p1"
    assert classified.json()["data"]["status"] == "reviewed"

    deleted = client.delete(f"/api/v1/knowledge-feedback/{submitted['id']}", headers=_headers(user_token))
    assert deleted.status_code == 200
    assert deleted.json()["data"] == {"id": submitted["id"], "deleted": True}
    assert client.get("/api/v1/knowledge-review-queue", headers=_headers(admin_token)).json()["data"]["items"] == []


def test_review_queue_syncs_source_release_and_review_age_triggers_and_purges_expired_feedback(feedback_client) -> None:
    client, session_factory, redis = feedback_client
    admin_token = asyncio.run(
        create_authenticated_test_token(session_factory, redis, username="trigger-admin", role="admin")
    )
    expired_at = datetime.now(UTC) - timedelta(days=181)

    async def seed() -> None:
        async with session_factory() as session:
            session.add(
                Document(
                    id="trigger-document",
                    filename="private-trigger-entry.md",
                    file_type="md",
                    file_size=10,
                    status="candidate",
                    chunk_strategy="agent",
                    published_generation=1,
                    candidate_generation=2,
                    candidate_chunk_strategy="agent",
                    candidate_chunk_count=1,
                )
            )
            session.add(
                DocumentChunk(
                    id="trigger-chunk",
                    document_id="trigger-document",
                    generation=1,
                    chunk_index=0,
                    content="private published body",
                    chunk_metadata={
                        "entry_id": "pae-operations-001",
                        "entry_title": "Bound Agent admission",
                        "domain": "operating-constraints",
                        "review_status": "approved",
                        "review_date": "2020-01-01",
                        "source_availability": "unavailable",
                    },
                )
            )
            session.add(
                KnowledgeFeedbackSignal(
                    id="expired-feedback",
                    answer_id="expired-answer",
                    user_id="expired-user",
                    entry_id="pae-old-001",
                    knowledge_edition="publication:v1",
                    label="helpful",
                    normalized_metadata={},
                    created_at=expired_at,
                    expires_at=expired_at + timedelta(days=180),
                )
            )
            session.add(
                ReviewWorkItem(
                    id="expired-review-item",
                    kind="feedback_signal",
                    dedupe_key="feedback:expired-feedback",
                    subject_id="pae-old-001",
                    signal_id="expired-feedback",
                    normalized_metadata={},
                    created_at=expired_at,
                    updated_at=expired_at,
                )
            )
            await session.commit()

    asyncio.run(seed())
    response = client.get("/api/v1/knowledge-review-queue", headers=_headers(admin_token))
    assert response.status_code == 200
    items = response.json()["data"]["items"]
    assert {item["kind"] for item in items} == {"source_link_failure", "release_change", "review_age"}
    assert {item["subject_id"] for item in items} == {"pae-operations-001"}
    serialized = json.dumps(items)
    assert "trigger-document" not in serialized
    assert "private-trigger-entry" not in serialized
    assert "private published body" not in serialized

    async def expired_counts() -> tuple[int, int]:
        async with session_factory() as session:
            signals = list((await session.scalars(select(KnowledgeFeedbackSignal))).all())
            work_items = list((await session.scalars(select(ReviewWorkItem))).all())
            return len(signals), len([item for item in work_items if item.id == "expired-review-item"])

    assert asyncio.run(expired_counts()) == (0, 0)
