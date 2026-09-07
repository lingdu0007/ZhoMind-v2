import asyncio
import json
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.knowledge_feedback.service import KnowledgeFeedbackService
from app.main import app
from app.model.answer_execution import AnswerExecutionModel
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.document import Document, DocumentChunk
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, ReviewWorkItem
from app.service.chat_service import ChatService
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


@pytest.fixture(autouse=True)
def closed_supported_answer_execution(monkeypatch) -> None:
    async def load_for_message(_self, **_kwargs):
        return SimpleNamespace(
            projection={
                "state": "completed",
                "outcome": "evidence_gated_answer",
            },
            result={
                "state": "completed",
                "outcome": "evidence_gated_answer",
            },
        )

    def summary_from_execution(_result):
        return {
            "coverage": "sufficient",
            "source_count": 1,
            "sources": [
                {
                    "entry_id": "pae-tools-001",
                    "publication_version": "v3",
                }
            ],
        }

    monkeypatch.setattr(
        "app.knowledge_feedback.service.AnswerExecutionStore.load_for_message",
        load_for_message,
    )
    monkeypatch.setattr(
        "app.knowledge_feedback.service.evidence_summary_from_execution",
        summary_from_execution,
    )


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
        assert data["outcome"] == "evidence_gated_answer"
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

    listed = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
    )
    assert listed.status_code == 200
    items = listed.json()["data"]["items"]
    assert len(items) == 4
    serialized = json.dumps(items)
    assert "private answer body" not in serialized
    assert "private evidence excerpt" not in serialized
    assert {item["label"] for item in items} == set(labels)
    assert {item["outcome"] for item in items} == {"evidence_gated_answer"}


@pytest.mark.parametrize(
    ("winner_note", "expected_status"),
    (
        ("The competing report keeps the same note.", 200),
        ("The competing report has a different note.", 409),
    ),
)
def test_gap_feedback_unique_constraint_race_returns_duplicate_or_conflict_through_authenticated_api(
    feedback_client,
    monkeypatch,
    winner_note: str,
    expected_status: int,
) -> None:
    client, session_factory, redis = feedback_client
    asyncio.run(_seed_answers(session_factory))
    token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="feedback-user"))
    submitted_note = "The competing report keeps the same note."
    gap_context = {
        "outcome": "insufficient_evidence_reply",
        "reason": "decision_not_covered",
        "query_condition_set_identity": "qcs-gap-race",
    }

    async def load_gap_execution(_self, **_kwargs):
        return SimpleNamespace(
            projection={
                "state": "completed",
                "outcome": "insufficient_evidence_reply",
                "query_condition_set": {"identity": gap_context["query_condition_set_identity"]},
                "insufficient_evidence_reply": {
                    "outcome": "insufficient_evidence_reply",
                    "reason": gap_context["reason"],
                    "query_condition_set_identity": gap_context["query_condition_set_identity"],
                },
            },
            result={"state": "completed", "outcome": "insufficient_evidence_reply"},
        )

    monkeypatch.setattr(
        "app.knowledge_feedback.service.AnswerExecutionStore.load_for_message",
        load_gap_execution,
    )
    now = datetime.now(UTC)
    database_path = session_factory.kw["bind"].url.database
    assert isinstance(database_path, str)
    winner_metadata = {
        "answer_id": "answer-1",
        "outcome": "insufficient_evidence_reply",
        "gap_context": gap_context,
        "label": "insufficient_evidence",
        "note": winner_note,
    }
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_feedback_signals (
                id, answer_id, user_id, entry_id, scope_key, knowledge_edition,
                label, note, normalized_metadata, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "feedback-gap-race-winner",
                "answer-1",
                "feedback-user",
                None,
                "gap:decision_not_covered",
                None,
                "insufficient_evidence",
                winner_note,
                json.dumps(winner_metadata),
                now.isoformat(),
                (now + timedelta(days=180)).isoformat(),
            ),
        )

    original_existing_submission = KnowledgeFeedbackService._existing_submission
    stale_precheck = True

    async def existing_submission_after_competing_commit(self, *args, **kwargs):
        nonlocal stale_precheck
        if stale_precheck:
            stale_precheck = False
            return None
        return await original_existing_submission(self, *args, **kwargs)

    monkeypatch.setattr(
        KnowledgeFeedbackService,
        "_existing_submission",
        existing_submission_after_competing_commit,
    )
    with TestClient(app, raise_server_exceptions=False) as race_client:
        response = race_client.post(
            "/api/v1/knowledge-feedback",
            headers=_headers(token),
            json={
                "answer_id": "answer-1",
                "label": "insufficient_evidence",
                "note": submitted_note,
            },
        )
    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json()["data"]["id"] == "feedback-gap-race-winner"
        assert response.json()["data"]["duplicate"] is True
    else:
        assert response.json()["code"] == "KNOWLEDGE_FEEDBACK_ALREADY_SUBMITTED"

    listed = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        params={"answer_id": "answer-1"},
    )
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["data"]["items"]] == ["feedback-gap-race-winner"]


def test_gap_feedback_does_not_reclassify_unrelated_integrity_errors_as_duplicates(
    feedback_client,
    monkeypatch,
) -> None:
    client, session_factory, redis = feedback_client
    asyncio.run(_seed_answers(session_factory))
    token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="feedback-user"))
    gap_context = {
        "outcome": "insufficient_evidence_reply",
        "reason": "decision_not_covered",
        "query_condition_set_identity": "qcs-gap-unrelated-integrity-error",
    }

    async def load_gap_execution(_self, **_kwargs):
        return SimpleNamespace(
            projection={
                "state": "completed",
                "outcome": "insufficient_evidence_reply",
                "query_condition_set": {"identity": gap_context["query_condition_set_identity"]},
                "insufficient_evidence_reply": {
                    "outcome": "insufficient_evidence_reply",
                    "reason": gap_context["reason"],
                    "query_condition_set_identity": gap_context["query_condition_set_identity"],
                },
            },
            result={"state": "completed", "outcome": "insufficient_evidence_reply"},
        )

    monkeypatch.setattr(
        "app.knowledge_feedback.service.AnswerExecutionStore.load_for_message",
        load_gap_execution,
    )
    now = datetime.now(UTC)
    database_path = session_factory.kw["bind"].url.database
    assert isinstance(database_path, str)
    winner_metadata = {
        "answer_id": "answer-1",
        "outcome": "insufficient_evidence_reply",
        "gap_context": gap_context,
        "label": "insufficient_evidence",
        "note": "An already submitted feedback signal.",
    }
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_feedback_signals (
                id, answer_id, user_id, entry_id, scope_key, knowledge_edition,
                label, note, normalized_metadata, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "feedback-gap-unrelated-integrity-winner",
                "answer-1",
                "feedback-user",
                None,
                "gap:decision_not_covered",
                None,
                "insufficient_evidence",
                "An already submitted feedback signal.",
                json.dumps(winner_metadata),
                now.isoformat(),
                (now + timedelta(days=180)).isoformat(),
            ),
        )

    original_existing_submission = KnowledgeFeedbackService._existing_submission
    stale_precheck = True

    async def existing_submission_after_stale_precheck(self, *args, **kwargs):
        nonlocal stale_precheck
        if stale_precheck:
            stale_precheck = False
            return None
        return await original_existing_submission(self, *args, **kwargs)

    original_flush = AsyncSession.flush

    async def unrelated_integrity_error_on_submission(self, *args, **kwargs):
        if any(isinstance(item, KnowledgeFeedbackSignal) for item in self.new):
            raise IntegrityError(
                "INSERT INTO review_work_items",
                {},
                sqlite3.IntegrityError("NOT NULL constraint failed: review_work_items.kind"),
            )
        return await original_flush(self, *args, **kwargs)

    monkeypatch.setattr(
        KnowledgeFeedbackService,
        "_existing_submission",
        existing_submission_after_stale_precheck,
    )
    monkeypatch.setattr(AsyncSession, "flush", unrelated_integrity_error_on_submission)
    with TestClient(app, raise_server_exceptions=False) as failing_client:
        response = failing_client.post(
            "/api/v1/knowledge-feedback",
            headers=_headers(token),
            json={
                "answer_id": "answer-1",
                "label": "insufficient_evidence",
                "note": "An already submitted feedback signal.",
            },
        )

    assert response.status_code == 500
    listed = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        params={"answer_id": "answer-1"},
    )
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["data"]["items"]] == [
        "feedback-gap-unrelated-integrity-winner"
    ]


def test_entry_feedback_requires_a_closed_supported_execution_not_legacy_trace(
    feedback_client,
    monkeypatch,
) -> None:
    client, session_factory, redis = feedback_client
    asyncio.run(_seed_answers(session_factory))
    token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="feedback-user"))

    async def no_closed_execution(_self, **_kwargs):
        return None

    monkeypatch.setattr(
        "app.knowledge_feedback.service.AnswerExecutionStore.load_for_message",
        no_closed_execution,
    )
    response = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        json={
            "answer_id": "answer-1",
            "entry_id": "pae-tools-001",
            "label": "helpful",
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "KNOWLEDGE_FEEDBACK_CLOSED_ANSWER_REQUIRED"

    listed = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(token),
        params={"answer_id": "answer-1"},
    )
    assert listed.status_code == 200
    assert listed.json()["data"] == {"items": []}


def test_member_lists_and_deletes_only_own_retained_feedback_without_conversation_or_note_content(
    feedback_client,
) -> None:
    client, session_factory, redis = feedback_client
    asyncio.run(_seed_answers(session_factory))
    user_token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="feedback-user"))
    other_token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="other-user"))
    admin_token = asyncio.run(
        create_authenticated_test_token(session_factory, redis, username="feedback-admin", role="admin")
    )

    submitted = client.post(
        "/api/v1/knowledge-feedback",
        headers=_headers(user_token),
        json={
            "answer_id": "answer-1",
            "entry_id": "pae-tools-001",
            "label": "helpful",
            "note": "The retained note must not reappear in the workspace list.",
        },
    )
    assert submitted.status_code == 200
    signal_id = submitted.json()["data"]["id"]

    listed = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(user_token),
        params={"answer_id": "answer-1"},
    )
    assert listed.status_code == 200
    assert listed.json()["data"] == {
        "items": [
            {
                "id": signal_id,
                "answer_id": "answer-1",
                "entry_id": "pae-tools-001",
                "knowledge_edition": "publication:v3",
                "outcome": "evidence_gated_answer",
                "label": "helpful",
                "created_at": listed.json()["data"]["items"][0]["created_at"],
                "expires_at": listed.json()["data"]["items"][0]["expires_at"],
                "retention_days": 180,
                "duplicate": False,
            }
        ]
    }
    serialized = listed.text
    assert "The retained note" not in serialized
    assert "private-session" not in serialized
    assert "private answer body" not in serialized

    for foreign_token in (other_token, admin_token):
        foreign_list = client.get(
            "/api/v1/knowledge-feedback",
            headers=_headers(foreign_token),
            params={"answer_id": "answer-1"},
        )
        assert foreign_list.status_code == 200
        assert foreign_list.json()["data"] == {"items": []}
        foreign_delete = client.delete(
            f"/api/v1/knowledge-feedback/{signal_id}",
            headers=_headers(foreign_token),
        )
        assert foreign_delete.status_code == 200
        assert foreign_delete.json()["data"] == {"id": signal_id, "deleted": False}

    owner_after_foreign_attempts = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(user_token),
        params={"answer_id": "answer-1"},
    )
    assert owner_after_foreign_attempts.status_code == 200
    assert owner_after_foreign_attempts.json()["data"] == listed.json()["data"]

    malformed_answer_filter = client.get(
        "/api/v1/knowledge-feedback",
        headers=_headers(user_token),
        params={"answer_id": ""},
    )
    assert malformed_answer_filter.status_code == 422
    assert malformed_answer_filter.json()["code"] == "KNOWLEDGE_FEEDBACK_ANSWER_INVALID"

    async def delete_conversation() -> None:
        async with session_factory() as session:
            await session.execute(delete(ChatMessage).where(ChatMessage.session_id == "private-session"))
            await session.execute(delete(ChatSession).where(ChatSession.id == "private-session"))
            await session.commit()

    asyncio.run(delete_conversation())
    retained = client.get("/api/v1/knowledge-feedback", headers=_headers(user_token))
    assert retained.status_code == 200
    assert retained.json()["data"] == listed.json()["data"]
    assert "The retained note" not in retained.text
    assert "private-session" not in retained.text
    assert "private answer body" not in retained.text
    deleted = client.delete(f"/api/v1/knowledge-feedback/{signal_id}", headers=_headers(user_token))
    assert deleted.status_code == 200
    assert deleted.json()["data"] == {"id": signal_id, "deleted": True}


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
            "outcome": "evidence_gated_answer",
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
                    scope_key="entry:pae-old-001",
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


def test_knowledge_feedback_migration_upgrades_and_downgrades(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "knowledge-feedback-migration.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    with monkeypatch.context() as settings_env:
        settings_env.setenv("DATABASE_URL", database_url)
        from app.common.config import get_settings

        get_settings.cache_clear()
        config = Config("alembic.ini")
        command.stamp(config, "20260802_0012")
        command.upgrade(config, "20260812_0013")

        sync_engine = create_engine(f"sqlite:///{db_path}")
        schema = inspect(sync_engine)
        assert {"knowledge_feedback_signals", "knowledge_review_work_items"}.issubset(schema.get_table_names())
        assert {item["name"] for item in schema.get_unique_constraints("knowledge_feedback_signals")} == {
            "uq_feedback_user_answer_entry"
        }
        assert {item["name"] for item in schema.get_unique_constraints("knowledge_review_work_items")} == {
            "uq_knowledge_review_work_item_dedupe"
        }

        command.downgrade(config, "20260802_0012")
        assert "knowledge_feedback_signals" not in inspect(sync_engine).get_table_names()
        assert "knowledge_review_work_items" not in inspect(sync_engine).get_table_names()
        sync_engine.dispose()
    get_settings.cache_clear()


def test_entry_free_gap_feedback_migration_preserves_existing_entry_feedback(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "knowledge-gap-feedback-migration.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    with monkeypatch.context() as settings_env:
        settings_env.setenv("DATABASE_URL", database_url)
        from app.common.config import get_settings

        get_settings.cache_clear()
        config = Config("alembic.ini")
        sync_engine = create_engine(f"sqlite:///{db_path}")
        with sync_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE users (
                        id CHAR(32) NOT NULL PRIMARY KEY,
                        username VARCHAR(64) NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(16) NOT NULL,
                        is_active BOOLEAN NOT NULL,
                        is_bootstrap_administrator BOOLEAN NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                    """
                )
            )
            connection.execute(text("CREATE UNIQUE INDEX ix_users_username ON users (username)"))
            connection.execute(
                text(
                    """
                    CREATE TABLE team_invitations (
                        id CHAR(32) NOT NULL PRIMARY KEY,
                        code_hash VARCHAR(64) NOT NULL,
                        created_by_user_id CHAR(32) NOT NULL,
                        expires_at DATETIME NOT NULL,
                        revoked_at DATETIME,
                        created_at DATETIME NOT NULL,
                        FOREIGN KEY(created_by_user_id) REFERENCES users (id),
                        UNIQUE(code_hash)
                    )
                    """
                )
            )
            connection.execute(
                text("CREATE INDEX ix_team_invitations_created_by_user_id ON team_invitations (created_by_user_id)")
            )
            connection.execute(text("CREATE INDEX ix_team_invitations_expires_at ON team_invitations (expires_at)"))
        command.stamp(config, "20260802_0012")
        command.upgrade(config, "20260812_0013")
        command.upgrade(config, "20260906_0019")

        with sync_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO knowledge_feedback_signals (
                        id, answer_id, user_id, entry_id, knowledge_edition, label,
                        note, normalized_metadata, created_at, expires_at
                    ) VALUES (
                        'entry-feedback', 'answer-001', 'user-001', 'pae-entry-001',
                        'publication:v1', 'helpful', NULL, '{}',
                        '2026-09-07 00:00:00', '2027-03-07 00:00:00'
                    )
                    """
                )
            )

        command.upgrade(config, "20260907_0020")

        upgraded = inspect(sync_engine)
        columns = {column["name"]: column for column in upgraded.get_columns("knowledge_feedback_signals")}
        assert columns["entry_id"]["nullable"] is True
        assert columns["knowledge_edition"]["nullable"] is True
        assert columns["scope_key"]["nullable"] is False
        assert {item["name"] for item in upgraded.get_unique_constraints("knowledge_feedback_signals")} == {
            "uq_feedback_user_answer_scope"
        }
        with sync_engine.connect() as connection:
            scope_key = connection.execute(
                text("SELECT scope_key FROM knowledge_feedback_signals WHERE id = 'entry-feedback'")
            ).scalar_one()
        assert scope_key == "entry:pae-entry-001"
        with pytest.raises(IntegrityError):
            with sync_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        INSERT INTO knowledge_feedback_signals (
                            id, answer_id, user_id, entry_id, scope_key, knowledge_edition, label,
                            note, normalized_metadata, created_at, expires_at
                        ) VALUES (
                            'duplicate-entry-feedback', 'answer-001', 'user-001', 'pae-entry-001',
                            'entry:pae-entry-001', 'publication:v1', 'helpful', NULL, '{}',
                            '2026-09-07 00:00:00', '2027-03-07 00:00:00'
                        )
                        """
                    )
                )
        with sync_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO knowledge_feedback_signals (
                        id, answer_id, user_id, entry_id, scope_key, knowledge_edition, label,
                        note, normalized_metadata, created_at, expires_at
                    ) VALUES (
                        'gap-feedback', 'answer-002', 'user-001', NULL,
                        'gap:decision_not_covered', NULL, 'insufficient_evidence', NULL, '{}',
                        '2026-09-07 00:00:00', '2027-03-07 00:00:00'
                    )
                    """
                )
            )

        command.downgrade(config, "20260906_0019")

        downgraded = inspect(sync_engine)
        downgraded_columns = {
            column["name"]: column for column in downgraded.get_columns("knowledge_feedback_signals")
        }
        assert downgraded_columns["entry_id"]["nullable"] is False
        assert downgraded_columns["knowledge_edition"]["nullable"] is False
        assert "scope_key" not in downgraded_columns
        assert {item["name"] for item in downgraded.get_unique_constraints("knowledge_feedback_signals")} == {
            "uq_feedback_user_answer_entry"
        }
        with sync_engine.connect() as connection:
            assert connection.execute(
                text("SELECT id FROM knowledge_feedback_signals ORDER BY id")
            ).scalars().all() == ["entry-feedback"]
        sync_engine.dispose()
    get_settings.cache_clear()


def test_session_list_isolates_one_unprojectable_execution_from_other_private_sessions(tmp_path, monkeypatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'session-summary.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def init() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            now = datetime.now(UTC)
            session.add_all(
                [
                    ChatSession(id="healthy-session", user_id="member", created_at=now, updated_at=now),
                    ChatSession(id="interrupted-session", user_id="member", created_at=now, updated_at=now),
                    ChatMessage(
                        id="healthy-question",
                        session_id="healthy-session",
                        user_id="member",
                        type="user",
                        content="健康会话的问题",
                        created_at=now,
                    ),
                    ChatMessage(
                        id="interrupted-question",
                        session_id="interrupted-session",
                        user_id="member",
                        type="user",
                        content="中断会话的问题",
                        created_at=now,
                    ),
                ]
            )
            await session.commit()

    batch_calls: list[str] = []

    async def execution_ids_for_messages(_self, *, messages):
        batch_calls.append("all-sessions")
        return ({message.id: f"{message.session_id}-execution" for message in messages}, frozenset())

    async def load_for_message(_self, *, message_id: str, **_kwargs):
        if message_id == "interrupted-question":
            raise ValueError("closed answer execution stream delivery was interrupted")
        return SimpleNamespace(projection={"state": "completed"})

    monkeypatch.setattr(
        "app.service.chat_service.AnswerExecutionStore.execution_ids_for_messages",
        execution_ids_for_messages,
    )
    monkeypatch.setattr("app.service.chat_service.AnswerExecutionStore.load_for_message", load_for_message)
    asyncio.run(init())
    try:
        async def list_items() -> list[dict]:
            async with session_factory() as session:
                return await ChatService(session).list_sessions(user_id="member")

        items = asyncio.run(list_items())
    finally:
        asyncio.run(engine.dispose())

    assert {item["session_id"] for item in items} == {"healthy-session", "interrupted-session"}
    by_id = {item["session_id"]: item for item in items}
    assert by_id["healthy-session"]["latest_execution_state"] == "completed"
    assert by_id["interrupted-session"]["latest_execution_state"] == "unavailable"
    assert batch_calls == ["all-sessions"]


def test_session_list_fails_closed_when_an_unindexed_message_is_bound_to_another_private_conversation(
    tmp_path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'cross-session-summary.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def init_and_list() -> list[dict]:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with session_factory() as session:
            now = datetime.now(UTC)
            session.add_all(
                [
                    ChatSession(id="bob-session", user_id="bob", created_at=now, updated_at=now),
                    ChatMessage(
                        id="moved-question",
                        session_id="bob-session",
                        user_id="bob",
                        type="user",
                        content="Alice private operating question",
                        answer_execution_id=None,
                        created_at=now,
                    ),
                    AnswerExecutionModel(
                        id="alice-execution",
                        session_id="alice-session",
                        user_id="alice",
                        initial_state="queued",
                        request={
                            "schema": "answer_execution_request/v1",
                            "request_id": "alice-request",
                            "user_id": "alice",
                            "session_id": "alice-session",
                            "user_message_id": "moved-question",
                            "question": "Alice private operating question",
                            "query_condition_set": {
                                "normalized_question": "Alice private operating question",
                                "conditions": [],
                            },
                            "condition_provenance": {"mode": "question_normalized"},
                        },
                        created_at=now,
                    ),
                ]
            )
            await session.commit()

        async with session_factory() as session:
            return await ChatService(session).list_sessions(user_id="bob")

    try:
        items = asyncio.run(init_and_list())
    finally:
        asyncio.run(engine.dispose())

    assert items == [
        {
            "session_id": "bob-session",
            "title": "未命名会话",
            "updated_at": items[0]["updated_at"],
            "message_count": 1,
            "latest_execution_state": "unavailable",
        }
    ]
