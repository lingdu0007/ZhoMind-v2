import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.model.chat import ChatMessage, ChatSession
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, ReviewWorkItem
from app.model.operational_event import OperationalEvent
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client


def test_retention_policy_is_versioned_non_content_and_admin_only(client: TestClient) -> None:
    admin = _headers(_register(client, username="retention-admin", role="admin"))
    member = _headers(_register(client, username="retention-user"))
    assert client.get("/api/v1/retention", headers=member).status_code == 403
    assert client.get("/api/v1/retention").status_code == 401
    response = client.get("/api/v1/retention", headers=admin)
    assert response.status_code == 200
    policy = response.json()["data"]["policy"]
    assert policy["days"] == {"conversations": 30, "operational_events": 30, "feedback_signals": 180}
    assert policy["version"] == 1
    assert policy["identity"].startswith("configuration:")
    assert policy["content_telemetry"] is False
    assert policy["administrator_transcript_access"] is False
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["policy"] == policy


def test_policy_change_requires_exact_version_and_is_retained_on_reload(client: TestClient) -> None:
    admin = _headers(_register(client, username="policy-admin", role="admin"))
    old = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]
    payload = {"expected_identity": old["identity"], "days": {
        "conversations": 7, "operational_events": 14, "feedback_signals": 90,
    }}
    response = client.put("/api/v1/retention/policy", headers=admin, json=payload)
    assert response.status_code == 200
    new = response.json()["data"]["policy"]
    assert new["version"] == 2
    assert new["identity"] != old["identity"]
    assert new["days"] == payload["days"]
    assert response.json()["data"]["reacceptance_required"] is True
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["policy"] == new
    stale = client.put("/api/v1/retention/policy", headers=admin, json=payload)
    assert stale.status_code == 409
    assert stale.json()["code"] == "RETENTION_POLICY_STALE"


@pytest.mark.parametrize("invalid", [0, -1, True, "30", 3651, None])
def test_policy_cannot_disable_retention_or_enable_content_access(client: TestClient, invalid) -> None:
    admin = _headers(_register(client, username="policy-invalid-admin", role="admin"))
    old = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]
    payload = {"expected_identity": old["identity"], "days": {**old["days"], "conversations": invalid}}
    assert client.put("/api/v1/retention/policy", headers=admin, json=payload).status_code == 422
    payload = {"expected_identity": old["identity"], "days": old["days"], "content_telemetry": True}
    assert client.put("/api/v1/retention/policy", headers=admin, json=payload).status_code == 422
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["policy"] == old


def test_independent_sweep_expires_all_classes_without_a_feedback_read(client: TestClient) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="sweep-admin", role="admin"))
    owner = _headers(_register(client, username="sweep-owner"))
    now = datetime.now(UTC)

    async def seed():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(ChatSession(id="expired-private", user_id="sweep-owner", created_at=now - timedelta(days=31)))
            session.add(ChatMessage(
                session_id="expired-private", user_id="sweep-owner", type="user", content="private-retention-fixture",
            ))
            session.add(OperationalEvent(
                request_id="old-operation", route_outcome="GET /api/v1/health:success", duration_ms=1,
                created_at=now - timedelta(days=31),
            ))
            session.add(KnowledgeFeedbackSignal(
                id="old-feedback", answer_id="retained-answer", user_id="sweep-owner", entry_id="entry-safe",
                scope_key="entry:entry-safe", label="outdated", note="private-feedback-fixture",
                created_at=now - timedelta(days=181), expires_at=now - timedelta(days=1),
            ))
            session.add(ReviewWorkItem(
                id="durable-review", kind="feedback_signal", dedupe_key="feedback:old-feedback",
                subject_id="entry-safe", signal_id="old-feedback", classification="content",
                normalized_metadata={"answer_id": "retained-answer", "note": "private-feedback-fixture"},
            ))
            await session.commit()

    asyncio.run(seed())
    result = asyncio.run(run_retention_sweep(client.app.state.test_auth_session_factory, now=now))
    assert set(result) == {"conversations", "operational_events", "feedback_signals"}
    for record_class in result.values():
        assert record_class["status"] == "verified"
        assert record_class["deleted_count"] >= 1
        assert record_class["remaining_expired"] == 0
    negative_read = client.get("/api/v1/sessions/expired-private", headers=owner)
    assert negative_read.json()["messages"] == []
    queue = client.get("/api/v1/knowledge-review-queue", headers=admin).json()["items"]
    retained = next(item for item in queue if item["id"] == "durable-review")
    assert retained["classification"] == "content"
    assert "private-feedback-fixture" not in str(retained)
    assert "retained-answer" not in str(retained)
    assert "old-feedback" not in str(retained)
    status = client.get("/api/v1/retention", headers=admin).json()["data"]
    assert status["privacy_blocked"] is False
    assert set(status["cleanup"]) == set(result)


def test_cleanup_failure_is_blocking_non_content_and_retryable(client: TestClient, caplog) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="retry-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory

    async def install_failure():
        async with factory() as session:
            session.add(ChatSession(
                id="failure-expired", user_id="absent-owner", created_at=datetime.now(UTC) - timedelta(days=31),
            ))
            await session.execute(text(
                "CREATE TRIGGER reject_private_cleanup BEFORE DELETE ON chat_sessions "
                "BEGIN SELECT RAISE(ABORT, 'private-content-in-database-error'); END"
            ))
            await session.commit()

    asyncio.run(install_failure())
    failed = asyncio.run(run_retention_sweep(factory))
    assert failed["conversations"]["status"] == "failed"
    assert failed["feedback_signals"]["status"] == "verified"
    status = client.get("/api/v1/retention", headers=admin).json()["data"]
    assert status["privacy_blocked"] is True
    assert "private-content-in-database-error" not in str(status)
    assert "private-content-in-database-error" not in caplog.text

    async def repair():
        async with factory() as session:
            await session.execute(text("DROP TRIGGER reject_private_cleanup"))
            await session.commit()

    asyncio.run(repair())
    retried = client.post("/api/v1/retention/cleanup", headers=admin)
    assert retried.status_code == 200
    data = retried.json()["data"]
    assert data["privacy_blocked"] is False
    assert data["cleanup"]["conversations"]["attempt"] > failed["conversations"]["attempt"]


def test_owner_deletion_denies_other_members_and_admin_and_severs_feedback_reference(client: TestClient) -> None:
    owner = _headers(_register(client, username="delete-owner"))
    other = _headers(_register(client, username="delete-other"))
    admin = _headers(_register(client, username="delete-admin", role="admin"))
    chat = client.post("/api/v1/chat", headers=owner, json={
        "session_id": "deletion-product-path", "message": "uncovered engineering decision",
    })
    assert chat.status_code == 200
    history = client.get("/api/v1/sessions/deletion-product-path", headers=owner).json()
    answer = next(message for message in history["messages"] if message["type"] == "assistant")
    feedback = client.post("/api/v1/knowledge-feedback", headers=owner, json={
        "answer_id": answer["id"], "label": "insufficient_evidence", "note": "explicit private feedback",
    })
    assert feedback.status_code == 200
    signal_id = feedback.json()["data"]["id"]
    queue = client.get("/api/v1/knowledge-review-queue", headers=admin).json()["items"]
    item_id = queue[0]["id"]
    classified = client.patch(f"/api/v1/knowledge-review-queue/{item_id}", headers=admin, json={
        "classification": "p1", "status": "reviewed",
    })
    assert classified.status_code == 200
    for denied in (other, admin):
        assert client.get("/api/v1/sessions/deletion-product-path", headers=denied).json()["messages"] == []
        assert client.delete("/api/v1/sessions/deletion-product-path", headers=denied).json()["deleted"] is False
        assert client.delete(f"/api/v1/knowledge-feedback/{signal_id}", headers=denied).json()["deleted"] is False
    assert client.delete("/api/v1/sessions/deletion-product-path", headers=owner).json()["deleted"] is True
    assert client.get("/api/v1/sessions/deletion-product-path", headers=owner).json()["messages"] == []
    assert client.delete(f"/api/v1/knowledge-feedback/{signal_id}", headers=owner).json()["deleted"] is True
    assert client.get("/api/v1/knowledge-feedback", headers=owner).json()["items"] == []
    retained = client.get("/api/v1/knowledge-review-queue", headers=admin).json()["items"]
    assert len(retained) == 1
    assert retained[0]["classification"] == "p1"
    assert retained[0]["metadata"] == {}
    assert "explicit private feedback" not in str(retained)
    assert answer["id"] not in str(retained)


def test_feedback_submission_uses_effective_policy_days(client: TestClient) -> None:
    owner = _headers(_register(client, username="short-feedback-owner"))
    admin = _headers(_register(client, username="short-feedback-admin", role="admin"))
    old = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]
    changed = client.put("/api/v1/retention/policy", headers=admin, json={
        "expected_identity": old["identity"], "days": {**old["days"], "feedback_signals": 7},
    })
    assert changed.status_code == 200
    result = client.post("/api/v1/chat", headers=owner, json={
        "session_id": "short-feedback-session", "message": "uncovered engineering request",
    })
    assert result.status_code == 200
    history = client.get("/api/v1/sessions/short-feedback-session", headers=owner).json()["messages"]
    answer = next(item for item in history if item["type"] == "assistant")
    response = client.post("/api/v1/knowledge-feedback", headers=owner, json={
        "answer_id": answer["id"], "label": "insufficient_evidence",
    })
    assert response.status_code == 200
    signal = response.json()["data"]
    assert signal["retention_days"] == 7
    assert datetime.fromisoformat(signal["expires_at"]) - datetime.fromisoformat(signal["created_at"]) == timedelta(days=7)


def test_policy_change_suspends_exact_bound_acceptance_without_reusing_its_evidence(client: TestClient) -> None:
    from tests.integration.test_delivery_acceptance import _record

    admin = _headers(_register(client, username="acceptance-policy-admin", role="admin"))
    old = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]
    payload = _record()
    assert old["identity"] in payload["product_identities"]
    created = client.post("/api/v1/acceptance/records", headers=admin, json=payload)
    assert created.status_code == 200
    identity = created.json()["data"]["record_id"]
    changed = client.put("/api/v1/retention/policy", headers=admin, json={
        "expected_identity": old["identity"], "days": {**old["days"], "conversations": 7},
    })
    assert changed.status_code == 200
    record = client.get(f"/api/v1/acceptance/records/{identity}", headers=admin).json()["data"]
    assert record["current_status"] == "suspended"
    assert record["accepted_scope"]["deployment_identity"] is None
    assert any(item.get("reason") == "retention_policy_changed" for item in record["blockers"])


def test_classification_cannot_expose_an_expired_raw_feedback_description(client: TestClient) -> None:
    admin = _headers(_register(client, username="expired-classification-admin", role="admin"))

    async def seed():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(KnowledgeFeedbackSignal(
                id="expired-classification-signal", answer_id="private-answer-id", user_id="private-owner",
                entry_id="entry-safe", scope_key="entry:entry-safe", label="outdated",
                note="expired private description", normalized_metadata={},
                created_at=datetime.now(UTC) - timedelta(days=181),
                expires_at=datetime.now(UTC) - timedelta(days=1),
            ))
            session.add(ReviewWorkItem(
                id="expired-classification-item", kind="feedback_signal", dedupe_key="feedback:expired-classification-signal",
                subject_id="entry-safe", signal_id="expired-classification-signal",
                normalized_metadata={"note": "expired private description", "answer_id": "private-answer-id"},
            ))
            await session.commit()

    asyncio.run(seed())
    response = client.patch("/api/v1/knowledge-review-queue/expired-classification-item", headers=admin, json={
        "classification": "p1", "status": "reviewed",
    })
    assert response.status_code == 404
    assert "expired private description" not in response.text


def test_cleanup_cannot_claim_success_when_database_ignores_expired_deletion(client: TestClient) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="survivor-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory

    async def seed():
        async with factory() as session:
            session.add(ChatSession(id="expiry-survivor", user_id="survivor", created_at=datetime.now(UTC) - timedelta(days=31)))
            await session.execute(text(
                "CREATE TRIGGER ignore_private_cleanup BEFORE DELETE ON chat_sessions BEGIN SELECT RAISE(IGNORE); END"
            ))
            await session.commit()

    asyncio.run(seed())
    result = asyncio.run(run_retention_sweep(factory))
    assert result["conversations"]["status"] == "failed"
    assert result["conversations"]["remaining_expired"] == 1
    assert result["conversations"]["normalized_error"] == "PRIVACY_EXPIRY_SURVIVED"
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["privacy_blocked"] is True


@pytest.mark.parametrize("supplied", [
    "password=private-header-fixture", "sk-" + "A" * 32, "ghp_" + "B" * 36, "private-project-question",
])
def test_private_request_id_is_replaced_before_response_and_telemetry(client: TestClient, supplied: str) -> None:
    response = client.get("/api/v1/health", headers={"x-request-id": supplied})
    assert response.status_code == 200
    assert response.headers["x-request-id"] != supplied
    assert supplied not in response.text


def test_review_projection_cannot_fall_back_to_a_private_copy_when_signal_is_missing(client: TestClient) -> None:
    admin = _headers(_register(client, username="orphan-review-admin", role="admin"))

    async def seed():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(ReviewWorkItem(
                id="orphan-review", kind="feedback_signal", dedupe_key="feedback:missing-signal",
                subject_id="entry-safe", signal_id="missing-signal", classification="p1",
                normalized_metadata={"note": "orphan-private-description", "answer_id": "private-answer"},
            ))
            await session.commit()

    asyncio.run(seed())
    response = client.get("/api/v1/knowledge-review-queue", headers=admin)
    assert response.status_code == 200
    assert response.json()["items"][0]["metadata"] == {}
    assert "orphan-private-description" not in response.text


@pytest.mark.parametrize("remaining,error,offset", [
    (1, None, 0), (0, "private-state-description", 0), (0, None, 3600),
])
def test_contradictory_cleanup_state_is_non_content_and_fail_closed(client: TestClient, remaining, error, offset) -> None:
    from app.retention.cleanup import run_retention_sweep
    from app.retention.models import RetentionCleanupState

    admin = _headers(_register(client, username="invalid-state-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory
    asyncio.run(run_retention_sweep(factory))

    async def corrupt():
        async with factory() as session:
            state = await session.get(RetentionCleanupState, "conversations")
            assert state is not None
            state.remaining_expired = remaining
            state.normalized_error = error
            state.checked_at = datetime.now(UTC) + timedelta(seconds=offset)
            await session.commit()

    asyncio.run(corrupt())
    response = client.get("/api/v1/retention", headers=admin)
    assert response.status_code == 200
    assert response.json()["data"]["privacy_blocked"] is True
    assert response.json()["data"]["cleanup"]["conversations"]["status"] == "unverified"
    assert "private-state-description" not in response.text


def test_changed_policy_enforces_distinct_class_cutoffs_without_member_reads(client: TestClient) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="distinct-cutoff-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory
    old = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]
    assert client.put("/api/v1/retention/policy", headers=admin, json={
        "expected_identity": old["identity"],
        "days": {"conversations": 7, "operational_events": 14, "feedback_signals": 90},
    }).status_code == 200
    now = datetime.now(UTC)

    async def seed():
        async with factory() as session:
            session.add(ChatSession(id="cutoff-conversation", user_id="cutoff-owner", created_at=now - timedelta(days=10)))
            session.add(OperationalEvent(
                request_id="cutoff-operation", route_outcome="GET /api/v1/health:success",
                duration_ms=1, created_at=now - timedelta(days=10),
            ))
            session.add(KnowledgeFeedbackSignal(
                id="cutoff-feedback", answer_id="cutoff-answer", user_id="cutoff-owner",
                entry_id="entry-safe", scope_key="entry:entry-safe", label="outdated",
                created_at=now - timedelta(days=100), expires_at=now + timedelta(days=80),
            ))
            await session.commit()

    asyncio.run(seed())
    result = asyncio.run(run_retention_sweep(factory, now=now))
    assert result["conversations"]["deleted_count"] == 1
    assert result["operational_events"]["deleted_count"] == 0
    assert result["feedback_signals"]["deleted_count"] == 1
    assert all(item["status"] == "verified" for item in result.values())


@pytest.mark.parametrize("reject_all_updates", [False, True])
def test_pending_write_failure_invalidates_prior_verified_observation(client: TestClient, reject_all_updates: bool) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="pending-failure-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory
    asyncio.run(run_retention_sweep(factory))

    async def seed():
        async with factory() as session:
            session.add(ChatSession(id="pending-failure-expired", user_id="absent", created_at=datetime.now(UTC) - timedelta(days=31)))
            condition = "" if reject_all_updates else " AND NEW.status = 'pending'"
            await session.execute(text(
                "CREATE TRIGGER reject_pending BEFORE UPDATE ON retention_cleanup_states "
                f"WHEN NEW.data_class = 'conversations'{condition} "
                "BEGIN SELECT RAISE(ABORT, 'private database detail'); END"
            ))
            await session.commit()

    asyncio.run(seed())
    failed = asyncio.run(run_retention_sweep(factory))
    assert failed["conversations"]["status"] == "failed"
    status = client.get("/api/v1/retention", headers=admin).json()["data"]
    assert status["privacy_blocked"] is True
    assert status["cleanup"]["conversations"]["status"] == "failed"
    assert status["cleanup"]["conversations"]["normalized_error"] == "RETENTION_CLEANUP_FAILED"
    assert client.post("/api/v1/retention/cleanup", headers=admin).json()["data"]["privacy_blocked"] is True


def test_feedback_reference_survivor_blocks_cleanup_and_is_found_on_retry(client: TestClient) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="reference-survivor-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory

    async def seed():
        async with factory() as session:
            session.add(KnowledgeFeedbackSignal(
                id="reference-survivor-signal", user_id="absent", answer_id="private-answer",
                entry_id="entry-safe", scope_key="entry:entry-safe", label="outdated",
                created_at=datetime.now(UTC) - timedelta(days=181), expires_at=datetime.now(UTC) - timedelta(days=1),
            ))
            session.add(ReviewWorkItem(
                id="reference-survivor-item", kind="feedback_signal",
                dedupe_key="feedback:reference-survivor-signal", signal_id="reference-survivor-signal", subject_id="entry-safe",
            ))
            await session.execute(text(
                "CREATE TRIGGER ignore_feedback_reference BEFORE DELETE ON knowledge_review_work_items "
                "BEGIN SELECT RAISE(IGNORE); END"
            ))
            await session.commit()

    asyncio.run(seed())
    failed = asyncio.run(run_retention_sweep(factory))
    assert failed["feedback_signals"]["status"] == "failed"
    assert failed["feedback_signals"]["remaining_expired"] >= 1
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["privacy_blocked"] is True

    async def repair():
        async with factory() as session:
            await session.execute(text("DROP TRIGGER ignore_feedback_reference"))
            await session.commit()

    asyncio.run(repair())
    retried = client.post("/api/v1/retention/cleanup", headers=admin).json()["data"]
    assert retried["privacy_blocked"] is False
    assert client.get("/api/v1/knowledge-review-queue", headers=admin).json()["items"] == []


@pytest.mark.parametrize("locally_suspended", [False, True])
def test_cleanup_retry_cannot_restore_pre_failure_acceptance(client: TestClient, locally_suspended: bool) -> None:
    from app.retention.cleanup import run_retention_sweep
    from tests.integration.test_delivery_acceptance import _activate, _active_local_record, _create, _record

    admin = _headers(_register(client, username="durable-suspension-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory
    asyncio.run(run_retention_sweep(factory))
    local = _active_local_record(client, admin)
    editorial = _activate(client, admin, _create(client, admin, _record("editorial_preview", local["record_id"]))["record_id"])
    pilot = _activate(client, admin, _create(client, admin, _record("limited_team_pilot", editorial["record_id"]))["record_id"])
    path = f"/api/v1/acceptance/records/{pilot['record_id']}"
    if locally_suspended:
        assert client.post(path + "/status", headers=admin, json={
            "status": "suspended", "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query", "reason": "entry evidence was withdrawn",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
                "evidence_links": ["evidence://retention/local-entry-failure"],
            },
        }).status_code == 200

    async def fail_cleanup():
        async with factory() as session:
            session.add(ChatSession(id="suspension-expired", user_id="absent", created_at=datetime.now(UTC) - timedelta(days=31)))
            await session.execute(text(
                "CREATE TRIGGER suspend_on_cleanup BEFORE DELETE ON chat_sessions "
                "BEGIN SELECT RAISE(IGNORE); END"
            ))
            await session.commit()

    asyncio.run(fail_cleanup())
    asyncio.run(run_retention_sweep(factory))
    failed = client.get(path, headers=admin).json()["data"]
    assert failed["current_status"] == "suspended"

    async def repair():
        async with factory() as session:
            await session.execute(text("DROP TRIGGER suspend_on_cleanup"))
            await session.commit()

    asyncio.run(repair())
    assert client.post("/api/v1/retention/cleanup", headers=admin).json()["data"]["privacy_blocked"] is False
    retained = client.get(path, headers=admin).json()["data"]
    assert retained["current_status"] == "suspended"
    assert retained["accepted_scope"]["deployment_identity"] is None
    assert retained["status_history"][-1]["reason_code"] == "integrity_failure"
    assert any(item.get("failure_kind") == "shared_privacy" for item in retained["blockers"])
    if locally_suspended:
        assert any(item.get("failure_kind") == "entry_specific" for item in retained["blockers"])
    assert client.post(path + "/status", headers=admin, json={
        "status": "active", "reason_code": "checks_verified",
        "verified_checks": [{"check_id": item["check_id"], "evidence_links": item["evidence_links"]} for item in pilot["checks"]],
    }).status_code == 409


@pytest.mark.parametrize("table", ["knowledge_feedback_signals", "knowledge_review_work_items"])
def test_owner_feedback_delete_does_not_claim_success_with_surviving_private_links(client: TestClient, table: str) -> None:
    owner = _headers(_register(client, username="verified-delete-owner"))
    factory = client.app.state.test_auth_session_factory

    async def seed():
        async with factory() as session:
            session.add(KnowledgeFeedbackSignal(
                id="verified-delete-signal", user_id="verified-delete-owner", answer_id="private-answer",
                entry_id="entry-safe", scope_key="entry:entry-safe", label="outdated",
                created_at=datetime.now(UTC), expires_at=datetime.now(UTC) + timedelta(days=180),
            ))
            session.add(ReviewWorkItem(
                id="verified-delete-item", kind="feedback_signal", dedupe_key="feedback:verified-delete-signal",
                signal_id="verified-delete-signal", subject_id="entry-safe",
            ))
            await session.execute(text(
                f"CREATE TRIGGER ignore_member_delete BEFORE DELETE ON {table} BEGIN SELECT RAISE(IGNORE); END"
            ))
            await session.commit()

    asyncio.run(seed())
    response = client.delete("/api/v1/knowledge-feedback/verified-delete-signal", headers=owner)
    assert response.status_code == 503
    assert response.json()["code"] == "PRIVACY_DELETE_UNVERIFIED"


def test_delayed_failure_recovery_cannot_overwrite_a_newer_successful_attempt(client: TestClient) -> None:
    from contextlib import asynccontextmanager

    from app.retention.cleanup import run_retention_sweep
    from tests.integration.test_delivery_acceptance import _activate, _active_local_record, _create, _record

    admin = _headers(_register(client, username="concurrent-cleanup-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory
    asyncio.run(run_retention_sweep(factory))
    local = _active_local_record(client, admin)
    editorial = _activate(client, admin, _create(client, admin, _record("editorial_preview", local["record_id"]))["record_id"])
    pilot = _activate(client, admin, _create(client, admin, _record("limited_team_pilot", editorial["record_id"]))["record_id"])

    async def exercise():
        await run_retention_sweep(factory)
        async with factory() as session:
            await session.execute(text(
                "CREATE TRIGGER reject_concurrent_pending BEFORE UPDATE ON retention_cleanup_states "
                "WHEN NEW.data_class = 'conversations' AND NEW.status = 'pending' "
                "BEGIN SELECT RAISE(ABORT, 'normalized test failure'); END"
            ))
            await session.commit()
        recovering = asyncio.Event()
        resume = asyncio.Event()
        calls = 0

        @asynccontextmanager
        async def delayed_factory():
            nonlocal calls
            calls += 1
            if calls == 2:
                recovering.set()
                await resume.wait()
            async with factory() as session:
                yield session

        older_task = asyncio.create_task(run_retention_sweep(delayed_factory))
        try:
            await asyncio.wait_for(recovering.wait(), timeout=5)
            async with factory() as session:
                await session.execute(text("DROP TRIGGER reject_concurrent_pending"))
                await session.commit()
            newer = await run_retention_sweep(factory)
        finally:
            resume.set()
            older = await asyncio.wait_for(older_task, timeout=5)
        assert older["conversations"] == newer["conversations"]

    asyncio.run(exercise())
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["privacy_blocked"] is False
    acceptance = client.get(f"/api/v1/acceptance/records/{pilot['record_id']}", headers=admin).json()["data"]
    assert acceptance["current_status"] == "suspended"
    assert acceptance["accepted_scope"]["deployment_identity"] is None
    assert acceptance["blockers"][0]["failure_kind"] == "shared_privacy"


@pytest.mark.parametrize("table", ["chat_sessions", "chat_messages", "answer_executions", "answer_execution_events"])
def test_owner_conversation_delete_verifies_every_private_record_layer(client: TestClient, table: str) -> None:
    owner = _headers(_register(client, username="verified-conversation-owner"))
    session_id = "verified-conversation-delete"
    assert client.post("/api/v1/chat", headers=owner, json={"session_id": session_id, "message": "uncovered decision"}).status_code == 200
    factory = client.app.state.test_auth_session_factory

    async def fail():
        async with factory() as session:
            await session.execute(text(
                f"CREATE TRIGGER ignore_conversation_delete BEFORE DELETE ON {table} BEGIN SELECT RAISE(IGNORE); END"
            ))
            await session.commit()

    asyncio.run(fail())
    response = client.delete(f"/api/v1/sessions/{session_id}", headers=owner)
    assert response.status_code == 503
    assert response.json()["code"] == "PRIVACY_DELETE_UNVERIFIED"
    assert client.get(f"/api/v1/sessions/{session_id}", headers=owner).json()["messages"]


def test_conversation_cleanup_retry_removes_orphans_after_parent_disappears(client: TestClient) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="conversation-orphan-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory

    async def seed():
        async with factory() as session:
            session.add(ChatSession(id="orphan-retry", user_id="absent", created_at=datetime.now(UTC) - timedelta(days=31)))
            session.add(ChatMessage(session_id="orphan-retry", user_id="absent", type="user", content="private orphan content"))
            await session.execute(text(
                "CREATE TRIGGER ignore_expired_message BEFORE DELETE ON chat_messages BEGIN SELECT RAISE(IGNORE); END"
            ))
            await session.commit()

    asyncio.run(seed())
    failed = asyncio.run(run_retention_sweep(factory))
    assert failed["conversations"]["status"] == "failed"

    async def repair():
        async with factory() as session:
            await session.execute(text("DROP TRIGGER ignore_expired_message"))
            await session.commit()

    asyncio.run(repair())
    retried = client.post("/api/v1/retention/cleanup", headers=admin).json()["data"]
    assert retried["cleanup"]["conversations"]["status"] == "verified"
    assert retried["cleanup"]["conversations"]["remaining_expired"] == 0


@pytest.mark.parametrize("cleanup_kind", ["owner_delete", "expiry", "orphan", "event_only"])
@pytest.mark.parametrize("surviving_message", [False, True])
def test_cleanup_removes_cross_session_messages_before_their_immutable_bindings(
    client: TestClient, cleanup_kind: str, surviving_message: bool,
) -> None:
    owner = _headers(_register(client, username="bound-cleanup-owner"))
    other = _headers(_register(client, username="bound-cleanup-other"))
    admin = _headers(_register(client, username="bound-cleanup-admin", role="admin"))
    original = "bound-cleanup-original"
    moved = "bound-cleanup-moved"
    assert client.post("/api/v1/chat", headers=owner, json={
        "session_id": original, "message": "private unindexed question",
    }).status_code == 200
    factory = client.app.state.test_auth_session_factory

    async def corrupt():
        async with factory() as session:
            session.add(ChatSession(id=moved, user_id="bound-cleanup-other"))
            await session.execute(text(
                "UPDATE chat_messages SET session_id = :moved, user_id = 'bound-cleanup-other', "
                "answer_execution_id = NULL WHERE session_id = :original AND (:event_only = 0 OR type = 'assistant')"
            ), {"moved": moved, "original": original, "event_only": cleanup_kind == "event_only"})
            if cleanup_kind in {"orphan", "event_only"}:
                await session.execute(text("DELETE FROM chat_sessions WHERE id = :id"), {"id": original})
                if cleanup_kind == "event_only":
                    await session.execute(text("DELETE FROM answer_executions WHERE session_id = :id"), {"id": original})
            elif cleanup_kind == "expiry":
                await session.execute(text("UPDATE chat_sessions SET created_at = :old WHERE id = :id"), {
                    "id": original, "old": datetime.now(UTC) - timedelta(days=31),
                })
            if surviving_message:
                await session.execute(text(
                    "CREATE TRIGGER ignore_bound_message BEFORE DELETE ON chat_messages "
                    "BEGIN SELECT RAISE(IGNORE); END"
                ))
            await session.commit()

    asyncio.run(corrupt())

    def cleanup():
        if cleanup_kind == "owner_delete":
            response = client.delete(f"/api/v1/sessions/{original}", headers=owner)
            return response.status_code == 200
        response = client.post("/api/v1/retention/cleanup", headers=admin)
        assert response.status_code == 200
        return response.json()["data"]["cleanup"]["conversations"]["status"] == "verified"

    assert cleanup() is not surviving_message
    if surviving_message:
        async def repair():
            async with factory() as session:
                await session.execute(text("DROP TRIGGER ignore_bound_message"))
                await session.commit()

        asyncio.run(repair())
        assert cleanup() is True
    history = client.get(f"/api/v1/sessions/{moved}", headers=other)
    assert history.status_code == 200
    assert history.json()["messages"] == []
    assert "private unindexed question" not in client.get("/api/v1/sessions", headers=other).text


def test_owner_deletion_cannot_use_a_corrupt_index_to_delete_another_members_message(client: TestClient) -> None:
    owner = _headers(_register(client, username="index-cleanup-owner"))
    other = _headers(_register(client, username="index-cleanup-other"))
    for headers, session_id in ((owner, "index-owner"), (other, "index-other")):
        assert client.post("/api/v1/chat", headers=headers, json={
            "session_id": session_id, "message": "retained private question",
        }).status_code == 200
    owner_history = client.get("/api/v1/sessions/index-owner", headers=owner).json()["messages"]
    other_history = client.get("/api/v1/sessions/index-other", headers=other).json()["messages"]
    original_message = next(item for item in other_history if item["type"] == "user")
    wrong_execution = owner_history[0]["answer_execution"]["id"]
    factory = client.app.state.test_auth_session_factory

    async def set_index(execution_id):
        async with factory() as session:
            await session.execute(text("UPDATE chat_messages SET answer_execution_id = :execution WHERE id = :id"), {
                "execution": execution_id, "id": original_message["id"],
            })
            await session.commit()

    asyncio.run(set_index(wrong_execution))
    assert client.delete("/api/v1/sessions/index-owner", headers=owner).status_code == 200
    summary = client.get("/api/v1/sessions", headers=other).json()
    assert "retained private question" not in str(summary)
    asyncio.run(set_index(original_message["answer_execution"]["id"]))
    retained = client.get("/api/v1/sessions/index-other", headers=other)
    assert retained.status_code == 200
    assert retained.json()["messages"] == other_history


def test_earlier_worker_start_cannot_hide_a_later_locked_attempt_failure(client: TestClient) -> None:
    from contextlib import asynccontextmanager

    from app.retention.cleanup import run_retention_sweep

    admin = _headers(_register(client, username="locked-order-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory

    async def exercise():
        waiting = asyncio.Event()
        resume = asyncio.Event()
        first = True

        @asynccontextmanager
        async def delayed_factory():
            nonlocal first
            if first:
                first = False
                waiting.set()
                await resume.wait()
            async with factory() as session:
                yield session

        now = datetime.now(UTC)
        delayed = asyncio.create_task(run_retention_sweep(delayed_factory, now=now - timedelta(seconds=1)))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
            await run_retention_sweep(factory, now=now)
            async with factory() as session:
                session.add(ChatSession(id="locked-order-expired", user_id="absent", created_at=now - timedelta(days=31)))
                await session.execute(text(
                    "CREATE TRIGGER reject_later_attempt BEFORE UPDATE ON retention_cleanup_states "
                    "WHEN NEW.data_class = 'conversations' BEGIN SELECT RAISE(ABORT, 'normalized failure'); END"
                ))
                await session.commit()
        finally:
            resume.set()
            await asyncio.wait_for(delayed, timeout=5)

    asyncio.run(exercise())
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["privacy_blocked"] is True


def test_normal_overlapping_sweeps_do_not_suspend_active_acceptance(client: TestClient) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.retention.cleanup import run_retention_sweep
    from tests.integration.test_delivery_acceptance import _activate, _active_local_record, _create, _record

    admin = _headers(_register(client, username="overlap-admin", role="admin"))
    factory = client.app.state.test_auth_session_factory
    asyncio.run(run_retention_sweep(factory))
    local = _active_local_record(client, admin)
    editorial = _activate(client, admin, _create(client, admin, _record("editorial_preview", local["record_id"]))["record_id"])
    pilot = _activate(client, admin, _create(client, admin, _record("limited_team_pilot", editorial["record_id"]))["record_id"])

    async def exercise():
        pending = asyncio.Event()
        resume = asyncio.Event()

        class PausingSession(AsyncSession):
            async def commit(self):
                await super().commit()
                if not pending.is_set():
                    pending.set()
                    await resume.wait()

        paused_factory = async_sessionmaker(factory.kw["bind"], class_=PausingSession, expire_on_commit=False)
        older = asyncio.create_task(run_retention_sweep(paused_factory))
        try:
            await asyncio.wait_for(pending.wait(), timeout=5)
            await run_retention_sweep(factory)
        finally:
            resume.set()
            await asyncio.wait_for(older, timeout=5)

    asyncio.run(exercise())
    assert client.get("/api/v1/retention", headers=admin).json()["data"]["privacy_blocked"] is False
    record = client.get(f"/api/v1/acceptance/records/{pilot['record_id']}", headers=admin).json()["data"]
    assert record["current_status"] == "active"
    assert record["accepted_scope"]["deployment_identity"] is not None
