import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client


def _gap(
    client: TestClient,
    headers: dict,
    session_id: str = "maintenance-gap",
    question: str = "uncovered engineering decision",
) -> dict:
    response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "session_id": session_id,
            "message": question,
        },
    )
    assert response.status_code == 200, response.text
    messages = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()["messages"]
    return next(message for message in reversed(messages) if message["type"] == "assistant")


@pytest.mark.parametrize("label", ["helpful", "insufficient_evidence", "outdated", "out_of_scope"])
def test_all_explicit_categories_bind_a_persisted_insufficient_outcome(client: TestClient, label: str) -> None:
    owner = _headers(_register(client, username="maintenance-signal-owner"))
    answer = _gap(client, owner)
    assert client.get("/api/v1/knowledge-feedback", headers=owner).json()["items"] == []
    payload = {"answer_id": answer["id"], "label": label}
    response = client.post("/api/v1/knowledge-feedback", headers=owner, json=payload)
    assert response.status_code == 200, response.text
    signal = response.json()["data"]
    assert signal["outcome"] == "insufficient_evidence_reply"
    assert signal["label"] == label
    assert signal["entry_id"] is None
    assert signal["retention_days"] == 180
    assert signal["answer_execution_id"] == answer["answer_execution"]["id"]
    assert signal["query_condition_set_identity"] == answer["answer_execution"]["query_condition_set"]["identity"]
    assert signal["knowledge_version_identities"] == answer["answer_execution"]["knowledge_version_identities"]
    assert signal["gap_context"]["reason"] == "no_eligible_published_evidence"
    repeated = client.post("/api/v1/knowledge-feedback", headers=owner, json=payload).json()["data"]
    assert repeated["id"] == signal["id"]
    assert repeated["duplicate"] is True


def _maintainer(client: TestClient) -> tuple[dict, dict, dict]:
    admin = _headers(_register(client, username="maintenance-admin", role="admin"))
    maintainer = _headers(_register(client, username="maintenance-editor"))
    worker = _headers(_register(client, username="maintenance-worker"))
    assigned = client.post(
        "/api/v1/maintenance/assignments",
        headers=admin,
        json={
            "username": "maintenance-editor",
        },
    )
    assert assigned.status_code == 200, assigned.text
    accepted = client.post(
        f"/api/v1/maintenance/assignments/{assigned.json()['data']['id']}/accept",
        headers=maintainer,
    )
    assert accepted.status_code == 200, accepted.text
    return admin, maintainer, worker


def _reproduction_case(client: TestClient) -> tuple[dict, dict, dict, str, dict]:
    _admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="replay-signal-owner"))
    answer = _gap(client, reporter, "replay-reported-gap")
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=reporter,
        json={
            "answer_id": answer["id"],
            "label": "insufficient_evidence",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    candidate = _gap(client, worker, "worker-replay-input")
    payload = {
        "expected_revision": 2,
        "signal_id": signal["id"],
        "answer_id": candidate["id"],
        "expected_outcome": "insufficient_evidence_reply",
        "confirmed_synthetic_fixture": True,
    }
    return maintainer, worker, reporter, base, payload


@pytest.mark.parametrize("interleaving", ["delete_signal", "change_revision"])
def test_reproduction_revalidates_retained_signal_and_revision_after_execution(client: TestClient, monkeypatch, interleaving: str) -> None:
    from app.service.chat_service import ChatService

    maintainer, worker, reporter, base, payload = _reproduction_case(client)
    original = ChatService.run_chat

    async def execute_then_change(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        if interleaving == "delete_signal":
            changed = await asyncio.to_thread(client.delete, f"/api/v1/knowledge-feedback/{payload['signal_id']}", headers=reporter)
        else:
            changed = await asyncio.to_thread(
                client.post,
                base + "/signals",
                headers=maintainer,
                json={
                    "expected_revision": 2,
                    "signal_ids": [payload["signal_id"]],
                },
            )
        assert changed.status_code == 200
        return result

    monkeypatch.setattr(ChatService, "run_chat", execute_then_change)
    result = client.post(base + "/reproductions", headers=worker, json=payload)
    assert result.status_code == 409, result.text
    assert result.json()["code"] == ("MAINTENANCE_EVIDENCE_REQUIRED" if interleaving == "delete_signal" else "MAINTENANCE_STALE")
    retained = client.get(base, headers=maintainer).json()["data"]
    assert retained["revision"] == (2 if interleaving == "delete_signal" else 3)
    assert retained.get("fixture_identity") is None
    if interleaving == "delete_signal":
        assert retained["signal_count"] == 0


def test_maintenance_context_projects_only_current_accepted_responsibility(client: TestClient) -> None:
    admin, maintainer, worker = _maintainer(client)
    active = client.get("/api/v1/maintenance/context", headers=maintainer)
    assert active.status_code == 200, active.text
    assert active.json()["data"]["is_maintainer"] is True
    assert active.json()["data"]["is_administrator"] is False
    unassigned = client.get("/api/v1/maintenance/context", headers=worker).json()["data"]
    assert unassigned["is_maintainer"] is False
    assert unassigned["assignment"] is None
    assigned = client.post(
        "/api/v1/maintenance/assignments",
        headers=admin,
        json={
            "username": "maintenance-worker",
        },
    ).json()["data"]
    pending = client.get("/api/v1/maintenance/context", headers=worker).json()["data"]
    assert pending["assignment"] == {"id": assigned["id"], "state": "assigned"}
    assert pending["is_maintainer"] is False
    assert client.post(f"/api/v1/maintenance/assignments/{assigned['id']}/accept", headers=worker).status_code == 200
    accepted = client.get("/api/v1/maintenance/context", headers=worker).json()["data"]
    assert accepted["is_maintainer"] is True
    assert accepted["assignment"]["state"] == "accepted"
    administrator = client.get("/api/v1/maintenance/context", headers=admin).json()["data"]
    assert administrator["is_administrator"] is True
    assert administrator["is_maintainer"] is False


def test_reproduction_rechecks_policy_shortened_during_authenticated_execution(client: TestClient, monkeypatch) -> None:
    from app.model.knowledge_feedback import KnowledgeFeedbackSignal
    from app.service.chat_service import ChatService

    maintainer, worker, _reporter, base, payload = _reproduction_case(client)
    admin = _headers(_register(client, username="replay-retention-admin", role="admin"))
    policy = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]

    async def age_signal():
        async with client.app.state.test_auth_session_factory() as session:
            signal = await session.get(KnowledgeFeedbackSignal, payload["signal_id"])
            assert signal is not None
            signal.created_at = datetime.now(UTC) - timedelta(days=2)
            await session.commit()

    asyncio.run(age_signal())
    original = ChatService.run_chat

    async def execute_then_shorten(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        changed = await asyncio.to_thread(
            client.put,
            "/api/v1/retention/policy",
            headers=admin,
            json={
                "expected_identity": policy["identity"],
                "days": {**policy["days"], "feedback_signals": 1},
            },
        )
        assert changed.status_code == 200, changed.text
        return result

    monkeypatch.setattr(ChatService, "run_chat", execute_then_shorten)
    result = client.post(base + "/reproductions", headers=worker, json=payload)
    assert result.status_code == 409, result.text
    assert result.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"
    retained = client.get(base, headers=maintainer).json()["data"]
    assert retained["revision"] == 2
    assert retained.get("fixture_identity") is None


def test_reproduction_inputs_are_owned_executions_and_deletable_non_content_targets(client: TestClient) -> None:
    maintainer, worker, reporter, base, payload = _reproduction_case(client)
    admin = _headers(_register(client, username="reproduction-input-admin", role="admin"))
    for forbidden in (maintainer, reporter, admin):
        assert client.get(base + "/reproduction-inputs", headers=forbidden).status_code == 403
    inputs = client.get(base + "/reproduction-inputs", headers=worker)
    assert inputs.status_code == 200, inputs.text
    choices = inputs.json()["data"]
    assert [target["signal_id"] for target in choices["targets"]] == [payload["signal_id"]]
    assert [answer["answer_id"] for answer in choices["answers"]] == [payload["answer_id"]]
    assert choices["answers"][0]["query_condition_set_identity"] == choices["targets"][0]["query_condition_set_identity"]
    for forbidden in ("uncovered engineering decision", "replay-signal-owner", "replay-reported-gap"):
        assert forbidden not in inputs.text
    fixture = client.post(base + "/reproductions", headers=worker, json=payload).json()["data"]
    summary = client.get(f"/api/v1/maintenance/fixtures/{fixture['id']}", headers=maintainer)
    assert summary.status_code == 200, summary.text
    assert summary.json()["data"]["expected_outcome"] == fixture["expected_outcome"]
    assert "replay_request" not in summary.text
    assert client.get(f"/api/v1/maintenance/fixtures/{fixture['id']}", headers=reporter).status_code == 403
    assert client.delete(f"/api/v1/knowledge-feedback/{payload['signal_id']}", headers=reporter).status_code == 200
    assert client.get(base + "/reproduction-inputs", headers=worker).json()["data"]["targets"] == []


@pytest.mark.parametrize("corruption", ["expected_outcome", "replay_request", "unknown_field"])
def test_fixture_projection_rejects_malformed_or_rebound_retained_evidence(client: TestClient, corruption: str) -> None:
    from sqlalchemy import update

    from app.model.canonical import CanonicalRecordModel

    maintainer, worker, _reporter, base, payload = _reproduction_case(client)
    fixture = client.post(base + "/reproductions", headers=worker, json=payload).json()["data"]

    async def corrupt_fixture():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, fixture["id"])
            assert record is not None
            if corruption == "replay_request":
                change = {"replay_request": {"message": "different private scenario", "query_conditions": []}}
            elif corruption == "expected_outcome":
                change = {"expected_outcome": "private fixture payload"}
            else:
                change = {"unknown_field": "private fixture payload"}
            connection = await session.connection()
            await connection.execute(
                update(CanonicalRecordModel.__table__)
                .where(
                    CanonicalRecordModel.stable_id == fixture["id"],
                )
                .values(payload={**record.payload, **change})
            )
            await session.commit()

    asyncio.run(corrupt_fixture())
    projection = client.get(f"/api/v1/maintenance/fixtures/{fixture['id']}", headers=maintainer)
    assert projection.status_code == 409, projection.text
    assert projection.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"
    assert "private fixture payload" not in projection.text
    diagnosis = client.post(
        base + "/diagnosis",
        headers=maintainer,
        json={
            "expected_revision": 3,
            "fixture_identity": fixture["id"],
            "observation": "coverage_gap",
        },
    )
    assert diagnosis.status_code == 409, diagnosis.text


def test_replay_failure_cannot_be_rewritten_as_success(client: TestClient) -> None:
    from sqlalchemy import update

    from app.model.canonical import CanonicalRecordModel

    maintainer, worker, _reporter, base, payload = _reproduction_case(client)
    fixture = client.post(
        base + "/reproductions",
        headers=worker,
        json={
            **payload,
            "expected_outcome": "evidence_gated_answer",
        },
    ).json()["data"]
    assert (
        client.post(
            base + "/diagnosis",
            headers=maintainer,
            json={
                "expected_revision": 3,
                "fixture_identity": fixture["id"],
                "observation": "coverage_gap",
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            base + "/findings",
            headers=maintainer,
            json={
                "expected_revision": 4,
                "fixture_identity": fixture["id"],
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            base + "/transition",
            headers=maintainer,
            json={
                "expected_revision": 5,
                "state": "in_progress",
            },
        ).status_code
        == 200
    )
    result = client.post(
        base + "/replays",
        headers=worker,
        json={
            "expected_revision": 6,
            "fixture_identity": fixture["id"],
            "answer_id": payload["answer_id"],
        },
    )
    assert result.status_code == 200, result.text
    replay = result.json()["data"]
    assert replay["passed"] is False
    assert replay["observed_outcome"] == "insufficient_evidence_reply"
    assert replay["expected_outcome"] == "evidence_gated_answer"
    assert (
        client.post(
            base + "/resolution",
            headers=maintainer,
            json={
                "expected_revision": 7,
                "disposition": "boundary-query",
                "artifact_identities": [fixture["id"], replay["id"]],
            },
        ).status_code
        == 409
    )

    async def rewrite_result():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, replay["id"])
            assert record is not None
            connection = await session.connection()
            await connection.execute(
                update(CanonicalRecordModel.__table__)
                .where(
                    CanonicalRecordModel.stable_id == replay["id"],
                )
                .values(payload={**record.payload, "passed": True, "observed_outcome": "evidence_gated_answer"})
            )
            await session.commit()

    asyncio.run(rewrite_result())
    retained = client.get(f"/api/v1/maintenance/replays/{replay['id']}", headers=maintainer)
    assert retained.status_code == 409, retained.text
    assert retained.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"


def test_helpful_confirmation_does_not_automatically_create_work(client: TestClient) -> None:
    admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    submitted = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "helpful",
        },
    )
    assert submitted.status_code == 200
    assert client.get("/api/v1/maintenance/items", headers=maintainer).json()["data"]["items"] == []
    queue = client.get("/api/v1/knowledge-review-queue", headers=admin)
    assert queue.status_code == 200, queue.text
    assert not any(item["kind"] == "feedback_signal" for item in queue.json()["items"])


def test_maintainer_creates_durable_item_without_copying_private_signal(client: TestClient) -> None:
    admin, maintainer, _worker = _maintainer(client)
    owner = _headers(_register(client, username="maintenance-reporting-user"))
    answer = _gap(client, owner)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=owner,
        json={
            "answer_id": answer["id"],
            "label": "out_of_scope",
            "note": "optional private description",
        },
    ).json()["data"]
    payload = {
        "classification": "coverage-gap",
        "severity": "p3",
        "disposition": "needs-reproduction",
        "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
        "work_owner_username": "maintenance-worker",
        "signal_ids": [signal["id"]],
    }
    assert client.post("/api/v1/maintenance/items", headers=owner, json=payload).status_code == 403
    assert client.post("/api/v1/maintenance/items", headers=admin, json=payload).status_code == 403
    created = client.post("/api/v1/maintenance/items", headers=maintainer, json=payload)
    assert created.status_code == 200, created.text
    item = created.json()["data"]
    assert item["state"] == "open"
    assert item["classification"] == "coverage-gap"
    assert item["disposition"] == "needs-reproduction"
    assert item["signal_count"] == 1
    assert item["verified_pattern"] is None
    for forbidden in (answer["id"], "optional private description", "maintenance-reporting-user", signal["created_at"]):
        assert forbidden not in created.text
    reloaded = client.get(f"/api/v1/maintenance/items/{item['id']}", headers=maintainer)
    assert reloaded.json()["data"] == item


def test_feedback_deletion_cannot_claim_success_when_maintenance_link_survives(client: TestClient) -> None:
    _admin, maintainer, _worker = _maintainer(client)
    owner = _headers(_register(client, username="maintenance-delete-owner"))
    answer = _gap(client, owner)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=owner,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
        },
    ).json()["data"]
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "source-freshness",
            "severity": "p2",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    )
    assert created.status_code == 200, created.text
    identity = created.json()["data"]["id"]

    async def database_fault(sql: str):
        async with client.app.state.test_auth_session_factory() as session:
            await session.execute(text(sql))
            await session.commit()

    asyncio.run(
        database_fault(
            "CREATE TRIGGER ignore_maintenance_link_delete BEFORE DELETE ON maintenance_signal_links BEGIN SELECT RAISE(IGNORE); END"
        )
    )
    failed = client.delete(f"/api/v1/knowledge-feedback/{signal['id']}", headers=owner)
    assert failed.status_code == 503
    assert failed.json()["code"] == "PRIVACY_DELETE_UNVERIFIED"
    assert len(client.get("/api/v1/knowledge-feedback", headers=owner).json()["items"]) == 1
    asyncio.run(database_fault("DROP TRIGGER ignore_maintenance_link_delete"))
    assert client.delete(f"/api/v1/knowledge-feedback/{signal['id']}", headers=owner).json()["deleted"] is True
    durable = client.get(f"/api/v1/maintenance/items/{identity}", headers=maintainer).json()["data"]
    assert durable["signal_count"] == 0
    assert durable["classification"] == "source-freshness"


def test_independent_expiry_detects_and_repairs_orphan_maintenance_links(client: TestClient) -> None:
    from app.retention.cleanup import run_retention_sweep

    admin, maintainer, _worker = _maintainer(client)
    owner = _headers(_register(client, username="maintenance-expiry-owner"))
    answer = _gap(client, owner)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=owner,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
        },
    ).json()["data"]
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "source-freshness",
            "severity": "p2",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    )
    assert created.status_code == 200
    factory = client.app.state.test_auth_session_factory

    async def fault():
        async with factory() as session:
            await session.execute(
                text("CREATE TRIGGER ignore_expired_link BEFORE DELETE ON maintenance_signal_links BEGIN SELECT RAISE(IGNORE); END")
            )
            await session.commit()

    asyncio.run(fault())
    future = datetime.now(UTC) + timedelta(days=181)
    failed = asyncio.run(run_retention_sweep(factory, now=future))
    assert failed["feedback_signals"]["status"] == "failed"

    async def repair_with_missing_raw_signal():
        async with factory() as session:
            await session.execute(text("DELETE FROM knowledge_feedback_signals"))
            await session.execute(text("DROP TRIGGER ignore_expired_link"))
            await session.commit()

    asyncio.run(repair_with_missing_raw_signal())
    retried = client.post("/api/v1/retention/cleanup", headers=admin).json()["data"]
    assert retried["cleanup"]["feedback_signals"]["status"] == "verified"
    durable = client.get(f"/api/v1/maintenance/items/{created.json()['data']['id']}", headers=maintainer).json()["data"]
    assert durable["signal_count"] == 0
    assert durable["classification"] == "source-freshness"


@pytest.mark.parametrize(
    "classification",
    [
        "confirmation",
        "content-integrity",
        "source-freshness",
        "coverage-gap",
        "retrieval-answer-behavior",
        "product-privacy-operations",
        "scope-roadmap",
    ],
)
def test_one_classification_and_append_only_triage_reject_stale_commands(client: TestClient, classification: str) -> None:
    admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "insufficient_evidence",
        },
    ).json()["data"]
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": classification,
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    )
    assert created.status_code == 200, created.text
    item = created.json()["data"]
    path = f"/api/v1/maintenance/items/{item['id']}/transition"
    assert client.post(path, headers=worker, json={"expected_revision": 1, "state": "triaged"}).status_code == 403
    assert client.post(path, headers=maintainer, json={"expected_revision": 1, "state": "resolved"}).status_code == 409
    revision = 1
    if classification in {"retrieval-answer-behavior", "product-privacy-operations"}:
        required = client.post(path, headers=maintainer, json={"expected_revision": revision, "state": "triaged"})
        assert required.status_code == 409
        assert required.json()["code"] == "MAINTENANCE_ADMINISTRATOR_REQUIRED"
        join_path = f"/api/v1/maintenance/items/{item['id']}/administrator"
        assert client.post(join_path, headers=worker, json={"expected_revision": 1}).status_code == 403
        joined = client.post(join_path, headers=admin, json={"expected_revision": 1})
        assert joined.status_code == 200, joined.text
        revision = 2
        assert client.get("/api/v1/maintenance/inbox", headers=admin).status_code == 403
        assert client.get("/api/v1/sessions/maintenance-gap", headers=admin).json()["messages"] == []
    triaged = client.post(path, headers=maintainer, json={"expected_revision": revision, "state": "triaged"})
    assert triaged.status_code == 200, triaged.text
    assert triaged.json()["data"]["revision"] == revision + 1
    assert triaged.json()["data"]["classification"] == classification
    assert triaged.json()["data"]["state"] == "triaged"
    stale = client.post(path, headers=maintainer, json={"expected_revision": 1, "state": "triaged"})
    assert stale.status_code == 409
    assert stale.json()["code"] == "MAINTENANCE_STALE"
    assert client.get(f"/api/v1/maintenance/items/{item['id']}", headers=worker).json()["data"]["state"] == "triaged"


def test_maintainer_inbox_and_consolidation_keep_raw_envelopes_separate(client: TestClient) -> None:
    _admin, maintainer, worker = _maintainer(client)
    signals = []
    for index in range(2):
        answer = _gap(client, worker, f"consolidation-{index}")
        signals.append(
            client.post(
                "/api/v1/knowledge-feedback",
                headers=worker,
                json={
                    "answer_id": answer["id"],
                    "label": "out_of_scope",
                },
            ).json()["data"]
        )
    assert client.get("/api/v1/maintenance/inbox", headers=worker).status_code == 403
    inbox = client.get("/api/v1/maintenance/inbox", headers=maintainer)
    assert inbox.status_code == 200, inbox.text
    assert {signal["id"] for signal in inbox.json()["data"]["signals"]} == {signal["id"] for signal in signals}
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "scope-roadmap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signals[0]["id"]],
        },
    ).json()["data"]
    combined = client.post(
        f"/api/v1/maintenance/items/{item['id']}/signals",
        headers=maintainer,
        json={
            "expected_revision": 1,
            "signal_ids": [signals[0]["id"], signals[1]["id"]],
        },
    )
    assert combined.status_code == 200, combined.text
    assert combined.json()["data"]["signal_count"] == 2
    assert combined.json()["data"]["revision"] == 2
    assert all(signal["id"] not in combined.text for signal in signals)
    listed = client.get("/api/v1/maintenance/items", headers=maintainer).json()["data"]["items"]
    assert [entry["id"] for entry in listed] == [item["id"]]


@pytest.mark.parametrize("severity", ["p0", "p1"])
def test_high_severity_rejects_entry_containment_for_an_unbound_gap(client: TestClient, severity: str) -> None:
    from tests.integration.test_delivery_acceptance import _active_local_record

    admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
        },
    ).json()["data"]
    record = _active_local_record(client, admin)
    payload = {
        "classification": "content-integrity",
        "severity": severity,
        "disposition": "needs-reproduction",
        "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
        "work_owner_username": "maintenance-worker",
        "signal_ids": [signal["id"]],
        "containment_record_identity": record["record_id"],
    }
    assert client.post("/api/v1/maintenance/items", headers=maintainer, json=payload).status_code == 409
    suspended = client.post(
        f"/api/v1/acceptance/records/{record['record_id']}/status",
        headers=admin,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "verified_entry_integrity",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
                "evidence_links": ["evidence://maintenance/verified-entry-integrity"],
            },
        },
    )
    assert suspended.status_code == 200, suspended.text
    created = client.post("/api/v1/maintenance/items", headers=maintainer, json=payload)
    assert created.status_code == 409, created.text
    assert client.get("/api/v1/maintenance/items", headers=maintainer).json()["data"]["items"] == []


def test_independent_authenticated_reproduction_checks_conditions_before_diagnosis(client: TestClient) -> None:
    _admin, maintainer, worker = _maintainer(client)
    owner = _headers(_register(client, username="diagnosis-reporting-user"))
    answer = _gap(client, owner, "reported-gap")
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=owner,
        json={
            "answer_id": answer["id"],
            "label": "insufficient_evidence",
            "note": "private reported details",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    triaged = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"})
    assert triaged.status_code == 200
    reproduced = _gap(client, worker, "independent-reproduction")
    request = {
        "expected_revision": 2,
        "signal_id": signal["id"],
        "answer_id": reproduced["id"],
        "expected_outcome": "insufficient_evidence_reply",
        "confirmed_synthetic_fixture": True,
    }
    denied = client.post(base + "/reproductions", headers=owner, json=request)
    assert denied.status_code == 403
    for unconfirmed in (1, "true", False):
        rejected = client.post(
            base + "/reproductions",
            headers=worker,
            json={
                **request,
                "confirmed_synthetic_fixture": unconfirmed,
            },
        )
        assert rejected.status_code == 422
    unsupported_diagnosis = client.post(
        base + "/reproductions",
        headers=worker,
        json={
            **request,
            "verified_observation": "retrieval_miss",
        },
    )
    assert unsupported_diagnosis.status_code == 409
    assert unsupported_diagnosis.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"
    prior_sessions = {session["session_id"] for session in client.get("/api/v1/sessions", headers=worker).json()["data"]["sessions"]}
    response = client.post(base + "/reproductions", headers=worker, json=request)
    assert response.status_code == 200, response.text
    fixture = response.json()["data"]
    current_sessions = {session["session_id"] for session in client.get("/api/v1/sessions", headers=worker).json()["data"]["sessions"]}
    replay_sessions = current_sessions - prior_sessions
    assert len(replay_sessions) == 1
    replay_history = client.get(f"/api/v1/sessions/{replay_sessions.pop()}", headers=worker).json()["data"]["messages"]
    replayed = next(message["answer_execution"] for message in reversed(replay_history) if message["type"] == "assistant")
    assert fixture["verification_method"] == "independent_authenticated_replay"
    assert fixture["observed_outcome"] == replayed["outcome"]
    assert fixture["query_condition_set_identity"] == replayed["query_condition_set"]["identity"]
    assert fixture["query_condition_set_identity"] == signal["query_condition_set_identity"]
    assert fixture["observed_outcome"] == "insufficient_evidence_reply"
    assert fixture["active_publication_identities"] == []
    assert "replay_request" not in fixture
    assert "uncovered engineering decision" not in response.text
    assert "private reported details" not in response.text
    assert "diagnosis-reporting-user" not in response.text
    diagnosis = client.post(
        base + "/diagnosis",
        headers=maintainer,
        json={
            "expected_revision": 3,
            "fixture_identity": fixture["id"],
            "observation": "coverage_gap",
        },
    )
    assert diagnosis.status_code == 200, diagnosis.text
    assert diagnosis.json()["data"]["disposition"] == "coverage-work"
    assert diagnosis.json()["data"]["classification"] == "coverage-gap"
    started = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 4, "state": "in_progress"})
    assert started.status_code == 200, started.text


def test_validated_finding_requires_independent_approval_and_survives_signal_deletion(client: TestClient) -> None:
    _admin, maintainer, worker = _maintainer(client)
    owner = _headers(_register(client, username="finding-reporting-user"))
    answer = _gap(client, owner, "finding-report")
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=owner,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
            "note": "private finding description",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    reproduced = _gap(client, worker, "finding-reproduction")
    fixture = client.post(
        base + "/reproductions",
        headers=worker,
        json={
            "expected_revision": 2,
            "signal_id": signal["id"],
            "answer_id": reproduced["id"],
            "expected_outcome": "insufficient_evidence_reply",
            "confirmed_synthetic_fixture": True,
        },
    ).json()["data"]
    assert (
        client.post(
            base + "/diagnosis",
            headers=maintainer,
            json={
                "expected_revision": 3,
                "fixture_identity": fixture["id"],
                "observation": "coverage_gap",
            },
        ).status_code
        == 200
    )
    payload = {"expected_revision": 4, "fixture_identity": fixture["id"]}
    assert client.post(base + "/findings", headers=worker, json=payload).status_code == 403
    assert client.post(base + "/findings", headers=maintainer, json={**payload, "question": "copied private content"}).status_code == 422
    verified = client.post(base + "/findings", headers=maintainer, json=payload)
    assert verified.status_code == 200, verified.text
    finding = verified.json()["data"]
    assert finding["pattern"]["observation"] == "coverage_gap"
    for forbidden in (
        "uncovered engineering decision",
        "private finding description",
        "finding-reporting-user",
        answer["id"],
        signal["created_at"],
    ):
        assert forbidden not in verified.text
    assert client.delete(f"/api/v1/knowledge-feedback/{signal['id']}", headers=owner).json()["deleted"] is True
    retained = client.get(f"/api/v1/maintenance/findings/{finding['id']}", headers=maintainer)
    assert retained.json()["data"] == finding
    durable = client.get(base, headers=maintainer).json()["data"]
    assert durable["signal_count"] == 0
    assert durable["verified_pattern"] == finding["pattern"]
    started = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"})
    assert started.status_code == 200
    premature = client.post(
        base + "/resolution",
        headers=maintainer,
        json={"expected_revision": 6, "disposition": "boundary-query", "artifact_identities": [fixture["id"]]},
    )
    assert premature.status_code == 409, premature.text
    replay_payload = {"expected_revision": 6, "fixture_identity": fixture["id"], "answer_id": reproduced["id"]}
    assert client.post(base + "/replays", headers=maintainer, json=replay_payload).status_code == 403
    assert (
        client.post(
            base + "/replays",
            headers=worker,
            json={
                **replay_payload,
                "expected_outcome": "evidence_gated_answer",
            },
        ).status_code
        == 422
    )
    replay_response = client.post(base + "/replays", headers=worker, json=replay_payload)
    assert replay_response.status_code == 200, replay_response.text
    replay = replay_response.json()["data"]
    assert replay["fixture_identity"] == fixture["id"]
    assert replay["expected_outcome"] == fixture["expected_outcome"]
    assert replay["query_condition_set_identity"] == fixture["query_condition_set_identity"]
    assert replay["observed_outcome"] == "insufficient_evidence_reply"
    assert replay["observed_state"] == "completed"
    assert replay["passed"] is True
    assert "uncovered engineering decision" not in replay_response.text
    assert client.get(f"/api/v1/maintenance/replays/{replay['id']}", headers=maintainer).json()["data"] == replay
    assert client.get(f"/api/v1/maintenance/replays/{replay['id']}", headers=owner).status_code == 403
    assert client.post(base + "/replays", headers=worker, json=replay_payload).status_code == 409
    resolved = client.post(
        base + "/resolution",
        headers=maintainer,
        json={
            "expected_revision": 7,
            "disposition": "boundary-query",
            "artifact_identities": [fixture["id"], replay["id"]],
        },
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["data"]["state"] == "resolved"
    assert resolved.json()["data"]["result_links"] == [
        f"evidence://maintenance/artifacts/{fixture['id']}",
        f"evidence://maintenance/artifacts/{replay['id']}",
    ]
    closed = client.post(
        base + "/transition",
        headers=maintainer,
        json={
            "expected_revision": 8,
            "state": "closed_confirmation",
        },
    )
    assert closed.status_code == 200, closed.text
    assert closed.json()["data"]["state"] == "closed_confirmation"
    assert (
        client.post(
            base + "/transition",
            headers=maintainer,
            json={
                "expected_revision": 9,
                "state": "triaged",
            },
        ).status_code
        == 409
    )


def test_roadmap_counts_distinct_executions_not_members_or_outcome_kinds(client: TestClient) -> None:
    _admin, maintainer, worker = _maintainer(client)
    signals = []
    for index in range(3):
        answer = _gap(client, worker, f"roadmap-{index}")
        signals.append(
            client.post(
                "/api/v1/knowledge-feedback",
                headers=worker,
                json={
                    "answer_id": answer["id"],
                    "label": "insufficient_evidence",
                },
            ).json()["data"]
        )
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"] for signal in signals[:2]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    request = {
        "expected_revision": 2,
        "owner_username": "maintenance-worker",
        "desired_outcome": "expand_coverage",
        "bounded_work_reason": "requires_separate_scope",
        "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
    }
    too_few = client.post(base + "/roadmap", headers=maintainer, json=request)
    assert too_few.status_code == 409
    assert too_few.json()["code"] == "ROADMAP_THRESHOLD_NOT_MET"
    consolidated = client.post(
        base + "/signals",
        headers=maintainer,
        json={
            "expected_revision": 2,
            "signal_ids": [signal["id"] for signal in signals],
        },
    )
    assert consolidated.status_code == 200
    qualified = client.post(base + "/roadmap", headers=maintainer, json={**request, "expected_revision": 3})
    assert qualified.status_code == 200, qualified.text
    candidate = qualified.json()["data"]
    assert candidate["qualification"]["distinct_executions_30_days"] == 3
    assert candidate["qualification"]["independent_findings"] == 0
    assert candidate["state"] == "deferred"
    for signal in signals:
        assert signal["id"] not in qualified.text
        assert signal["answer_id"] not in qualified.text
        assert signal["created_at"] not in qualified.text
    assert "maintenance-worker" not in qualified.text
    assert client.get(base, headers=maintainer).json()["data"]["state"] == "deferred"


@pytest.mark.parametrize("action", ["renew", "close", "start_wayfinder"])
def test_monthly_roadmap_owner_can_renew_close_or_start_a_separate_map(client: TestClient, action: str) -> None:
    _admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "out_of_scope",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "scope-roadmap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    candidate = client.post(
        base + "/roadmap",
        headers=maintainer,
        json={
            "expected_revision": 2,
            "owner_username": "maintenance-worker",
            "desired_outcome": "clarify_scope",
            "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    ).json()["data"]
    path = f"/api/v1/maintenance/roadmap/{candidate['id']}"
    assert client.get(path, headers=worker).json()["data"] == candidate
    review = {
        "expected_revision": 1,
        "action": action,
        "rationale": "separate_discovery_needed"
        if action == "start_wayfinder"
        else "still_outside_scope"
        if action == "renew"
        else "no_longer_needed",
        "owner_username": "maintenance-worker",
        "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat() if action == "renew" else None,
    }
    denied = client.post(path + "/review", headers=maintainer, json=review)
    assert denied.status_code == 403
    response = client.post(path + "/review", headers=worker, json=review)
    assert response.status_code == 200, response.text
    reviewed = response.json()["data"]
    assert reviewed["revision"] == 2
    assert reviewed["state"] == {"renew": "deferred", "close": "closed", "start_wayfinder": "mapped"}[action]
    assert client.post(path + "/review", headers=worker, json=review).status_code == 409
    if action == "start_wayfinder":
        artifact = client.get(f"/api/v1/maintenance/maps/{reviewed['map_identity']}", headers=worker)
        assert artifact.status_code == 200, artifact.text
        assert artifact.json()["data"]["markdown"].startswith("# Wayfinder:")
        assert "Status: open" in artifact.json()["data"]["markdown"]
        assert "uncovered engineering decision" not in artifact.text


def _scope_roadmap(client: TestClient) -> tuple[dict, dict, dict, dict, dict]:
    admin, maintainer, worker = _maintainer(client)
    owner = _headers(_register(client, username="roadmap-independent-owner"))
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "out_of_scope",
        },
    ).json()["data"]
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "scope-roadmap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    qualified = client.post(
        base + "/roadmap",
        headers=maintainer,
        json={
            "expected_revision": 2,
            "owner_username": "roadmap-independent-owner",
            "desired_outcome": "clarify_scope",
            "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    )
    assert qualified.status_code == 200, qualified.text
    return admin, maintainer, worker, owner, qualified.json()["data"]


@pytest.mark.parametrize("corruption", ["private_field", "foreign_actor", "revision_gap", "wrong_from_state"])
def test_roadmap_history_rejects_unknown_content_and_unqualified_decisions(client: TestClient, corruption: str) -> None:
    from app.model.canonical import CanonicalEventModel

    _admin, maintainer, _worker, owner, candidate = _scope_roadmap(client)
    changes = {
        "state": "deferred",
        "owner_identity": candidate["owner_identity"],
        "rationale": "still_outside_scope",
        "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
    }
    if corruption == "private_field":
        changes["description"] = "private roadmap description"
    maintainer_identity = client.get("/api/v1/maintenance/context", headers=maintainer).json()["data"]["member_identity"]

    async def corrupt_trail():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                CanonicalEventModel(
                    aggregate_id=candidate["id"],
                    aggregate_kind="maintenance_item",
                    event_type="roadmap_reviewed",
                    from_state="open" if corruption == "wrong_from_state" else "deferred",
                    to_state="deferred",
                    recorded_by=maintainer_identity if corruption == "foreign_actor" else candidate["owner_identity"],
                    payload={"schema": "roadmap_review/v1", "revision": 3 if corruption == "revision_gap" else 2, "changes": changes},
                )
            )
            await session.commit()

    asyncio.run(corrupt_trail())
    response = client.get(f"/api/v1/maintenance/roadmap/{candidate['id']}", headers=owner)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "MAINTENANCE_EVENT_INVALID"
    assert "private roadmap description" not in response.text


def test_roadmap_listing_follows_current_ownership_without_exposing_foreign_work(client: TestClient) -> None:
    _admin, maintainer, _worker, owner, candidate = _scope_roadmap(client)
    successor = _headers(_register(client, username="roadmap-next-owner"))
    assert client.get("/api/v1/maintenance/items", headers=owner).json()["data"]["items"] == []
    listing = client.get("/api/v1/maintenance/roadmap", headers=owner)
    assert listing.status_code == 200, listing.text
    assert listing.json()["data"]["candidates"] == [candidate]
    assert client.get("/api/v1/maintenance/roadmap", headers=successor).json()["data"]["candidates"] == []
    changed = client.post(
        f"/api/v1/maintenance/roadmap/{candidate['id']}/review",
        headers=owner,
        json={
            "expected_revision": 1,
            "action": "renew",
            "rationale": "still_outside_scope",
            "owner_username": "roadmap-next-owner",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    )
    assert changed.status_code == 200, changed.text
    assert client.get("/api/v1/maintenance/roadmap", headers=owner).json()["data"]["candidates"] == []
    for reader in (successor, maintainer):
        retained = client.get("/api/v1/maintenance/roadmap", headers=reader)
        assert retained.json()["data"]["candidates"] == [changed.json()["data"]]
        for forbidden in ("uncovered engineering decision", "answer_id", "signal_id", "submitted_at"):
            assert forbidden not in retained.text


def test_wayfinder_map_requires_exact_qualifying_monthly_decision(client: TestClient) -> None:
    from sqlalchemy import update

    from app.model.canonical import CanonicalRecordModel

    _admin, _maintainer, _worker, owner, candidate = _scope_roadmap(client)
    forged_id = "maintenance_item:" + "f" * 32

    async def add_unqualified_map():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                CanonicalRecordModel(
                    stable_id=forged_id,
                    identity_kind="maintenance_item",
                    identity_value="f" * 32,
                    state="open",
                    record_class="authoritative",
                    payload={
                        "schema": "maintenance_wayfinder_map/v1",
                        "candidate_identity": candidate["id"],
                        "markdown": "private map content",
                    },
                )
            )
            await session.commit()

    asyncio.run(add_unqualified_map())
    unqualified = client.get(f"/api/v1/maintenance/maps/{forged_id}", headers=owner)
    assert unqualified.status_code == 409, unqualified.text
    assert "private map content" not in unqualified.text
    mapped = client.post(
        f"/api/v1/maintenance/roadmap/{candidate['id']}/review",
        headers=owner,
        json={
            "expected_revision": 1,
            "action": "start_wayfinder",
            "rationale": "separate_discovery_needed",
            "owner_username": "roadmap-independent-owner",
        },
    )
    assert mapped.status_code == 200, mapped.text
    identity = mapped.json()["data"]["map_identity"]
    assert client.get(f"/api/v1/maintenance/maps/{identity}", headers=owner).status_code == 200

    async def replace_map_content():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, identity)
            assert record is not None
            connection = await session.connection()
            await connection.execute(
                update(CanonicalRecordModel.__table__)
                .where(
                    CanonicalRecordModel.stable_id == identity,
                )
                .values(payload={**record.payload, "markdown": "private map content"})
            )
            await session.commit()

    asyncio.run(replace_map_content())
    replaced = client.get(f"/api/v1/maintenance/maps/{identity}", headers=owner)
    assert replaced.status_code == 409, replaced.text
    assert "private map content" not in replaced.text


def test_cadence_records_weekly_monthly_quarterly_and_seven_day_triage(client: TestClient, monkeypatch) -> None:
    from tests.integration.test_delivery_acceptance import _active_local_record

    admin, maintainer, worker = _maintainer(client)
    acceptance = _active_local_record(client, admin)
    answer = _gap(client, worker)
    client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
        },
    )
    for period in ("weekly", "monthly", "quarterly"):
        response = client.post(
            "/api/v1/maintenance/cadence",
            headers=maintainer,
            json={
                "period": period,
                "item_revisions": {},
                "context_sha256": client.get("/api/v1/maintenance/review-context", headers=maintainer).json()["data"]["context_sha256"],
                "sample_acceptance_identities": [acceptance["record_id"]] if period == "quarterly" else [],
                "evidence_links": [f"evidence://maintenance/{period}-review"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["period"] == period
    current = client.get("/api/v1/maintenance/dashboard", headers=maintainer).json()["data"]
    assert current["overdue_triage_count"] == 0
    assert current["cadence_due"] == {"weekly": False, "monthly": False, "quarterly": False}
    from app.maintenance import cadence

    future = datetime.now(UTC) + timedelta(days=8)
    monkeypatch.setattr(cadence, "utcnow", lambda: future)
    overdue = client.get("/api/v1/maintenance/dashboard", headers=maintainer).json()["data"]
    assert overdue["overdue_triage_count"] == 1
    assert overdue["cadence_due"] == {"weekly": True, "monthly": False, "quarterly": False}


def test_cadence_excludes_logically_expired_signals_before_cleanup(client: TestClient, monkeypatch) -> None:
    from app.maintenance import cadence

    admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    response = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={"answer_id": answer["id"], "label": "outdated"},
    )
    assert response.status_code == 200, response.text
    policy = client.get("/api/v1/retention", headers=admin).json()["data"]["policy"]
    changed = client.put(
        "/api/v1/retention/policy",
        headers=admin,
        json={"expected_identity": policy["identity"], "days": {**policy["days"], "feedback_signals": 10}},
    )
    assert changed.status_code == 200, changed.text
    now = datetime.now(UTC)
    monkeypatch.setattr(cadence, "utcnow", lambda: now + timedelta(days=8))
    assert client.get("/api/v1/maintenance/dashboard", headers=maintainer).json()["data"]["overdue_triage_count"] == 1
    monkeypatch.setattr(cadence, "utcnow", lambda: now + timedelta(days=11))
    result = client.get("/api/v1/maintenance/dashboard", headers=maintainer)
    assert result.status_code == 200, result.text
    assert result.json()["data"]["overdue_triage_count"] == 0


def test_cadence_freezes_health_deferrals_and_exact_sampled_acceptance(client: TestClient) -> None:
    from tests.integration.test_delivery_acceptance import _active_local_record

    admin, maintainer, worker, owner, candidate = _scope_roadmap(client)
    acceptance = _active_local_record(client, admin)
    for forbidden in (admin, worker, owner):
        assert client.get("/api/v1/maintenance/review-context", headers=forbidden).status_code == 403
    initial = client.get("/api/v1/maintenance/review-context", headers=maintainer)
    assert initial.status_code == 200, initial.text
    context = initial.json()["data"]
    assert context["deferrals"][0]["id"] == candidate["id"]
    assert context["knowledge_health"]["published_entries"] == []
    sample = context["sample_options"][0]
    assert sample["record_identity"] == acceptance["record_id"]
    assert sample["status_event_id"] == acceptance["status_history"][-1]["event_id"]
    assert sample["accepted_scope"] == acceptance["accepted_scope"]
    assert "checks" not in sample
    assert (
        client.post(
            f"/api/v1/maintenance/roadmap/{candidate['id']}/review",
            headers=owner,
            json={
                "expected_revision": 1,
                "action": "renew",
                "rationale": "still_outside_scope",
                "owner_username": "roadmap-independent-owner",
                "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
            },
        ).status_code
        == 200
    )
    payload = {
        "period": "monthly",
        "item_revisions": context["item_revisions"],
        "context_sha256": context["context_sha256"],
        "evidence_links": ["evidence://maintenance/monthly-review"],
    }
    stale = client.post("/api/v1/maintenance/cadence", headers=maintainer, json=payload)
    assert stale.status_code == 409, stale.text
    current = client.get("/api/v1/maintenance/review-context", headers=maintainer).json()["data"]
    recorded = client.post(
        "/api/v1/maintenance/cadence",
        headers=maintainer,
        json={
            **payload,
            "context_sha256": current["context_sha256"],
        },
    )
    assert recorded.status_code == 200, recorded.text
    snapshot = recorded.json()["data"]["review_snapshot"]
    assert snapshot["deferrals"][0]["revision"] == 2
    assert snapshot["knowledge_health"] == current["knowledge_health"]
    assert (
        client.post(
            f"/api/v1/maintenance/roadmap/{candidate['id']}/review",
            headers=owner,
            json={
                "expected_revision": 2,
                "action": "close",
                "rationale": "no_longer_needed",
                "owner_username": "roadmap-independent-owner",
            },
        ).status_code
        == 200
    )
    later = client.get("/api/v1/maintenance/dashboard", headers=maintainer).json()["data"]
    assert later["cadence_records"][0]["review_snapshot"] == snapshot
    quarter_context = client.get("/api/v1/maintenance/review-context", headers=maintainer).json()["data"]
    quarterly = client.post(
        "/api/v1/maintenance/cadence",
        headers=maintainer,
        json={
            "period": "quarterly",
            "item_revisions": quarter_context["item_revisions"],
            "context_sha256": quarter_context["context_sha256"],
            "sample_acceptance_identities": [sample["record_identity"]],
            "evidence_links": ["evidence://maintenance/quarterly-review"],
        },
    )
    assert quarterly.status_code == 200, quarterly.text
    assert quarterly.json()["data"]["sampled_acceptances"] == [sample]


def test_cadence_evidence_reference_cannot_retain_private_description(client: TestClient) -> None:
    _admin, maintainer, _worker = _maintainer(client)
    rejected = client.post(
        "/api/v1/maintenance/cadence",
        headers=maintainer,
        json={
            "period": "weekly",
            "item_revisions": {},
            "context_sha256": client.get("/api/v1/maintenance/review-context", headers=maintainer).json()["data"]["context_sha256"],
            "evidence_links": ["evidence://maintenance/private-user-description"],
        },
    )
    assert rejected.status_code == 422
    dashboard = client.get("/api/v1/maintenance/dashboard", headers=maintainer)
    assert dashboard.json()["data"]["cadence_records"] == []
    assert "private-user-description" not in dashboard.text


def test_cadence_requires_one_unambiguous_recording_event(client: TestClient) -> None:
    from app.model.canonical import CanonicalEventModel

    _admin, maintainer, _worker = _maintainer(client)
    context = client.get("/api/v1/maintenance/review-context", headers=maintainer).json()["data"]
    recorded = client.post(
        "/api/v1/maintenance/cadence",
        headers=maintainer,
        json={
            "period": "weekly",
            "item_revisions": {},
            "context_sha256": context["context_sha256"],
            "evidence_links": ["evidence://maintenance/weekly-review"],
        },
    )
    assert recorded.status_code == 200, recorded.text

    async def append_duplicate():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                CanonicalEventModel(
                    aggregate_id=recorded.json()["data"]["id"],
                    aggregate_kind="maintenance_item",
                    event_type="cadence_reviewed",
                    to_state="reviewed",
                    recorded_by=context["maintainer_identity"],
                    payload={"schema": "maintenance_cadence_event/v1", "record_sha256": "0" * 64},
                )
            )
            await session.commit()

    asyncio.run(append_duplicate())
    dashboard = client.get("/api/v1/maintenance/dashboard", headers=maintainer)
    assert dashboard.status_code == 409, dashboard.text
    assert dashboard.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"


def test_cadence_due_is_scoped_to_the_accountable_maintainer(client: TestClient) -> None:
    admin, maintainer, worker = _maintainer(client)
    assigned = client.post(
        "/api/v1/maintenance/assignments",
        headers=admin,
        json={
            "username": "maintenance-worker",
        },
    ).json()["data"]
    assert client.post(f"/api/v1/maintenance/assignments/{assigned['id']}/accept", headers=worker).status_code == 200
    recorded = client.post(
        "/api/v1/maintenance/cadence",
        headers=maintainer,
        json={
            "period": "weekly",
            "item_revisions": {},
            "context_sha256": client.get("/api/v1/maintenance/review-context", headers=maintainer).json()["data"]["context_sha256"],
            "evidence_links": ["evidence://maintenance/weekly-review"],
        },
    )
    assert recorded.status_code == 200
    second = client.get("/api/v1/maintenance/dashboard", headers=worker).json()["data"]
    assert second["cadence_due"]["weekly"] is True
    assert second["cadence_records"] == []


def test_cadence_does_not_count_another_maintainers_completed_triage_as_overdue(client: TestClient, monkeypatch) -> None:
    from app.maintenance import cadence

    admin, maintainer, second = _maintainer(client)
    _register(client, username="cadence-other-work-owner")
    assignment = client.post("/api/v1/maintenance/assignments", headers=admin, json={"username": "maintenance-worker"}).json()["data"]
    assert client.post(f"/api/v1/maintenance/assignments/{assignment['id']}/accept", headers=second).status_code == 200
    answer = _gap(client, second)
    signal = client.post("/api/v1/knowledge-feedback", headers=second, json={"answer_id": answer["id"], "label": "outdated"}).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "source-freshness",
            "severity": "p2",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "cadence-other-work-owner",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    future = datetime.now(UTC) + timedelta(days=8)
    monkeypatch.setattr(cadence, "utcnow", lambda: future)
    assert client.get("/api/v1/maintenance/dashboard", headers=second).json()["data"]["overdue_triage_count"] == 1
    assert (
        client.post(
            f"/api/v1/maintenance/items/{item['id']}/transition",
            headers=maintainer,
            json={
                "expected_revision": 1,
                "state": "triaged",
            },
        ).status_code
        == 200
    )
    dashboard = client.get("/api/v1/maintenance/dashboard", headers=second).json()["data"]
    assert dashboard["items"] == []
    assert dashboard["overdue_triage_count"] == 0
    assert dashboard["cadence_due"] == {"weekly": True, "monthly": True, "quarterly": True}


def test_roadmap_review_exposes_overdue_accountable_deferral(client: TestClient, monkeypatch) -> None:
    from app.maintenance import roadmap

    _admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "out_of_scope",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "scope-roadmap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    candidate = client.post(
        base + "/roadmap",
        headers=maintainer,
        json={
            "expected_revision": 2,
            "owner_username": "maintenance-worker",
            "desired_outcome": "clarify_scope",
            "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=1)).isoformat(),
        },
    ).json()["data"]
    assert candidate["overdue_review"] is False
    future = datetime.now(UTC) + timedelta(days=2)
    monkeypatch.setattr(roadmap, "utcnow", lambda: future)
    reloaded = client.get(f"/api/v1/maintenance/roadmap/{candidate['id']}", headers=worker).json()["data"]
    assert reloaded["overdue_review"] is True
    assert reloaded["owner_identity"] == candidate["owner_identity"]


def test_malformed_maintenance_history_fails_closed_without_projecting_content(client: TestClient) -> None:
    from app.model.canonical import CanonicalEventModel

    _admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]

    async def corrupt_trail():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                CanonicalEventModel(
                    aggregate_id=item["id"],
                    aggregate_kind="maintenance_item",
                    event_type="diagnosed",
                    from_state="open",
                    to_state="triaged",
                    recorded_by=item["accountable_maintainer"],
                    payload={"schema": "maintenance_event/v1", "revision": 2, "changes": {"note": "private forged content"}},
                )
            )
            await session.commit()

    asyncio.run(corrupt_trail())
    response = client.get(f"/api/v1/maintenance/items/{item['id']}", headers=maintainer)
    assert response.status_code == 409
    assert response.json()["code"] == "MAINTENANCE_EVENT_INVALID"
    assert "private forged content" not in response.text


@pytest.mark.parametrize("corruption", ["private_field", "rewritten_outcome"])
def test_approved_finding_rejects_rewritten_retained_payload(client: TestClient, corruption: str) -> None:
    from sqlalchemy import update

    from app.model.canonical import CanonicalRecordModel

    maintainer, worker, _reporter, base, payload = _reproduction_case(client)
    fixture = client.post(base + "/reproductions", headers=worker, json=payload).json()["data"]
    diagnosed = client.post(
        base + "/diagnosis",
        headers=maintainer,
        json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "coverage_gap"},
    )
    assert diagnosed.status_code == 200, diagnosed.text
    approved = client.post(
        base + "/findings",
        headers=maintainer,
        json={"expected_revision": 4, "fixture_identity": fixture["id"]},
    )
    assert approved.status_code == 200, approved.text
    identity = approved.json()["data"]["id"]

    async def corrupt_record():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, identity)
            assert record is not None
            fields = dict(record.payload)
            if corruption == "private_field":
                fields["note"] = "private retained finding corruption"
            else:
                fields["expected_outcome"] = "evidence_gated_answer"
            connection = await session.connection()
            await connection.execute(
                update(CanonicalRecordModel.__table__).where(CanonicalRecordModel.stable_id == identity).values(payload=fields)
            )
            await session.commit()

    asyncio.run(corrupt_record())
    response = client.get(f"/api/v1/maintenance/findings/{identity}", headers=maintainer)
    assert response.status_code == 409, response.text
    assert "private retained finding corruption" not in response.text
    replay = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": 5, "fixture_identity": fixture["id"], "answer_id": payload["answer_id"]},
    )
    assert replay.status_code == 409, replay.text


@pytest.mark.parametrize(
    ("schema", "resource"),
    [
        ("validated_finding/v1", "findings"),
        ("knowledge_roadmap_candidate/v1", "roadmap"),
    ],
)
def test_unqualified_maintenance_artifact_cannot_project_private_fields(client: TestClient, schema: str, resource: str) -> None:
    from uuid import uuid4

    from app.model.canonical import CanonicalRecordModel

    _admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "outdated",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    identity = f"maintenance_item:{uuid4().hex}"

    async def insert_unqualified_record():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(
                CanonicalRecordModel(
                    stable_id=identity,
                    identity_kind="maintenance_item",
                    identity_value=identity.split(":")[1],
                    state="deferred",
                    record_class="immutable",
                    payload={
                        "schema": schema,
                        "item_identity": item["id"],
                        "owner_identity": item["work_owner"],
                        "state": "deferred",
                        "revision": 1,
                        "review_date": (datetime.now(UTC).date() + timedelta(days=1)).isoformat(),
                        "note": "private unqualified artifact",
                    },
                )
            )
            await session.commit()

    asyncio.run(insert_unqualified_record())
    response = client.get(f"/api/v1/maintenance/{resource}/{identity}", headers=maintainer)
    assert response.status_code == 409
    assert "private unqualified artifact" not in response.text


@pytest.mark.parametrize("independent_scenarios", [False, True])
def test_roadmap_requires_independent_findings_not_changed_expectations(client: TestClient, independent_scenarios: bool) -> None:
    _admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="independent-finding-reporter"))
    base = None
    findings = []
    for index in range(2):
        question = f"uncovered engineering decision {index if independent_scenarios else 0}"
        reported = _gap(client, reporter, f"independence-report-{index}", question)
        signal = client.post(
            "/api/v1/knowledge-feedback",
            headers=reporter,
            json={
                "answer_id": reported["id"],
                "label": "insufficient_evidence",
            },
        ).json()["data"]
        if base is None:
            item = client.post(
                "/api/v1/maintenance/items",
                headers=maintainer,
                json={
                    "classification": "coverage-gap",
                    "severity": "p3",
                    "disposition": "needs-reproduction",
                    "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
                    "work_owner_username": "maintenance-worker",
                    "signal_ids": [signal["id"]],
                },
            ).json()["data"]
            base = f"/api/v1/maintenance/items/{item['id']}"
            assert (
                client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
            )
        else:
            revision = client.get(base, headers=maintainer).json()["data"]["revision"]
            assert (
                client.post(
                    base + "/signals",
                    headers=maintainer,
                    json={
                        "expected_revision": revision,
                        "signal_ids": [signal["id"]],
                    },
                ).status_code
                == 200
            )
        reproduced = _gap(client, worker, f"independence-check-{index}", question)
        revision = client.get(base, headers=maintainer).json()["data"]["revision"]
        response = client.post(
            base + "/reproductions",
            headers=worker,
            json={
                "expected_revision": revision,
                "signal_id": signal["id"],
                "answer_id": reproduced["id"],
                "expected_outcome": "insufficient_evidence_reply" if index == 0 else "evidence_gated_answer",
                "confirmed_synthetic_fixture": True,
            },
        )
        assert response.status_code == 200, response.text
        fixture = response.json()["data"]
        assert (
            client.post(
                base + "/diagnosis",
                headers=maintainer,
                json={
                    "expected_revision": revision + 1,
                    "fixture_identity": fixture["id"],
                    "observation": "coverage_gap",
                },
            ).status_code
            == 200
        )
        approved = client.post(
            base + "/findings",
            headers=maintainer,
            json={
                "expected_revision": revision + 2,
                "fixture_identity": fixture["id"],
            },
        )
        assert approved.status_code == 200, approved.text
        findings.append(approved.json()["data"]["id"])
    assert base is not None
    assert len(set(findings)) == (2 if independent_scenarios else 1)
    item = client.get(base, headers=maintainer).json()["data"]
    qualified = client.post(
        base + "/roadmap",
        headers=maintainer,
        json={
            "expected_revision": item["revision"],
            "owner_username": "maintenance-worker",
            "desired_outcome": "expand_coverage",
            "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    )
    assert qualified.status_code == (200 if independent_scenarios else 409), qualified.text
    if independent_scenarios:
        assert qualified.json()["data"]["qualification"]["independent_findings"] == 2
    else:
        assert qualified.json()["code"] == "ROADMAP_THRESHOLD_NOT_MET"


def test_roadmap_owner_transfer_returns_a_receipt_before_old_owner_loses_read_access(client: TestClient) -> None:
    _admin, maintainer, worker = _maintainer(client)
    prior_owner = _headers(_register(client, username="roadmap-prior-owner"))
    answer = _gap(client, worker)
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={
            "answer_id": answer["id"],
            "label": "out_of_scope",
        },
    ).json()["data"]
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "scope-roadmap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"]],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    candidate = client.post(
        base + "/roadmap",
        headers=maintainer,
        json={
            "expected_revision": 2,
            "owner_username": "roadmap-prior-owner",
            "desired_outcome": "clarify_scope",
            "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    ).json()["data"]
    path = f"/api/v1/maintenance/roadmap/{candidate['id']}"
    transferred = client.post(
        path + "/review",
        headers=prior_owner,
        json={
            "expected_revision": 1,
            "action": "renew",
            "rationale": "still_outside_scope",
            "owner_username": "maintenance-worker",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    )
    assert transferred.status_code == 200, transferred.text
    assert transferred.json()["data"]["revision"] == 2
    assert client.get(path, headers=worker).json()["data"] == transferred.json()["data"]
    assert client.get(path, headers=prior_owner).status_code == 403


def test_roadmap_three_execution_threshold_uses_a_rolling_thirty_day_window(client: TestClient, monkeypatch) -> None:
    from app.maintenance import roadmap
    from app.model import answer_execution

    _admin, maintainer, worker = _maintainer(client)
    completed_at = datetime.now(UTC)

    class ExecutionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return completed_at.astimezone(tz) if tz else completed_at.replace(tzinfo=None)

    signals = []
    for index in range(3):
        with monkeypatch.context() as clock:
            clock.setattr(answer_execution, "datetime", ExecutionClock)
            answer = _gap(client, worker, f"rolling-window-{index}")
        signals.append(
            client.post(
                "/api/v1/knowledge-feedback",
                headers=worker,
                json={
                    "answer_id": answer["id"],
                    "label": "insufficient_evidence",
                },
            ).json()["data"]
        )
    item = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal["id"] for signal in signals],
        },
    ).json()["data"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    future = completed_at + timedelta(days=30, microseconds=1)
    monkeypatch.setattr(roadmap, "utcnow", lambda: future)
    payload = {
        "expected_revision": 2,
        "owner_username": "maintenance-worker",
        "desired_outcome": "expand_coverage",
        "bounded_work_reason": "requires_separate_scope",
        "review_date": (future.date() + timedelta(days=1)).isoformat(),
    }
    expired_window = client.post(base + "/roadmap", headers=maintainer, json=payload)
    assert expired_window.status_code == 409
    assert expired_window.json()["code"] == "ROADMAP_THRESHOLD_NOT_MET"
    future = completed_at + timedelta(days=30)
    included_boundary = client.post(base + "/roadmap", headers=maintainer, json=payload)
    assert included_boundary.status_code == 200, included_boundary.text
    assert included_boundary.json()["data"]["qualification"]["distinct_executions_30_days"] == 3
