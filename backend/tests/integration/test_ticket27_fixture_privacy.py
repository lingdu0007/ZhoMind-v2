import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from app.model.chat import ChatSession
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _gap, _reproduction_case


@pytest.mark.parametrize("operation", ["reproduction", "replay"])
def test_expired_owned_input_cannot_be_replayed_into_a_new_session(client, operation):
    maintainer, worker, _reporter, base, payload = _reproduction_case(client)
    answer = _gap(client, worker, "privacy-expired-input", question="bounded independent coverage scenario")
    payload["answer_id"] = answer["id"]
    if operation == "replay":
        reproduced = client.post(base + "/reproductions", headers=worker, json=payload)
        assert reproduced.status_code == 200, reproduced.text
        fixture = reproduced.json()["data"]
        assert client.post(
            base + "/diagnosis", headers=maintainer,
            json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "coverage_gap"},
        ).status_code == 200
        assert client.post(
            base + "/findings", headers=maintainer,
            json={"expected_revision": 4, "fixture_identity": fixture["id"]},
        ).status_code == 200
        assert client.post(
            base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"},
        ).status_code == 200
        payload = {"expected_revision": 6, "fixture_identity": fixture["id"], "answer_id": answer["id"]}

    async def expire_session():
        async with client.app.state.test_auth_session_factory() as session:
            await session.execute(update(ChatSession).where(ChatSession.id == "privacy-expired-input").values(
                created_at=datetime.now(UTC) - timedelta(days=1000),
            ))
            await session.commit()
            return set((await session.scalars(select(ChatSession.id))).all())

    previous_sessions = asyncio.run(expire_session())
    endpoint = "/reproductions" if operation == "reproduction" else "/replays"
    rejected = client.post(base + endpoint, headers=worker, json=payload)
    assert rejected.status_code == 409, rejected.text
    assert rejected.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"
    current = client.get(base, headers=maintainer).json()["data"]
    assert current["revision"] == (2 if operation == "reproduction" else 6)

    async def session_identities():
        async with client.app.state.test_auth_session_factory() as session:
            return set((await session.scalars(select(ChatSession.id))).all())

    assert asyncio.run(session_identities()).issubset(previous_sessions), "expired input was copied into a new session"


def test_independent_question_cannot_change_reported_structured_conditions(client):
    maintainer, worker, _, base, payload = _reproduction_case(client)
    different = _gap(client, worker, "privacy-changed-conditions", question="independent scenario deployment=production")
    rejected = client.post(base + "/reproductions", headers=worker, json={**payload, "answer_id": different["id"]})
    assert rejected.status_code == 409, rejected.text
    retained = client.get(base, headers=maintainer).json()["data"]
    assert retained["revision"] == 2
    assert retained.get("fixture_identity") is None


@pytest.mark.parametrize("independent_question", [
    "uncovered engineering decision",
    "independently authored bounded coverage scenario",
])
def test_durable_fixture_retains_no_question_and_replay_requires_owned_input(client, independent_question):
    maintainer, worker, reporter, base, payload = _reproduction_case(client)
    answer = _gap(client, worker, "privacy-independent", question=independent_question)
    payload["answer_id"] = answer["id"]
    reproduced = client.post(base + "/reproductions", headers=worker, json=payload)
    assert reproduced.status_code == 200, reproduced.text
    fixture = reproduced.json()["data"]
    assert "replay_request" not in fixture
    assert independent_question not in reproduced.text
    assert "uncovered engineering decision" not in reproduced.text
    diagnosed = client.post(
        base + "/diagnosis",
        headers=maintainer,
        json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "coverage_gap"},
    )
    assert diagnosed.status_code == 200, diagnosed.text
    finding = client.post(
        base + "/findings",
        headers=maintainer,
        json={"expected_revision": 4, "fixture_identity": fixture["id"]},
    )
    assert finding.status_code == 200, finding.text
    assert independent_question not in finding.text
    deleted = client.delete(f"/api/v1/knowledge-feedback/{payload['signal_id']}", headers=reporter)
    assert deleted.status_code == 200, deleted.text
    retained = client.get(f"/api/v1/maintenance/fixtures/{fixture['id']}", headers=maintainer)
    assert retained.status_code == 200, retained.text
    assert independent_question not in retained.text
    started = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"})
    assert started.status_code == 200, started.text
    replay_payload = {"expected_revision": 6, "fixture_identity": fixture["id"]}
    assert client.post(base + "/replays", headers=worker, json=replay_payload).status_code == 422
    foreign = _gap(client, reporter, "privacy-foreign", question=independent_question)
    rejected = client.post(base + "/replays", headers=worker, json={**replay_payload, "answer_id": foreign["id"]})
    assert rejected.status_code == 409, rejected.text
    different = _gap(client, worker, "privacy-different", question="a different bounded scenario")
    rejected = client.post(base + "/replays", headers=worker, json={**replay_payload, "answer_id": different["id"]})
    assert rejected.status_code == 409, rejected.text
    removed = client.delete("/api/v1/sessions/privacy-independent", headers=worker)
    assert removed.status_code == 200, removed.text
    rejected = client.post(base + "/replays", headers=worker, json={**replay_payload, "answer_id": answer["id"]})
    assert rejected.status_code == 409, rejected.text
    replacement = _gap(client, worker, "privacy-recreated", question=independent_question)
    replayed = client.post(base + "/replays", headers=worker, json={**replay_payload, "answer_id": replacement["id"]})
    assert replayed.status_code == 200, replayed.text
    assert replayed.json()["data"]["passed"] is True
    assert independent_question not in replayed.text
    assert replayed.json()["data"]["query_condition_set_identity"] == fixture["query_condition_set_identity"]
