import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.operations.chat_capacity import get_chat_admission_gate
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize("failure", ["deactivation", "timeout"])
def test_queued_maintenance_failure_cannot_start_provider_execution(client, monkeypatch, failure):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    _publish(client, publisher, entry_id="maintenance-admission")
    admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="admission-reporter"))
    provider = _activate_route(client, monkeypatch, admin)
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "admission-reported", question)
    independent = _gap(client, worker, "admission-independent", question)
    signal = client.post("/api/v1/knowledge-feedback", headers=reporter, json={
        "answer_id": reported["id"], "entry_id": "maintenance-admission", "label": "insufficient_evidence",
    })
    assert signal.status_code == 200
    created = client.post("/api/v1/maintenance/items", headers=maintainer, json={
        "classification": "product-privacy-operations", "severity": "p2", "disposition": "needs-reproduction",
        "coverage_position": "provider_failure_and_observability",
        "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
    })
    assert created.status_code == 200
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    assert client.post(base + "/administrator", headers=admin, json={"expected_revision": 1}).status_code == 200
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 2, "state": "triaged"}).status_code == 200
    attempts = provider.attempts
    gate = get_chat_admission_gate()
    if failure == "timeout":
        monkeypatch.setattr(gate, "queue_timeout_seconds", 0.2)
    held = [gate.reserve(member_id=f"busy-{index}") for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, base + "/reproductions", headers=worker, json={
            "expected_revision": 3, "answer_id": independent["id"], "signal_id": signal.json()["data"]["id"],
            "expected_outcome": "evidence_gated_answer", "confirmed_synthetic_fixture": True,
            "verified_observation": "provider_failure",
        })
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = client.get("/api/v1/operations", headers=admin).json()["data"]["admission"]
                if state["queued"] == 1:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("maintenance request did not enter the queue")
            if failure == "deactivation":
                assert client.post("/api/v1/members/maintenance-worker/deactivate", headers=admin).status_code == 200
            else:
                response = future.result(timeout=5)
        finally:
            for reservation in held:
                gate.finish(reservation)
        response = future.result(timeout=5)
    if failure == "deactivation":
        assert response.status_code in {401, 403, 409}
    assert provider.attempts == attempts
    if failure == "timeout":
        assert response.status_code == 503
        assert response.json()["code"] == "CHAT_QUEUE_TIMEOUT"
        operations = client.get("/api/v1/operations", headers=admin).json()["data"]
        assert any(item["kind"] == "queue" and item["code"] == "CHAT_QUEUE_TIMEOUT"
                   for item in operations["failures"])
