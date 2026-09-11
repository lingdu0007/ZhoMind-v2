from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _gap, _maintainer


def test_maintenance_creation_rejects_unknown_or_multiple_classifications_and_noncanonical_fields(client):
    _admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker, "vocabulary-boundary")
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=worker,
        json={"answer_id": answer["id"], "label": "insufficient_evidence"},
    )
    assert signal.status_code == 200, signal.text
    payload = {
        "classification": "coverage-gap", "severity": "p2", "disposition": "needs-reproduction",
        "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
        "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
    }
    for invalid in (
        {"classification": "needs-reproduction"},
        {"classification": "unknown"},
        {"classification": ["confirmation", "coverage-gap"]},
        {"classification": None},
        {"severity": "p4"},
        {"disposition": "retrieval-experiment"},
        {"question": "unapproved copied conversation"},
    ):
        rejected = client.post("/api/v1/maintenance/items", headers=maintainer, json={**payload, **invalid})
        assert rejected.status_code == 422, rejected.text
        assert client.get("/api/v1/maintenance/items", headers=maintainer).json()["data"]["items"] == []
    accepted = client.post("/api/v1/maintenance/items", headers=maintainer, json=payload)
    assert accepted.status_code == 200, accepted.text
    item = accepted.json()["data"]
    assert item["classification"] == "coverage-gap"
    assert item["severity"] == "p2"
    assert item["disposition"] == "needs-reproduction"
    assert item["state"] == "open"
