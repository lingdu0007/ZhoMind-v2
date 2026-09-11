import pytest

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize("artifact", ["matching", "foreign"])
def test_published_coverage_repair_closes_only_with_replayed_knowledge(client, monkeypatch, artifact):
    admin, maintainer, worker = _maintainer(client)
    _activate_route(client, monkeypatch, admin)
    reporter = _headers(_register(client, username="coverage-reporter"))
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "coverage-reported", question)
    independent = _gap(client, worker, "coverage-independent", question)
    assert reported["answer_execution"]["outcome"] == "insufficient_evidence_reply"
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": reported["id"], "label": "insufficient_evidence"},
    )
    assert signal.status_code == 200, signal.text
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "coverage-gap", "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"

    def mutate(endpoint, actor, fields):
        revision = client.get(base, headers=maintainer).json()["data"]["revision"]
        response = client.post(base + endpoint, headers=actor, json={"expected_revision": revision, **fields})
        assert response.status_code == 200, response.text
        return response.json()["data"]

    mutate("/transition", maintainer, {"state": "triaged"})
    fixture = mutate("/reproductions", worker, {
        "answer_id": independent["id"], "signal_id": signal.json()["data"]["id"],
        "expected_outcome": "evidence_gated_answer", "verified_observation": "coverage_gap",
        "confirmed_synthetic_fixture": True,
    })
    mutate("/diagnosis", maintainer, {"fixture_identity": fixture["id"], "observation": "coverage_gap"})
    mutate("/findings", maintainer, {"fixture_identity": fixture["id"]})
    mutate("/transition", maintainer, {"state": "in_progress"})
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    foreign = _publish(client, publisher, entry_id="coverage-other")
    matching = _publish(client, publisher, entry_id="coverage-repaired")
    replay = mutate("/replays", worker, {"fixture_identity": fixture["id"], "answer_id": independent["id"]})
    assert replay["passed"] is True
    publications = {matching: "coverage-repaired", foreign: "coverage-other"}
    assert len(replay["evidence_publication_identities"]) == 1
    actual = replay["evidence_publication_identities"][0]
    assert actual in publications
    projected = client.get(f"/api/v1/maintenance/replays/{replay['id']}", headers=maintainer)
    assert projected.status_code == 200, projected.text
    assert projected.json()["data"]["evidence_publications"] == [{
        "publication_identity": actual, "entry_identity": f"entry:{publications[actual]}",
        "revision_identity": f"editorial_revision:{publications[actual]}.r1",
    }]
    publication = actual if artifact == "matching" else next(identity for identity in publications if identity != actual)
    entry = publications[publication]
    resolved = client.post(
        base + "/resolution", headers=maintainer,
        json={
            "expected_revision": 7, "disposition": "entry-revision",
            "artifact_identities": [fixture["id"], replay["id"], publication, f"editorial_revision:{entry}.r1"],
        },
    )
    if artifact == "foreign":
        assert resolved.status_code == 409, resolved.text
        return
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["data"]["state"] == "resolved"
    mutate("/transition", maintainer, {"state": "closed_confirmation"})
    assert client.get(base, headers=worker).json()["data"]["state"] == "closed_confirmation"
