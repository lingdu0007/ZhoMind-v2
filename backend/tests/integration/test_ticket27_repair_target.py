import pytest

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize("observation", ["stale_source", "wrong_content"])
@pytest.mark.parametrize("target", ["reported-entry", "foreign-entry"])
def test_reproduction_binds_selected_feedback_to_the_repaired_entry(client, monkeypatch, observation, target):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="reported-entry")
    admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="target-reporter"))
    _activate_route(client, monkeypatch, admin)
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "target-report", question)
    assert reported["answer_execution"]["knowledge_version_identities"] == [publication]
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": reported["id"], "entry_id": "reported-entry", "label": "outdated"},
    )
    assert signal.status_code == 200, signal.text
    _publish(client, publisher, entry_id="foreign-entry")
    for entry in ("reported-entry", "foreign-entry"):
        owner = _headers(_register(client, username=f"{entry}-maintainer"))
        path = f"/api/v1/editorial/entries/{entry}"
        current = client.get(path, headers=owner).json()["data"]
        if observation == "stale_source":
            changed = client.post(
                path + f"/sources/{current['sources'][0]['source_id']}/availability", headers=owner,
                json={"availability": "changed_or_unreachable_awaiting_review"},
            )
        else:
            changed = client.post(
                path + "/integrity-review", headers=owner,
                json={
                    "revision_identity": current["revision_identity"],
                    "publication_identity": current["integrity_review"]["publication_identity"],
                    "source_identity": current["integrity_review"]["source_identities"][0],
                    "defect": "integrity_defect", "confirmed_independent_review": True,
                },
            )
        assert changed.status_code == 200, changed.text
    independent = _gap(client, worker, "target-independent", question)
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "source-freshness" if observation == "stale_source" else "content-integrity",
            "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    reproduced = client.post(
        base + "/reproductions", headers=worker,
        json={
            "expected_revision": 2, "signal_id": signal.json()["data"]["id"],
            "answer_id": independent["id"], "expected_outcome": "evidence_gated_answer",
            "confirmed_synthetic_fixture": True, "verified_observation": observation,
            "entry_identity": f"entry:{target}",
        },
    )
    if target == "foreign-entry":
        assert reproduced.status_code == 409, reproduced.text
        retained = client.get(base, headers=maintainer).json()["data"]
        assert retained["revision"] == 2
        assert retained.get("fixture_identity") is None
    else:
        assert reproduced.status_code == 200, reproduced.text
        fixture = reproduced.json()["data"]
        assert fixture["publication_review"]["entry_identity"] == "entry:reported-entry"
        assert fixture["affected_scope"] == created.json()["data"]["affected_scope"]
        assert client.delete(f"/api/v1/knowledge-feedback/{signal.json()['data']['id']}", headers=reporter).status_code == 200
        retained = client.get(f"/api/v1/maintenance/fixtures/{fixture['id']}", headers=maintainer)
        assert retained.status_code == 200, retained.text
        assert retained.json()["data"]["affected_scope"] == fixture["affected_scope"]
