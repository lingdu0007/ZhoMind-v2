import pytest

from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _reproduction_case


@pytest.mark.parametrize("later_expectation", ["insufficient_evidence_reply", "evidence_gated_answer"])
def test_approved_fixture_remains_replayable_after_a_duplicate_reproduction(client, later_expectation):
    maintainer, worker, _reporter, base, reproduction = _reproduction_case(client)

    def revision():
        response = client.get(base, headers=maintainer)
        assert response.status_code == 200, response.text
        return response.json()["data"]["revision"]

    def reproduce_and_approve(expected):
        response = client.post(
            base + "/reproductions",
            headers=worker,
            json={**reproduction, "expected_revision": revision(), "expected_outcome": expected},
        )
        assert response.status_code == 200, response.text
        fixture = response.json()["data"]
        diagnosed = client.post(
            base + "/diagnosis",
            headers=maintainer,
            json={"expected_revision": revision(), "fixture_identity": fixture["id"], "observation": "coverage_gap"},
        )
        assert diagnosed.status_code == 200, diagnosed.text
        approved = client.post(
            base + "/findings",
            headers=maintainer,
            json={"expected_revision": revision(), "fixture_identity": fixture["id"]},
        )
        assert approved.status_code == 200, approved.text
        return fixture, approved.json()["data"]

    original, finding = reproduce_and_approve("insufficient_evidence_reply")
    started = client.post(
        base + "/transition", headers=maintainer, json={"expected_revision": revision(), "state": "in_progress"},
    )
    assert started.status_code == 200, started.text
    replayed = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": revision(), "fixture_identity": original["id"], "answer_id": reproduction["answer_id"]},
    )
    assert replayed.status_code == 200, replayed.text
    replay = replayed.json()["data"]
    duplicate, repeated_finding = reproduce_and_approve(later_expectation)
    assert duplicate["id"] != original["id"]
    assert repeated_finding == finding
    assert repeated_finding["fixture_identity"] == original["id"]

    historical = client.get(f"/api/v1/maintenance/replays/{replay['id']}", headers=maintainer)
    assert historical.status_code == 200, historical.text
    assert historical.json()["data"] == replay
    unapproved = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": revision(), "fixture_identity": duplicate["id"], "answer_id": reproduction["answer_id"]},
    )
    assert unapproved.status_code == 409, unapproved.text
    repeated = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": revision(), "fixture_identity": original["id"], "answer_id": reproduction["answer_id"]},
    )
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["data"]["expected_outcome"] == "insufficient_evidence_reply"
    assert repeated.json()["data"]["passed"] is True
