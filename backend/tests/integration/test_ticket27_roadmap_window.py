from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _gap, _maintainer


@pytest.mark.parametrize("elapsed_days, qualifies", [(0, True), (7, False)])
def test_roadmap_window_uses_execution_time_not_recent_feedback(client, monkeypatch, elapsed_days, qualifies):
    from app.maintenance import roadmap
    from app.model import answer_execution

    _, maintainer, worker = _maintainer(client)
    now = datetime.now(UTC)
    execution_time = now - timedelta(days=29)

    class ExecutionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return execution_time.astimezone(tz) if tz else execution_time.replace(tzinfo=None)

    signals = []
    for index in range(3):
        with monkeypatch.context() as clock:
            clock.setattr(answer_execution, "datetime", ExecutionClock)
            answer = _gap(client, worker, f"roadmap-window-{index}")
        submitted = client.post(
            "/api/v1/knowledge-feedback",
            headers=worker,
            json={"answer_id": answer["id"], "label": "insufficient_evidence"},
        )
        assert submitted.status_code == 200, submitted.text
        signals.append(submitted.json()["data"]["id"])
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "coverage-gap",
            "severity": "p3",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": signals,
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    triaged = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"})
    assert triaged.status_code == 200, triaged.text
    review_time = datetime.now(UTC) + timedelta(days=elapsed_days)
    monkeypatch.setattr(roadmap, "utcnow", lambda: review_time)
    qualified = client.post(
        base + "/roadmap",
        headers=maintainer,
        json={
            "expected_revision": 2,
            "owner_username": "maintenance-worker",
            "desired_outcome": "expand_coverage",
            "bounded_work_reason": "requires_separate_scope",
            "review_date": (review_time.date() + timedelta(days=30)).isoformat(),
        },
    )
    if qualifies:
        assert qualified.status_code == 200, qualified.text
        assert qualified.json()["data"]["qualification"]["distinct_executions_30_days"] == 3
    else:
        assert qualified.status_code == 409, qualified.text
        assert qualified.json()["code"] == "ROADMAP_THRESHOLD_NOT_MET"
        retained = client.get(base, headers=maintainer).json()["data"]
        assert retained["state"] == "triaged"
        assert retained["revision"] == 2
