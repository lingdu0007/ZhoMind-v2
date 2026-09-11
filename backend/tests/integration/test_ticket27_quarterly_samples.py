from datetime import datetime, timedelta

import pytest

from tests.integration.test_delivery_acceptance import _active_local_record
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _maintainer


@pytest.mark.parametrize("period", ["previous", "future", "current"])
def test_quarterly_sample_requires_current_verified_results(client, monkeypatch, period):
    from app.maintenance import cadence, cadence_context

    admin, maintainer, _worker = _maintainer(client)
    acceptance = _active_local_record(client, admin)
    activation = acceptance["status_history"][-1]
    verified_at = datetime.fromisoformat(activation["occurred_at"])
    now = {
        "previous": verified_at + timedelta(days=100),
        "future": verified_at - timedelta(seconds=1),
        "current": verified_at + timedelta(seconds=1),
    }[period]
    monkeypatch.setattr(cadence_context, "utcnow", lambda: now, raising=False)
    monkeypatch.setattr(cadence, "utcnow", lambda: now)
    response = client.get("/api/v1/maintenance/review-context", headers=maintainer)
    assert response.status_code == 200, response.text
    context = response.json()["data"]
    recorded = client.post(
        "/api/v1/maintenance/cadence", headers=maintainer,
        json={
            "period": "quarterly", "item_revisions": context["item_revisions"],
            "context_sha256": context["context_sha256"],
            "sample_acceptance_identities": [acceptance["record_id"]],
            "evidence_links": ["evidence://maintenance/quarterly-review"],
        },
    )
    if period != "current":
        assert context["sample_options"] == [], "a historical Active record is not a current-quarter sample"
        assert recorded.status_code == 409, recorded.text
        return
    assert recorded.status_code == 200, recorded.text
    sample = recorded.json()["data"]["sampled_acceptances"][0]
    assert datetime.fromisoformat(sample["verified_at"]) == verified_at
    assert sample["verified_by"] == activation["recorded_by"]
    expected_results = {check["check_id"]: check["result"] for check in acceptance["checks"]}
    assert sample["verified_checks"] == [
        {**check, "result": expected_results[check["check_id"]]}
        for check in activation["verified_checks"]
    ]
    assert sample["status_event_id"] == activation["event_id"]
    assert sample["accepted_scope"] == acceptance["accepted_scope"]
    history = client.get("/api/v1/maintenance/dashboard", headers=maintainer)
    assert history.status_code == 200, history.text
    assert history.json()["data"]["cadence_records"][0]["sampled_acceptances"] == [sample]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "empty_evidence"])
def test_quarterly_sample_rejects_incomplete_verification_attachments(client, monkeypatch, defect):
    from app.delivery_acceptance.service import DeliveryAcceptanceService

    admin, maintainer, _worker = _maintainer(client)
    acceptance = _active_local_record(client, admin)
    original = DeliveryAcceptanceService.get_projection

    async def projection(service, identity):
        result = await original(service, identity)
        if identity == acceptance["record_id"]:
            activation = result["status_history"][-1]
            if defect == "missing":
                activation.pop("verified_checks")
            elif defect == "duplicate":
                activation["verified_checks"].append(activation["verified_checks"][0])
            else:
                activation["verified_checks"][0]["evidence_links"] = []
        return result

    monkeypatch.setattr(DeliveryAcceptanceService, "get_projection", projection)
    response = client.get("/api/v1/maintenance/review-context", headers=maintainer)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["sample_options"] == []
