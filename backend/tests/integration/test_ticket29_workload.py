from datetime import UTC, datetime, timedelta

from app.operations.pilot_measurement import MeasurementBinding
from app.settings.generation_routes import GenerationRouteService
from tests.integration.test_approved_generation_settings import (
    acceptance_for_route,
    local_activation_boundary,  # noqa: F401
    route_payload,
)
from tests.integration.test_system_settings_flow import _headers, _register
from tests.integration.test_system_settings_flow import client as client


def test_entry_runner_makes_120_authenticated_requests_and_does_not_certify_empty_corpus(client, monkeypatch):
    from app.operations.pilot_runner import collect_probe, run_entry_requests

    admin_token = _register(client, username="workload-admin", role="admin")
    tokens = tuple(_register(client, username=f"workload-member-{index}") for index in range(8))
    headers = _headers(admin_token)

    async def validated(self, payload):
        return None

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    monkeypatch.setattr(client.app.state, "operational_event_session_factory", client.app.state.settings_session_factory)
    path = "/api/v1/settings/generation-route"
    route = client.put(path, headers=headers, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, headers, route)
    activation = client.post(path + "/activate", headers=headers, json={
        "route_identity": route["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    })
    assert activation.status_code == 200
    snapshot_response = client.get("/api/v1/operations/measurement-snapshot", headers=headers)
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()["data"]
    now = datetime.now(UTC)
    bound = MeasurementBinding(
        **{key: value for key, value in snapshot.items() if key != "background_workers"},
        concurrency=1, window_start=now, window_end=now + timedelta(hours=1), mode="local_deterministic",
    )
    report = run_entry_requests(client, binding=bound, member_tokens=tokens, administrator_token=admin_token, question="hello")
    assert [item["eligible_count"] for item in report["request_reports"]] == [40, 40, 40]
    assert len(report["samples"]) == 120
    assert report["observed_members"] == 8
    assert report["profile_status"] == "unavailable"
    assert any(item["objective"] == "workload_profile" and item["required_action"] == "complete_measurement"
               for item in report["follow_up_decisions"])
    assert report["live_acceptance"] is False
    assert all(sample["outcome"] == "non_knowledge_base_reply" for sample in report["samples"])
    serialized = str(report)
    assert admin_token not in serialized
    assert all(token not in serialized for token in tokens)
    assert "hello" not in serialized
    probe = collect_probe(client, binding=bound, member_token=tokens[0], administrator_token=admin_token,
                          question="hello", product_check=True)
    assert probe.healthy is True
    assert probe.product_checked is True
    assert probe.product_outcome is None
