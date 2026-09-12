from uuid import uuid4

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client


def test_measurement_reads_exact_request_non_content_timings_with_admin_authority(client):
    member = _headers(_register(client, username="measurement-member"))
    admin = _headers(_register(client, username="measurement-admin", role="admin"))
    response = client.post("/api/v1/chat/stream", headers=member, json={"message": "hello"})
    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    path = f"/api/v1/operations/requests/{request_id}"
    assert client.get(path, headers=member).status_code == 403
    assert client.get(path).status_code == 401
    observed = client.get(path, headers=admin)
    assert observed.status_code == 200
    data = observed.json()["data"]
    assert data["request_id"] == request_id
    assert data["dimensions"]["outcome"] == "non_knowledge_base_reply"
    assert data["dimensions"]["stage_durations_ms"]["provider"] == 0
    assert data["dimensions"]["stage_durations_ms"]["retrieval"] == 0
    assert data["dimensions"]["stage_durations_ms"]["persistence"] >= 0
    assert data["created_at"]
    assert all(key not in data for key in ("question", "answer", "content", "messages", "username"))
    assert "hello" not in observed.text
    assert client.get(f"/api/v1/operations/requests/{uuid4()}", headers=admin).status_code == 404


def test_measurement_snapshot_is_current_server_derived_and_content_free(client):
    admin = _headers(_register(client, username="snapshot-admin", role="admin"))
    member = _headers(_register(client, username="snapshot-member"))
    path = "/api/v1/operations/measurement-snapshot"
    assert client.get(path, headers=member).status_code == 403
    response = client.get(path, headers=admin)
    assert response.status_code == 200
    snapshot = response.json()["data"]
    assert snapshot["active_entries"] == 0
    assert snapshot["eligible_chunks"] == 0
    assert snapshot["provider_route"] is None
    assert snapshot["corpus"].startswith("corpus:")
    assert snapshot["retrieval_profile"].startswith("retrieval_profile:")
    assert snapshot["background_workers"] == 1
    assert snapshot["configuration"].startswith("configuration:")
