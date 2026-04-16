from fastapi.testclient import TestClient

from app.main import app


def test_request_id_generated_when_missing() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.headers.get("x-request-id")
    assert response.json()["request_id"] == response.headers["x-request-id"]


def test_request_id_passthrough() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health", headers={"x-request-id": "req-123"})
    assert response.headers["x-request-id"] == "req-123"
    assert response.json()["request_id"] == "req-123"
