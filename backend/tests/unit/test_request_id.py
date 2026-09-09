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
    request_id = "cba9d526-19f9-4571-bc2a-f1162433696a"
    response = client.get("/api/v1/health", headers={"x-request-id": request_id})
    assert response.headers["x-request-id"] == request_id
    assert response.json()["request_id"] == request_id
