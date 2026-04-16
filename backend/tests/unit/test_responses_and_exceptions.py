from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.common.exceptions import AppError
from app.main import app


def test_app_error_shape() -> None:
    router = APIRouter()

    @router.get("/api/v1/_raise-app")
    def raise_app() -> None:
        raise AppError(status_code=403, code="AUTH_FORBIDDEN", message="forbidden")

    app.include_router(router)
    client = TestClient(app)
    response = client.get("/api/v1/_raise-app")
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "AUTH_FORBIDDEN"
    assert body["message"] == "forbidden"
    assert "request_id" in body
