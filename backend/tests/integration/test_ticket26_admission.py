import pytest
from fastapi.testclient import TestClient

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client


@pytest.mark.parametrize("content", [
    "password=fixture-private-password",
    "email: private.person@example.com",
    "ssn: 123-45-6789",
    "github_pat_" + "A" * 24,
])
def test_authenticated_upload_rejects_private_material_without_retaining_it(client: TestClient, content) -> None:
    admin = _headers(_register(client, username="upload-safety-admin", role="admin"))
    response = client.post("/api/v1/documents/upload", headers=admin, files={
        "file": ("private-fixture.md", content.encode(), "text/markdown"),
    })
    assert response.status_code == 422
    assert response.json()["code"] == "CONTENT_BOUNDARY_REJECTED"
    assert content not in response.text
    assert "private-fixture.md" not in client.get("/api/v1/documents", headers=admin).text
