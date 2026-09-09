import pytest
from fastapi.testclient import TestClient

from tests.integration.test_editorial_authority_api import _member_headers, _review_ready_payload
from tests.integration.test_editorial_authority_api import client as client


@pytest.mark.parametrize("prefix", [
    "%73%6b%2d", "%252573%25256b%25252d", "%25252573%2525256b%2525252d",
])
def test_editorial_rejects_encoded_path_credentials_before_draft_persistence(client: TestClient, prefix):
    author = _member_headers(client, "encoded-source-author")
    payload = _review_ready_payload()
    private_url = "https://example.com/docs/" + prefix + "A" * 32
    payload["sources"][0]["public_url"] = private_url
    response = client.post("/api/v1/editorial/entries", headers=author, json=payload)
    assert response.status_code == 422
    assert response.json()["code"] == "EDITORIAL_SECRET_REJECTED"
    assert private_url not in response.text
    assert "A" * 32 not in response.text
    absent = client.get(f"/api/v1/editorial/entries/{payload['entry_id']}", headers=author)
    assert absent.status_code == 404


def test_editorial_accepts_a_safe_encoded_source_path(client: TestClient):
    author = _member_headers(client, "safe-source-author")
    payload = _review_ready_payload()
    payload["sources"][0]["public_url"] = "https://example.com/docs/%E6%96%87%E6%A1%A3"
    response = client.post("/api/v1/editorial/entries", headers=author, json=payload)
    assert response.status_code == 200
    retained = client.get(f"/api/v1/editorial/entries/{payload['entry_id']}", headers=author)
    assert retained.status_code == 200
