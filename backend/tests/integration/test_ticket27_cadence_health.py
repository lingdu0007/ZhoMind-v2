from tests.browser_acceptance_api import _browser_ticket24_entry
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _maintainer


def test_monthly_health_retains_unknown_source_on_unpublished_successor(client):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-health")
    _, maintainer, _ = _maintainer(client)
    author = _headers(_register(client, username="maintenance-health-author"))
    entry = _browser_ticket24_entry("maintenance-health").model_dump(mode="json")
    entry["approving_reviewer_username"] = "maintenance-health-reviewer"
    entry["accountable_maintainer_username"] = "maintenance-health-maintainer"
    entry["sources"].append({
        **entry["sources"][0],
        "source_id": "maintenance-health-new-source",
    })
    revised = client.post(
        "/api/v1/editorial/entries/maintenance-health/revisions",
        headers=author,
        json={"entry": entry, "change_kind": "material"},
    )
    assert revised.status_code == 200, revised.text
    assert revised.json()["data"]["answer_eligible"] is False
    context_response = client.get("/api/v1/maintenance/review-context", headers=maintainer)
    assert context_response.status_code == 200, context_response.text
    context = context_response.json()["data"]
    health = context["knowledge_health"]["published_entries"]
    assert len(health) == 1
    assert health[0]["publication_identity"] == publication
    assert health[0]["current_revision_identity"] == revised.json()["data"]["revision_identity"]
    assert health[0]["current_revision_answer_eligible"] is False
    assert health[0]["source_states"] == ["unknown", "verified_usable"]
    recorded = client.post(
        "/api/v1/maintenance/cadence",
        headers=maintainer,
        json={
            "period": "monthly",
            "item_revisions": context["item_revisions"],
            "context_sha256": context["context_sha256"],
            "evidence_links": ["evidence://maintenance/monthly-review"],
        },
    )
    assert recorded.status_code == 200, recorded.text
    assert recorded.json()["data"]["review_snapshot"]["knowledge_health"] == context["knowledge_health"]
