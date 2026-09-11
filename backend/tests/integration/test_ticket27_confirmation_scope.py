import asyncio

import pytest
from sqlalchemy import delete

from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from tests.integration import test_ticket25_withdrawal as publication_helpers
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize("coverage", [
    "partial", "complete", "in_progress", "forged_partial", "legacy_partial", "missing_finding", "later_source_loss",
])
def test_direct_confirmation_requires_findings_for_all_retained_targets(client, monkeypatch, coverage):
    original_entry = publication_helpers._browser_ticket24_entry

    def scoped_entry(entry_id):
        entry = original_entry(entry_id)
        payload = entry.model_dump(mode="json")
        topic = "production" if entry_id == "confirmation-a" else "staging"
        payload["body"] = {key: f"{value} Deployment: {topic}." for key, value in payload["body"].items()}
        payload["applicability_conditions"][0]["value"] = topic
        payload["non_applicability_conditions"][0]["value"] = "staging" if topic == "production" else "production"
        supported = payload["acceptance_material"]["supported_queries"][0]
        supported["query"] = f"Which Candidate publication contract applies? deployment={topic}"
        supported["query_conditions"][0]["value"] = topic
        return type(entry).model_validate(payload)

    monkeypatch.setattr(publication_helpers, "_browser_ticket24_entry", scoped_entry)
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    _publish(client, publisher, entry_id="confirmation-a")
    admin, maintainer, worker = _maintainer(client)
    _activate_route(client, monkeypatch, admin)
    reporter = _headers(_register(client, username="confirmation-reporter"))

    def question(entry):
        deployment = "production" if entry == "confirmation-a" else "staging"
        return f"Which Candidate publication contract applies? deployment={deployment}"

    def report(entry):
        answer = _gap(client, reporter, f"confirmation-report-{entry}", question(entry))
        response = client.post(
            "/api/v1/knowledge-feedback", headers=reporter,
            json={"answer_id": answer["id"], "entry_id": entry, "label": "helpful"},
        )
        assert response.status_code == 200, (response.text, answer)
        return response.json()["data"]["id"]

    signal_a = report("confirmation-a")
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "confirmation", "severity": "p3", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal_a],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"

    def mutate(endpoint, actor, fields):
        revision = client.get(base, headers=maintainer).json()["data"]["revision"]
        response = client.post(base + endpoint, headers=actor, json={"expected_revision": revision, **fields})
        assert response.status_code == 200, response.text
        return response.json()["data"]

    def approve(signal, entry):
        independent = _gap(client, worker, f"confirmation-independent-{entry}", question(entry))
        fixture = mutate("/reproductions", worker, {
            "answer_id": independent["id"], "signal_id": signal,
            "expected_outcome": "evidence_gated_answer", "verified_observation": "confirmation",
            "confirmed_synthetic_fixture": True,
        })
        assert all(target["publication_identity"] in fixture["evidence_publication_identities"]
                   for target in fixture["affected_scope"]["entry_versions"]), fixture
        mutate("/diagnosis", maintainer, {"fixture_identity": fixture["id"], "observation": "confirmation"})
        return mutate("/findings", maintainer, {"fixture_identity": fixture["id"]})

    mutate("/transition", maintainer, {"state": "triaged"})
    findings = [approve(signal_a, "confirmation-a")]
    _publish(client, publisher, entry_id="confirmation-b")
    signal_b = report("confirmation-b")
    mutate("/signals", maintainer, {"signal_ids": [signal_b]})
    if coverage not in {"partial", "forged_partial", "legacy_partial"}:
        findings.append(approve(signal_b, "confirmation-b"))
    for entry in ("confirmation-a", "confirmation-b"):
        editorial_owner = _headers(_register(client, username=f"{entry}-maintainer"))
        editorial = client.get(f"/api/v1/editorial/entries/{entry}", headers=editorial_owner)
        assert editorial.status_code == 200, editorial.text
        assert editorial.json()["data"]["answer_eligible"] is True, editorial.text
    if coverage == "in_progress":
        mutate("/transition", maintainer, {"state": "in_progress"})
    before = client.get(base, headers=maintainer).json()["data"]
    assert len(before["affected_scope"]["entry_versions"]) == 2
    if coverage in {"forged_partial", "legacy_partial"}:
        async def forge_closure():
            async with client.app.state.test_auth_session_factory() as session:
                changes = {"result_links": findings[0]["result_links"]} if coverage == "forged_partial" else {}
                session.add(CanonicalEventModel(
                    aggregate_id=before["id"], aggregate_kind="maintenance_item",
                    event_type="confirmation_closed" if coverage == "forged_partial" else "transition",
                    from_state=before["state"], to_state="closed_confirmation",
                    recorded_by=before["accountable_maintainer"],
                    payload={"schema": "maintenance_event/v1", "revision": before["revision"] + 1, "changes": changes},
                ))
                await session.commit()
        asyncio.run(forge_closure())
        assert client.get(base, headers=maintainer).status_code == 409
        return
    closed = client.post(
        base + "/transition", headers=maintainer,
        json={"expected_revision": before["revision"], "state": "closed_confirmation"},
    )
    if coverage == "partial":
        assert closed.status_code == 409, closed.text
        retained = client.get(base, headers=maintainer).json()["data"]
        assert retained["state"] == "triaged"
        assert retained["revision"] == before["revision"]
        return
    assert closed.status_code == 200, closed.text
    retained = client.get(base, headers=worker).json()["data"]
    assert retained["state"] == "closed_confirmation"
    for finding in findings:
        assert finding["result_links"][0] in retained["result_links"]
    if coverage == "missing_finding":
        async def delete_finding():
            async with client.app.state.test_auth_session_factory() as session:
                connection = await session.connection()
                await connection.execute(
                    delete(CanonicalRecordModel.__table__).where(CanonicalRecordModel.stable_id == findings[1]["id"]),
                )
                await session.commit()
        asyncio.run(delete_finding())
        assert client.get(base, headers=maintainer).status_code == 409
    elif coverage == "later_source_loss":
        owner = _headers(_register(client, username="confirmation-a-maintainer"))
        changed = client.post(
            "/api/v1/editorial/entries/confirmation-a/sources/source-confirmation-a/availability",
            headers=owner, json={"availability": "unavailable_for_new_evidence"},
        )
        assert changed.status_code == 200, changed.text
        historical = client.get(base, headers=maintainer)
        assert historical.status_code == 200, historical.text
        assert historical.json()["data"]["result_links"] == retained["result_links"]
