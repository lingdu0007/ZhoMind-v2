import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


def test_gap_scope_survives_raw_deletion_consolidation_and_roadmap(client):
    _, maintainer, worker = _maintainer(client)
    scopes = []
    item = None
    for index in range(2):
        answer = _gap(client, worker, f"scope-{index}", question=f"independent coverage scenario {index}")
        signal = client.post(
            "/api/v1/knowledge-feedback", headers=worker,
            json={"answer_id": answer["id"], "label": "out_of_scope", "note": "private reporter note"},
        ).json()["data"]
        scopes.append({
            "reason": signal["gap_context"]["reason"],
            "query_condition_set_identity": signal["query_condition_set_identity"],
        })
        if item is None:
            response = client.post(
                "/api/v1/maintenance/items", headers=maintainer,
                json={
                    "classification": "scope-roadmap", "severity": "p3", "disposition": "needs-reproduction",
                    "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
                    "work_owner_username": "maintenance-worker", "signal_ids": [signal["id"]],
                },
            )
        else:
            response = client.post(
                f"/api/v1/maintenance/items/{item['id']}/signals", headers=maintainer,
                json={"expected_revision": item["revision"], "signal_ids": [signal["id"]]},
            )
        assert response.status_code == 200, response.text
        item = response.json()["data"]
        expected = {"entry_versions": [], "gap_contexts": sorted(scopes, key=lambda scope: scope["query_condition_set_identity"])}
        assert item["affected_scope"] == expected
        assert "private reporter note" not in response.text
        assert "independent coverage scenario" not in response.text
        assert client.delete(f"/api/v1/knowledge-feedback/{signal['id']}", headers=worker).status_code == 200
        retained = client.get(f"/api/v1/maintenance/items/{item['id']}", headers=maintainer).json()["data"]
        assert retained["signal_count"] == 0
        assert retained["affected_scope"] == expected
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 2, "state": "triaged"}).status_code == 200
    qualified = client.post(
        base + "/roadmap", headers=maintainer,
        json={
            "expected_revision": 3, "owner_username": "maintenance-worker",
            "desired_outcome": "clarify_scope", "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=30)).isoformat(),
        },
    )
    assert qualified.status_code == 200, qualified.text
    candidate = qualified.json()["data"]
    assert candidate["affected_scope"] == expected
    loaded = client.get(f"/api/v1/maintenance/roadmap/{candidate['id']}", headers=worker)
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["data"]["affected_scope"] == expected


@pytest.mark.parametrize("binding", ["exact", "expiry", "missing_publication", "wrong_entry"])
def test_entry_scope_retains_exact_publication_without_reporter_envelope(client, monkeypatch, binding):
    from app.model.canonical import CanonicalRecordModel
    from app.model.knowledge_feedback import KnowledgeFeedbackSignal

    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="decision-entry-001")
    admin, maintainer, worker = _maintainer(client)
    _activate_route(client, monkeypatch, admin)
    answer = _gap(
        client, worker, "affected-entry", "Which Candidate publication contract applies? deployment=production",
    )
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=worker,
        json={"answer_id": answer["id"], "entry_id": "decision-entry-001", "label": "outdated"},
    ).json()["data"]

    async def prepare():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, publication)
            expected = {
                "entry_identity": "entry:decision-entry-001",
                "publication_identity": publication,
                "revision_identity": record.payload["editorial_revision_identity"],
            }
            raw = await session.get(KnowledgeFeedbackSignal, signal["id"])
            if binding == "missing_publication":
                raw.normalized_metadata = {
                    **raw.normalized_metadata, "knowledge_version_identities": ["published_knowledge_version:missing-scope"],
                }
            elif binding == "wrong_entry":
                raw.entry_id = "foreign-entry"
            await session.commit()
            return expected

    expected = {"entry_versions": [asyncio.run(prepare())], "gap_contexts": []}
    response = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "content-integrity", "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal["id"]],
        },
    )
    if binding not in {"exact", "expiry"}:
        assert response.status_code == 409, response.text
        assert client.get("/api/v1/maintenance/items", headers=maintainer).json()["data"]["items"] == []
        return
    assert response.status_code == 200, response.text
    item = response.json()["data"]
    assert item["affected_scope"] == expected
    if binding == "expiry":
        async def expire():
            async with client.app.state.test_auth_session_factory() as session:
                raw = await session.get(KnowledgeFeedbackSignal, signal["id"])
                raw.expires_at = datetime.now(UTC) - timedelta(seconds=1)
                await session.commit()
        asyncio.run(expire())
        assert client.get("/api/v1/knowledge-feedback", headers=worker).json()["items"] == []
    else:
        assert client.delete(f"/api/v1/knowledge-feedback/{signal['id']}", headers=worker).status_code == 200
    retained = client.get(f"/api/v1/maintenance/items/{item['id']}", headers=maintainer)
    assert retained.status_code == 200, retained.text
    assert retained.json()["data"]["affected_scope"] == expected
    assert retained.json()["data"]["signal_count"] == 0
    assert answer["id"] not in retained.text
    assert signal["id"] not in retained.text


@pytest.mark.parametrize("corruption", ["private_field", "scope_removed"])
def test_retained_scope_rejects_private_fields_and_loss_on_consolidation(client, corruption):
    from sqlalchemy import select, update

    from app.model.canonical import CanonicalEventModel
    from tests.integration.test_ticket27_maintenance import _reproduction_case

    maintainer, _, _, base, payload = _reproduction_case(client)
    combined = client.post(
        base + "/signals", headers=maintainer,
        json={"expected_revision": 2, "signal_ids": [payload["signal_id"]]},
    )
    assert combined.status_code == 200, combined.text
    identity = combined.json()["data"]["id"]

    async def corrupt():
        async with client.app.state.test_auth_session_factory() as session:
            event = await session.scalar(select(CanonicalEventModel).where(
                CanonicalEventModel.aggregate_id == identity, CanonicalEventModel.event_type == "signals_consolidated",
            ))
            scope = event.payload["changes"]["affected_scope"]
            if corruption == "private_field":
                scope = {**scope, "question": "injected private question"}
            else:
                scope = {**scope, "gap_contexts": []}
            connection = await session.connection()
            await connection.execute(
                update(CanonicalEventModel.__table__).where(CanonicalEventModel.id == event.id).values(
                    payload={**event.payload, "changes": {"affected_scope": scope}},
                ),
            )
            await session.commit()

    asyncio.run(corrupt())
    rejected = client.get(base, headers=maintainer)
    assert rejected.status_code == 409, rejected.text
    assert "injected private question" not in rejected.text
