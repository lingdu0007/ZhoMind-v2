import asyncio

import pytest
from sqlalchemy import delete, update

from app.model.canonical import CanonicalRecordModel
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize("coverage", ["partial", "complete", "superseded", "missing_failed", "damaged_failed", "diagnose_all_first"])
def test_consolidated_source_repair_requires_all_affected_targets(client, monkeypatch, coverage):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    entries = ["scope-source-a", "scope-source-b"]
    admin, maintainer, worker = _maintainer(client)
    provider = _activate_route(client, monkeypatch, admin)
    reporter = _headers(_register(client, username="scope-reporter"))
    question = "Which Candidate publication contract applies? deployment=production"
    signals = {}
    sources = {}
    for entry in entries:
        _publish(client, publisher, entry_id=entry)
        reported = _gap(client, reporter, f"scope-reported-{entry}", question)
        signal = client.post(
            "/api/v1/knowledge-feedback", headers=reporter,
            json={"answer_id": reported["id"], "entry_id": entry, "label": "outdated"},
        )
        assert signal.status_code == 200, signal.text
        signals[entry] = signal.json()["data"]["id"]
        owner = _headers(_register(client, username=f"{entry}-maintainer"))
        editorial = client.get(f"/api/v1/editorial/entries/{entry}", headers=owner).json()["data"]
        source_id = editorial["sources"][0]["source_id"]
        sources[entry] = (owner, source_id)
        changed = client.post(
            f"/api/v1/editorial/entries/{entry}/sources/{source_id}/availability", headers=owner,
            json={"availability": "changed_or_unreachable_awaiting_review"},
        )
        assert changed.status_code == 200, changed.text
    independent = _gap(client, worker, "scope-independent", question)
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "source-freshness", "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signals[entries[0]]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"

    def mutate(endpoint, actor, fields):
        revision = client.get(base, headers=maintainer).json()["data"]["revision"]
        response = client.post(base + endpoint, headers=actor, json={"expected_revision": revision, **fields})
        assert response.status_code == 200, response.text
        return response.json()["data"]

    mutate("/transition", maintainer, {"state": "triaged"})
    artifacts = []
    pending = []
    findings = []

    def repair(entry, fixture):
        owner, source_id = sources[entry]
        repaired = client.post(
            f"/api/v1/editorial/entries/{entry}/sources/{source_id}/availability", headers=owner,
            json={"availability": "verified_usable"},
        )
        assert repaired.status_code == 200, repaired.text
        replay = mutate("/replays", worker, {"fixture_identity": fixture["id"], "answer_id": independent["id"]})
        assert replay["passed"] is True
        artifacts.extend([fixture["id"], replay["id"], f"source:{source_id}"])
        projected = client.get(f"/api/v1/maintenance/fixtures/{fixture['id']}", headers=maintainer)
        assert projected.status_code == 200, projected.text
        assert projected.json()["data"]["latest_replay_identity"] == replay["id"]

    for index, entry in enumerate(entries):
        if index:
            mutate("/signals", maintainer, {"signal_ids": [signals[entry]]})
            if coverage == "partial":
                break
        fixture = mutate("/reproductions", worker, {
            "answer_id": independent["id"], "signal_id": signals[entry],
            "expected_outcome": "evidence_gated_answer", "confirmed_synthetic_fixture": True,
            "verified_observation": "stale_source", "entry_identity": f"entry:{entry}",
        })
        mutate("/diagnosis", maintainer, {"fixture_identity": fixture["id"], "observation": "stale_source"})
        finding = mutate("/findings", maintainer, {"fixture_identity": fixture["id"]})
        assert finding["fixture_identity"] == fixture["id"], "approval deduplicated a different verified target"
        findings.append(finding)
        if coverage == "diagnose_all_first":
            pending.append((entry, fixture))
            continue
        if not index:
            mutate("/transition", maintainer, {"state": "in_progress"})
        repair(entry, fixture)
    if pending:
        assert len({finding["id"] for finding in findings}) == 2
        assert len({finding["verification_fingerprint"] for finding in findings}) == 1
        mutate("/transition", maintainer, {"state": "in_progress"})
        for entry, fixture in pending:
            repair(entry, fixture)
    if coverage in {"superseded", "missing_failed", "damaged_failed"}:
        provider.unavailable = True
        failed = mutate("/replays", worker, {"fixture_identity": artifacts[0], "answer_id": independent["id"]})
        assert failed["passed"] is False
        if coverage != "superseded":
            async def damage_latest_record():
                async with client.app.state.test_auth_session_factory() as session:
                    connection = await session.connection()
                    if coverage == "missing_failed":
                        await connection.execute(delete(CanonicalRecordModel.__table__).where(
                            CanonicalRecordModel.stable_id == failed["id"],
                        ))
                    else:
                        record = await session.get(CanonicalRecordModel, failed["id"])
                        assert record is not None
                        await connection.execute(update(CanonicalRecordModel.__table__).where(
                            CanonicalRecordModel.stable_id == failed["id"],
                        ).values(payload={**record.payload, "fixture_identity": f"maintenance_item:{'0' * 32}"}))
                    await session.commit()

            asyncio.run(damage_latest_record())
    before = client.get(base, headers=maintainer).json()["data"]
    assert len(before["affected_scope"]["entry_versions"]) == 2
    resolved = client.post(
        base + "/resolution", headers=maintainer,
        json={"expected_revision": before["revision"], "disposition": "source-change", "artifact_identities": artifacts},
    )
    if coverage not in {"complete", "diagnose_all_first"}:
        assert resolved.status_code == 409, resolved.text
        current = client.get(base, headers=maintainer).json()["data"]
        assert current["state"] == "in_progress"
        assert current["revision"] == before["revision"]
    else:
        assert resolved.status_code == 200, resolved.text
        mutate("/transition", maintainer, {"state": "closed_confirmation"})
        assert client.get(base, headers=worker).json()["data"]["state"] == "closed_confirmation"
