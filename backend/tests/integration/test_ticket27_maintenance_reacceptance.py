import asyncio

import pytest

from tests.integration.test_delivery_acceptance import _activate, _create, _record
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


def _publication_acceptance(client, admin, publication, replay_identity=None, link_scope="both"):
    from app.delivery_acceptance.schemas import CandidatePublicationBindingInput
    from app.model.canonical import CanonicalRecordModel

    async def binding():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, publication)
            assert record is not None
            return {
                "candidate_identity": record.payload["candidate_id"],
                "published_knowledge_version_identity": publication,
                **{key: record.payload[key] for key in (
                    "inspection_record_identity", "acceptance_record_identity", "entry_identity",
                    "configuration_identity", "bundle_sha256", "frozen_input_sha256",
                )},
            }

    bound = asyncio.run(binding())
    identities = CandidatePublicationBindingInput.model_validate(bound).exact_identities
    declaration = _record()
    declaration["candidate_publication_binding"] = bound
    declaration["affected_scope"]["entry_identities"] = [bound["entry_identity"], publication]
    declaration["affected_scope"]["blocking_scope_identity"] = publication
    declaration["affected_scope"]["configuration_identities"].append(bound["configuration_identity"])
    declaration["product_identities"].append(bound["configuration_identity"])
    declaration["content_identities"] = sorted(identities - {bound["configuration_identity"]})
    links = [f"evidence://maintenance/artifacts/{replay_identity}"] if replay_identity else []
    if link_scope != "check_only":
        declaration["evidence_links"].extend(links)
    for check in declaration["checks"]:
        if "entry:decision-entry-001" in check.get("identity_dependencies", []):
            check["identity_dependencies"] = sorted(identities)
            if link_scope != "record_only":
                check["evidence_links"].extend(links)
    record = _create(client, admin, declaration)
    _activate(client, admin, record["record_id"])
    return record["record_id"]


@pytest.mark.parametrize("severity", ["p0", "p1"])
@pytest.mark.parametrize("reacceptance", ["fresh_bound", "preexisting", "fresh_unbound", "record_only", "check_only"])
def test_high_severity_source_repair_requires_acceptance_of_its_actual_replay(client, monkeypatch, severity, reacceptance):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-reacceptance")
    admin, maintainer, worker = _maintainer(client)
    provider = _activate_route(client, monkeypatch, admin)
    reporter = _headers(_register(client, username="reacceptance-reporter"))
    source_owner = _headers(_register(client, username="maintenance-reacceptance-maintainer"))
    entry_url = "/api/v1/editorial/entries/maintenance-reacceptance"
    entry = client.get(entry_url, headers=source_owner).json()["data"]
    source_id = entry["sources"][0]["source_id"]
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "reacceptance-reported", question)
    assert reported["answer_execution"]["outcome"] == "evidence_gated_answer"
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": reported["id"], "entry_id": "maintenance-reacceptance", "label": "outdated"},
    )
    assert signal.status_code == 200, signal.text
    containment = _publication_acceptance(client, admin, publication)
    old_acceptance = _publication_acceptance(client, admin, publication) if reacceptance == "preexisting" else None
    lost = client.post(
        entry_url + f"/sources/{source_id}/availability", headers=source_owner,
        json={"availability": "changed_or_unreachable_awaiting_review"},
    )
    assert lost.status_code == 200, lost.text
    suspended = client.post(
        f"/api/v1/acceptance/records/{containment}/status", headers=admin,
        json={
            "status": "suspended", "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query", "reason": "verified source integrity failure",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": publication},
                "evidence_links": ["evidence://maintenance/source-integrity"],
            },
        },
    )
    assert suspended.status_code == 200, suspended.text
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "source-freshness", "severity": severity, "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
            "containment_record_identity": containment,
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    independent = _gap(client, worker, "reacceptance-independent", question)
    reproduced = client.post(
        base + "/reproductions", headers=worker,
        json={
            "expected_revision": 2, "answer_id": independent["id"], "signal_id": signal.json()["data"]["id"],
            "expected_outcome": "evidence_gated_answer", "confirmed_synthetic_fixture": True,
            "verified_observation": "stale_source", "entry_identity": "entry:maintenance-reacceptance",
        },
    )
    assert reproduced.status_code == 200, reproduced.text
    fixture = reproduced.json()["data"]
    diagnosed = client.post(
        base + "/diagnosis", headers=maintainer,
        json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "stale_source"},
    )
    assert diagnosed.status_code == 200, diagnosed.text
    approved = client.post(
        base + "/findings", headers=maintainer, json={"expected_revision": 4, "fixture_identity": fixture["id"]},
    )
    assert approved.status_code == 200, approved.text
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"}).status_code == 200
    restored = client.post(
        entry_url + f"/sources/{source_id}/availability", headers=source_owner, json={"availability": "verified_usable"},
    )
    assert restored.status_code == 200, restored.text
    ordinary = _gap(client, worker, "ordinary-before-reacceptance", question)
    assert ordinary["answer_execution"]["outcome"] == "insufficient_evidence_reply"
    replayed = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": 6, "fixture_identity": fixture["id"], "answer_id": independent["id"]},
    )
    assert replayed.status_code == 200, replayed.text
    replay = replayed.json()["data"]
    assert replay["passed"] is True, {"replay": replay, "provider_attempts": provider.attempts}
    artifacts = [fixture["id"], replay["id"], f"source:{source_id}"]
    missing = client.post(
        base + "/resolution", headers=maintainer,
        json={"expected_revision": 7, "disposition": "source-change", "artifact_identities": artifacts},
    )
    assert missing.status_code == 409, missing.text
    accepted = old_acceptance or _publication_acceptance(
        client, admin, publication, replay["id"] if reacceptance != "fresh_unbound" else None,
        reacceptance if reacceptance in {"record_only", "check_only"} else "both",
    )
    resolved = client.post(
        base + "/resolution", headers=maintainer,
        json={"expected_revision": 7, "disposition": "source-change", "artifact_identities": [*artifacts, accepted]},
    )
    if reacceptance != "fresh_bound":
        assert resolved.status_code == 409, resolved.text
        assert client.get(base, headers=maintainer).json()["data"]["state"] == "in_progress"
        return
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["data"]["state"] == "resolved"
    assert f"evidence://maintenance/artifacts/{accepted}" in resolved.json()["data"]["result_links"]
    acceptance = client.get(f"/api/v1/acceptance/records/{accepted}", headers=admin).json()["data"]
    activation_id = next(event["event_id"] for event in reversed(acceptance["status_history"]) if event["status"] == "active")
    assert f"evidence://maintenance/artifacts/{accepted}:{activation_id}" in resolved.json()["data"]["result_links"]
    closed = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 8, "state": "closed_confirmation"})
    assert closed.status_code == 200, closed.text
    assert client.get(base, headers=worker).json()["data"]["state"] == "closed_confirmation"
    restored_answer = _gap(client, worker, "ordinary-after-reacceptance", question)
    assert restored_answer["answer_execution"]["outcome"] == "evidence_gated_answer"
    assert restored_answer["answer_execution"]["knowledge_version_identities"] == [publication]
