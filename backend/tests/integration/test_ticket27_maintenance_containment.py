import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from tests.integration.test_delivery_acceptance import _activate, _create, _record
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize("severity", ["p0", "p1"])
@pytest.mark.parametrize(
    "scope_case",
    [
        "matching_entry", "during_provider", "during_provider_sse", "unrelated_entry",
        "unrelated_version", "broader_collection", "foreign_collection",
    ],
)
def test_entry_containment_binds_the_reported_publication_and_minimum_scope(client, monkeypatch, severity, scope_case):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="decision-entry-001")
    admin, maintainer, worker = _maintainer(client)
    provider = _activate_route(client, monkeypatch, admin)
    matching_scope = scope_case in {"matching_entry", "during_provider", "during_provider_sse"}
    answer = _gap(
        client, worker, "contained-publication", "Which Candidate publication contract applies? deployment=production"
    )
    assert answer["answer_execution"]["outcome"] == "evidence_gated_answer"
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=worker,
        json={"answer_id": answer["id"], "entry_id": "decision-entry-001", "label": "outdated"},
    )
    assert signal.status_code == 200, signal.text
    matching_signal = None
    gap_signal = None
    if matching_scope:
        matching_answer = _gap(
            client, worker, "second-contained-publication",
            "Which Candidate publication contract applies? deployment=production",
        )
        matching_signal = client.post(
            "/api/v1/knowledge-feedback", headers=worker,
            json={"answer_id": matching_answer["id"], "entry_id": "decision-entry-001", "label": "outdated"},
        )
        assert matching_signal.status_code == 200, matching_signal.text
        gap_answer = _gap(client, worker, "unbound-gap", "An uncovered decision about lunar deployment?")
        assert gap_answer["answer_execution"]["outcome"] == "insufficient_evidence_reply"
        gap_signal = client.post(
            "/api/v1/knowledge-feedback", headers=worker,
            json={"answer_id": gap_answer["id"], "label": "insufficient_evidence"},
        )
        assert gap_signal.status_code == 200, gap_signal.text
    unrelated_publication = (
        _publish(client, publisher, entry_id="unrelated-entry") if not matching_scope else None
    )
    from app.delivery_acceptance.schemas import CandidatePublicationBindingInput
    from app.model.canonical import CanonicalRecordModel

    acceptance_publication = unrelated_publication if scope_case == "unrelated_version" else publication

    async def publication_binding():
        async with client.app.state.test_auth_session_factory() as session:
            record = await session.get(CanonicalRecordModel, acceptance_publication)
            assert record is not None
            return {
                "candidate_identity": record.payload["candidate_id"],
                "published_knowledge_version_identity": acceptance_publication,
                **{
                    key: record.payload[key]
                    for key in (
                        "inspection_record_identity", "acceptance_record_identity", "entry_identity",
                        "configuration_identity", "bundle_sha256", "frozen_input_sha256",
                    )
                },
            }

    binding = asyncio.run(publication_binding())
    identities = CandidatePublicationBindingInput.model_validate(binding).exact_identities
    declaration = _record()
    if scope_case == "foreign_collection":
        declaration["affected_scope"]["collection_identities"].append("collection:unrelated-corpus")
    declaration["candidate_publication_binding"] = binding
    declaration["affected_scope"]["entry_identities"].extend([acceptance_publication, "entry:unrelated-entry"])
    declaration["affected_scope"]["configuration_identities"].append(binding["configuration_identity"])
    declaration["product_identities"].append(binding["configuration_identity"])
    declaration["content_identities"] = sorted(
        (identities - {binding["configuration_identity"]}) | {"entry:decision-entry-001", "entry:unrelated-entry"}
    )
    for check in declaration["checks"]:
        if "entry:decision-entry-001" in check.get("identity_dependencies", []):
            check["identity_dependencies"] = sorted(identities | {"entry:decision-entry-001", "entry:unrelated-entry"})
    record = _create(client, admin, declaration)
    _activate(client, admin, record["record_id"])
    blocking = (
        {"scope": "collection", "identity": "collection:production-rag-agent-engineering"}
        if scope_case == "broader_collection"
        else {"scope": "entry_version", "identity": "entry:unrelated-entry" if scope_case == "unrelated_entry" else publication}
    )
    if scope_case == "unrelated_version":
        blocking = {"scope": "entry_version", "identity": unrelated_publication}
    if scope_case == "foreign_collection":
        blocking = {"scope": "collection", "identity": "collection:unrelated-corpus"}
    suspended = client.post(
        f"/api/v1/acceptance/records/{record['record_id']}/status",
        headers=admin,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "verified_integrity_failure",
                "failure_kind": (
                    "shared_privacy" if scope_case == "broader_collection"
                    else "collection_retrieval" if scope_case == "foreign_collection" else "entry_specific"
                ),
                "blocking_scope": blocking,
                "evidence_links": ["evidence://maintenance/verified-entry-integrity"],
            },
        },
    )
    assert suspended.status_code == 200, suspended.text
    creation = {
            "classification": "content-integrity", "severity": severity, "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance", "work_owner_username": "maintenance-worker",
            "signal_ids": [signal.json()["data"]["id"]], "containment_record_identity": record["record_id"],
    }
    if scope_case.startswith("during_provider"):
        created = None
        original_complete = provider.complete

        async def contain_before_completion(*args, **kwargs):
            nonlocal created
            result = await original_complete(*args, **kwargs)
            created = await asyncio.to_thread(client.post, "/api/v1/maintenance/items", headers=maintainer, json=creation)
            return result

        with monkeypatch.context() as interception:
            interception.setattr(provider, "complete", contain_before_completion)
            late = client.post(
                "/api/v1/chat/stream" if scope_case.endswith("_sse") else "/api/v1/chat", headers=worker,
                json={
                    "session_id": "containment-during-provider",
                    "message": "Which Candidate publication contract applies? deployment=production",
                },
            )
        assert created is not None and created.status_code == 200
        if scope_case.endswith("_sse"):
            from tests.integration.test_ticket20_answer_execution_persistence import _sse_event_data

            assert late.status_code == 200, late.text
            assert "event: outcome" not in late.text
            assert "event: content" not in late.text
            streamed = _sse_event_data(late.text, "answer_execution")["answer_execution"]
            assert streamed["state"] == "failed"
            assert streamed.get("outcome") is None
            assert _sse_event_data(late.text, "error")["code"] == "ANSWER_EVIDENCE_WITHDRAWN"
        else:
            assert late.status_code == 409, late.json().get("code")
        history = client.get("/api/v1/sessions/containment-during-provider", headers=worker).json()["messages"]
        terminal = next(message["answer_execution"] for message in reversed(history) if message["type"] == "assistant")
        assert terminal["state"] == "failed"
        assert terminal.get("outcome") is None
        if scope_case.endswith("_sse"):
            assert terminal == streamed
    else:
        created = client.post("/api/v1/maintenance/items", headers=maintainer, json=creation)
    if not matching_scope:
        assert created.status_code == 409, created.text
        return
    assert created.status_code == 200, created.text
    spoofed = client.post(
        "/api/v1/chat", headers=worker,
        json={
            "session_id": "forged-maintenance-context",
            "message": "Which Candidate publication contract applies? deployment=production",
            "containment_verification": {"item_identity": created.json()["data"]["id"]},
        },
    )
    assert spoofed.status_code == 422, spoofed.text
    contained = _gap(
        client, worker, "after-maintenance-containment",
        "Which Candidate publication contract applies? deployment=production",
    )
    assert contained["answer_execution"]["outcome"] == "insufficient_evidence_reply", contained["answer_execution"]
    assert contained["answer_execution"]["knowledge_version_identities"] == []
    unaffected_publication = _publish(client, publisher, entry_id="unrelated-entry")
    unaffected = _gap(
        client, worker, "unaffected-maintenance-publication",
        "Which Candidate publication contract applies? deployment=production",
    )
    assert unaffected["answer_execution"]["outcome"] == "evidence_gated_answer", unaffected["answer_execution"]
    assert unaffected["answer_execution"]["knowledge_version_identities"] == [unaffected_publication]
    item = created.json()["data"]
    assert item["blocking_scope"] == blocking
    assert item["containment_event_id"] == suspended.json()["data"]["status_history"][-1]["event_id"]
    assert item["administrator_identity"] == suspended.json()["data"]["status_history"][-1]["recorded_by"]
    assert publication not in suspended.json()["data"]["accepted_scope"]["entry_identities"]
    assert "entry:unrelated-entry" in suspended.json()["data"]["accepted_scope"]["entry_identities"]
    base = f"/api/v1/maintenance/items/{item['id']}"
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"}).status_code == 200
    assert client.post(base + "/transition", headers=maintainer, json={"expected_revision": 2, "state": "deferred"}).status_code == 409
    roadmap = client.post(
        base + "/roadmap", headers=maintainer,
        json={
            "expected_revision": 2, "owner_username": "maintenance-worker",
            "desired_outcome": "expand_coverage", "bounded_work_reason": "requires_separate_scope",
            "review_date": (datetime.now(UTC).date() + timedelta(days=1)).isoformat(),
        },
    )
    assert roadmap.status_code == 409, roadmap.text
    assert roadmap.json()["code"] == "ROADMAP_INTEGRITY_DEFERRAL_FORBIDDEN"
    assert client.get(base, headers=maintainer).json()["data"]["revision"] == 2
    assert client.get("/api/v1/maintenance/roadmap", headers=maintainer).json()["data"]["candidates"] == []
    assert matching_signal is not None and gap_signal is not None
    matching = client.post(
        base + "/signals", headers=maintainer,
        json={"expected_revision": 2, "signal_ids": [matching_signal.json()["data"]["id"]]},
    )
    assert matching.status_code == 200, matching.text
    rejected = client.post(
        base + "/signals", headers=maintainer,
        json={"expected_revision": 3, "signal_ids": [gap_signal.json()["data"]["id"]]},
    )
    assert rejected.status_code == 409, rejected.text
    assert rejected.json()["code"] == "MAINTENANCE_CONTAINMENT_REQUIRED"
    retained = client.get(base, headers=maintainer).json()["data"]
    assert retained["revision"] == 3
    assert retained["signal_count"] == 2
    successor = _create(client, admin, {**declaration, "replaces_record_identity": record["record_id"]})
    _activate(client, admin, successor["record_id"])
    superseded = client.post(
        f"/api/v1/acceptance/records/{record['record_id']}/status", headers=admin,
        json={
            "status": "superseded", "reason_code": "superseded_by_record",
            "superseding_record_identity": successor["record_id"],
        },
    )
    assert superseded.status_code == 200, superseded.text
    stale = client.post(
        base + "/signals", headers=maintainer,
        json={"expected_revision": 3, "signal_ids": [matching_signal.json()["data"]["id"]]},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["code"] == "MAINTENANCE_CONTAINMENT_REQUIRED"
    assert client.get(base, headers=maintainer).json()["data"]["revision"] == 3
