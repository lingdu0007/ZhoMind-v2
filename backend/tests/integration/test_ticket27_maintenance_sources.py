import asyncio

import pytest

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


@pytest.mark.parametrize(
    "source_changes_again",
    ["unchanged", "before_diagnosis", "after_replay", "retrieval_claim", "tied_events", "boundary_bypass"],
)
def test_source_diagnosis_uses_published_revision_after_a_later_editorial_draft(client, monkeypatch, source_changes_again):
    from sqlalchemy import select

    from app.editorial_authority.schemas import ReviseEditorialEntryRequest
    from app.editorial_authority.service import EditorialAuthorityService
    from app.model.user import User
    from tests.browser_acceptance_api import _browser_ticket24_entry

    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-source")
    admin, maintainer, worker = _maintainer(client)
    provider = _activate_route(client, monkeypatch, admin)
    source_owner = _headers(_register(client, username="maintenance-source-maintainer"))
    reporter = _headers(_register(client, username="maintenance-source-reporter"))
    entry_url = "/api/v1/editorial/entries/maintenance-source"
    original = client.get(entry_url, headers=source_owner).json()["data"]
    original_revision = original["revision_identity"]
    source_id = original["sources"][0]["source_id"]

    async def revise():
        async with client.app.state.test_auth_session_factory() as session:
            author = await session.scalar(select(User).where(User.username == "maintenance-source-author"))
            assert author is not None
            entry = _browser_ticket24_entry("maintenance-source").model_copy(
                update={
                    "approving_reviewer_username": "maintenance-source-reviewer",
                    "accountable_maintainer_username": "maintenance-source-maintainer",
                },
                deep=True,
            )
            entry.body["recommendation_or_reviewed_branches"] += " A later editorial draft remains unpublished."
            return await EditorialAuthorityService(session).revise_entry(
                "maintenance-source",
                ReviseEditorialEntryRequest(entry=entry, change_kind="material"),
                author,
            )

    revised = asyncio.run(revise())
    assert revised["revision_identity"] != original_revision
    accepted = client.post(entry_url + "/maintainer-acceptance", headers=source_owner)
    assert accepted.status_code == 200, accepted.text
    author_headers = _headers(_register(client, username="maintenance-source-author"))
    self_approval = client.post(entry_url + "/approve", headers=author_headers)
    assert self_approval.status_code == 409, self_approval.text
    assert self_approval.json()["code"] == "EDITORIAL_SELF_APPROVAL_FORBIDDEN"
    lost = client.post(
        entry_url + f"/sources/{source_id}/availability",
        headers=source_owner,
        json={"availability": "changed_or_unreachable_awaiting_review"},
    )
    assert lost.status_code == 200, lost.text
    assert lost.json()["data"]["answer_eligible"] is False

    def answer(headers, session_id, outcome="insufficient_evidence_reply"):
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "session_id": session_id,
                "message": "Which Candidate publication contract applies? deployment=production",
                "query_conditions": [
                    {"condition_id": "ticket24-production", "field": "deployment", "operator": "equals", "value": "production"}
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["outcome"] == outcome
        messages = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()["messages"]
        return next(message for message in reversed(messages) if message["type"] == "assistant")

    reference = None
    if source_changes_again == "retrieval_claim":
        for availability in ("verified_usable", "changed_or_unreachable_awaiting_review"):
            changed = client.post(
                entry_url + f"/sources/{source_id}/availability",
                headers=source_owner,
                json={"availability": availability},
            )
            assert changed.status_code == 200, changed.text
            if availability == "verified_usable":
                reference = answer(worker, "source-historical-supported", "evidence_gated_answer")
    reported = answer(reporter, "source-reported")
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=reporter,
        json={"answer_id": reported["id"], "label": "outdated"},
    )
    assert signal.status_code == 200, signal.text
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "source-freshness",
            "severity": "p2",
            "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    triaged = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"})
    assert triaged.status_code == 200, triaged.text
    independent = answer(worker, "source-independent")
    reproduced = client.post(
        base + "/reproductions",
        headers=worker,
        json={
            "expected_revision": 2,
            "answer_id": independent["id"],
            "signal_id": signal.json()["data"]["id"],
            "expected_outcome": (
                "insufficient_evidence_reply" if source_changes_again == "boundary_bypass" else "evidence_gated_answer"
            ),
            "confirmed_synthetic_fixture": True,
            "verified_observation": "retrieval_miss" if reference else "stale_source",
            "entry_identity": "entry:maintenance-source",
            **({"reference_answer_id": reference["id"]} if reference else {}),
        },
    )
    if reference:
        assert reproduced.status_code == 409, reproduced.text
        return
    assert reproduced.status_code == 200, reproduced.text
    fixture = reproduced.json()["data"]
    assert fixture["publication_review"]["publication_identity"] == publication
    assert fixture["publication_review"]["revision_identity"] == original_revision
    assert fixture["publication_review"]["source_states"] == ["changed_or_unreachable_awaiting_review"]
    if source_changes_again == "before_diagnosis":
        for availability in ("verified_usable", "changed_or_unreachable_awaiting_review"):
            changed = client.post(
                entry_url + f"/sources/{source_id}/availability",
                headers=source_owner,
                json={"availability": availability},
            )
            assert changed.status_code == 200, changed.text
    diagnosed = client.post(
        base + "/diagnosis",
        headers=maintainer,
        json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "stale_source"},
    )
    if source_changes_again == "before_diagnosis":
        assert diagnosed.status_code == 409, diagnosed.text
        return
    assert diagnosed.status_code == 200, diagnosed.text
    assert diagnosed.json()["data"]["disposition"] == "source-change"
    approved = client.post(base + "/findings", headers=maintainer, json={"expected_revision": 4, "fixture_identity": fixture["id"]})
    assert approved.status_code == 200, approved.text
    started = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"})
    assert started.status_code == 200, started.text
    if source_changes_again == "boundary_bypass":
        replayed = client.post(
            base + "/replays", headers=worker,
            json={"expected_revision": 6, "fixture_identity": fixture["id"], "answer_id": independent["id"]},
        )
        assert replayed.status_code == 200, replayed.text
        replay = replayed.json()["data"]
        assert replay["passed"] is True
        attempted = client.post(
            base + "/resolution",
            headers=maintainer,
            json={
                "expected_revision": 7,
                "disposition": "boundary-query",
                "artifact_identities": [fixture["id"], replay["id"]],
            },
        )
        assert attempted.status_code == 409, attempted.text
        retained = client.get(base, headers=worker).json()["data"]
        assert retained["state"] == "in_progress"
        assert retained["revision"] == 7
        assert retained["disposition"] == "source-change"
        return
    restored = client.post(
        entry_url + f"/sources/{source_id}/availability",
        headers=source_owner,
        json={"availability": "verified_usable"},
    )
    assert restored.status_code == 200, restored.text
    replayed = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": 6, "fixture_identity": fixture["id"], "answer_id": independent["id"]},
    )
    assert replayed.status_code == 200, replayed.text
    replay = replayed.json()["data"]
    if replay["passed"] is not True:
        pytest.fail(str({
            "replay": replay,
            "provider_attempts": provider.attempts,
        }))
    assert replay["evidence_publication_identities"] == [publication]
    missing_change = client.post(
        base + "/resolution",
        headers=maintainer,
        json={"expected_revision": 7, "disposition": "source-change", "artifact_identities": [fixture["id"], replay["id"]]},
    )
    assert missing_change.status_code == 409, missing_change.text
    if source_changes_again == "after_replay":
        for availability in ("changed_or_unreachable_awaiting_review", "verified_usable"):
            changed = client.post(
                entry_url + f"/sources/{source_id}/availability",
                headers=source_owner,
                json={"availability": availability},
            )
            assert changed.status_code == 200, changed.text
    if source_changes_again == "tied_events":
        from app.model.canonical import CanonicalEventModel

        async def append_tied_events():
            async with client.app.state.test_auth_session_factory() as session:
                event = await session.get(CanonicalEventModel, replay["publication_review"]["source_facts"][0]["status_event_id"])
                assert event is not None
                for identity, previous, target in [
                    ("0" * 31 + "1", "verified_usable", "changed_or_unreachable_awaiting_review"),
                    ("0" * 31 + "2", "changed_or_unreachable_awaiting_review", "verified_usable"),
                ]:
                    assert identity < event.id
                    session.add(
                        CanonicalEventModel(
                            id=identity,
                            aggregate_id=event.aggregate_id,
                            aggregate_kind=event.aggregate_kind,
                            event_type=event.event_type,
                            from_state=previous,
                            to_state=target,
                            recorded_by=event.recorded_by,
                            occurred_at=event.occurred_at,
                            payload=dict(event.payload),
                        )
                    )
                await session.commit()

        asyncio.run(append_tied_events())
    resolved = client.post(
        base + "/resolution",
        headers=maintainer,
        json={
            "expected_revision": 7,
            "disposition": "source-change",
            "artifact_identities": [fixture["id"], replay["id"], f"source:{source_id}"],
        },
    )
    if source_changes_again in {"after_replay", "tied_events"}:
        assert resolved.status_code == 409, resolved.text
        return
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["data"]["state"] == "resolved"
    assert any(f"source:{source_id}:" in link for link in resolved.json()["data"]["result_links"])
    closed = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 8, "state": "closed_confirmation"})
    assert closed.status_code == 200, closed.text
    assert client.get(base, headers=worker).json()["data"]["state"] == "closed_confirmation"
