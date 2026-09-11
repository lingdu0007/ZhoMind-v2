import asyncio

import pytest

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


def _publish_content_repair(client, publisher, original, *, suffix="repair"):
    from sqlalchemy import select

    from app.editorial_authority.schemas import CreateEditorialEntryRequest, ReviseEditorialEntryRequest
    from app.editorial_authority.service import EditorialAuthorityService
    from app.model.user import User
    from app.service.identity_audit_service import IdentityAuditService
    from tests.browser_acceptance_api import _build_ticket24_candidate
    from tests.integration.test_ticket24_candidate_publication_api import _inspect_and_accept, _publish_selection

    entry_id = original["entry_id"]
    base = f"/api/v1/editorial/entries/{entry_id}"
    author = _headers(_register(client, username=f"{entry_id}-author"))
    reviewer = _headers(_register(client, username=f"{entry_id}-reviewer"))
    owner = _headers(_register(client, username=f"{entry_id}-maintainer"))
    entry = CreateEditorialEntryRequest.model_validate(original["entry"])
    entry.body["recommendation_or_reviewed_branches"] += (
        " The independently reviewed repair requires explicit administrator publication confirmation."
    )

    async def revise():
        async with client.app.state.test_auth_session_factory() as session:
            actor = await session.scalar(select(User).where(User.username == f"{entry_id}-author"))
            assert actor is not None
            return await EditorialAuthorityService(session).revise_entry(
                entry_id, ReviseEditorialEntryRequest(entry=entry, change_kind="material"), actor,
            )

    revised = asyncio.run(revise())
    assert revised["revision_identity"] != original["revision_identity"]
    accepted = client.post(base + "/maintainer-acceptance", headers=owner)
    assert accepted.status_code == 200, accepted.text
    denied = client.post(base + "/approve", headers=author)
    assert denied.status_code == 409, denied.text
    assert denied.json()["code"] == "EDITORIAL_SELF_APPROVAL_FORBIDDEN"
    approved = client.post(base + "/approve", headers=reviewer)
    assert approved.status_code == 200, approved.text
    exported = client.post(base + "/export", headers=publisher)
    assert exported.status_code == 200, exported.text

    async def build():
        async with client.app.state.test_auth_session_factory() as session:
            actor = await session.scalar(select(User).where(User.username == "ticket25-admin"))
            assert actor is not None
            identity = await IdentityAuditService(session).ensure_member_record(actor, admission_path="test")
            return await _build_ticket24_candidate(
                session, artifact=exported.json()["data"]["artifact"],
                bundle_id=f"{entry_id}-{suffix}", operation="replace", actor_identity=identity,
            )

    candidate = asyncio.run(build())
    _inspect_and_accept(client, candidate_id=candidate, headers=publisher)
    eligible = client.get(
        f"/api/v1/reviewed-release-bundles/candidates/{candidate}/publication-eligibility", headers=publisher,
    )
    assert eligible.status_code == 200, eligible.text
    published = client.post(
        "/api/v1/reviewed-release-bundles/publication-batches", headers=publisher,
        json={"confirmation_id": f"{entry_id}-{suffix}", "selected_items": [_publish_selection(candidate, eligible.json()["data"])]},
    )
    assert published.status_code == 200, published.text
    assert published.json()["data"]["batch_complete"], published.text
    return revised["revision_identity"], published.json()["data"]["published"][0]["publication_identity"]


@pytest.mark.parametrize("defect", ["known_contradiction", "integrity_defect"])
@pytest.mark.parametrize("freshness_grace", [False, True])
@pytest.mark.parametrize("finish", ["corruption", "repair", "repair_stale", "repair_chain"])
def test_content_diagnosis_requires_explicit_published_revision_integrity_review(client, monkeypatch, defect, freshness_grace, finish):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-content")
    admin, maintainer, worker = _maintainer(client)
    provider = _activate_route(client, monkeypatch, admin)
    owner = _headers(_register(client, username="maintenance-content-maintainer"))
    reporter = _headers(_register(client, username="content-reporter"))
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "content-reported", question)
    independent = _gap(client, worker, "content-independent", question)
    assert reported["answer_execution"]["outcome"] == "evidence_gated_answer"
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": reported["id"], "entry_id": "maintenance-content", "label": "insufficient_evidence"},
    )
    assert signal.status_code == 200, signal.text
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "content-integrity", "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    triaged = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"})
    assert triaged.status_code == 200, triaged.text
    reproduction = {
        "expected_revision": 2, "answer_id": independent["id"], "signal_id": signal.json()["data"]["id"],
        "expected_outcome": "evidence_gated_answer", "confirmed_synthetic_fixture": True,
        "verified_observation": "wrong_content", "entry_identity": "entry:maintenance-content",
    }
    unsupported = client.post(base + "/reproductions", headers=worker, json=reproduction)
    assert unsupported.status_code == 409, unsupported.text
    editorial = "/api/v1/editorial/entries/maintenance-content"
    original = client.get(editorial, headers=owner).json()["data"]
    if freshness_grace:
        started = client.post(
            editorial + "/freshness-review", headers=owner,
            json={
                "revision_identity": original["revision_identity"], "publication_identity": publication,
                "trigger_id": original["entry"]["freshness_triggers"][0]["trigger_id"],
            },
        )
        assert started.status_code == 200, started.text
        assert started.json()["data"]["answer_eligible"] is True
    review = {
        "revision_identity": original["revision_identity"], "publication_identity": publication,
        "source_identity": original["sources"][0]["source_identity"], "defect": defect,
        "confirmed_independent_review": True,
    }
    unauthorized = client.post(editorial + "/integrity-review", headers=maintainer, json=review)
    assert unauthorized.status_code == 403, unauthorized.text
    for patch in (
        {"publication_identity": "published_knowledge_version:" + "0" * 64},
        {"revision_identity": original["revision_identity"] + ".stale"},
        {"source_identity": "source:unrelated-source"},
    ):
        rejected = client.post(editorial + "/integrity-review", headers=owner, json={**review, **patch})
        assert rejected.status_code == 409, rejected.text
    for confirmation in (False, 1, "true"):
        unconfirmed = client.post(
            editorial + "/integrity-review", headers=owner, json={**review, "confirmed_independent_review": confirmation},
        )
        assert unconfirmed.status_code == 422, unconfirmed.text
    for _ in range(2):
        reviewed = client.post(editorial + "/integrity-review", headers=owner, json=review)
        assert reviewed.status_code == 200, reviewed.text
        assert reviewed.json()["data"]["answer_eligible"] is False
    refused = _gap(client, worker, "content-after-review", question)
    assert refused["answer_execution"]["outcome"] == "insufficient_evidence_reply"
    reproduced = client.post(base + "/reproductions", headers=worker, json=reproduction)
    assert reproduced.status_code == 200, reproduced.text
    fixture = reproduced.json()["data"]
    assert fixture["publication_review"]["integrity_review_event_id"]
    assert fixture["publication_review"]["publication_identity"] == publication
    diagnosed = client.post(
        base + "/diagnosis", headers=maintainer,
        json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "wrong_content"},
    )
    assert diagnosed.status_code == 200, diagnosed.text
    assert diagnosed.json()["data"]["disposition"] == "entry-revision"
    approved = client.post(
        base + "/findings", headers=maintainer, json={"expected_revision": 4, "fixture_identity": fixture["id"]},
    )
    assert approved.status_code == 200, approved.text
    retained = client.get(editorial, headers=owner).json()["data"]
    assert retained["integrity_review"]["publication_identity"] == publication
    assert retained["integrity_review"]["event_id"] == fixture["publication_review"]["integrity_review_event_id"]

    if finish.startswith("repair"):
        started = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"})
        assert started.status_code == 200, started.text
        premature = client.post(
            base + "/resolution", headers=maintainer,
            json={
                "expected_revision": 6, "disposition": "entry-revision",
                "artifact_identities": [fixture["id"], original["revision_identity"], publication],
            },
        )
        assert premature.status_code == 409, premature.text
        revision, replacement = _publish_content_repair(client, publisher, original)
        if finish == "repair_chain":
            next_original = client.get(editorial, headers=owner).json()["data"]
            revision, replacement = _publish_content_repair(client, publisher, next_original, suffix="repair-next")
        assert replacement != publication
        replayed = client.post(
            base + "/replays", headers=worker,
            json={"expected_revision": 6, "fixture_identity": fixture["id"], "answer_id": independent["id"]},
        )
        assert replayed.status_code == 200, replayed.text
        replay = replayed.json()["data"]
        assert replay["passed"] is True, replay
        assert replay["evidence_publication_identities"] == [replacement]
        assert replay["publication_review"]["revision_identity"] == revision
        for artifacts in (
            [fixture["id"], replay["id"]],
            [fixture["id"], replay["id"], revision],
            [fixture["id"], replay["id"], original["revision_identity"], publication],
        ):
            rejected = client.post(
                base + "/resolution", headers=maintainer,
                json={"expected_revision": 7, "disposition": "entry-revision", "artifact_identities": artifacts},
            )
            assert rejected.status_code == 409, rejected.text
            unchanged = client.get(base, headers=maintainer).json()["data"]
            assert unchanged["revision"] == 7
            assert unchanged["state"] == "in_progress"
            assert unchanged["result_links"] == []
        if finish == "repair_stale":
            changed = client.post(
                editorial + "/integrity-review", headers=owner,
                json={**review, "revision_identity": revision, "publication_identity": replacement},
            )
            assert changed.status_code == 200, changed.text
        resolved = client.post(
            base + "/resolution", headers=maintainer,
            json={
                "expected_revision": 7, "disposition": "entry-revision",
                "artifact_identities": [fixture["id"], replay["id"], revision, replacement],
            },
        )
        if finish == "repair_stale":
            assert resolved.status_code == 409, resolved.text
            assert client.get(base, headers=maintainer).json()["data"]["state"] == "in_progress"
            return
        assert resolved.status_code == 200, resolved.text
        assert resolved.json()["data"]["state"] == "resolved"
        assert f"evidence://maintenance/artifacts/{replacement}" in resolved.json()["data"]["result_links"]
        closed = client.post(
            base + "/transition", headers=maintainer, json={"expected_revision": 8, "state": "closed_confirmation"},
        )
        assert closed.status_code == 200, closed.text
        return

    async def corrupt_retained_review():
        from sqlalchemy import update

        from app.model.canonical import CanonicalEventModel

        async with client.app.state.test_auth_session_factory() as session:
            event = await session.get(CanonicalEventModel, retained["integrity_review"]["event_id"])
            assert event is not None
            connection = await session.connection()
            await connection.execute(
                update(CanonicalEventModel).where(CanonicalEventModel.id == event.id).values(
                    payload={**event.payload, "integrity_defect": False, "private_note": "private injected material"},
                ),
            )
            await session.commit()

    asyncio.run(corrupt_retained_review())
    for response in (
        client.get(editorial, headers=owner),
        client.post(editorial + "/integrity-review", headers=owner, json=review),
    ):
        assert response.status_code == 409, response.text
        assert "private injected material" not in response.text
    attempts = provider.attempts
    corrupt_answer = _gap(client, worker, "content-corrupted-review", question)
    assert corrupt_answer["answer_execution"]["outcome"] == "insufficient_evidence_reply"
    assert provider.attempts == attempts
