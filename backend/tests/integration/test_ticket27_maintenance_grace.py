import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


def test_concurrent_freshness_registration_retains_one_review_clock(client, monkeypatch):
    from sqlalchemy import select

    from app.editorial_authority.service import EditorialAuthorityService
    from app.model.canonical import CanonicalEventModel

    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="concurrent-grace")
    owner = _headers(_register(client, username="concurrent-grace-maintainer"))
    base = "/api/v1/editorial/entries/concurrent-grace"
    original = client.get(base, headers=owner).json()["data"]
    payload = {
        "revision_identity": original["revision_identity"], "publication_identity": publication,
        "trigger_id": original["entry"]["freshness_triggers"][0]["trigger_id"],
    }
    first_read, second_read, release = Event(), Event(), Event()
    original_read = EditorialAuthorityService._entry_and_events

    async def paused_read(self, entry_id, **kwargs):
        result = await original_read(self, entry_id, **kwargs)
        if entry_id == "concurrent-grace" and kwargs.get("for_update"):
            if not first_read.is_set():
                first_read.set()
                assert await asyncio.to_thread(release.wait, 10)
            else:
                second_read.set()
        return result

    monkeypatch.setattr(EditorialAuthorityService, "_entry_and_events", paused_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, base + "/freshness-review", headers=owner, json=payload)
        try:
            assert first_read.wait(10)
            second = pool.submit(client.post, base + "/freshness-review", headers=owner, json=payload)
            assert not second_read.wait(0.5), "second registration read before the first committed"
        finally:
            release.set()
        responses = [first.result(timeout=15), second.result(timeout=15)]
    for response in responses:
        assert response.status_code == 200, response.text

    async def retained_reviews():
        async with client.app.state.test_auth_session_factory() as session:
            return list((await session.scalars(select(CanonicalEventModel).where(
                CanonicalEventModel.aggregate_id == "entry:concurrent-grace",
                CanonicalEventModel.payload["action"].as_string() == "freshness_review_requested",
            ))).all())

    events = asyncio.run(retained_reviews())
    assert len(events) == 1
    retained = client.get(base, headers=owner).json()["data"]
    assert retained["freshness_review"]["started_at"] == events[0].payload["needs_review_at"]


@pytest.mark.parametrize("boundary", ["within", "exact", "expired", "source_lost"])
def test_p2_freshness_review_uses_real_authority_without_resetting_grace(client, monkeypatch, boundary):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-grace")
    admin, maintainer, reporter = _maintainer(client)
    source_owner = _headers(_register(client, username="maintenance-grace-maintainer"))
    _activate_route(client, monkeypatch, admin)
    question = "Which Candidate publication contract applies? deployment=production"
    answer = _gap(client, reporter, "grace-report", question)
    assert answer["answer_execution"]["outcome"] == "evidence_gated_answer"
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": answer["id"], "entry_id": "maintenance-grace", "label": "outdated"},
    )
    assert signal.status_code == 200, signal.text
    item = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "source-freshness", "severity": "p2", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-grace-maintainer", "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert item.status_code == 200, item.text
    base = "/api/v1/editorial/entries/maintenance-grace"
    original = client.get(base, headers=source_owner).json()["data"]
    request = {
        "revision_identity": original["revision_identity"],
        "publication_identity": publication,
        "trigger_id": original["entry"]["freshness_triggers"][0]["trigger_id"],
    }
    clock = datetime.now(UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz) if tz else clock.replace(tzinfo=None)

    monkeypatch.setattr("app.editorial_authority.service.datetime", Clock)
    denied = client.post(base + "/freshness-review", headers=maintainer, json=request)
    assert denied.status_code == 403, denied.text
    for patch in (
        {"revision_identity": original["revision_identity"] + ".stale"},
        {"publication_identity": "published_knowledge_version:" + "0" * 64},
        {"trigger_id": "unreviewed-trigger"},
    ):
        invalid = client.post(base + "/freshness-review", headers=source_owner, json={**request, **patch})
        assert invalid.status_code == 409, invalid.text
    started = client.post(base + "/freshness-review", headers=source_owner, json=request)
    assert started.status_code == 200, started.text
    assert started.json()["data"]["answer_eligible"] is True
    clock += {
        "within": timedelta(days=6, hours=23),
        "exact": timedelta(days=7),
        "expired": timedelta(days=7, seconds=1),
        "source_lost": timedelta(days=1),
    }[boundary]
    if boundary == "source_lost":
        lost = client.post(
            base + f"/sources/{original['sources'][0]['source_id']}/availability", headers=source_owner,
            json={"availability": "unavailable_for_new_evidence"},
        )
        assert lost.status_code == 200, lost.text
    repeated = client.post(base + "/freshness-review", headers=source_owner, json=request)
    assert repeated.status_code == 200, repeated.text
    eligible = boundary not in {"expired", "source_lost"}
    assert repeated.json()["data"]["answer_eligible"] is eligible
    replay = _gap(client, source_owner, "grace-replay", question)
    expected = "evidence_gated_answer" if eligible else "insufficient_evidence_reply"
    assert replay["answer_execution"]["outcome"] == expected, replay["answer_execution"]
    retained = client.get(f"/api/v1/maintenance/items/{item.json()['data']['id']}", headers=source_owner)
    assert retained.status_code == 200, retained.text
    assert retained.json()["data"]["severity"] == "p2"
    assert retained.json()["data"]["state"] == "open"


def test_p3_triaged_work_remains_in_the_named_owners_queue_without_claiming_resolution(client):
    _admin, maintainer, worker = _maintainer(client)
    answer = _gap(client, worker, "p3-queue-report", "Which deployment validation evidence is available?")
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=worker,
        json={"answer_id": answer["id"], "label": "out_of_scope"},
    )
    assert signal.status_code == 200, signal.text
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "content-integrity", "severity": "p3", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    identity = created.json()["data"]["id"]
    triaged = client.post(
        f"/api/v1/maintenance/items/{identity}/transition", headers=maintainer,
        json={"expected_revision": 1, "state": "triaged"},
    )
    assert triaged.status_code == 200, triaged.text
    context = client.get("/api/v1/maintenance/context", headers=worker).json()["data"]
    for headers in (maintainer, worker):
        listing = client.get("/api/v1/maintenance/items", headers=headers)
        assert listing.status_code == 200, listing.text
        queued = next(item for item in listing.json()["data"]["items"] if item["id"] == identity)
        assert queued["state"] == "triaged"
        assert queued["severity"] == "p3"
        assert queued["work_owner"] == context["member_identity"]
        assert queued["disposition"] == "needs-reproduction"
        assert queued["result_links"] == []


@pytest.mark.parametrize("corruption", ["actor", "acceptance", "publication", "timestamp", "extra", "duplicate"])
def test_freshness_review_read_and_retry_require_qualified_retained_event(client, monkeypatch, corruption):
    from sqlalchemy import select, update

    from app.model.canonical import CanonicalEventModel

    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="review-authority")
    provider = _activate_route(client, monkeypatch, publisher)
    owner = _headers(_register(client, username="review-authority-maintainer"))
    base = "/api/v1/editorial/entries/review-authority"
    original = client.get(base, headers=owner).json()["data"]
    payload = {
        "revision_identity": original["revision_identity"], "publication_identity": publication,
        "trigger_id": original["entry"]["freshness_triggers"][0]["trigger_id"],
    }
    started = client.post(base + "/freshness-review", headers=owner, json=payload)
    assert started.status_code == 200, started.text

    async def corrupt():
        async with client.app.state.test_auth_session_factory() as session:
            event = await session.scalar(select(CanonicalEventModel).where(
                CanonicalEventModel.aggregate_id == "entry:review-authority",
                CanonicalEventModel.payload["action"].as_string() == "freshness_review_requested",
            ))
            assert event is not None
            if corruption == "duplicate":
                session.add(CanonicalEventModel(
                    aggregate_id=event.aggregate_id, aggregate_kind=event.aggregate_kind,
                    event_type=event.event_type, from_state=event.from_state, to_state=event.to_state,
                    recorded_by=event.recorded_by, payload=dict(event.payload), occurred_at=event.occurred_at,
                ))
            else:
                patch = {
                    "acceptance": {"maintainer_acceptance_event_id": "0" * 32},
                    "publication": {"publication_identity": "published_knowledge_version:" + "0" * 64},
                    "timestamp": {"needs_review_at": (datetime.now(UTC) + timedelta(days=30)).isoformat()},
                    "extra": {"note": "private review corruption"},
                }.get(corruption, {})
                values = {"recorded_by": "member:" + "0" * 32} if corruption == "actor" else {"payload": {**event.payload, **patch}}
                connection = await session.connection()
                await connection.execute(update(CanonicalEventModel).where(CanonicalEventModel.id == event.id).values(**values))
            await session.commit()

    asyncio.run(corrupt())
    for response in (
        client.get(base, headers=owner),
        client.post(base + "/freshness-review", headers=owner, json=payload),
    ):
        assert response.status_code == 409, response.text
        assert "private review corruption" not in response.text
    answer = _gap(
        client, owner, "invalid-review-answer",
        "Which Candidate publication contract applies? deployment=production",
    )
    assert answer["answer_execution"]["outcome"] == "insufficient_evidence_reply", answer["answer_execution"]
    assert provider.attempts == 0
