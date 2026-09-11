import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from sqlalchemy import text

from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.editorial_authority.service import EditorialAuthorityService
from app.maintenance import resolution
from tests.integration import test_ticket25_withdrawal as publication_helpers
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_ticket27_confirmation_scope import (
    test_direct_confirmation_requires_findings_for_all_retained_targets as confirmation_case,
)
from tests.integration.test_ticket27_reacceptance_postgres import client as client
from tests.unit.test_editorial_authority import _release_assurance_authority_records


@pytest.mark.parametrize("writer_kind", ["integrity", "release_acceptance"])
def test_confirmation_serializes_current_eligibility_against_integrity_review(client, monkeypatch, writer_kind):
    original_reviews = resolution.eligible_reference_reviews
    original_integrity = EditorialAuthorityService.record_integrity_review
    original_status = DeliveryAcceptanceService.update_status
    writer_ready = Event()
    writer_pid = None
    writer = None
    observed_lock = False
    acceptance_identity = "delivery_acceptance_record:confirmation-release-assurance"
    administrator = _headers(_register(client, username="confirmation-release-admin", role="admin"))
    if writer_kind == "release_acceptance":
        assurance = {
            "contract_identity": "product_path:confirmation-contract",
            "calibration_identity": "configuration:confirmation-calibration",
            "frozen_acceptance_identity": acceptance_identity,
            "named_gate": "capability:confirmation-gate",
        }

        async def seed_assurance():
            async with client.app.state.test_auth_session_factory() as session:
                session.add_all(_release_assurance_authority_records(
                    entry_identity="entry:confirmation-a", contract_identity=assurance["contract_identity"],
                    calibration_identity=assurance["calibration_identity"], acceptance_identity=acceptance_identity,
                    named_gate=assurance["named_gate"], qualified_active_status=True,
                ))
                await session.commit()

        asyncio.run(seed_assurance())
        original_entry = publication_helpers._browser_ticket24_entry

        def release_entry(entry_id):
            entry = original_entry(entry_id)
            if entry_id == "confirmation-a":
                return entry.model_copy(update={"assurance_level": "release_assured", "release_assurance": assurance})
            return entry

        monkeypatch.setattr(publication_helpers, "_browser_ticket24_entry", release_entry)

    async def integrity(service, entry_id, payload, actor):
        nonlocal writer_pid
        writer_pid = await service.session.scalar(text("SELECT pg_backend_pid()"))
        writer_ready.set()
        return await original_integrity(service, entry_id, payload, actor)

    async def update_status(service, identity, payload, actor):
        nonlocal writer_pid
        if identity == acceptance_identity and payload.status.value == "suspended":
            writer_pid = await service.session.scalar(text("SELECT pg_backend_pid()"))
            writer_ready.set()
        return await original_status(service, identity, payload, actor)

    def record_integrity(review):
        if writer_kind == "release_acceptance":
            return client.post(
                f"/api/v1/acceptance/records/{acceptance_identity}/status", headers=administrator,
                json={
                    "status": "suspended", "reason_code": "integrity_failure",
                    "status_failure": {
                        "check_id": "check:entry-supported-query", "reason": "concurrent verified integrity failure",
                        "failure_kind": "entry_specific",
                        "blocking_scope": {"scope": "entry_version", "identity": "entry:confirmation-a"},
                        "evidence_links": ["evidence://maintenance/concurrent-confirmation"],
                    },
                },
            )
        owner = _headers(_register(client, username="confirmation-a-maintainer"))
        return client.post(
            "/api/v1/editorial/entries/confirmation-a/integrity-review", headers=owner,
            json={
                "publication_identity": review["publication_identity"],
                "revision_identity": review["revision_identity"],
                "source_identity": "source:source-confirmation-a",
                "defect": "integrity_defect", "confirmed_independent_review": True,
            },
        )

    def assert_invalidated(response):
        assert response.status_code == 200, response.text
        if writer_kind == "release_acceptance":
            assert response.json()["data"]["current_status"] == "suspended"
        else:
            assert response.json()["data"]["answer_eligible"] is False

    with ThreadPoolExecutor(max_workers=1) as executor:
        async def reviews(session, identities):
            nonlocal writer, observed_lock
            result = await original_reviews(session, identities)
            target = next((review for review in result if review["entry_identity"] == "entry:confirmation-a"), None)
            if target is None or writer is not None:
                return result
            writer = executor.submit(record_integrity, target)
            assert await asyncio.to_thread(writer_ready.wait, 10), "integrity writer did not reach its database session"
            deadline = asyncio.get_running_loop().time() + 10
            async with client.app.state.test_auth_session_factory() as observer:
                while asyncio.get_running_loop().time() < deadline:
                    waiting = await observer.scalar(
                        text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"), {"pid": writer_pid},
                    )
                    if waiting == "Lock":
                        observed_lock = True
                        break
                    if writer.done():
                        response = writer.result()
                        assert_invalidated(response)
                        break
                    await asyncio.sleep(0.01)
            assert observed_lock or writer.done(), "integrity writer neither completed nor entered a lock wait"
            return result

        monkeypatch.setattr(resolution, "eligible_reference_reviews", reviews)
        monkeypatch.setattr(EditorialAuthorityService, "record_integrity_review", integrity)
        monkeypatch.setattr(DeliveryAcceptanceService, "update_status", update_status)
        confirmation_case(client, monkeypatch, "complete")
        assert writer is not None
        response = writer.result(timeout=10)
        assert_invalidated(response)
        assert observed_lock, "confirmation closed after its supporting entry was already made ineligible"
