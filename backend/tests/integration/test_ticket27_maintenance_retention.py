import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest
from sqlalchemy import select

from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket27_maintenance import _reproduction_case


def test_feedback_expiring_during_containment_verification_cannot_be_linked_to_new_work(client, monkeypatch):
    from app.maintenance.service import MaintenanceService
    from app.model.canonical import CanonicalRecordModel
    from app.model.knowledge_feedback import MaintenanceSignalLink

    maintainer, _worker, _reporter, _base, reproduction = _reproduction_case(client)
    clock = datetime.now(UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz) if tz else clock.replace(tzinfo=None)

    original = MaintenanceService.containment

    async def verify_then_expire(self, **kwargs):
        nonlocal clock
        result = await original(self, **kwargs)
        clock += timedelta(days=181)
        return result

    monkeypatch.setattr("app.knowledge_feedback.service.datetime", Clock)
    monkeypatch.setattr("app.maintenance.service.datetime", Clock)
    monkeypatch.setattr(MaintenanceService, "containment", verify_then_expire)
    result = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "coverage-gap", "severity": "p3", "disposition": "needs-reproduction",
            "coverage_position": "evidence_sufficiency_refusal_and_acceptance",
            "work_owner_username": "maintenance-worker", "signal_ids": [reproduction["signal_id"]],
        },
    )
    assert result.status_code == 409, result.text
    assert result.json()["code"] == "MAINTENANCE_SIGNAL_UNAVAILABLE"

    async def retained_items():
        async with client.app.state.test_auth_session_factory() as session:
            items = list((await session.scalars(select(CanonicalRecordModel).where(
                CanonicalRecordModel.payload["schema"].as_string() == "maintenance_item/v1",
            ))).all())
            links = list((await session.scalars(select(MaintenanceSignalLink))).all())
            return len(items), len(links)

    assert asyncio.run(retained_items()) == (1, 1)


@pytest.mark.parametrize("operation", ["delete", "expiry_cleanup"])
def test_feedback_mutation_cannot_report_success_when_the_shared_write_fence_is_unavailable(client, monkeypatch, operation):
    from app.common.exceptions import AppError
    from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink
    from app.repository.chat_repository import ChatRepository

    _maintainer, _worker, reporter, _base, reproduction = _reproduction_case(client)

    async def unavailable(self):
        raise AppError(status_code=503, code="PRIVACY_WRITE_FENCE_UNAVAILABLE", message="privacy write fence unavailable")

    monkeypatch.setattr(ChatRepository, "acquire_private_conversation_write_fence", unavailable)
    if operation == "delete":
        response = client.delete(f"/api/v1/knowledge-feedback/{reproduction['signal_id']}", headers=reporter)
    else:
        response = client.get("/api/v1/knowledge-feedback", headers=reporter)
    assert response.status_code == 503, response.text
    assert response.json()["code"] == "PRIVACY_WRITE_FENCE_UNAVAILABLE"

    async def retained():
        async with client.app.state.test_auth_session_factory() as session:
            signal = await session.get(KnowledgeFeedbackSignal, reproduction["signal_id"])
            link = await session.scalar(select(MaintenanceSignalLink).where(
                MaintenanceSignalLink.signal_id == reproduction["signal_id"],
            ))
            return signal is not None and link is not None

    assert asyncio.run(retained())


@pytest.mark.parametrize("operation", ["delete", "expiry_cleanup"])
def test_consolidation_waits_for_started_deletion_and_cannot_restore_its_signal_link(client, monkeypatch, operation):
    from app.knowledge_feedback.service import KnowledgeFeedbackService
    from app.maintenance.service import MaintenanceService
    from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink
    from app.repository.chat_repository import ChatRepository

    maintainer, _worker, reporter, base, reproduction = _reproduction_case(client)
    if operation == "expiry_cleanup":
        async def expire():
            async with client.app.state.test_auth_session_factory() as session:
                signal = await session.get(KnowledgeFeedbackSignal, reproduction["signal_id"])
                assert signal is not None
                signal.expires_at = datetime.now(UTC) - timedelta(seconds=1)
                await session.commit()
        asyncio.run(expire())
    deleting, waiting, resume = Event(), Event(), Event()
    original_detach = KnowledgeFeedbackService.detach_signals
    original_consolidate = MaintenanceService.consolidate
    original_fence = ChatRepository.acquire_private_conversation_write_fence

    async def paused_detach(self, signal_ids, *, now):
        if reproduction["signal_id"] in signal_ids:
            deleting.set()
            if not await asyncio.to_thread(resume.wait, 5):
                raise AssertionError("deletion was not resumed")
        return await original_detach(self, signal_ids, now=now)

    async def marked_consolidation(self, *args, **kwargs):
        self.session.info["maintenance_retention_consolidation"] = True
        return await original_consolidate(self, *args, **kwargs)

    async def observed_fence(self):
        if self.session.info.get("maintenance_retention_consolidation"):
            waiting.set()
        await original_fence(self)

    monkeypatch.setattr(KnowledgeFeedbackService, "detach_signals", paused_detach)
    monkeypatch.setattr(MaintenanceService, "consolidate", marked_consolidation)
    monkeypatch.setattr(ChatRepository, "acquire_private_conversation_write_fence", observed_fence)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = (
            pool.submit(client.delete, f"/api/v1/knowledge-feedback/{reproduction['signal_id']}", headers=reporter)
            if operation == "delete" else pool.submit(client.get, "/api/v1/knowledge-feedback", headers=reporter)
        )
        try:
            assert deleting.wait(5), "deletion did not reach detachment"
            consolidation = pool.submit(
                client.post, base + "/signals", headers=maintainer,
                json={"expected_revision": 2, "signal_ids": [reproduction["signal_id"]]},
            )
            assert waiting.wait(5), "consolidation did not attempt its fence"
            with pytest.raises(TimeoutError):
                consolidation.result(timeout=0.1)
        finally:
            resume.set()
        deleted = deletion.result(timeout=5)
        rejected = consolidation.result(timeout=5)
    assert deleted.status_code == 200, deleted.text
    if operation == "delete":
        assert deleted.json()["deleted"] is True
    else:
        assert deleted.json()["items"] == []
    assert rejected.status_code == 409, rejected.text
    assert rejected.json()["code"] == "MAINTENANCE_SIGNAL_UNAVAILABLE"
    retained = client.get(base, headers=maintainer).json()["data"]
    assert retained["revision"] == 2
    assert retained["signal_count"] == 0

    async def survivors():
        async with client.app.state.test_auth_session_factory() as session:
            signal = await session.get(KnowledgeFeedbackSignal, reproduction["signal_id"])
            links = list((await session.scalars(select(MaintenanceSignalLink).where(
                MaintenanceSignalLink.signal_id == reproduction["signal_id"],
            ))).all())
            return signal, links

    assert asyncio.run(survivors()) == (None, [])
