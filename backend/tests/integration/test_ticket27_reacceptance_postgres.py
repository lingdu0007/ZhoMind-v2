import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from app.delivery_acceptance.service import DeliveryAcceptanceService
from tests.integration import test_privacy_operations_boundaries as boundaries
from tests.integration import test_ticket27_maintenance_reacceptance as reacceptance_cases


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = os.getenv("TICKET27_POSTGRES_TEST_URL")
    if not url:
        pytest.skip("TICKET27_POSTGRES_TEST_URL is required for PostgreSQL row-lock verification")
    parsed = make_url(url)
    if parsed.drivername != "postgresql+asyncpg" or not (parsed.database or "").startswith("ticket27_test"):
        raise ValueError("an isolated ticket27_test PostgreSQL database is required")
    schema = f"ticket27_{uuid4().hex}"
    engine = create_async_engine(
        url, poolclass=NullPool, connect_args={"server_settings": {"search_path": schema}},
    )

    async def create_schema():
        async with engine.begin() as connection:
            await connection.execute(CreateSchema(schema))

    async def remove_schema():
        try:
            async with engine.begin() as connection:
                await connection.execute(DropSchema(schema, cascade=True))
        finally:
            await engine.dispose()

    asyncio.run(create_schema())
    try:
        monkeypatch.setattr(boundaries, "create_async_engine", lambda _url: engine)
        yield from boundaries.client.__wrapped__(tmp_path)
    finally:
        asyncio.run(remove_schema())


@pytest.mark.parametrize("severity", ["p0", "p1"])
def test_resolution_serializes_reacceptance_against_administrator_suspension(client, monkeypatch, severity):
    original_accept = reacceptance_cases._publication_acceptance
    original_projection = DeliveryAcceptanceService.get_projection
    original_update = DeliveryAcceptanceService.update_status
    target = {}
    writer_ready = Event()
    writer_pid = None
    writer = None
    observed_lock = False

    def accept(*args, **kwargs):
        identity = original_accept(*args, **kwargs)
        replay_identity = args[3] if len(args) > 3 else kwargs.get("replay_identity")
        if replay_identity:
            target.update(identity=identity, admin=args[1], publication=args[2])
        return identity

    async def update_status(service, identity, payload, administrator):
        nonlocal writer_pid
        if identity == target.get("identity") and payload.status.value == "suspended":
            writer_pid = await service.session.scalar(text("SELECT pg_backend_pid()"))
            writer_ready.set()
        return await original_update(service, identity, payload, administrator)

    with ThreadPoolExecutor(max_workers=1) as executor:
        async def projection(service, identity):
            nonlocal writer, observed_lock
            result = await original_projection(service, identity)
            if identity != target.get("identity") or writer is not None:
                return result
            writer = executor.submit(
                client.post, f"/api/v1/acceptance/records/{identity}/status", headers=target["admin"],
                json={
                    "status": "suspended", "reason_code": "integrity_failure",
                    "status_failure": {
                        "check_id": "check:entry-supported-query", "reason": "concurrent verified integrity failure",
                        "failure_kind": "entry_specific",
                        "blocking_scope": {"scope": "entry_version", "identity": target["publication"]},
                        "evidence_links": ["evidence://maintenance/concurrent-reacceptance"],
                    },
                },
            )
            assert await asyncio.to_thread(writer_ready.wait, 10), "administrator update did not reach its database session"
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
                        assert response.status_code == 200, response.text
                        assert response.json()["data"]["current_status"] == "suspended"
                        break
                    await asyncio.sleep(0.01)
            assert observed_lock or writer.done(), "administrator update did not reach an observable terminal or lock wait"
            return result

        monkeypatch.setattr(reacceptance_cases, "_publication_acceptance", accept)
        monkeypatch.setattr(DeliveryAcceptanceService, "get_projection", projection)
        monkeypatch.setattr(DeliveryAcceptanceService, "update_status", update_status)
        reacceptance_cases.test_high_severity_source_repair_requires_acceptance_of_its_actual_replay(
            client, monkeypatch, severity, "fresh_bound",
        )
        assert writer is not None
        response = writer.result(timeout=10)
        assert response.status_code == 200, response.text
        assert response.json()["data"]["current_status"] == "suspended"
        assert observed_lock, "resolution closed after its repair acceptance was already suspended"
