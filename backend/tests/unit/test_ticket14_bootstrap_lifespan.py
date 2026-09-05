import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app import main
from app.service.member_admission_service import MemberAdmissionService


def test_lifespan_fails_closed_when_configured_bootstrap_cannot_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def session_factory():
        yield object()

    async def noop_purge(_session_factory) -> None:
        return None

    async def wait_for_cancellation(_session_factory) -> None:
        await asyncio.Event().wait()

    async def noop_restore(_self) -> None:
        return None

    async def reject_bootstrap(_self, *, username: str, password: str) -> None:
        del username, password
        raise SQLAlchemyError("bootstrap persistence failed")

    application = FastAPI()
    application.state.settings_session_factory = session_factory
    monkeypatch.setattr(main, "_purge_expired_records", noop_purge)
    monkeypatch.setattr(main, "_retention_loop", wait_for_cancellation)
    monkeypatch.setattr(main.SystemSettingsDraftService, "restore_active_application", noop_restore)
    monkeypatch.setattr(main, "get_extension_registry", lambda: None)
    monkeypatch.setattr(MemberAdmissionService, "create_bootstrap_administrator", reject_bootstrap)

    async def run_lifespan() -> None:
        with pytest.raises(SQLAlchemyError, match="bootstrap persistence failed"):
            async with main.lifespan(application):
                raise AssertionError("lifespan must not yield after bootstrap persistence fails")

    asyncio.run(run_lifespan())
