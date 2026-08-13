import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.router import router as api_v1_router
from app.common.config import get_settings
from app.common.exceptions import register_exception_handlers
from app.common.logger import configure_logging
from app.common.request_id import RequestIdMiddleware
from app.extensions.registry import get_extension_registry
from app.infra.db import SessionLocal
from app.operations.events import OperationalEventService
from app.operations.middleware import OperationalEventMiddleware
from app.service.member_admission_service import MemberAdmissionService
from app.settings.service import SystemSettingsDraftService

settings = get_settings()
configure_logging()


async def _purge_expired_records(session_factory) -> None:
    try:
        async with session_factory() as session:
            await OperationalEventService(session).purge_expired()
            await session.commit()
    except (OSError, SQLAlchemyError):
        # The retention tables may not exist before migrations have run.
        pass


async def _retention_loop(session_factory) -> None:
    while True:
        await asyncio.sleep(60)
        await _purge_expired_records(session_factory)


@asynccontextmanager
async def lifespan(application: FastAPI):
    session_factory = getattr(application.state, "settings_session_factory", SessionLocal)
    await _purge_expired_records(session_factory)
    retention_task = asyncio.create_task(_retention_loop(session_factory), name="conversation-retention-cleanup")
    try:
        async with session_factory() as session:
            await SystemSettingsDraftService(session).restore_active_application()
    except (OSError, SQLAlchemyError):
        # The settings tables may not exist before migrations have run.
        pass
    # Resolver artifacts are process-owned. A configured but untrusted profile
    # must fail startup instead of silently degrading on the first chat request.
    get_extension_registry()
    try:
        async with session_factory() as session:
            settings = get_settings()
            await MemberAdmissionService(session, redis=None).create_bootstrap_administrator(
                username=settings.bootstrap_admin_username,
                password=settings.bootstrap_admin_password,
            )
    except (OSError, SQLAlchemyError):
        # The users table may not exist before migrations have run.
        pass
    try:
        yield
    finally:
        retention_task.cancel()
        with suppress(asyncio.CancelledError):
            await retention_task


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.state.settings_session_factory = SessionLocal
app.state.operational_event_session_factory = SessionLocal
app.add_middleware(RequestIdMiddleware)
app.add_middleware(OperationalEventMiddleware)
register_exception_handlers(app)
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
