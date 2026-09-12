import asyncio
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.router import router as api_v1_router
from app.common.config import get_settings
from app.common.exceptions import register_exception_handlers
from app.common.logger import configure_logging
from app.common.request_id import RequestIdMiddleware
from app.common.stream_delivery import StreamDeliveryMiddleware
from app.extensions.registry import get_extension_registry
from app.infra.db import SessionLocal, get_db_session
from app.operations.middleware import OperationalEventMiddleware
from app.repository.chat_repository import ChatRepository
from app.retention.cleanup import run_retention_sweep
from app.reviewed_bundles.runtime import candidate_build_runtime
from app.service.answer_execution_store import AnswerExecutionStore
from app.service.member_admission_service import MemberAdmissionService
from app.settings.service import SystemSettingsDraftService

settings = get_settings()
configure_logging()


async def _purge_expired_records(session_factory) -> None:
    await run_retention_sweep(session_factory)


async def _retention_loop(session_factory) -> None:
    while True:
        await asyncio.sleep(60)
        await _purge_expired_records(session_factory)


async def _recover_candidate_builds(session_factory) -> None:
    try:
        await candidate_build_runtime.recover(session_factory)
    except SQLAlchemyError as exc:
        if _candidate_tables_are_not_migrated(exc):
            return
        raise


def _lifespan_session_factory(application: FastAPI):
    override = application.dependency_overrides.get(get_db_session)
    if override is None:
        return getattr(application.state, "settings_session_factory", SessionLocal)

    @asynccontextmanager
    async def override_session_scope() -> AsyncIterator[AsyncSession]:
        value = override()
        if inspect.isasyncgen(value):
            try:
                yield _require_lifespan_session(await anext(value))
            finally:
                await value.aclose()
            return
        session = _require_lifespan_session(await value if inspect.isawaitable(value) else value)
        try:
            yield session
        finally:
            await session.close()

    return override_session_scope


def _require_lifespan_session(value: object) -> AsyncSession:
    if not isinstance(value, AsyncSession):
        raise TypeError("get_db_session override must provide an AsyncSession")
    return value


def _candidate_tables_are_not_migrated(error: SQLAlchemyError) -> bool:
    message = str(error).lower()
    missing_table = "no such table" in message or "does not exist" in message
    return missing_table and ("candidate_build_jobs" in message or "candidate_build_chunks" in message)


@asynccontextmanager
async def lifespan(application: FastAPI):
    session_factory = _lifespan_session_factory(application)
    await _purge_expired_records(session_factory)
    retention_task = asyncio.create_task(_retention_loop(session_factory), name="conversation-retention-cleanup")
    try:
        try:
            async with session_factory() as session:
                await SystemSettingsDraftService(session).restore_active_application()
        except (OSError, SQLAlchemyError):
            # The settings tables may not exist before migrations have run.
            pass
        # Resolver artifacts are process-owned. A configured but untrusted profile
        # must fail startup instead of silently degrading on the first chat request.
        get_extension_registry()
        async with session_factory() as session:
            settings = get_settings()
            await MemberAdmissionService(session, redis=None).create_bootstrap_administrator(
                username=settings.bootstrap_admin_username,
                password=settings.bootstrap_admin_password,
            )
        async with session_factory() as session:
            await AnswerExecutionStore(session, ChatRepository(session)).recover_interrupted()
            await session.commit()
        await _recover_candidate_builds(session_factory)
        yield
    finally:
        retention_task.cancel()
        with suppress(asyncio.CancelledError):
            await retention_task


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.state.settings_session_factory = SessionLocal
app.state.operational_event_session_factory = SessionLocal
app.add_middleware(RequestIdMiddleware)
app.add_middleware(StreamDeliveryMiddleware)
app.add_middleware(OperationalEventMiddleware)
register_exception_handlers(app)
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
