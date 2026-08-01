from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.router import router as api_v1_router
from app.common.config import get_settings
from app.common.exceptions import register_exception_handlers
from app.common.logger import configure_logging
from app.common.request_id import RequestIdMiddleware
from app.infra.db import SessionLocal
from app.service.member_admission_service import MemberAdmissionService
from app.settings.service import SystemSettingsDraftService

settings = get_settings()
configure_logging()


@asynccontextmanager
async def lifespan(application: FastAPI):
    session_factory = getattr(application.state, "settings_session_factory", SessionLocal)
    try:
        async with session_factory() as session:
            await SystemSettingsDraftService(session).restore_active_application()
    except (OSError, SQLAlchemyError):
        # The settings tables may not exist before migrations have run.
        pass
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
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.state.settings_session_factory = SessionLocal
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
