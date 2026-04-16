from fastapi import FastAPI

from app.api.v1.router import router as api_v1_router
from app.common.config import get_settings
from app.common.exceptions import register_exception_handlers
from app.common.logger import configure_logging
from app.common.request_id import RequestIdMiddleware

settings = get_settings()
configure_logging()
app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(RequestIdMiddleware)
register_exception_handlers(app)
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
