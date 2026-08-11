from time import perf_counter

from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.common.config import get_settings
from app.infra.db import SessionLocal
from app.operations.events import OperationalEventService


class OperationalEventMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", None) or "unmatched"
        if route_path != "unmatched":
            route_path = f"{get_settings().api_v1_prefix.rstrip('/')}{route_path}"
        outcome = "success" if response.status_code < 400 else "client_error" if response.status_code < 500 else "server_error"
        context = getattr(request.state, "operational_event", {})
        if not isinstance(context, dict):
            context = {}
        session_factory = getattr(request.app.state, "operational_event_session_factory", SessionLocal)
        try:
            async with session_factory() as session:
                await OperationalEventService(session).record(
                    request_id=response.headers.get("x-request-id", ""),
                    route_outcome=f"{request.method} {route_path}:{outcome}",
                    duration_ms=round((perf_counter() - started) * 1000),
                    gate_outcome=context.get("gate_outcome"),
                    provider_identity=context.get("provider_identity"),
                    normalized_error=context.get("normalized_error"),
                    candidate_count=context.get("candidate_count"),
                )
        except (OSError, SQLAlchemyError):
            # Operational recording is never allowed to replace the user response.
            pass
        return response
