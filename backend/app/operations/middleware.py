from time import perf_counter

from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from app.common.config import get_settings
from app.infra.db import SessionLocal
from app.operations.events import OperationalEventService


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None) or "unmatched"
    if route_path != "unmatched":
        route_path = f"{get_settings().api_v1_prefix.rstrip('/')}{route_path}"
    return route_path


def _outcome_for_status(status_code: int) -> str:
    if status_code < 400:
        return "success"
    if status_code < 500:
        return "client_error"
    return "server_error"


async def _record_operational_event(
    *,
    request: Request,
    request_id: str,
    route_outcome: str,
    duration_ms: int,
    context: dict,
) -> None:
    session_factory = getattr(request.app.state, "operational_event_session_factory", SessionLocal)
    try:
        async with session_factory() as session:
            await OperationalEventService(session).record(
                request_id=request_id,
                route_outcome=route_outcome,
                duration_ms=duration_ms,
                gate_outcome=context.get("gate_outcome"),
                provider_identity=context.get("provider_identity"),
                normalized_error=context.get("normalized_error"),
                candidate_count=context.get("candidate_count"),
            )
    except (OSError, SQLAlchemyError):
        # Operational recording is never allowed to replace the user response.
        pass


class _DeferredOperationalEventResponse:
    """ASGI wrapper that records the operational event after the body streams.

    Streaming endpoints (e.g. /chat/stream) set ``request.state.defer_operational_event``
    when the operational context (gate outcome, provider identity) is only
    produced while the response body streams. BaseHTTPMiddleware regains control
    at ``http.response.start``, before the body completes, so recording must wait
    until the wrapped response finishes sending.
    """

    def __init__(self, inner: ASGIApp, *, request: Request, started: float) -> None:
        self._inner = inner
        self._request = request
        self._started = started
        self.status_code = getattr(inner, "status_code", 200)
        self.raw_headers = getattr(inner, "raw_headers", [])

    def _request_id(self) -> str:
        for name, value in self.raw_headers:
            if name == b"x-request-id":
                return value.decode("latin-1")
        return ""

    async def _record(self) -> None:
        context = getattr(self._request.state, "operational_event", {})
        if not isinstance(context, dict):
            context = {}
        await _record_operational_event(
            request=self._request,
            request_id=self._request_id(),
            route_outcome=f"{self._request.method} {_route_path(self._request)}:{_outcome_for_status(self.status_code)}",
            duration_ms=round((perf_counter() - self._started) * 1000),
            context=context,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self._inner(scope, receive, send)
        finally:
            await self._record()


class OperationalEventMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = perf_counter()
        response = await call_next(request)
        if getattr(request.state, "defer_operational_event", False):
            return _DeferredOperationalEventResponse(response, request=request, started=started)
        context = getattr(request.state, "operational_event", {})
        if not isinstance(context, dict):
            context = {}
        await _record_operational_event(
            request=request,
            request_id=response.headers.get("x-request-id", ""),
            route_outcome=f"{request.method} {_route_path(request)}:{_outcome_for_status(response.status_code)}",
            duration_ms=round((perf_counter() - started) * 1000),
            context=context,
        )
        return response
