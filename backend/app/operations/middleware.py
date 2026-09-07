from time import perf_counter

from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.common.config import get_settings
from app.common.generation_audit import generation_audit_sink
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
                generation_route=context.get("generation_route"),
            )
    except (OSError, SQLAlchemyError):
        # Operational recording is never allowed to replace the user response.
        pass


class OperationalEventMiddleware:
    """Record after the inner stream-delivery observer finishes durable cleanup."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        started = perf_counter()
        status = 500
        context: dict = {}
        request.state.operational_event = context

        async def observed_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        def observe(record: dict) -> None:
            context["generation_route"] = record
            context["normalized_error"] = record.get("route_reason") if record.get("route_reason") != "succeeded" else None
            attempts = record.get("provider_attempts") or []
            context["provider_identity"] = attempts[-1].get("provider") if attempts else None

        token = generation_audit_sink.set(observe)
        try:
            await self.app(scope, receive, observed_send)
        finally:
            generation_audit_sink.reset(token)
            context = getattr(request.state, "operational_event", {})
            await _record_operational_event(
                request=request,
                request_id=getattr(request.state, "request_id", ""),
                route_outcome=f"{request.method} {_route_path(request)}:{_outcome_for_status(status)}",
                duration_ms=round((perf_counter() - started) * 1000),
                context=context if isinstance(context, dict) else {},
            )
