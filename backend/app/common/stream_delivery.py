"""Observe terminal stream delivery at the outer ASGI transport boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

STREAM_DELIVERY_OBSERVER_STATE_KEY = "answer_execution_stream_delivery_observer"


@runtime_checkable
class StreamDeliveryObserver(Protocol):
    """Receives only transport facts after downstream middleware has run."""

    async def observe_network_message(self, message: Message) -> None: ...

    async def finish_network_response(self) -> None: ...

    async def interrupt_network_response(self) -> None: ...


def attach_stream_delivery_observer(scope: Scope, observer: StreamDeliveryObserver) -> None:
    state = scope.setdefault("state", {})
    if not isinstance(state, dict):
        raise TypeError("HTTP scope state must be a dictionary")
    state[STREAM_DELIVERY_OBSERVER_STATE_KEY] = observer


def _observer_from_scope(scope: Scope) -> StreamDeliveryObserver | None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return None
    observer = state.get(STREAM_DELIVERY_OBSERVER_STATE_KEY)
    return observer if isinstance(observer, StreamDeliveryObserver) else None


class StreamDeliveryMiddleware:
    """Observe real ASGI writes outside every response-buffering middleware."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_delivery_observation(message: Message) -> None:
            await send(message)
            observer = _observer_from_scope(scope)
            if observer is not None:
                await observer.observe_network_message(message)

        try:
            await self.app(scope, receive, send_with_delivery_observation)
        except BaseException:
            observer = _observer_from_scope(scope)
            if observer is not None:
                await observer.interrupt_network_response()
            raise
        else:
            observer = _observer_from_scope(scope)
            if observer is not None:
                await observer.finish_network_response()
