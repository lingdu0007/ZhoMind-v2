from collections.abc import Callable
from contextvars import ContextVar

# The request installs a sink inherited by its execution task, including SSE.
# This channel carries only route metadata, never a generation result or prompt.
generation_audit_sink: ContextVar[Callable[[dict], None] | None] = ContextVar("generation_audit_sink", default=None)


def publish_generation_audit(record: dict) -> None:
    sink = generation_audit_sink.get()
    if sink is not None:
        sink(record)
