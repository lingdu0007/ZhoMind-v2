import re
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.canonical import AnswerExecutionState, AnswerOutcome
from app.extensions.provider_router import ADVANCE_REASONS, STOP_REASONS
from app.model.operational_event import OperationalEvent
from app.retention.policy import read_policy

_GATE_OUTCOMES = {"passed", "rejected", "unavailable"}
_ERROR_CODES = ADVANCE_REASONS | STOP_REASONS | {
    "PROVIDER_TIMEOUT", "CHAT_QUEUE_FULL", "CHAT_MEMBER_LIMIT", "CHAT_QUEUE_TIMEOUT",
    "ANSWER_EXECUTION_INTERRUPTED", "ANSWER_EXECUTION_STOPPED", "ANSWER_EXECUTION_FAILED",
    "ANSWER_EXECUTION_PERSISTENCE_FAILED", "CHAT_STREAM_INTERRUPTED", "RETRIEVAL_FAILED",
    "APPLICATION_FAILED", "ANSWER_EVIDENCE_WITHDRAWN",
}


class OperationalEventService:
    """Stores the fixed, content-free Operational Event projection."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def record(
        self,
        *,
        request_id: str,
        route_outcome: str,
        duration_ms: int,
        gate_outcome: object = None,
        provider_identity: object = None,
        normalized_error: object = None,
        candidate_count: object = None,
        generation_route: object = None,
        dimensions: object = None,
    ) -> None:
        policy = await read_policy(self.session)
        event = OperationalEvent(
            request_id=self.request_identity(request_id),
            route_outcome=self.route_class(route_outcome),
            duration_ms=max(0, int(duration_ms)),
            gate_outcome=gate_outcome if isinstance(gate_outcome, str) and gate_outcome in _GATE_OUTCOMES else None,
            provider_identity=(
                provider_identity if isinstance(provider_identity, str)
                and re.fullmatch(r"configuration:[0-9a-f]{64}", provider_identity) else None
            ),
            normalized_error=self.error_code(normalized_error),
            candidate_count=self.nonnegative_integer(candidate_count),
            generation_route=self.route_observation(generation_route),
            dimensions={**self.dimensions(dimensions), "policy_identity": policy["identity"]},
        )
        self.session.add(event)
        await self.purge_expired()
        await self.session.commit()

    async def purge_expired(self, *, now: datetime | None = None, retention_days: int | None = None) -> None:
        days = retention_days if retention_days is not None else (await read_policy(self.session))["days"]["operational_events"]
        cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
        await self.session.execute(delete(OperationalEvent).where(OperationalEvent.created_at <= cutoff))

    @staticmethod
    def route_observation(value: object) -> dict | None:
        if not isinstance(value, dict):
            return None
        identity = value.get("route_identity")
        reason = value.get("route_reason")
        if not isinstance(reason, str) or reason not in ADVANCE_REASONS | STOP_REASONS | {"succeeded"}:
            return None
        if identity is not None and not re.fullmatch(r"provider_route:[0-9a-f]{64}", str(identity)):
            return None
        attempts = []
        raw_attempts = value.get("provider_attempts")
        for item in (raw_attempts if isinstance(raw_attempts, list) else [])[:4]:
            if not isinstance(item, dict):
                continue
            safe = {}
            for key, pattern in {
                "approval_identity": r"configuration:[0-9a-f]{64}",
                "payload_sha256": r"[0-9a-f]{64}",
                "snapshot_sha256": r"[0-9a-f]{64}",
            }.items():
                candidate = item.get(key)
                if isinstance(candidate, str) and re.fullmatch(pattern, candidate):
                    safe[key] = candidate
            for key in ("attempt", "latency_ms"):
                candidate = OperationalEventService.nonnegative_integer(item.get(key))
                if candidate is not None:
                    safe[key] = candidate
            code = item.get("error_code")
            safe["error_code"] = code if isinstance(code, str) and code in ADVANCE_REASONS | STOP_REASONS else None
            attempts.append(safe)
        return {"route_identity": identity, "route_reason": reason, "attempts": attempts}

    @staticmethod
    def retained_route(value: object) -> dict | None:
        if not isinstance(value, dict):
            return None
        return OperationalEventService.route_observation({
            "route_identity": value.get("route_identity"), "route_reason": value.get("route_reason"),
            "provider_attempts": value.get("attempts"),
        })

    @staticmethod
    def request_identity(value: object) -> str:
        if isinstance(value, str):
            try:
                parsed = UUID(value)
                if parsed.version == 4 and str(parsed) == value:
                    return value
            except ValueError:
                pass
        return "unknown-request"

    @staticmethod
    def error_code(value: object) -> str | None:
        if value is None:
            return None
        return value if isinstance(value, str) and value in _ERROR_CODES else "APPLICATION_FAILED"

    @staticmethod
    def failure_category(value: object) -> str:
        code = OperationalEventService.error_code(value)
        if code == "application_failure":
            return "application"
        if code in ADVANCE_REASONS | STOP_REASONS | {"PROVIDER_TIMEOUT"}:
            return "generation_provider"
        return {
            "CHAT_QUEUE_FULL": "queue", "CHAT_MEMBER_LIMIT": "queue", "CHAT_QUEUE_TIMEOUT": "queue",
            "RETRIEVAL_FAILED": "retrieval", "ANSWER_EXECUTION_PERSISTENCE_FAILED": "persistence",
            "CHAT_STREAM_INTERRUPTED": "stream",
        }.get(code or "", "application")

    @staticmethod
    def route_class(value: object) -> str:
        # Compare against registered templates, never a request path or free text.
        from app.main import app

        for path, operations in app.openapi()["paths"].items():
            for method in operations:
                for outcome in ("success", "client_error", "server_error"):
                    expected = f"{method.upper()} {path}:{outcome}"
                    if value == expected:
                        return expected
        return "unmatched"

    @staticmethod
    def dimensions(value: object) -> dict:
        if not isinstance(value, dict):
            return {}
        safe: dict = {}
        for key, allowed in {
            "execution_state": {state.value for state in AnswerExecutionState},
            "outcome": {outcome.value for outcome in AnswerOutcome},
        }.items():
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate in allowed:
                safe[key] = candidate
        for key in ("configuration_identity", "policy_identity"):
            candidate = value.get(key)
            if isinstance(candidate, str) and re.fullmatch(r"configuration:[0-9a-f]{64}", candidate):
                safe[key] = candidate
        count = OperationalEventService.nonnegative_integer(value.get("evidence_count"))
        if count is not None:
            safe["evidence_count"] = count
        raw_timings = value.get("stage_durations_ms")
        if isinstance(raw_timings, dict):
            safe["stage_durations_ms"] = {
                key: duration for key in ("queue", "retrieval", "provider", "persistence", "stream", "application")
                if (duration := OperationalEventService.nonnegative_integer(raw_timings.get(key))) is not None
            }
        return safe

    @staticmethod
    def nonnegative_integer(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
