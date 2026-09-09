import re
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.extensions.provider_router import ADVANCE_REASONS, STOP_REASONS
from app.model.operational_event import OperationalEvent
from app.retention.policy import read_policy

_SAFE_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_GATE_OUTCOMES = {"passed", "rejected", "unavailable"}


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
    ) -> None:
        event = OperationalEvent(
            request_id=self._code(request_id) or "unknown-request",
            route_outcome=route_outcome[:128],
            duration_ms=max(0, int(duration_ms)),
            gate_outcome=str(gate_outcome) if gate_outcome in _GATE_OUTCOMES else None,
            provider_identity=self._code(provider_identity),
            normalized_error=self._code(normalized_error),
            candidate_count=self._count(candidate_count),
            generation_route=self.route_observation(generation_route),
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
        if reason not in ADVANCE_REASONS | STOP_REASONS | {"succeeded"}:
            return None
        if identity is not None and not re.fullmatch(r"provider_route:[0-9a-f]{64}", str(identity)):
            return None
        attempts = []
        for item in (value.get("provider_attempts") or [])[:4]:
            if not isinstance(item, dict):
                continue
            safe = {}
            for key, pattern in {
                "provider": r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
                "approval_identity": r"configuration:[0-9a-f]{64}",
                "payload_sha256": r"[0-9a-f]{64}",
                "snapshot_sha256": r"[0-9a-f]{64}",
            }.items():
                candidate = item.get(key)
                if isinstance(candidate, str) and re.fullmatch(pattern, candidate):
                    safe[key] = candidate
            for key in ("attempt", "latency_ms"):
                candidate = OperationalEventService._count(item.get(key))
                if candidate is not None:
                    safe[key] = candidate
            code = item.get("error_code")
            safe["error_code"] = code if code in ADVANCE_REASONS | STOP_REASONS else None
            attempts.append(safe)
        return {"route_identity": identity, "route_reason": reason, "attempts": attempts}

    @staticmethod
    def _code(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        code = value.strip()
        return code if _SAFE_CODE.fullmatch(code) else None

    @staticmethod
    def _count(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
