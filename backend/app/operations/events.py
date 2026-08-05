import re
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.operational_event import OperationalEvent
from app.repository.chat_repository import ChatRepository

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
    ) -> None:
        event = OperationalEvent(
            request_id=self._code(request_id) or "unknown-request",
            route_outcome=route_outcome[:128],
            duration_ms=max(0, int(duration_ms)),
            gate_outcome=str(gate_outcome) if gate_outcome in _GATE_OUTCOMES else None,
            provider_identity=self._code(provider_identity),
            normalized_error=self._code(normalized_error),
            candidate_count=self._count(candidate_count),
        )
        self.session.add(event)
        await self.purge_expired()
        await self.session.commit()

    async def purge_expired(self, *, now: datetime | None = None) -> None:
        cutoff = (now or datetime.now(UTC)) - timedelta(days=30)
        await ChatRepository(self.session).purge_expired_sessions(now=now)
        await self.session.execute(delete(OperationalEvent).where(OperationalEvent.created_at <= cutoff))

    @staticmethod
    def _code(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        code = value.strip()
        return code if _SAFE_CODE.fullmatch(code) else None

    @staticmethod
    def _count(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
