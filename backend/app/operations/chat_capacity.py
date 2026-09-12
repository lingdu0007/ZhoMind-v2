import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from time import monotonic

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.model.user import User


@dataclass
class AdmissionReservation:
    identity: str
    member_id: str
    state: str
    reason: str | None = None
    queue_deadline: float | None = None


class ChatAdmissionGate:
    """Single-process Pilot scheduling; private execution events survive restart."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._reservations: dict[str, AdmissionReservation] = {}
        self.queue_timeout_seconds = 30.0

    def reserve(self, *, member_id: str) -> AdmissionReservation:
        with self._lock:
            self._expire_queued()
            running = sum(item.state == "running" for item in self._reservations.values())
            queued = sum(item.state == "queued" for item in self._reservations.values())
            member_states = {item.state for item in self._reservations.values() if item.member_id == member_id}
            state = "running" if running < 2 and "running" not in member_states else "queued"
            reason = None
            if "queued" in member_states:
                state, reason = "throttled", "CHAT_MEMBER_LIMIT"
            elif state == "queued" and queued >= 2:
                state, reason = "throttled", "CHAT_QUEUE_FULL"
            reservation = AdmissionReservation(
                identity=uuid.uuid4().hex, member_id=member_id, state=state,
                reason=reason,
                queue_deadline=monotonic() + self.queue_timeout_seconds if state == "queued" else None,
            )
            if state != "throttled":
                self._reservations[reservation.identity] = reservation
            return reservation

    def finish(self, reservation: AdmissionReservation) -> None:
        with self._lock:
            self._expire_queued()
            if self._reservations.get(reservation.identity) is not reservation:
                return
            del self._reservations[reservation.identity]
            reservation.state = "released"
            active_members = {item.member_id for item in self._reservations.values() if item.state == "running"}
            for item in self._reservations.values():
                if len(active_members) >= 2:
                    break
                if item.state == "queued" and item.member_id not in active_members:
                    item.state = "running"
                    active_members.add(item.member_id)

    def observe(self, reservation: AdmissionReservation) -> dict:
        with self._lock:
            self._expire_queued()
            queue = [item.identity for item in self._reservations.values() if item.state == "queued"]
            return {
                "state": reservation.state,
                "position": queue.index(reservation.identity) + 1 if reservation.identity in queue else None,
                "retryable": reservation.state == "throttled",
                "reason": reservation.reason,
            }

    def snapshot(self) -> dict:
        with self._lock:
            self._expire_queued()
            return {
                "executing": sum(item.state == "running" for item in self._reservations.values()),
                "queued": sum(item.state == "queued" for item in self._reservations.values()),
                "configuration": self.configuration(),
            }

    def _expire_queued(self) -> None:
        now = monotonic()
        for identity, item in list(self._reservations.items()):
            if item.state == "queued" and item.queue_deadline is not None and now >= item.queue_deadline:
                item.state, item.reason = "failed", "CHAT_QUEUE_TIMEOUT"
                del self._reservations[identity]

    def configuration(self) -> dict:
        values = {
            "schema": "pilot_admission/v1", "version": 1, "max_executing": 2,
            "max_queued": 2, "member_executing": 1, "member_queued": 1,
            "queue_timeout_seconds": self.queue_timeout_seconds,
        }
        return {**values, "identity": f"configuration:{canonical_json_sha256(values)}"}

    async def wait(
        self, reservation: AdmissionReservation,
        *, progress: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        deadline = reservation.queue_deadline if reservation.queue_deadline is not None else monotonic()
        previous_position = None
        while True:
            observation = self.observe(reservation)
            if observation["reason"] == "CHAT_QUEUE_TIMEOUT":
                raise AppError(status_code=503, code="CHAT_QUEUE_TIMEOUT", message="queue wait expired",
                               detail={"state": "failed", "retryable": True})
            if observation["state"] == "running":
                if progress is not None:
                    await progress("running", "Request started")
                return
            if observation["state"] != "queued":
                raise AppError(status_code=429, code="CHAT_QUEUE_FULL", message="retry the request",
                               detail={"state": "throttled", "retryable": True})
            if progress is not None and observation["position"] != previous_position:
                previous_position = observation["position"]
                await progress("queued", f"Waiting in queue: {previous_position}")
            if monotonic() >= deadline:
                raise AppError(status_code=503, code="CHAT_QUEUE_TIMEOUT", message="queue wait expired",
                               detail={"state": "failed", "retryable": True})
            await asyncio.sleep(min(0.05, max(0, deadline - monotonic())))

    async def wait_for_interactive_idle(self) -> None:
        while self.snapshot()["executing"] or self.snapshot()["queued"]:
            await asyncio.sleep(0.05)

    async def wait_for_member(
        self, reservation: AdmissionReservation, *, session: AsyncSession,
        progress: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        await self.wait(reservation, progress=progress)
        if reservation.queue_deadline is not None:
            active = await session.scalar(select(User.is_active).where(User.username == reservation.member_id))
            if active is not True:
                raise AppError(
                    status_code=401, code="AUTH_INVALID_TOKEN",
                    message="member authorization is no longer active",
                )

_chat_admission_gate = ChatAdmissionGate()


def get_chat_admission_gate() -> ChatAdmissionGate:
    return _chat_admission_gate
