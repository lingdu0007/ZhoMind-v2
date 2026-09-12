import asyncio

import pytest

from app.common.exceptions import AppError
from app.operations.chat_capacity import ChatAdmissionGate


def test_four_request_burst_has_two_executing_two_queued_and_retryable_fifth():
    gate = ChatAdmissionGate()
    reservations = [gate.reserve(member_id=f"member-{index}") for index in range(4)]
    assert [gate.observe(item)["state"] for item in reservations] == [
        "running", "running", "queued", "queued",
    ]
    denied = gate.reserve(member_id="member-4")
    assert gate.observe(denied) == {
        "state": "throttled", "position": None, "retryable": True,
        "reason": "CHAT_QUEUE_FULL",
    }
    assert gate.snapshot()["executing"] == 2
    assert gate.snapshot()["queued"] == 2


@pytest.mark.asyncio
async def test_background_build_waits_for_interactive_work_to_leave():
    gate = ChatAdmissionGate()
    request = gate.reserve(member_id="member-a")
    build = asyncio.create_task(gate.wait_for_interactive_idle())
    try:
        await asyncio.sleep(0)
        assert not build.done()
        gate.finish(request)
        await asyncio.wait_for(build, timeout=1)
    finally:
        build.cancel()
        await asyncio.gather(build, return_exceptions=True)


def test_member_has_one_execution_and_one_queue_slot_and_release_promotes_fairly():
    gate = ChatAdmissionGate()
    first = gate.reserve(member_id="member-a")
    second = gate.reserve(member_id="member-a")
    assert gate.observe(second)["state"] == "queued"
    denied = gate.reserve(member_id="member-a")
    assert gate.observe(denied)["reason"] == "CHAT_MEMBER_LIMIT"
    other = gate.reserve(member_id="member-b")
    assert gate.observe(other)["state"] == "running"
    waiting = gate.reserve(member_id="member-c")
    gate.finish(other)
    assert gate.observe(second)["state"] == "queued"
    assert gate.observe(waiting)["state"] == "running"
    gate.finish(first)
    assert gate.observe(second)["state"] == "running"
    gate.finish(first)
    assert gate.snapshot()["executing"] == 2


@pytest.mark.parametrize("elapsed", [30.0, 31.0])
async def test_expired_reservation_cannot_start_when_release_races_the_waiter(monkeypatch, elapsed):
    now = [0.0]
    monkeypatch.setattr("app.operations.chat_capacity.monotonic", lambda: now[0])
    gate = ChatAdmissionGate()
    held = [gate.reserve(member_id=f"busy-{index}") for index in range(2)]
    queued = gate.reserve(member_id="waiting")
    now[0] = elapsed
    gate.finish(held[0])
    with pytest.raises(AppError) as failure:
        await gate.wait(queued)
    assert failure.value.code == "CHAT_QUEUE_TIMEOUT"
    assert gate.snapshot()["executing"] == 1
    assert gate.snapshot()["queued"] == 0
    gate.finish(queued)
    gate.finish(held[1])
