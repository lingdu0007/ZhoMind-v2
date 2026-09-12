import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.model.operational_event import OperationalEvent
from app.operations.chat_capacity import get_chat_admission_gate
from app.operations.events import OperationalEventService
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client


@pytest.mark.parametrize("route", ["/api/v1/chat", "/api/v1/chat/stream"])
def test_authenticated_queue_timeout_is_visible_and_reloadable(client, monkeypatch, route):
    token = _register(client, username="queued-member")
    admin = _register(client, username="queue-admin", role="admin")
    gate = get_chat_admission_gate()
    held = [gate.reserve(member_id=f"busy-{index}") for index in range(2)]
    monkeypatch.setattr(gate, "queue_timeout_seconds", 0.01, raising=False)
    try:
        response = client.post(
            route, headers=_headers(token),
            json={"message": "hello", "session_id": "queued-conversation"},
        )
        assert "CHAT_QUEUE_TIMEOUT" in response.text
        history = client.get("/api/v1/sessions/queued-conversation", headers=_headers(token))
        executions = [item["answer_execution"] for item in history.json()["messages"]]
        assert executions
        assert all(item["state"] == "failed" for item in executions)
        assert all(item["failure_code"] == "CHAT_QUEUE_TIMEOUT" for item in executions)
        assert all(item.get("outcome") is None for item in executions)
        operations = client.get("/api/v1/operations", headers=_headers(admin)).json()["data"]
        assert any(item["kind"] == "queue" and item["code"] == "CHAT_QUEUE_TIMEOUT"
                   for item in operations["failures"])
        if route.endswith("stream"):
            assert '"stage": "queued"' in response.text
            assert "event: done" in response.text
    finally:
        for item in held:
            gate.finish(item)


@pytest.mark.parametrize("gate_value", ["PRIVATE_GATE", ["PRIVATE_GATE"], {"content": "PRIVATE_GATE"}])
def test_content_in_operational_fields_never_reaches_administrator(client, gate_value):
    token = _register(client, username="operations-owner", role="admin")

    async def inject():
        async with client.app.state.test_auth_session_factory() as session:
            await OperationalEventService(session).record(
                request_id="PRIVATE_REQUEST", route_outcome="PRIVATE_ROUTE", duration_ms=1,
                provider_identity="PRIVATE_PROVIDER", normalized_error="PRIVATE_ERROR",
                gate_outcome=gate_value, candidate_count="PRIVATE_COUNT",
            )

    asyncio.run(inject())
    response = client.get("/api/v1/operations", headers=_headers(token))
    assert response.status_code == 200
    assert "PRIVATE_" not in response.text


def test_admin_observes_versioned_queue_and_completed_event_dimensions(client):
    member = _register(client, username="dimension-member")
    admin = _register(client, username="dimension-admin", role="admin")
    answer = client.post("/api/v1/chat", headers=_headers(member), json={"message": "hello"})
    assert answer.status_code == 200
    response = client.get("/api/v1/operations", headers=_headers(admin))
    data = response.json()["data"]
    assert data["admission"]["configuration"]["max_executing"] == 2
    assert data["admission"]["configuration"]["max_queued"] == 2
    assert data["admission"]["configuration"]["version"] == 1
    assert data["limits"]["concurrent_chats"] == 2
    event = next(item for item in data["events"] if item["request_id"] == answer.headers["x-request-id"])
    assert event["dimensions"]["execution_state"] == "completed"
    assert event["dimensions"]["outcome"] == "non_knowledge_base_reply"
    assert event["dimensions"]["stage_durations_ms"]["queue"] >= 0
    assert event["dimensions"]["stage_durations_ms"]["persistence"] >= 0
    assert event["dimensions"]["configuration_identity"] == data["admission"]["configuration"]["identity"]
    assert event["dimensions"]["policy_identity"].startswith("configuration:")


@pytest.mark.parametrize("duration", ["PRIVATE_DURATION_CONTENT", -1, 1.5])
def test_legacy_operational_content_is_sanitized_on_read(client, duration):
    admin = _register(client, username="legacy-operations-admin", role="admin")

    async def seed():
        async with client.app.state.test_auth_session_factory() as session:
            session.add(OperationalEvent(
                request_id="PRIVATE_REQUEST", route_outcome="PRIVATE_ROUTE",
                duration_ms=duration, normalized_error="PRIVATE_ERROR",
                generation_route={"route_identity": "provider_route:" + "a" * 64,
                                  "route_reason": "succeeded", "PRIVATE_KEY": "PRIVATE_BODY",
                                  "attempts": [{"provider": "PRIVATE_PROVIDER"}]},
            ))
            await session.commit()

    asyncio.run(seed())
    response = client.get("/api/v1/operations", headers=_headers(admin))
    assert response.status_code == 200
    assert "PRIVATE_" not in response.text
    legacy = next(item for item in response.json()["data"]["events"] if item["request_id"] == "unknown-request")
    assert legacy["duration_ms"] is None


def test_operations_distinguishes_provider_retrieval_persistence_stream_queue_and_application(client):
    admin = _register(client, username="failure-category-admin", role="admin")
    expected = {
        "PROVIDER_TIMEOUT": "generation_provider", "RETRIEVAL_FAILED": "retrieval",
        "ANSWER_EXECUTION_PERSISTENCE_FAILED": "persistence", "CHAT_STREAM_INTERRUPTED": "stream",
        "CHAT_QUEUE_TIMEOUT": "queue", "APPLICATION_FAILED": "application",
    }

    async def record():
        async with client.app.state.test_auth_session_factory() as session:
            for error in expected:
                await OperationalEventService(session).record(
                    request_id="52c05c25-2bcf-4463-909f-c009d98348de",
                    route_outcome="POST /api/v1/chat:server_error", duration_ms=10, normalized_error=error,
                )

    asyncio.run(record())
    data = client.get("/api/v1/operations", headers=_headers(admin)).json()["data"]
    assert {item["code"]: item["kind"] for item in data["failures"]} == expected


def test_actual_retrieval_failure_has_a_sanitized_distinct_operational_category(client, monkeypatch):
    from app.service.chat_service import ChatService

    class BrokenRetriever:
        async def retrieve(self, query, top_k):
            raise RuntimeError("PRIVATE_RETRIEVAL_EXCEPTION")

    monkeypatch.setattr(ChatService, "_resolve_retriever", lambda self: (BrokenRetriever(), "failure-fixture"))
    member = _register(client, username="retrieval-failure-member")
    admin = _register(client, username="retrieval-failure-admin", role="admin")
    response = client.post("/api/v1/chat/stream", headers=_headers(member), json={"message": "unsupported engineering query"})
    assert "PRIVATE_" not in response.text
    data = client.get("/api/v1/operations", headers=_headers(admin)).json()["data"]
    assert any(item["kind"] == "retrieval" for item in data["failures"])


def test_queued_request_cannot_start_after_member_deactivation(client):
    member = _register(client, username="deactivated-queue-member")
    admin = _register(client, username="deactivated-queue-admin", role="admin")
    gate = get_chat_admission_gate()
    held = [gate.reserve(member_id=f"held-{index}") for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, "/api/v1/chat/stream", headers=_headers(member), json={"message": "hello"})
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = client.get("/api/v1/operations", headers=_headers(admin)).json()["data"]["admission"]
                if state["queued"] == 1:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("request did not become queued")
            denied = client.post("/api/v1/members/deactivated-queue-member/deactivate", headers=_headers(admin))
            assert denied.status_code == 200
        finally:
            for item in held:
                gate.finish(item)
        response = future.result(timeout=5)
    assert "AUTH_INVALID_TOKEN" in response.text
    assert '"outcome": "non_knowledge_base_reply"' not in response.text


@pytest.mark.parametrize("fault", ["persistence", "stream"])
def test_actual_terminal_fault_has_distinct_content_free_operations(client, monkeypatch, fault):
    from app.api.v1.chat import _DeliveryAwareStreamingResponse
    from app.service.answer_execution_store import AnswerExecutionStore

    if fault == "persistence":
        async def fail_complete(self, **kwargs):
            raise OSError("PRIVATE_PERSISTENCE_BODY")
        monkeypatch.setattr(AnswerExecutionStore, "complete", fail_complete)
    else:
        original = _DeliveryAwareStreamingResponse.stream_response

        async def fail_send(self, send):
            async def broken_send(message):
                if b"event: content" in message.get("body", b""):
                    raise OSError("PRIVATE_STREAM_BODY")
                await send(message)
            await original(self, broken_send)
        monkeypatch.setattr(_DeliveryAwareStreamingResponse, "stream_response", fail_send)
    member = _register(client, username=f"{fault}-member")
    admin = _register(client, username=f"{fault}-admin", role="admin")
    if fault == "stream":
        with pytest.raises(OSError):
            client.post("/api/v1/chat/stream", headers=_headers(member), json={"message": "hello"})
    else:
        client.post("/api/v1/chat/stream", headers=_headers(member), json={"message": "hello"})
    response = client.get("/api/v1/operations", headers=_headers(admin))
    assert "PRIVATE_" not in response.text
    assert any(item["kind"] == fault for item in response.json()["data"]["failures"])
