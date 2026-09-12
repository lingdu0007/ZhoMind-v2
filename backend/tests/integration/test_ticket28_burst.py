import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.service.chat_service import ChatService
from app.settings.generation_routes import GenerationRouteService
from tests.integration.test_approved_generation_settings import (
    acceptance_for_route,
    local_activation_boundary,  # noqa: F401
    route_payload,
)
from tests.integration.test_system_settings_flow import _headers, _register
from tests.integration.test_system_settings_flow import client as client
from tests.unit.test_ticket19_evidence_execution import _SufficientPilotRetriever, _valid_answer


def test_authenticated_four_request_burst_preserves_member_slots_and_sse_completion(client, monkeypatch):
    admin = _headers(_register(client, username="burst-admin", role="admin"))
    members = [_headers(_register(client, username=f"burst-member-{index}")) for index in range(4)]
    release = threading.Event()
    calls = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append(1)
            while not release.is_set():
                await asyncio.sleep(0.01)
            return _valid_answer()

    async def validated(self, payload):
        return None

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    monkeypatch.setattr("app.settings.generation_routes.build_generation_provider", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(ChatService, "_resolve_retriever", lambda self: (_SufficientPilotRetriever(), "published-fixture"))
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda self: (None, "disabled"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda self: (None, "disabled"))
    monkeypatch.setattr(client.app.state, "operational_event_session_factory", client.app.state.settings_session_factory)
    path = "/api/v1/settings/generation-route"
    route = client.put(path, headers=admin, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, admin, route)
    assert client.post(path + "/activate", headers=admin, json={
        "route_identity": route["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    }).status_code == 200
    question = "Which reviewed operating decision applies for environment=production?"

    def request(index, member, streaming=False):
        return client.post(
            "/api/v1/chat/stream" if streaming else "/api/v1/chat", headers=members[member],
            json={"message": question, "session_id": f"burst-session-{index}"},
        )

    def await_counts(executing, queued):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            data = client.get("/api/v1/operations", headers=admin).json()["data"]["admission"]
            if (data["executing"], data["queued"]) == (executing, queued):
                return
            time.sleep(0.01)
        raise AssertionError("authenticated queue state did not become ready")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        try:
            futures.extend([pool.submit(request, 0, 0), pool.submit(request, 1, 1)])
            await_counts(2, 0)
            futures.extend([pool.submit(request, 2, 0, True), pool.submit(request, 3, 2, True)])
            await_counts(2, 2)
            denied = request(4, 3)
            assert denied.status_code == 429
            assert denied.json()["code"] == "CHAT_QUEUE_FULL"
            same_member = request(5, 0)
            assert same_member.status_code == 429
            assert same_member.json()["code"] == "CHAT_MEMBER_LIMIT"
            failures = client.get("/api/v1/operations", headers=admin).json()["data"]["failures"]
            assert {"CHAT_QUEUE_FULL", "CHAT_MEMBER_LIMIT"} <= {item["code"] for item in failures}
            assert len(calls) <= 2
        finally:
            release.set()
        responses = [future.result(timeout=10) for future in futures]
    assert len(calls) == 4
    for response in responses[:2]:
        assert response.status_code == 200, response.text
        assert response.json()["data"]["outcome"] == "evidence_gated_answer"
    for response in responses[2:]:
        assert '"stage": "queued"' in response.text
        assert '"stage": "running"' in response.text
        assert '"outcome": "evidence_gated_answer"' in response.text
        assert response.text.endswith("event: done\ndata: [DONE]\n\n")
    await_counts(0, 0)
