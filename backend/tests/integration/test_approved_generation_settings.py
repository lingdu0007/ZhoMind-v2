import secrets

import pytest

from tests.integration.test_system_settings_flow import _headers, _register
from tests.integration.test_system_settings_flow import client as client


@pytest.fixture(autouse=True)
def local_activation_boundary(monkeypatch):
    from app.common.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "generation_validation_mode", "local_development", raising=False)
    monkeypatch.setattr(settings, "generation_deployment_identity", "deployment:editorial-preview-20260905", raising=False)
    monkeypatch.setattr(settings, "generation_product_revision", "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9", raising=False)


def route_payload():
    return {
        "data_scope": "team_shared_pilot",
        "providers": [{
            "provider": "primary",
            "provider_type": "openai",
            "model": "model-v1",
            "service_url": "https://provider.example.test/v1",
            "endpoint_class": "public_https",
            "data_scope": "team_shared_pilot",
            "timeout_seconds": 10,
            "provider_api_key": secrets.token_urlsafe(32),
        }],
        "max_attempts": 1,
        "total_timeout_seconds": 20,
    }


def test_admin_can_retain_a_versioned_inactive_route_without_returning_credentials(client):
    member = _register(client, username="route-member")
    admin = _register(client, username="route-admin", role="admin")
    payload = route_payload()
    secret = payload["providers"][0]["provider_api_key"]
    path = "/api/v1/settings/generation-route"

    denied = client.put(path, headers=_headers(member), json=payload)
    assert denied.status_code == 403
    saved = client.put(path, headers=_headers(admin), json=payload)
    assert saved.status_code == 200, saved.text
    record = saved.json()["data"]
    assert record["route_identity"].startswith("provider_route:")
    assert record["activation_status"] == "inactive"
    assert record["providers"][0]["credential_configured"] is True
    assert record["providers"][0]["validation_evidence"] is None
    assert secret not in saved.text
    loaded = client.get(path, headers=_headers(admin))
    assert loaded.status_code == 200
    assert loaded.json()["data"]["draft"] == record
    assert loaded.json()["data"]["active"] is None
    assert secret not in loaded.text


def acceptance_for_route(client, headers, route):
    from tests.integration.test_delivery_acceptance import _activate, _create
    from tests.support.generation import generation_acceptance_payload

    record = _create(client, headers, generation_acceptance_payload(route))
    return _activate(client, headers, record["record_id"])["record_id"]


def test_activation_requires_exact_verified_acceptance_and_preserves_prior_route_on_validation_failure(client, monkeypatch):
    headers = _headers(_register(client, username="route-admin", role="admin"))
    path = "/api/v1/settings/generation-route"
    first = client.put(path, headers=headers, json=route_payload()).json()["data"]
    missing = client.post(path + "/activate", headers=headers, json={
        "route_identity": first["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": "delivery_acceptance_record:missing",
    })
    assert missing.status_code == 409
    assert client.get(path, headers=headers).json()["data"]["active"] is None

    acceptance = acceptance_for_route(client, headers, first)
    accepted = client.get(f"/api/v1/acceptance/records/{acceptance}", headers=headers).json()["data"]
    assert accepted["current_status"] == "active"
    assert accepted["approver_identities"]
    assert not accepted["blockers"]
    assert first["route_identity"] in accepted["product_identities"]

    async def validated(self, payload):
        return None

    monkeypatch.setattr("app.settings.generation_routes.GenerationRouteService.validate_connection", validated)
    activated = client.post(path + "/activate", headers=headers, json={
        "route_identity": first["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    })
    assert activated.status_code == 200, activated.text
    active = client.get(path, headers=headers).json()["data"]["active"]
    assert active["route_identity"] == first["route_identity"]
    assert active["providers"][0]["validation_evidence"]["record_identity"] == acceptance

    second = client.put(path, headers=headers, json=route_payload()).json()["data"]
    wrong_evidence = client.post(path + "/activate", headers=headers, json={
        "route_identity": second["route_identity"], "expected_active_identity": first["route_identity"],
        "acceptance_record_identity": acceptance,
    })
    assert wrong_evidence.status_code == 409
    second_acceptance = acceptance_for_route(client, headers, second)

    async def failed(self, payload):
        raise RuntimeError("raw credential and private provider response")

    monkeypatch.setattr("app.settings.generation_routes.GenerationRouteService.validate_connection", failed)
    rejected = client.post(path + "/activate", headers=headers, json={
        "route_identity": second["route_identity"], "expected_active_identity": first["route_identity"],
        "acceptance_record_identity": second_acceptance,
    })
    assert rejected.status_code == 409
    assert "credential" not in rejected.text
    assert client.get(path, headers=headers).json()["data"]["active"]["route_identity"] == first["route_identity"]


def test_captured_route_survives_atomic_replacement_and_new_capture_uses_new_identity(client, monkeypatch):
    import asyncio

    from app.settings.generation_routes import GenerationRouteService

    headers = _headers(_register(client, username="route-admin", role="admin"))
    path = "/api/v1/settings/generation-route"

    async def validated(self, payload):
        return None

    class Provider:
        def __init__(self, model):
            self.model = model

        async def complete(self, prompt, *, system_prompt=None):
            return self.model

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    monkeypatch.setattr(
        "app.settings.generation_routes.build_generation_provider",
        lambda settings, **kwargs: Provider(settings.openai_model),
    )

    async def capture():
        async with client.app.state.settings_session_factory() as session:
            return await GenerationRouteService(session).capture()

    previous = None
    captured = []
    for model in ("model-v1", "model-v2"):
        payload = route_payload()
        payload["providers"][0]["model"] = model
        route = client.put(path, headers=headers, json=payload).json()["data"]
        acceptance = acceptance_for_route(client, headers, route)
        response = client.post(path + "/activate", headers=headers, json={
            "route_identity": route["route_identity"], "expected_active_identity": previous,
            "acceptance_record_identity": acceptance,
        })
        assert response.status_code == 200
        previous = route["route_identity"]
        captured.append(asyncio.run(capture()))

    first = asyncio.run(captured[0].complete(primary="ignored", fallbacks=[], prompt="payload"))
    second = asyncio.run(captured[1].complete(primary="ignored", fallbacks=[], prompt="payload"))
    assert first["text"] == "model-v1"
    assert second["text"] == "model-v2"
    assert first["route_identity"] != second["route_identity"]


def test_authenticated_chat_uses_persisted_approved_route_in_http_sse_and_history(client, monkeypatch):
    from app.service.chat_service import ChatService
    from app.settings.generation_routes import GenerationRouteService
    from tests.integration.test_ticket20_answer_execution_persistence import _sse_event_data
    from tests.unit.test_ticket19_evidence_execution import _SufficientPilotRetriever, _valid_answer

    admin_headers = _headers(_register(client, username="route-admin", role="admin"))
    member_headers = _headers(_register(client, username="route-member"))
    monkeypatch.setattr(
        client.app.state, "operational_event_session_factory",
        client.app.state.settings_session_factory, raising=False,
    )
    calls = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append((prompt, system_prompt))
            return _valid_answer()

    async def validated(self, payload):
        return None

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    monkeypatch.setattr("app.settings.generation_routes.build_generation_provider", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(ChatService, "_resolve_retriever", lambda self: (_SufficientPilotRetriever(), "published-fixture"))
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda self: (None, "disabled"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda self: (None, "disabled"))
    path = "/api/v1/settings/generation-route"
    route = client.put(path, headers=admin_headers, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, admin_headers, route)
    assert client.post(path + "/activate", headers=admin_headers, json={
        "route_identity": route["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    }).status_code == 200
    question = "Which reviewed operating decision applies for environment=production?"
    normal = client.post("/api/v1/chat", headers=member_headers, json={"message": question, "session_id": "route-normal"})
    assert normal.status_code == 200, normal.text
    data = normal.json()["data"]
    assert data["outcome"] == "evidence_gated_answer"
    streamed = client.post("/api/v1/chat/stream", headers=member_headers, json={"message": question, "session_id": "route-stream"})
    assert _sse_event_data(streamed.text, "outcome") == {"outcome": "evidence_gated_answer"}
    assert calls[0] == calls[1]
    for session_id in ("route-normal", "route-stream"):
        history = client.get(f"/api/v1/sessions/{session_id}", headers=member_headers)
        assert history.status_code == 200, history.text
        assert data["answer_execution"]["snapshot_ids"][0] in history.text
    assert client.get("/api/v1/operations", headers=member_headers).status_code == 403
    for _ in range(21):
        assert client.get("/api/v1/operations", headers=admin_headers).status_code == 200
    operations = client.get("/api/v1/operations", headers=admin_headers)
    assert operations.status_code == 200, operations.text
    observations = operations.json()["data"]["generation"]["route_executions"]
    assert len(observations) == 2
    assert len({item["request_id"] for item in observations}) == 2
    for observed in observations:
        assert observed["route_identity"] == route["route_identity"]
        assert observed["route_reason"] == "succeeded"
        assert observed["attempts"][0]["approval_identity"] == route["providers"][0]["approval_identity"]
        assert len(observed["attempts"][0]["payload_sha256"]) == 64
        assert len(observed["attempts"][0]["snapshot_sha256"]) == 64
    assert question not in operations.text
    assert _valid_answer() not in operations.text


def test_suspended_validation_evidence_stops_new_capture_and_is_not_projected_active(client, monkeypatch):
    import asyncio

    from app.settings.generation_routes import GenerationRouteService

    headers = _headers(_register(client, username="route-admin", role="admin"))
    path = "/api/v1/settings/generation-route"
    route = client.put(path, headers=headers, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, headers, route)

    async def validated(self, payload):
        return None

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    assert client.post(path + "/activate", headers=headers, json={
        "route_identity": route["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    }).status_code == 200
    suspended = client.post(f"/api/v1/acceptance/records/{acceptance}/status", headers=headers, json={
        "status": "suspended", "reason_code": "integrity_failure", "status_failure": {
            "check_id": "check:generation-privacy", "reason": "privacy validation revoked",
            "failure_kind": "shared_privacy",
            "blocking_scope": {"scope": "collection", "identity": "collection:production-rag-agent-engineering"},
            "evidence_links": ["evidence://ticket22/privacy-revoked"],
        },
    })
    assert suspended.status_code == 200, suspended.text

    async def capture():
        async with client.app.state.settings_session_factory() as session:
            return await GenerationRouteService(session).capture()

    assert asyncio.run(capture()).approved_route is None
    projection = client.get(path, headers=headers).json()["data"]
    assert projection["active"] is None
    assert projection["draft"]["activation_status"] == "inactive"


def test_new_draft_during_connection_validation_rejects_stale_activation(client, monkeypatch):
    from app.settings.generation_routes import GenerationRouteService

    member = _register(client, username="route-admin", role="admin")
    headers = _headers(member)
    path = "/api/v1/settings/generation-route"
    original = client.put(path, headers=headers, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, headers, original)
    replacement = []

    async def replaced(self, payload):
        async with client.app.state.settings_session_factory() as session:
            replacement.append(await GenerationRouteService(session).save(actor=payload["configured_by"], payload=route_payload()))

    monkeypatch.setattr(GenerationRouteService, "validate_connection", replaced)
    response = client.post(path + "/activate", headers=headers, json={
        "route_identity": original["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    })
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "GENERATION_ROUTE_STALE"
    state = client.get(path, headers=headers).json()["data"]
    assert state["active"] is None
    assert state["draft"]["route_identity"] == replacement[0]["route_identity"]


@pytest.mark.parametrize("changed", ["mode", "deployment", "revision"])
def test_local_or_foreign_evidence_cannot_authorize_current_live_boundary(client, monkeypatch, changed):
    from app.common.config import get_settings
    from app.settings.generation_routes import GenerationRouteService

    headers = _headers(_register(client, username="route-admin", role="admin"))
    path = "/api/v1/settings/generation-route"
    route = client.put(path, headers=headers, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, headers, route)
    calls = []

    async def validated(self, payload):
        calls.append(payload)

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    field, value = {
        "mode": ("generation_validation_mode", "controlled_live"),
        "deployment": ("generation_deployment_identity", "deployment:other-20260908"),
        "revision": ("generation_product_revision", "product_revision:" + "f" * 40),
    }[changed]
    monkeypatch.setattr(get_settings(), field, value)
    response = client.post(path + "/activate", headers=headers, json={
        "route_identity": route["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    })
    assert response.status_code == 409, response.text
    assert not calls
    assert client.get(path, headers=headers).json()["data"]["active"] is None


@pytest.mark.parametrize("failure,transport", [
    ("application_failure", "sse"), ("malformed_envelope", "sse"),
    ("application_failure", "http"), ("application_failure", "http_with_id"),
    ("outer_send_failure", "sse"),
])
def test_failed_stream_still_records_non_content_attempt_identity(client, monkeypatch, failure, transport):
    import asyncio
    import json

    from app.rag.answer_execution import AnswerExecutionContractError
    from app.rag.interfaces import GenerationCompletion
    from app.service.chat_service import ChatService
    from app.settings.generation_routes import GenerationRouteService
    from tests.unit.test_ticket19_evidence_execution import _SufficientPilotRetriever

    headers = _headers(_register(client, username="route-admin", role="admin"))
    monkeypatch.setattr(client.app.state, "operational_event_session_factory",
                        client.app.state.settings_session_factory, raising=False)
    started = asyncio.Event()
    cancelled = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            if failure == "outer_send_failure":
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.append(True)
            if failure == "application_failure":
                raise ValueError("private input parsing detail")
            return GenerationCompletion("private generated content", {"bad": "private content"})

    async def validated(self, payload):
        return None

    monkeypatch.setattr(GenerationRouteService, "validate_connection", validated)
    monkeypatch.setattr("app.settings.generation_routes.build_generation_provider", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(ChatService, "_resolve_retriever", lambda self: (_SufficientPilotRetriever(), "published-fixture"))
    monkeypatch.setattr(ChatService, "_resolve_reranker", lambda self: (None, "disabled"))
    monkeypatch.setattr(ChatService, "_resolve_judge", lambda self: (None, "disabled"))
    path = "/api/v1/settings/generation-route"
    route = client.put(path, headers=headers, json=route_payload()).json()["data"]
    acceptance = acceptance_for_route(client, headers, route)
    assert client.post(path + "/activate", headers=headers, json={
        "route_identity": route["route_identity"], "expected_active_identity": None,
        "acceptance_record_identity": acceptance,
    }).status_code == 200
    question = {"message": "Which reviewed operating decision applies for environment=production?"}
    response_request_id = None
    if failure == "outer_send_failure" or transport.startswith("http"):
        path = "/api/v1/chat/stream" if failure == "outer_send_failure" else "/api/v1/chat"
        if transport == "http_with_id":
            headers["X-Request-ID"] = "cba9d526-19f9-4571-bc2a-f1162433696a"

        async def call():
            body = json.dumps(question).encode()
            received = False
            messages = []
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": path, "raw_path": path.encode(), "query_string": b"",
                "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                            *((key.lower().encode(), value.encode()) for key, value in headers.items())],
                "client": ("testclient", 50000), "server": ("testserver", 80),
            }

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.Future()

            async def send(message):
                messages.append(message)
                if failure == "outer_send_failure" and b"generating" in message.get("body", b""):
                    await asyncio.wait_for(started.wait(), timeout=3)
                    raise OSError("simulated outer transport failure")

            expected_error = OSError if failure == "outer_send_failure" else AnswerExecutionContractError
            with pytest.raises(expected_error):
                await client.app(scope, receive, send)
            return messages

        messages = asyncio.run(call())
        if transport.startswith("http"):
            assert messages[0]["status"] == 500
            response = json.loads(b"".join(message.get("body", b"") for message in messages))
            response_request_id = response["request_id"]
        else:
            assert cancelled == [True]
    else:
        streamed = client.post("/api/v1/chat/stream", headers=headers, json=question)
        assert "CHAT_STREAM_FAILED" in streamed.text
    observed = client.get("/api/v1/operations", headers=headers)
    events = observed.json()["data"]["generation"]["route_executions"]
    assert len(events) == 1
    assert events[0]["route_identity"] == route["route_identity"]
    assert events[0]["route_reason"] == ("user_cancellation" if cancelled else "application_failure")
    assert events[0]["attempts"][0]["approval_identity"] == route["providers"][0]["approval_identity"]
    assert "private" not in observed.text
    assert events[0]["request_id"] not in {"", "unknown-request"}
    if response_request_id is not None:
        assert events[0]["request_id"] == response_request_id
    if transport == "http_with_id":
        assert response_request_id == "cba9d526-19f9-4571-bc2a-f1162433696a"
