import pytest

from tests.browser_acceptance_api import _DeterministicLlm
from tests.integration.test_approved_generation_settings import acceptance_for_route, route_payload
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _maintainer


def _activate_route(client, monkeypatch, admin):
    from cryptography.fernet import Fernet

    from app.common.config import get_settings
    from app.settings.generation_routes import GenerationRouteService

    settings = get_settings()
    monkeypatch.setattr(settings, "system_settings_encryption_key", Fernet.generate_key().decode("ascii"))
    monkeypatch.setattr(settings, "generation_validation_mode", "local_development")
    monkeypatch.setattr(settings, "generation_deployment_identity", "deployment:editorial-preview-20260905")
    monkeypatch.setattr(settings, "generation_product_revision", "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9")

    class Provider(_DeterministicLlm):
        unavailable = False
        attempts = 0

        async def complete(self, prompt, *, system_prompt=None):
            if prompt != "Connection validation. Reply with OK.":
                self.attempts += 1
                if self.unavailable:
                    raise TimeoutError("deterministic provider unavailable")
            return await super().complete(prompt, system_prompt=system_prompt)

    provider = Provider()
    acceptance_failures = []
    original_verify = GenerationRouteService._verify_acceptance

    async def verify_acceptance(service, identity, route):
        try:
            return await original_verify(service, identity, route)
        except Exception as exc:
            cause = exc.__cause__
            acceptance_failures.append({
                "type": type(cause).__name__,
                "reason": str(cause) if isinstance(cause, (ValueError, KeyError)) else getattr(cause, "code", None),
            })
            raise

    monkeypatch.setattr(GenerationRouteService, "_verify_acceptance", verify_acceptance)
    monkeypatch.setattr("app.settings.generation_routes.build_generation_provider", lambda *args, **kwargs: provider)
    route_url = "/api/v1/settings/generation-route"
    saved = client.put(route_url, headers=admin, json=route_payload())
    assert saved.status_code == 200, saved.text
    route = saved.json()["data"]
    acceptance = acceptance_for_route(client, admin, route)
    activated = client.post(
        route_url + "/activate",
        headers=admin,
        json={"route_identity": route["route_identity"], "expected_active_identity": None, "acceptance_record_identity": acceptance},
    )
    if activated.status_code != 200:
        retained = client.get(f"/api/v1/acceptance/records/{acceptance}", headers=admin)
        projection = retained.json().get("data", {})
        pytest.fail(str({
            "activation": activated.json(),
            "verification_failures": acceptance_failures,
            "acceptance_read_status": retained.status_code,
            "acceptance": {
                key: projection.get(key) for key in (
                    "record_id", "current_status", "status_history", "stage", "blockers",
                    "conditions", "product_identities", "affected_scope",
                )
            },
        }))
    return provider


@pytest.mark.parametrize("reactivate", ["unchanged", "reactivated", "draft_only"])
def test_published_provider_failure_is_independently_diagnosed_and_replayed_after_recovery(client, monkeypatch, reactivate):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="maintenance-provider")
    admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="provider-reporter"))
    provider = _activate_route(client, monkeypatch, admin)

    def answer(headers, session_id):
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={
                "session_id": session_id,
                "message": "Which Candidate publication contract applies? deployment=production",
                "query_conditions": [
                    {"condition_id": "ticket24-production", "field": "deployment", "operator": "equals", "value": "production"}
                ],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["outcome"] == "evidence_gated_answer", response.text
        messages = client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()["messages"]
        result = next(message for message in reversed(messages) if message["type"] == "assistant")
        assert result["answer_execution"]["knowledge_version_identities"] == [publication]
        return result

    reported = answer(reporter, "provider-reported")
    independent = answer(worker, "provider-independent")
    signal = client.post(
        "/api/v1/knowledge-feedback",
        headers=reporter,
        json={"answer_id": reported["id"], "entry_id": "maintenance-provider", "label": "insufficient_evidence"},
    )
    assert signal.status_code == 200, signal.text
    created = client.post(
        "/api/v1/maintenance/items",
        headers=maintainer,
        json={
            "classification": "product-privacy-operations",
            "severity": "p2",
            "disposition": "needs-reproduction",
            "coverage_position": "provider_failure_and_observability",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal.json()["data"]["id"]],
        },
    )
    assert created.status_code == 200, created.text
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    joined = client.post(base + "/administrator", headers=admin, json={"expected_revision": 1})
    assert joined.status_code == 200, joined.text
    triaged = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 2, "state": "triaged"})
    assert triaged.status_code == 200, triaged.text
    provider.unavailable = True
    reproduced = client.post(
        base + "/reproductions",
        headers=worker,
        json={
            "expected_revision": 3,
            "answer_id": independent["id"],
            "signal_id": signal.json()["data"]["id"],
            "expected_outcome": "evidence_gated_answer",
            "confirmed_synthetic_fixture": True,
            "verified_observation": "provider_failure",
        },
    )
    assert reproduced.status_code == 200, reproduced.text
    fixture = reproduced.json()["data"]
    assert fixture["observed_outcome"] == "generation_unavailable"
    assert fixture["evidence_publication_identities"] == [publication]
    diagnosed = client.post(
        base + "/diagnosis",
        headers=maintainer,
        json={"expected_revision": 4, "fixture_identity": fixture["id"], "observation": "provider_failure"},
    )
    assert diagnosed.status_code == 200, diagnosed.text
    assert diagnosed.json()["data"]["disposition"] == "provider-work"
    approved = client.post(
        base + "/findings", headers=maintainer, json={"expected_revision": 5, "fixture_identity": fixture["id"]}
    )
    assert approved.status_code == 200, approved.text
    started = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 6, "state": "in_progress"})
    assert started.status_code == 200, started.text
    provider.unavailable = False
    replayed = client.post(
        base + "/replays", headers=worker,
        json={"expected_revision": 7, "fixture_identity": fixture["id"], "answer_id": independent["id"]},
    )
    assert replayed.status_code == 200, replayed.text
    replay = replayed.json()["data"]
    assert replay["passed"] is True
    assert replay["observed_outcome"] == "evidence_gated_answer"
    assert replay["active_publication_identities"] == [publication]
    assert replay["request_sha256"] == fixture["request_sha256"]
    assert replay["query_condition_set_identity"] == fixture["query_condition_set_identity"]
    assert provider.attempts >= 4
    active = client.get("/api/v1/settings/generation-route", headers=admin).json()["data"]["active"]
    acceptance = active["providers"][0]["validation_evidence"]["record_identity"]
    missing = client.post(
        base + "/resolution",
        headers=maintainer,
        json={"expected_revision": 8, "disposition": "provider-work", "artifact_identities": [fixture["id"], replay["id"]]},
    )
    assert missing.status_code == 409, missing.text
    if reactivate == "draft_only":
        draft = client.put("/api/v1/settings/generation-route", headers=admin, json=route_payload())
        assert draft.status_code == 200, draft.text
        assert draft.json()["data"]["route_identity"] != active["route_identity"]
    if reactivate == "reactivated":
        activated = client.post(
            "/api/v1/settings/generation-route/activate",
            headers=admin,
            json={
                "route_identity": active["route_identity"],
                "expected_active_identity": active["route_identity"],
                "acceptance_record_identity": acceptance,
            },
        )
        assert activated.status_code == 200, activated.text
    resolved = client.post(
        base + "/resolution",
        headers=maintainer,
        json={
            "expected_revision": 8,
            "disposition": "provider-work",
            "artifact_identities": [fixture["id"], replay["id"], active["route_identity"], acceptance],
        },
    )
    if reactivate == "reactivated":
        assert resolved.status_code == 409, resolved.text
        return
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["data"]["state"] == "resolved"
    closed = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 9, "state": "closed_confirmation"})
    assert closed.status_code == 200, closed.text
