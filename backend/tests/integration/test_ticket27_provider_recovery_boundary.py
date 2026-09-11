import asyncio

import pytest

from tests.integration.test_approved_generation_settings import acceptance_for_route
from tests.integration.test_privacy_operations_boundaries import _headers, _register
from tests.integration.test_privacy_operations_boundaries import client as client
from tests.integration.test_ticket25_withdrawal import _publish
from tests.integration.test_ticket27_maintenance import _gap, _maintainer
from tests.integration.test_ticket27_maintenance_provider import _activate_route


async def _corrupt_authorization(client, identity):
    from sqlalchemy import update

    from app.model.canonical import CanonicalRecordModel

    async with client.app.state.test_auth_session_factory() as session:
        grant = await session.get(CanonicalRecordModel, identity)
        assert grant is not None
        connection = await session.connection()
        await connection.execute(
            update(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == grant.stable_id).values(
                payload={**grant.payload, "private_note": "private authorization corruption"},
            ),
        )
        await session.commit()


@pytest.mark.parametrize("severity", ["p0", "p1"])
@pytest.mark.parametrize("authorize", [
    False, True, "execute", "execute_independent", "execute_corrupt", "execute_record_only", "execute_check_only", "execute_foreign_admin",
    "execute_deleted", "execute_expired", "execute_deleted_after_grant",
    "execute_deleted_unapproved",
])
def test_suspended_route_cannot_be_used_or_misdiagnosed_by_ordinary_maintenance_reproduction(client, monkeypatch, severity, authorize):
    publisher = _headers(_register(client, username="ticket25-admin", role="admin"))
    publication = _publish(client, publisher, entry_id="provider-recovery-boundary")
    admin, maintainer, worker = _maintainer(client)
    reporter = _headers(_register(client, username="recovery-boundary-reporter"))
    provider = _activate_route(client, monkeypatch, admin)
    question = "Which Candidate publication contract applies? deployment=production"
    reported = _gap(client, reporter, "recovery-boundary-report", question)
    independent_question = f"Independent publication check: {question}" if authorize == "execute_independent" else question
    independent = _gap(client, worker, "recovery-boundary-independent", independent_question)
    assert reported["answer_execution"]["outcome"] == "evidence_gated_answer"
    assert independent["answer_execution"]["knowledge_version_identities"] == [publication]
    signal = client.post(
        "/api/v1/knowledge-feedback", headers=reporter,
        json={"answer_id": reported["id"], "entry_id": "provider-recovery-boundary", "label": "insufficient_evidence"},
    )
    assert signal.status_code == 200, signal.text
    route = client.get("/api/v1/settings/generation-route", headers=admin).json()["data"]["active"]
    acceptance = route["providers"][0]["validation_evidence"]["record_identity"]
    suspended = client.post(
        f"/api/v1/acceptance/records/{acceptance}/status", headers=admin,
        json={
            "status": "suspended", "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:generation-privacy", "reason": "independently verified provider privacy boundary failure",
                "failure_kind": "shared_privacy",
                "blocking_scope": {"scope": "collection", "identity": "collection:production-rag-agent-engineering"},
                "evidence_links": ["evidence://maintenance/provider-recovery-boundary"],
            },
        },
    )
    assert suspended.status_code == 200, suspended.text
    assert client.get("/api/v1/settings/generation-route", headers=admin).json()["data"]["active"] is None
    attempts = provider.attempts
    blocked = _gap(client, reporter, "recovery-boundary-blocked", question)
    assert blocked["answer_execution"]["outcome"] == "generation_unavailable"
    assert provider.attempts == attempts
    created = client.post(
        "/api/v1/maintenance/items", headers=maintainer,
        json={
            "classification": "product-privacy-operations", "severity": severity, "disposition": "needs-reproduction",
            "coverage_position": "provider_failure_and_observability",
            "work_owner_username": "maintenance-worker",
            "signal_ids": [signal.json()["data"]["id"]], "containment_record_identity": acceptance,
        },
    )
    if created.status_code != 200:
        retained = client.get(f"/api/v1/acceptance/records/{acceptance}", headers=admin).json().get("data", {})
        pytest.fail(str({
            "response": created.json(),
            "acceptance_status": retained.get("current_status"),
            "status_history": [
                {key: event.get(key) for key in ("event_id", "status", "occurred_at", "recorded_at", "reason_code")}
                for event in retained.get("status_history", [])
            ],
        }))
    base = f"/api/v1/maintenance/items/{created.json()['data']['id']}"
    triaged = client.post(base + "/transition", headers=maintainer, json={"expected_revision": 1, "state": "triaged"})
    assert triaged.status_code == 200, triaged.text
    payload = {
            "expected_revision": 2, "answer_id": independent["id"], "signal_id": signal.json()["data"]["id"],
            "expected_outcome": "evidence_gated_answer", "confirmed_synthetic_fixture": True,
            "verified_observation": "provider_failure",
    }
    if authorize:
        admission = acceptance_for_route(client, admin, route)
        authorization = {
            "expected_revision": 2, "route_identity": route["route_identity"],
            "admission_acceptance_identity": admission,
            "query_condition_set_identity": independent["answer_execution"]["query_condition_set"]["identity"],
        }
        granted = client.post(
            base + "/provider-verification-authorizations", headers=admin, json=authorization,
        )
        assert granted.status_code == 200, granted.text
        record = granted.json()["data"]
        assert record["acceptance_record_identity"] == admission
        assert record["route_identity"] == route["route_identity"]
        assert record["active_publication_identities"] == [publication]
        assert record["query_condition_set_identity"] == authorization["query_condition_set_identity"]
        assert record["item_revision"] == 2
        assert record["approved_fixture_identity"] is None
        assert question not in granted.text
        assert independent_question not in granted.text
        assert independent["id"] not in granted.text
        assert provider.attempts == attempts
        repeated = client.post(base + "/provider-verification-authorizations", headers=admin, json=authorization)
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["data"] == record
        grant_path = f"/api/v1/maintenance/provider-verification-authorizations/{record['id']}"
        read = client.get(grant_path, headers=worker)
        assert read.status_code == 200, read.text
        assert read.json()["data"] == record
        assert client.get(grant_path, headers=reporter).status_code == 403
        for headers in (worker, maintainer):
            denied = client.post(base + "/provider-verification-authorizations", headers=headers, json=authorization)
            assert denied.status_code == 403, denied.text
        for patch in (
            {"expected_revision": 1}, {"admission_acceptance_identity": acceptance},
            {"query_condition_set_identity": "0" * 64},
        ):
            rejected = client.post(
                base + "/provider-verification-authorizations", headers=admin, json={**authorization, **patch},
            )
            assert rejected.status_code == 409, rejected.text
        if isinstance(authorize, str) and authorize.startswith("execute"):
            provider.unavailable = True
            reproduced = client.post(
                base + "/reproductions", headers=worker,
                json={**payload, "provider_verification_authorization_identity": record["id"]},
            )
            assert reproduced.status_code == 200, reproduced.text
            fixture = reproduced.json()["data"]
            assert fixture["observed_outcome"] == "generation_unavailable"
            assert fixture["evidence_publication_identities"] == [publication]
            context = fixture["generation_context"]
            assert context["authorization_identity"] == record["id"]
            assert context["route_identity"] == route["route_identity"]
            assert context["acceptance_record_identity"] == admission
            assert "activation_event_id" not in context
            assert provider.attempts > attempts
            if authorize == "execute_corrupt":
                asyncio.run(_corrupt_authorization(client, record["id"]))
            diagnosed = client.post(
                base + "/diagnosis", headers=maintainer,
                json={"expected_revision": 3, "fixture_identity": fixture["id"], "observation": "provider_failure"},
            )
            if authorize == "execute_corrupt":
                assert diagnosed.status_code == 409, diagnosed.text
                assert "private authorization corruption" not in diagnosed.text
                assert client.get(base, headers=worker).json()["data"]["revision"] == 3
                return
            assert diagnosed.status_code == 200, diagnosed.text
            assert diagnosed.json()["data"]["disposition"] == "provider-work"
            assert client.get("/api/v1/settings/generation-route", headers=admin).json()["data"]["active"] is None
            attempts = provider.attempts
            if authorize == "execute_deleted_unapproved":
                deleted = client.delete(f"/api/v1/knowledge-feedback/{signal.json()['data']['id']}", headers=reporter)
                assert deleted.status_code == 200, deleted.text
                denied_grant = client.post(
                    base + "/provider-verification-authorizations", headers=admin,
                    json={**authorization, "expected_revision": 4},
                )
                assert denied_grant.status_code == 409, denied_grant.text
                assert provider.attempts == attempts
                assert client.get(base, headers=worker).json()["data"]["revision"] == 4
            stale = client.post(
                base + "/reproductions", headers=worker,
                json={**payload, "expected_revision": 4, "provider_verification_authorization_identity": record["id"]},
            )
            assert stale.status_code == 409, stale.text
            assert provider.attempts == attempts
            approved = client.post(
                base + "/findings", headers=maintainer,
                json={"expected_revision": 4, "fixture_identity": fixture["id"]},
            )
            assert approved.status_code == 200, approved.text
            started = client.post(
                base + "/transition", headers=maintainer, json={"expected_revision": 5, "state": "in_progress"},
            )
            assert started.status_code == 200, started.text
            if authorize in {"execute_deleted", "execute_expired"}:
                if authorize == "execute_expired":
                    from datetime import UTC, datetime, timedelta

                    from app.model.knowledge_feedback import KnowledgeFeedbackSignal

                    async def expire_feedback():
                        async with client.app.state.test_auth_session_factory() as session:
                            raw = await session.get(KnowledgeFeedbackSignal, signal.json()["data"]["id"])
                            raw.expires_at = datetime.now(UTC) - timedelta(seconds=1)
                            await session.commit()
                    asyncio.run(expire_feedback())
                    assert client.get("/api/v1/knowledge-feedback", headers=reporter).json()["items"] == []
                else:
                    deleted = client.delete(f"/api/v1/knowledge-feedback/{signal.json()['data']['id']}", headers=reporter)
                    assert deleted.status_code == 200, deleted.text
                assert client.get(base, headers=worker).json()["data"]["signal_count"] == 0
            repair_grant = client.post(
                base + "/provider-verification-authorizations", headers=admin,
                json={**authorization, "expected_revision": 6},
            )
            assert repair_grant.status_code == 200, repair_grant.text
            repair_authorization = repair_grant.json()["data"]["id"]
            assert repair_grant.json()["data"]["approved_fixture_identity"] == fixture["id"]
            assert repair_authorization != record["id"]
            if authorize == "execute_deleted_after_grant":
                deleted = client.delete(f"/api/v1/knowledge-feedback/{signal.json()['data']['id']}", headers=reporter)
                assert deleted.status_code == 200, deleted.text
                assert client.get(base, headers=worker).json()["data"]["signal_count"] == 0
            provider.unavailable = False
            replayed = client.post(
                base + "/replays", headers=worker,
                json={
                    "expected_revision": 6, "fixture_identity": fixture["id"],
                    "answer_id": payload["answer_id"],
                    "provider_verification_authorization_identity": repair_authorization,
                },
            )
            assert replayed.status_code == 200, replayed.text
            replay = replayed.json()["data"]
            assert replay["passed"] is True
            assert replay["observed_outcome"] == "evidence_gated_answer"
            assert replay["evidence_publication_identities"] == [publication]
            assert replay["request_sha256"] == fixture["request_sha256"]
            assert replay["query_condition_set_identity"] == fixture["query_condition_set_identity"]
            assert replay["generation_context"]["authorization_identity"] == repair_authorization
            assert provider.attempts > attempts
            retained_replay = client.get(f"/api/v1/maintenance/replays/{replay['id']}", headers=worker)
            assert retained_replay.status_code == 200, retained_replay.text
            assert retained_replay.json()["data"] == replay
            assert client.get("/api/v1/settings/generation-route", headers=admin).json()["data"]["active"] is None
            premature = client.post(
                base + "/resolution", headers=maintainer,
                json={
                    "expected_revision": 7, "disposition": "provider-work",
                    "artifact_identities": [fixture["id"], replay["id"], route["route_identity"], admission],
                },
            )
            assert premature.status_code == 409, premature.text
            from tests.integration.test_delivery_acceptance import _activate, _create
            from tests.support.generation import generation_acceptance_payload

            repair_acceptance = generation_acceptance_payload(route)
            replay_link = f"evidence://maintenance/artifacts/{replay['id']}"
            if authorize != "execute_check_only":
                repair_acceptance["evidence_links"].append(replay_link)
            if authorize != "execute_record_only":
                next(check for check in repair_acceptance["checks"] if check["check_id"] == "check:generation-privacy")[
                    "evidence_links"
                ].append(replay_link)
            accepted_repair = _create(client, admin, repair_acceptance)
            reacceptance = _activate(client, admin, accepted_repair["record_id"])
            resolution = {
                "expected_revision": 7, "disposition": "provider-work",
                "artifact_identities": [fixture["id"], replay["id"], route["route_identity"], reacceptance["record_id"]],
            }
            inactive = client.post(base + "/resolution", headers=maintainer, json=resolution)
            assert inactive.status_code == 409, inactive.text
            activation_admin = (
                _headers(_register(client, username="other-recovery-administrator", role="admin"))
                if authorize == "execute_foreign_admin" else admin
            )
            activated = client.post(
                "/api/v1/settings/generation-route/activate", headers=activation_admin,
                json={
                    "route_identity": route["route_identity"], "expected_active_identity": route["route_identity"],
                    "acceptance_record_identity": reacceptance["record_id"],
                },
            )
            assert activated.status_code == 200, activated.text
            resolved = client.post(base + "/resolution", headers=maintainer, json=resolution)
            if authorize in {"execute_record_only", "execute_check_only", "execute_foreign_admin"}:
                assert resolved.status_code == 409, resolved.text
                retained = client.get(base, headers=worker).json()["data"]
                assert retained["state"] == "in_progress"
                assert retained["revision"] == 7
                assert retained["result_links"] == []
                return
            assert resolved.status_code == 200, resolved.text
            assert resolved.json()["data"]["state"] == "resolved"
            expected_link = (
                f"evidence://maintenance/artifacts/{reacceptance['record_id']}:"
                f"{reacceptance['status_history'][-1]['event_id']}"
            )
            assert expected_link in resolved.json()["data"]["result_links"]
            closed = client.post(
                base + "/transition", headers=maintainer,
                json={"expected_revision": 8, "state": "closed_confirmation"},
            )
            assert closed.status_code == 200, closed.text
            assert client.get(base, headers=worker).json()["data"]["state"] == "closed_confirmation"
            return
        asyncio.run(_corrupt_authorization(client, record["id"]))
        invalid = client.get(grant_path, headers=worker)
        assert invalid.status_code == 409, invalid.text
        assert "private authorization corruption" not in invalid.text
        assert client.post(base + "/provider-verification-authorizations", headers=admin, json=authorization).status_code == 409
    reproduced = client.post(base + "/reproductions", headers=worker, json=payload)
    assert reproduced.status_code == 409, reproduced.text
    assert reproduced.json()["code"] == "MAINTENANCE_EVIDENCE_REQUIRED"
    assert provider.attempts == attempts
    retained = client.get(base, headers=worker).json()["data"]
    assert retained["revision"] == 2
    assert retained["disposition"] == "needs-reproduction"
    assert retained.get("fixture_identity") is None
    assert retained.get("finding_identities", []) == []
    assert client.get("/api/v1/settings/generation-route", headers=admin).json()["data"]["active"] is None
