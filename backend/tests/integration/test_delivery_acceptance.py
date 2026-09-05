import asyncio
import fnmatch
from collections.abc import Generator
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.canonical import CanonicalRecordModel


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self.hashes)

    async def scan_iter(self, match: str):
        for key in list(self.hashes):
            if fnmatch.fnmatch(key, match):
                yield key

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.hashes:
                del self.hashes[key]
                deleted += 1
        return deleted


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "delivery-acceptance.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()
    settings = get_settings()
    monkeypatch.setattr(settings, "bootstrap_admin_username", "bootstrap-admin", raising=False)
    monkeypatch.setattr(settings, "bootstrap_admin_password", "bootstrap-password", raising=False)

    async def initialize_database() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(initialize_database())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.settings_session_factory = session_factory
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = SessionLocal
    asyncio.run(engine.dispose())


def _administrator_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "bootstrap-admin", "password": "bootstrap-password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['data']['access_token']}"}


def _record(stage: str = "local_development", predecessor_record_identity: str | None = None) -> dict:
    record = {
        "stage": stage,
        "predecessor_record_identity": predecessor_record_identity,
        "affected_scope": {
            "entry_identities": ["entry:decision-entry-001"],
            "collection_identities": ["collection:production-rag-agent-engineering"],
            "product_path_identities": ["product_path:chat-evidence-gate-v1"],
            "configuration_identities": ["configuration:retrieval-profile-20260905"],
            "protected_capability_identities": [],
            "public_claim_identities": [],
            "deployment_identity": "deployment:editorial-preview-20260905",
            "expected_blocking_scope": "entry_version",
            "blocking_scope_identity": "entry:decision-entry-001",
        },
        "content_identities": [
            "entry:decision-entry-001",
            "published_knowledge_version:decision-entry-001-v1",
        ],
        "product_identities": [
            "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9",
            "configuration:retrieval-profile-20260905",
        ],
        "conditions": {"language": "zh-cn", "audience": "internal-editorial-preview"},
        "assumptions": ["one reviewed entry", "approved retrieval profile"],
        "checks": [
            {
                "check_id": "check:impact-declaration",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/entry-001/impact"],
            },
            {
                "check_id": "check:bundle-secret-scan",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/entry-001/secret-scan"],
            },
            {
                "check_id": "check:product-path-impact",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/product-path/impact"],
            },
            {
                "check_id": "check:configuration-impact",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/configuration/impact"],
            },
            {
                "check_id": "check:product-revision",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/product-revision/impact"],
            },
            {
                "check_id": "check:entry-supported-query",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/entry-001/supported-query"],
                "identity_dependencies": ["entry:decision-entry-001"],
            },
            {
                "check_id": "check:entry-boundary-query",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/entry-001/boundary-query"],
                "identity_dependencies": ["entry:decision-entry-001"],
            },
            {
                "check_id": "check:evidence-citation-identity",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/entry-001/evidence-citation-identity"],
                "identity_dependencies": ["entry:decision-entry-001"],
            },
            {
                "check_id": "check:collection-retrieval",
                "result": "passed",
                "evidence_links": ["evidence://editorial-preview/collection/retrieval"],
            }
        ],
        "known_limits": ["not a Pilot Entry Baseline"],
        "risks": ["limited to one reviewed entry"],
        "evidence_links": ["evidence://editorial-preview/entry-001"],
        "reacceptance_triggers": ["entry material revision", "retrieval profile change"],
    }
    if stage != "local_development":
        record["checks"].extend(
            {
                "check_id": check_id,
                "result": "passed",
                "evidence_links": [f"evidence://acceptance/{stage}/{check_id.removeprefix('check:')}"],
            }
            for check_id in (
                "check:candidate-inspection",
                "check:editorial-review-evidence",
                "check:freshness-assurance",
                "check:explicit-publication",
            )
        )
    if stage in {"limited_team_pilot", "daily_use_release", "public_evidence_release"}:
        record["checks"].extend(
            {
                "check_id": check_id,
                "result": "passed",
                "evidence_links": [f"evidence://acceptance/{stage}/{check_id.removeprefix('check:')}"],
            }
            for check_id in (
                "check:authorization",
                "check:evidence-identity",
                "check:privacy",
                "check:publication",
                "check:policy-isolation",
                "check:closed-outcome",
                "check:pilot-workload",
                "check:recovery-evidence",
            )
        )
    if stage == "daily_use_release":
        record["checks"].extend(
            {
                "check_id": check_id,
                "result": "passed",
                "evidence_links": [f"evidence://acceptance/{stage}/{check_id.removeprefix('check:')}"],
            }
            for check_id in (
                "check:daily-use-adoption",
                "check:daily-use-service-objective",
            )
        )
    if stage == "public_evidence_release":
        record["checks"].append(
            {
                "check_id": "check:public-claim",
                "result": "passed",
                "evidence_links": ["evidence://acceptance/public-evidence-release/public-claim"],
            }
        )
    return record


def _create(client: TestClient, headers: dict[str, str], payload: dict) -> dict:
    response = client.post("/api/v1/acceptance/records", headers=headers, json=payload)
    assert response.status_code == 200, response.json()
    return response.json()["data"]


def _activate(client: TestClient, headers: dict[str, str], record_id: str) -> dict:
    projection = client.get(f"/api/v1/acceptance/records/{record_id}", headers=headers)
    assert projection.status_code == 200, projection.json()
    verified_checks = [
        {"check_id": check["check_id"], "evidence_links": check["evidence_links"]}
        for check in projection.json()["data"]["checks"]
        if check["result"] in {"passed", "carried_forward"}
    ]
    response = client.post(
        f"/api/v1/acceptance/records/{record_id}/status",
        headers=headers,
        json={
            "status": "active",
            "reason_code": "checks_verified",
            "verified_checks": verified_checks,
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()["data"]


def _active_local_record(client: TestClient, headers: dict[str, str]) -> dict:
    created = _create(client, headers, _record())
    assert created["current_status"] == "at_risk"
    return _activate(client, headers, created["record_id"])


def _pilot_entry_baseline(predecessor_record_identity: str) -> dict:
    record = _record("limited_team_pilot", predecessor_record_identity)
    record["is_pilot_entry_baseline"] = True
    return _bind_pilot_entry_baseline_scope(record)


def _bind_pilot_entry_baseline_scope(record: dict) -> dict:
    record["product_identities"].extend(
        [
            "user_boundary:internal-pilot-team-v1",
            "data_boundary:non-sensitive-internal-v1",
            "host:pilot-host-class-v1",
            "corpus:pilot-corpus-envelope-v1",
            "concurrency:two-active-two-queued-v1",
        ]
    )
    record["checks"].extend(
        {
            "check_id": check_id,
            "result": "passed",
            "evidence_links": [f"evidence://pilot-entry-baseline/{check_id.removeprefix('check:')}"],
        }
        for check_id in (
            "check:deployment-boundary",
            "check:workload-profile",
        )
    )
    return record


def _bind_entry_evidence(payload: dict, entry_identity: str) -> None:
    for check_id in (
        "check:entry-supported-query",
        "check:entry-boundary-query",
        "check:evidence-citation-identity",
    ):
        _check(payload, check_id)["identity_dependencies"].append(entry_identity)


def _check(payload: dict, check_id: str) -> dict:
    return next(check for check in payload["checks"] if check["check_id"] == check_id)


def test_administrator_records_server_derived_actors_then_explicitly_verifies_checks(client: TestClient) -> None:
    headers = _administrator_headers(client)
    created = _create(client, headers, _record())

    assert created["record_id"].startswith("delivery_acceptance_record:")
    assert created["stage"] == "local_development"
    assert created["current_status"] == "at_risk"
    assert created["blockers"] == []
    assert created["change_owner_identity"] == created["evaluator_identity"]
    assert created["approver_identities"] == []
    assert created["evaluator_identity"].startswith("member:")
    created_event = created["status_history"][0]
    assert created_event["status"] == "at_risk"
    assert created_event["reason_code"] == "record_created_pending_verification"
    assert created_event["event_type"] == "created"
    assert created_event["recorded_by"] == created["evaluator_identity"]
    assert created_event["event_id"]
    assert created_event["occurred_at"].endswith("+00:00")
    assert created_event["from_status"] is None

    active = _activate(client, headers, created["record_id"])
    projection = client.get(f"/api/v1/acceptance/records/{created['record_id']}", headers=headers)

    assert active["current_status"] == "active"
    assert active["approver_identities"] == [active["evaluator_identity"]]
    active_event = active["status_history"][-1]
    assert active_event["status"] == "active"
    assert active_event["reason_code"] == "checks_verified"
    assert active_event["event_type"] == "status_changed"
    assert active_event["recorded_by"] == active["evaluator_identity"]
    assert len(active_event["verified_checks"]) == 9
    assert projection.status_code == 200
    assert projection.json()["data"] == active


def test_stage_progression_requires_an_active_exact_scope_predecessor(client: TestClient) -> None:
    headers = _administrator_headers(client)
    rejected = client.post("/api/v1/acceptance/records", headers=headers, json=_record("daily_use_release"))
    assert rejected.status_code == 422

    pending_local = _create(client, headers, _record())
    pending_editorial = _record("editorial_preview", pending_local["record_id"])
    blocked = client.post("/api/v1/acceptance/records", headers=headers, json=pending_editorial)
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "ACCEPTANCE_PREDECESSOR_NOT_ACTIVE"

    local = _activate(client, headers, pending_local["record_id"])
    stages = [local]
    for stage in ("editorial_preview", "limited_team_pilot", "daily_use_release", "public_evidence_release"):
        staged_payload = (
            _pilot_entry_baseline(stages[-1]["record_id"])
            if stage == "limited_team_pilot"
            else _record(stage, stages[-1]["record_id"])
        )
        if stage in {"daily_use_release", "public_evidence_release"}:
            _bind_pilot_entry_baseline_scope(staged_payload)
        if stage == "daily_use_release":
            staged_payload["baseline_record_identity"] = stages[-1]["record_id"]
        if stage == "public_evidence_release":
            staged_payload["affected_scope"]["public_claim_identities"] = ["public_claim:pilot-summary-v1"]
        staged = _create(client, headers, staged_payload)
        assert staged["stage"] == stage
        stages.append(_activate(client, headers, staged["record_id"]))

    altered_scope = _record("editorial_preview", local["record_id"])
    altered_scope["affected_scope"]["configuration_identities"] = ["configuration:changed-profile-20260905"]
    rejected_scope = client.post("/api/v1/acceptance/records", headers=headers, json=altered_scope)
    assert rejected_scope.status_code == 409
    assert rejected_scope.json()["code"] == "ACCEPTANCE_PREDECESSOR_SCOPE_MISMATCH"

    public_claim_failure = _record("public_evidence_release", stages[-2]["record_id"])
    _bind_pilot_entry_baseline_scope(public_claim_failure)
    public_claim_failure["affected_scope"]["public_claim_identities"] = ["public_claim:pilot-summary-v1"]
    _check(public_claim_failure, "check:public-claim").update(
        {
            "result": "failed",
            "reason": "the claimed public scope has no approved evidence",
            "failure_kind": "public_claim",
            "blocking_scope": {
                "scope": "public_claim",
                "identity": "public_claim:pilot-summary-v1",
            },
            "evidence_links": ["evidence://public-evidence-release/claim/failure"],
        }
    )
    assert _create(client, headers, public_claim_failure)["current_status"] == "suspended"


def test_failed_checks_record_their_own_minimum_blocking_scope_and_cannot_be_reactivated(client: TestClient) -> None:
    headers = _administrator_headers(client)
    entry_failure = _record()
    _check(entry_failure, "check:entry-supported-query").update(
        {
            "result": "failed",
            "reason": "one entry no longer has sufficient evidence",
            "failure_kind": "entry_specific",
            "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
            "evidence_links": ["evidence://editorial-preview/entry-001/failure"],
        }
    )
    entry = _create(client, headers, entry_failure)
    assert entry["current_status"] == "at_risk"
    assert entry["blockers"][-1] == {
        "check_id": "check:entry-supported-query",
        "result": "failed",
        "reason": "one entry no longer has sufficient evidence",
        "failure_kind": "entry_specific",
        "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
    }
    assert client.post(
        f"/api/v1/acceptance/records/{entry['record_id']}/status",
        headers=headers,
        json={
            "status": "active",
            "reason_code": "checks_verified",
            "verified_checks": [
                {"check_id": check["check_id"], "evidence_links": check["evidence_links"]}
                for check in entry["checks"]
                if check["result"] in {"passed", "carried_forward"}
            ],
        },
    ).status_code == 409

    shared_failure = _record()
    shared_failure["affected_scope"]["public_claim_identities"] = ["public_claim:pilot-summary-v1"]
    shared_failure["checks"].append(
        {
            "check_id": "check:authorization",
            "result": "failed",
            "reason": "authorization invariant failed",
            "failure_kind": "shared_authorization",
            "blocking_scope": {
                "scope": "deployment",
                "identity": "deployment:editorial-preview-20260905",
            },
            "evidence_links": ["evidence://editorial-preview/authorization/failure"],
        }
    )
    shared = _create(client, headers, shared_failure)
    assert shared["current_status"] == "suspended"
    assert shared["accepted_scope"]["entry_identities"] == []
    assert shared["accepted_scope"]["public_claim_identities"] == []

    collection_failure = _record()
    _check(collection_failure, "check:collection-retrieval").update(
        {
            "result": "failed",
            "reason": "collection retrieval missed a required result",
            "failure_kind": "collection_retrieval",
            "blocking_scope": {
                "scope": "collection",
                "identity": "collection:production-rag-agent-engineering",
            },
            "evidence_links": ["evidence://editorial-preview/collection/retrieval-failure"],
        }
    )
    collection = _create(client, headers, collection_failure)
    assert collection["current_status"] == "at_risk"
    assert collection["blockers"][-1]["blocking_scope"]["scope"] == "collection"
    assert collection["accepted_scope"]["collection_identities"] == []

    performance_failure = _record()
    performance_failure["checks"].append(
        {
            "check_id": "check:performance-objective",
            "result": "failed",
            "reason": "p95 exceeds the declared objective",
            "failure_kind": "performance",
            "performance_objective_identity": "objective:answer-p95",
            "blocking_scope": {
                "scope": "deployment",
                "identity": "deployment:editorial-preview-20260905",
            },
            "evidence_links": ["evidence://editorial-preview/performance/failure"],
        }
    )
    performance = _create(client, headers, performance_failure)
    assert performance["current_status"] == "at_risk"
    assert performance["accepted_scope"]["entry_identities"] == []


def test_carry_forward_requires_the_full_affected_scope_and_matching_check_dependencies(client: TestClient) -> None:
    headers = _administrator_headers(client)
    source_payload = _record()
    source_payload["conditions"]["concurrency"] = "1"
    _check(source_payload, "check:entry-supported-query").update(
        {
            "applicability_conditions": {"language": "zh-cn"},
            "assumptions": ["one reviewed entry"],
            "identity_dependencies": [
                "entry:decision-entry-001",
                "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9",
            ],
        }
    )
    source = _activate(client, headers, _create(client, headers, source_payload)["record_id"])

    successor_payload = _record("editorial_preview", source["record_id"])
    successor_payload["conditions"]["concurrency"] = "5"
    successor_payload["assumptions"].append("five concurrent answer executions")
    _check(successor_payload, "check:entry-supported-query").update(
        {
            "result": "carried_forward",
            "carried_forward_from": source["record_id"],
            "evidence_links": ["evidence://editorial-preview/entry-001/supported-query"],
            "applicability_conditions": {"language": "zh-cn"},
            "assumptions": ["one reviewed entry"],
            "identity_dependencies": [
                "entry:decision-entry-001",
                "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9",
            ],
        }
    )
    successor_payload["checks"].append(
        {
            "check_id": "check:concurrency-objective",
            "result": "required",
            "evidence_links": ["evidence://pilot/concurrency-objective"],
            "applicability_conditions": {"concurrency": "5"},
            "assumptions": ["five concurrent answer executions"],
        }
    )
    successor = _create(client, headers, successor_payload)
    assert successor["current_status"] == "at_risk"
    assert _check(successor, "check:entry-supported-query")["result"] == "carried_forward"
    assert _check(successor, "check:concurrency-objective")["result"] == "required"

    changed_identity = deepcopy(successor_payload)
    changed_identity["affected_scope"]["protected_capability_identities"] = ["capability:identity-v2"]
    changed_identity["checks"].append(
        {
            "check_id": "check:protected-capability-activation",
            "result": "passed",
            "evidence_links": ["evidence://editorial-preview/capability/identity-v2"],
        }
    )
    assert client.post("/api/v1/acceptance/records", headers=headers, json=changed_identity).status_code == 409

    stale_check = deepcopy(successor_payload)
    stale_check["conditions"]["language"] = "en"
    _check(stale_check, "check:entry-supported-query")["applicability_conditions"] = {"language": "en"}
    stale_result = client.post("/api/v1/acceptance/records", headers=headers, json=stale_check)
    assert stale_result.status_code == 409
    assert stale_result.json()["code"] == "ACCEPTANCE_CARRY_FORWARD_INVALID"


def test_status_history_requires_existing_successors_and_records_remain_immutable_and_admin_only(client: TestClient) -> None:
    administrator_headers = _administrator_headers(client)
    original = _active_local_record(client, administrator_headers)
    successor_payload = _record()
    successor_payload["replaces_record_identity"] = original["record_id"]
    successor_payload["product_identities"] = [
        "product_revision:8fe565873608c7dcb3f824355dacf77bb1b277c9",
        "configuration:retrieval-profile-20260906",
    ]
    successor_payload["affected_scope"]["configuration_identities"] = ["configuration:retrieval-profile-20260906"]
    successor = _activate(
        client,
        administrator_headers,
        _create(client, administrator_headers, successor_payload)["record_id"],
    )

    missing_successor = client.post(
        f"/api/v1/acceptance/records/{original['record_id']}/status",
        headers=administrator_headers,
        json={
            "status": "superseded",
            "reason_code": "superseded_by_record",
            "superseding_record_identity": "delivery_acceptance_record:missing-successor-001",
        },
    )
    assert missing_successor.status_code == 404

    assert client.post(
        f"/api/v1/acceptance/records/{original['record_id']}/status",
        headers=administrator_headers,
        json={
            "status": "at_risk",
            "reason_code": "reacceptance_due",
            "reacceptance_trigger": "entry material revision",
            "invalidated_check_ids": ["check:entry-supported-query"],
        },
    ).status_code == 200
    assert client.post(
        f"/api/v1/acceptance/records/{original['record_id']}/status",
        headers=administrator_headers,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "fresh evidence invalidated the entry",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
                "evidence_links": ["evidence://editorial-preview/entry-001/runtime-failure"],
            },
        },
    ).status_code == 200
    superseded = client.post(
        f"/api/v1/acceptance/records/{original['record_id']}/status",
        headers=administrator_headers,
        json={
            "status": "superseded",
            "reason_code": "superseded_by_record",
            "superseding_record_identity": successor["record_id"],
        },
    )
    assert superseded.status_code == 200
    supersession_event = superseded.json()["data"]["status_history"][-1]
    assert supersession_event["status"] == "superseded"
    assert supersession_event["reason_code"] == "superseded_by_record"
    assert supersession_event["event_type"] == "superseded"
    assert supersession_event["superseding_record_identity"] == successor["record_id"]
    assert supersession_event["recorded_by"] == original["evaluator_identity"]

    async def reject_record_mutation() -> None:
        async with client.app.state.settings_session_factory() as session:
            persisted = await session.scalar(
                select(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == original["record_id"])
            )
            assert persisted is not None
            persisted.payload = {"rewritten": True}
            with pytest.raises(ValueError, match="immutable"):
                await session.flush()

    asyncio.run(reject_record_mutation())

    invitation = client.post("/api/v1/members/invitations", headers=administrator_headers, json={})
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "username": "knowledge-user",
            "password": "safe-password",
            "invitation_code": invitation.json()["data"]["invitation_code"],
        },
    )
    knowledge_user_headers = {"Authorization": f"Bearer {registration.json()['data']['access_token']}"}
    assert client.get(
        f"/api/v1/acceptance/records/{original['record_id']}",
        headers=knowledge_user_headers,
    ).status_code == 403

    fake_authority = _record()
    fake_authority["change_owner_identity"] = "member:forged-owner-001"
    assert client.post("/api/v1/acceptance/records", headers=administrator_headers, json=fake_authority).status_code == 422
    fake_editorial = _record()
    fake_editorial["editorial_review_approval"] = True
    assert client.post("/api/v1/acceptance/records", headers=administrator_headers, json=fake_editorial).status_code == 422


def test_exact_identities_closed_status_reasons_and_all_check_results_are_retained(client: TestClient) -> None:
    headers = _administrator_headers(client)
    branch_payload = _record()
    branch_payload["product_identities"] = ["main"]
    assert client.post("/api/v1/acceptance/records", headers=headers, json=branch_payload).status_code == 422
    latest_payload = _record()
    latest_payload["product_identities"] = ["product_revision:latest"]
    assert client.post("/api/v1/acceptance/records", headers=headers, json=latest_payload).status_code == 422

    payload = _record()
    payload["checks"].extend(
        [
        {
            "check_id": "check:pilot-workload",
            "result": "required",
            "evidence_links": ["evidence://pilot/workload"],
        },
        {
            "check_id": "check:public-evidence-export",
            "result": "skipped",
            "reason": "public evidence is not in the Local Development scope",
            "evidence_links": [],
        },
        {
            "check_id": "check:daily-use-adoption",
            "result": "not_applicable",
            "reason": "Daily-Use Release has not been requested",
            "evidence_links": [],
        },
        ]
    )
    record = _create(client, headers, payload)
    assert _check(record, "check:entry-supported-query")["result"] == "passed"
    assert _check(record, "check:pilot-workload")["result"] == "required"
    assert _check(record, "check:public-evidence-export")["result"] == "skipped"
    assert _check(record, "check:daily-use-adoption")["result"] == "not_applicable"
    assert record["blockers"][-1] == {"check_id": "check:pilot-workload", "result": "required"}
    assert client.post(
        f"/api/v1/acceptance/records/{record['record_id']}/status",
        headers=headers,
        json={
            "status": "active",
            "reason_code": "integrity_failure",
            "verified_checks": [
                {"check_id": check["check_id"], "evidence_links": check["evidence_links"]}
                for check in record["checks"]
                if check["result"] in {"passed", "carried_forward"}
            ],
        },
    ).status_code == 422


def test_activation_rechecks_ancestors_and_fails_closed_for_malformed_persisted_checks(client: TestClient) -> None:
    headers = _administrator_headers(client)
    local = _active_local_record(client, headers)
    editorial = _create(client, headers, _record("editorial_preview", local["record_id"]))
    assert client.post(
        f"/api/v1/acceptance/records/{local['record_id']}/status",
        headers=headers,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "entry evidence was withdrawn",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
                "evidence_links": ["evidence://editorial-preview/entry-001/withdrawn"],
            },
        },
    ).status_code == 200
    assert client.post(
        f"/api/v1/acceptance/records/{editorial['record_id']}/status",
        headers=headers,
        json={
            "status": "active",
            "reason_code": "checks_verified",
            "verified_checks": [
                {"check_id": check["check_id"], "evidence_links": check["evidence_links"]}
                for check in editorial["checks"]
                if check["result"] in {"passed", "carried_forward"}
            ],
        },
    ).status_code == 409

    malformed = _create(client, headers, _record())

    async def corrupt_checks() -> None:
        async with client.app.state.settings_session_factory() as session:
            with pytest.raises(ValueError, match="cannot be updated or deleted"):
                await session.execute(
                    update(CanonicalRecordModel)
                    .where(CanonicalRecordModel.stable_id == malformed["record_id"])
                    .values(payload={"schema": "delivery_acceptance_record/v1", "checks": "invalid"})
                )
            await session.execute(
                text("UPDATE canonical_records SET payload = :payload WHERE stable_id = :record_id"),
                {
                    "payload": '{"schema":"delivery_acceptance_record/v1","checks":"invalid"}',
                    "record_id": malformed["record_id"],
                },
            )
            await session.commit()

    asyncio.run(corrupt_checks())
    rejected = client.post(
        f"/api/v1/acceptance/records/{malformed['record_id']}/status",
        headers=headers,
        json={
            "status": "active",
            "reason_code": "checks_verified",
            "verified_checks": [
                {
                    "check_id": "check:impact-declaration",
                    "evidence_links": ["evidence://editorial-preview/entry-001/impact"],
                }
            ],
        },
    )
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "ACCEPTANCE_RECORD_INVALID"


def test_entry_failure_cannot_cover_unrelated_entries_and_carry_forward_declares_dependencies(client: TestClient) -> None:
    headers = _administrator_headers(client)
    overbroad = _record()
    overbroad["affected_scope"]["entry_identities"].append("entry:unrelated-entry-002")
    overbroad["content_identities"].append("entry:unrelated-entry-002")
    _bind_entry_evidence(overbroad, "entry:unrelated-entry-002")
    _check(overbroad, "check:entry-supported-query").update(
        {
            "result": "failed",
            "reason": "one entry no longer has sufficient evidence",
            "failure_kind": "entry_specific",
            "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
            "evidence_links": ["evidence://editorial-preview/entry-001/failure"],
        }
    )
    assert client.post("/api/v1/acceptance/records", headers=headers, json=overbroad).status_code == 200

    undeclared_dependencies = _record()
    _check(undeclared_dependencies, "check:entry-supported-query").update(
        {
            "result": "carried_forward",
            "carried_forward_from": "delivery_acceptance_record:source-record-001",
            "evidence_links": ["evidence://editorial-preview/entry-001/supported-query"],
        }
    )
    assert client.post(
        "/api/v1/acceptance/records",
        headers=headers,
        json=undeclared_dependencies,
    ).status_code == 422


def test_acceptance_requires_evidence_and_rejects_mutable_identity_aliases(client: TestClient) -> None:
    headers = _administrator_headers(client)
    without_evidence = _record()
    without_evidence["checks"][0]["evidence_links"] = []
    assert client.post("/api/v1/acceptance/records", headers=headers, json=without_evidence).status_code == 422

    alias = _record()
    alias["product_identities"] = ["git_branch:ticket-15"]
    assert client.post("/api/v1/acceptance/records", headers=headers, json=alias).status_code == 422

    secret = _record()
    secret["risks"] = ["token=acceptance-secret"]
    assert client.post("/api/v1/acceptance/records", headers=headers, json=secret).status_code == 422


def test_failure_scope_mapping_and_multi_entry_projection_are_minimum_sufficient(client: TestClient) -> None:
    headers = _administrator_headers(client)
    collection_failure = _record()
    collection_failure["checks"].append(
        {
            "check_id": "check:authorization",
            "result": "failed",
            "reason": "authorization invariant failed",
            "failure_kind": "shared_authorization",
            "blocking_scope": {
                "scope": "collection",
                "identity": "collection:production-rag-agent-engineering",
            },
            "evidence_links": ["evidence://editorial-preview/authorization/failure"],
        }
    )
    shared = _create(client, headers, collection_failure)
    assert shared["current_status"] == "suspended"
    assert shared["blockers"][-1]["blocking_scope"] == {
        "scope": "collection",
        "identity": "collection:production-rag-agent-engineering",
    }

    multi_entry = _record()
    multi_entry["affected_scope"]["entry_identities"].append("entry:unrelated-entry-002")
    multi_entry["content_identities"].append("entry:unrelated-entry-002")
    _bind_entry_evidence(multi_entry, "entry:unrelated-entry-002")
    _check(multi_entry, "check:entry-supported-query").update(
        {
            "result": "failed",
            "reason": "one entry no longer has sufficient evidence",
            "failure_kind": "entry_specific",
            "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
            "evidence_links": ["evidence://editorial-preview/entry-001/failure"],
        }
    )
    record = _create(client, headers, multi_entry)
    assert record["current_status"] == "at_risk"
    assert record["accepted_scope"]["entry_identities"] == []


def test_carry_forward_requires_ancestor_source_and_preserves_evidence_lineage(client: TestClient) -> None:
    headers = _administrator_headers(client)
    source_payload = _record()
    _check(source_payload, "check:entry-supported-query").update(
        {
            "applicability_conditions": {"language": "zh-cn"},
            "assumptions": ["one reviewed entry"],
            "identity_dependencies": [
                "entry:decision-entry-001",
                "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9",
            ],
        }
    )
    source = _create(client, headers, source_payload)
    source = _activate(client, headers, source["record_id"])

    successor = _record("editorial_preview", source["record_id"])
    _check(successor, "check:entry-supported-query").update(
        {
            "result": "carried_forward",
            "carried_forward_from": source["record_id"],
            "evidence_links": ["evidence://different/evidence"],
            "applicability_conditions": {"language": "zh-cn"},
            "assumptions": ["one reviewed entry"],
            "identity_dependencies": [
                "entry:decision-entry-001",
                "product_revision:3ec565873608c7dcb3f824355dacf77bb1b277c9",
            ],
        }
    )
    rejected = client.post("/api/v1/acceptance/records", headers=headers, json=successor)
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "ACCEPTANCE_CARRY_FORWARD_INVALID"


def test_status_events_bind_new_failures_and_auditable_event_metadata(client: TestClient) -> None:
    headers = _administrator_headers(client)
    record = _active_local_record(client, headers)
    suspended = client.post(
        f"/api/v1/acceptance/records/{record['record_id']}/status",
        headers=headers,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "fresh evidence invalidated the entry",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
                "evidence_links": ["evidence://editorial-preview/entry-001/runtime-failure"],
            },
        },
    )
    assert suspended.status_code == 200
    history = suspended.json()["data"]["status_history"][-1]
    assert history["event_id"]
    assert history["occurred_at"].endswith("+00:00")
    assert history["from_status"] == "active"
    assert history["status_failure"]["blocking_scope"]["identity"] == "entry:decision-entry-001"
    assert suspended.json()["data"]["blockers"][-1]["check_id"] == "check:entry-supported-query"


def test_replacement_and_stage_paths_are_explicit_and_ordered(client: TestClient) -> None:
    headers = _administrator_headers(client)
    local = _active_local_record(client, headers)
    editorial = _activate(
        client,
        headers,
        _create(client, headers, _record("editorial_preview", local["record_id"]))["record_id"],
    )
    pilot = _activate(
        client,
        headers,
        _create(client, headers, _pilot_entry_baseline(editorial["record_id"]))["record_id"],
    )

    direct_daily = _bind_pilot_entry_baseline_scope(_record("daily_use_release", editorial["record_id"]))
    direct_daily["change_classification"] = "ordinary_content"
    direct_daily["baseline_record_identity"] = pilot["record_id"]
    assert _create(client, headers, direct_daily)["stage"] == "daily_use_release"

    protected_daily = deepcopy(direct_daily)
    protected_daily["change_classification"] = "protected_product_path"
    protected_daily["affected_scope"]["protected_capability_identities"] = ["capability:retrieval-v2"]
    protected_daily["checks"].append(
        {
            "check_id": "check:protected-capability-activation",
            "result": "passed",
            "evidence_links": ["evidence://daily-use-release/capability-activation"],
        }
    )
    assert client.post("/api/v1/acceptance/records", headers=headers, json=protected_daily).status_code == 409

    replacement = _record()
    replacement["replaces_record_identity"] = local["record_id"]
    replacement["product_identities"] = [
        "product_revision:8fe565873608c7dcb3f824355dacf77bb1b277c9",
        "configuration:retrieval-profile-20260906",
    ]
    replacement["affected_scope"]["configuration_identities"] = ["configuration:retrieval-profile-20260906"]
    replacement = _activate(client, headers, _create(client, headers, replacement)["record_id"])

    superseded = client.post(
        f"/api/v1/acceptance/records/{local['record_id']}/status",
        headers=headers,
        json={
            "status": "superseded",
            "reason_code": "superseded_by_record",
            "superseding_record_identity": replacement["record_id"],
        },
    )
    assert superseded.status_code == 200


def test_invalid_status_transition_is_a_controlled_conflict(client: TestClient) -> None:
    headers = _administrator_headers(client)
    record = _active_local_record(client, headers)
    rejected = client.post(
        f"/api/v1/acceptance/records/{record['record_id']}/status",
        headers=headers,
        json={
            "status": "active",
            "reason_code": "checks_verified",
            "verified_checks": [
                {"check_id": check["check_id"], "evidence_links": check["evidence_links"]}
                for check in record["checks"]
                if check["result"] in {"passed", "carried_forward"}
            ],
        },
    )
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "ACCEPTANCE_STATUS_TRANSITION_INVALID"


def test_mandatory_stage_checks_and_semantic_identity_fields_cannot_be_bypassed(client: TestClient) -> None:
    headers = _administrator_headers(client)
    local = _active_local_record(client, headers)
    pilot = _record("limited_team_pilot", _activate(
        client,
        headers,
        _create(client, headers, _record("editorial_preview", local["record_id"]))["record_id"],
    )["record_id"])
    _check(pilot, "check:authorization").update(
        {"result": "skipped", "reason": "must not bypass the Pilot authority check"}
    )
    assert client.post("/api/v1/acceptance/records", headers=headers, json=pilot).status_code == 422

    swapped_scope = _record()
    swapped_scope["affected_scope"]["entry_identities"] = ["deployment:not-an-entry"]
    assert client.post("/api/v1/acceptance/records", headers=headers, json=swapped_scope).status_code == 422

    mutable_deployment = _record()
    mutable_deployment["affected_scope"]["deployment_identity"] = "deployment:production"
    assert client.post("/api/v1/acceptance/records", headers=headers, json=mutable_deployment).status_code == 422


def test_accepted_scope_requires_approval_and_performance_never_suspends(client: TestClient) -> None:
    headers = _administrator_headers(client)
    pending = _create(client, headers, _record())
    assert pending["accepted_scope"]["entry_identities"] == []

    active = _active_local_record(client, headers)
    performance = client.post(
        f"/api/v1/acceptance/records/{active['record_id']}/status",
        headers=headers,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "p95 exceeds the declared objective",
                "failure_kind": "performance",
                "performance_objective_identity": "objective:answer-p95",
                "blocking_scope": {
                    "scope": "deployment",
                    "identity": "deployment:editorial-preview-20260905",
                },
                "evidence_links": ["evidence://editorial-preview/performance/runtime-failure"],
            },
        },
    )
    assert performance.status_code == 422

    at_risk = client.post(
        f"/api/v1/acceptance/records/{active['record_id']}/status",
        headers=headers,
        json={
            "status": "at_risk",
            "reason_code": "performance_objective_missed",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "p95 exceeds the declared objective",
                "failure_kind": "performance",
                "performance_objective_identity": "objective:answer-p95",
                "blocking_scope": {
                    "scope": "deployment",
                    "identity": "deployment:editorial-preview-20260905",
                },
                "evidence_links": ["evidence://editorial-preview/performance/runtime-failure"],
            },
        },
    )
    assert at_risk.status_code == 200
    assert at_risk.json()["data"]["accepted_scope"]["entry_identities"] == ["entry:decision-entry-001"]


def test_replacement_cannot_lower_an_accepted_stage(client: TestClient) -> None:
    headers = _administrator_headers(client)
    local = _active_local_record(client, headers)
    editorial = _activate(
        client,
        headers,
        _create(client, headers, _record("editorial_preview", local["record_id"]))["record_id"],
    )
    pilot = _activate(
        client,
        headers,
        _create(client, headers, _pilot_entry_baseline(editorial["record_id"]))["record_id"],
    )
    daily_payload = _bind_pilot_entry_baseline_scope(_record("daily_use_release", pilot["record_id"]))
    daily_payload["baseline_record_identity"] = pilot["record_id"]
    daily = _activate(
        client,
        headers,
        _create(
            client,
            headers,
            daily_payload,
        )["record_id"],
    )
    lower_replacement = _record()
    lower_replacement["replaces_record_identity"] = daily["record_id"]
    lower = _activate(client, headers, _create(client, headers, lower_replacement)["record_id"])
    rejected = client.post(
        f"/api/v1/acceptance/records/{daily['record_id']}/status",
        headers=headers,
        json={
            "status": "superseded",
            "reason_code": "superseded_by_record",
            "superseding_record_identity": lower["record_id"],
        },
    )
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "ACCEPTANCE_SUCCESSOR_STAGE_INVALID"


def test_acceptance_requires_exact_environment_boundaries_dependency_checks_and_active_pilot_baseline(
    client: TestClient,
) -> None:
    headers = _administrator_headers(client)

    missing_deployment = _record()
    missing_deployment["affected_scope"]["deployment_identity"] = None
    assert client.post("/api/v1/acceptance/records", headers=headers, json=missing_deployment).status_code == 422

    missing_dependency_check = _record()
    missing_dependency_check["checks"] = [
        check
        for check in missing_dependency_check["checks"]
        if check["check_id"] != "check:configuration-impact"
    ]
    assert client.post(
        "/api/v1/acceptance/records",
        headers=headers,
        json=missing_dependency_check,
    ).status_code == 422

    complete_boundary = _record()
    complete_boundary["product_identities"].extend(
        [
            "user_boundary:internal-pilot-team-v1",
            "data_boundary:non-sensitive-internal-v1",
            "concurrency:two-active-two-queued-v1",
        ]
    )
    complete_boundary["checks"].extend(
        {
            "check_id": check_id,
            "result": "passed",
            "evidence_links": [f"evidence://editorial-preview/{check_id.removeprefix('check:')}"],
        }
        for check_id in (
            "check:deployment-boundary",
            "check:workload-profile",
        )
    )
    assert client.post("/api/v1/acceptance/records", headers=headers, json=complete_boundary).status_code == 200

    local = _active_local_record(client, headers)
    editorial = _activate(
        client,
        headers,
        _create(client, headers, _record("editorial_preview", local["record_id"]))["record_id"],
    )
    non_baseline_pilot = _activate(
        client,
        headers,
        _create(client, headers, _record("limited_team_pilot", editorial["record_id"]))["record_id"],
    )

    daily_without_baseline = _record("daily_use_release", editorial["record_id"])
    assert client.post(
        "/api/v1/acceptance/records",
        headers=headers,
        json=daily_without_baseline,
    ).status_code == 422
    daily_with_non_baseline = _record("daily_use_release", editorial["record_id"])
    daily_with_non_baseline["baseline_record_identity"] = non_baseline_pilot["record_id"]
    rejected_baseline = client.post(
        "/api/v1/acceptance/records",
        headers=headers,
        json=daily_with_non_baseline,
    )
    assert rejected_baseline.status_code == 409
    assert rejected_baseline.json()["code"] == "ACCEPTANCE_BASELINE_KIND_INVALID"

    baseline = _activate(client, headers, _create(client, headers, _pilot_entry_baseline(editorial["record_id"]))["record_id"])
    daily_with_baseline = _bind_pilot_entry_baseline_scope(_record("daily_use_release", editorial["record_id"]))
    daily_with_baseline["baseline_record_identity"] = baseline["record_id"]
    assert _create(client, headers, daily_with_baseline)["stage"] == "daily_use_release"


def test_entry_failure_preserves_an_already_accepted_unrelated_entry(client: TestClient) -> None:
    headers = _administrator_headers(client)
    under_evidenced = _record()
    under_evidenced["affected_scope"]["entry_identities"].append("entry:unrelated-entry-002")
    under_evidenced["content_identities"].append("entry:unrelated-entry-002")
    assert client.post("/api/v1/acceptance/records", headers=headers, json=under_evidenced).status_code == 422

    multi_entry = deepcopy(under_evidenced)
    _bind_entry_evidence(multi_entry, "entry:unrelated-entry-002")
    active = _activate(client, headers, _create(client, headers, multi_entry)["record_id"])

    suspended = client.post(
        f"/api/v1/acceptance/records/{active['record_id']}/status",
        headers=headers,
        json={
            "status": "suspended",
            "reason_code": "integrity_failure",
            "status_failure": {
                "check_id": "check:entry-supported-query",
                "reason": "one entry no longer has sufficient evidence",
                "failure_kind": "entry_specific",
                "blocking_scope": {"scope": "entry_version", "identity": "entry:decision-entry-001"},
                "evidence_links": ["evidence://editorial-preview/entry-001/runtime-failure"],
            },
        },
    )
    assert suspended.status_code == 200
    assert suspended.json()["data"]["accepted_scope"]["entry_identities"] == ["entry:unrelated-entry-002"]
