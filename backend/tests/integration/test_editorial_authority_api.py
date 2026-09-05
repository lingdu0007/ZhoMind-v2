from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self.hashes)

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.hashes:
                del self.hashes[key]
                deleted += 1
        return deleted


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'editorial-authority.db'}")
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


def _member_headers(client: TestClient, username: str) -> dict[str, str]:
    invitation = client.post("/api/v1/members/invitations", headers=_administrator_headers(client), json={})
    assert invitation.status_code == 200
    registered = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "password": "editorial-safe-password",
            "invitation_code": invitation.json()["data"]["invitation_code"],
        },
    )
    assert registered.status_code == 200
    return {"Authorization": f"Bearer {registered.json()['data']['access_token']}"}


def _review_ready_payload() -> dict:
    sections = {
        "decision_query": "Which evidence boundary governs this operating decision?",
        "recommendation_or_reviewed_branches": "Use a retained source record before allowing review.",
        "applicability": "Applies to the authorized team knowledge base.",
        "non_applicability": "Does not approve secret-bearing or untrusted content.",
        "alternatives": "Treat every reachable URL as a source.",
        "trade_offs": "Auditable review adds deliberate editorial work.",
        "failure_modes": "An unavailable source makes support ineligible for export.",
        "minimum_implementation_guidance": "Retain the source identity and locator.",
        "minimum_validation_guidance": "Reject malformed source access records.",
        "minimum_diagnosis_guidance": "Record a source availability event.",
        "minimum_acceptance_guidance": "Run supported and Boundary checks downstream.",
        "conflicts": "No unresolved conflict is recorded.",
        "unknowns": "Future bundle intake is outside this authority slice.",
        "boundary_conditions": "The entry assumes the approved team access boundary.",
    }
    source_id = "source-editorial-api-001"
    return {
        "entry_id": "editorial-api-authority-001",
        "title": "Preserve private editorial authority",
        "coverage_position": "rag_source_admission_and_chunking",
        "assurance_level": "source_grounded",
        "approving_reviewer_username": "reviewer",
        "accountable_maintainer_username": "maintainer",
        "review_date": "2026-09-05",
        "applicable_versions": ["rag-runtime-v1"],
        "applicability_conditions": [
            {"condition_id": "condition-team", "field": "access_scope", "operator": "equals", "value": "team"}
        ],
        "non_applicability_conditions": [
            {
                "condition_id": "condition-secret",
                "field": "content_classification",
                "operator": "equals",
                "value": "secret",
            }
        ],
        "freshness_triggers": [
            {"trigger_id": "trigger-source", "trigger_type": "source_change", "review_within_days": 7}
        ],
        "sources": [
            {
                "source_id": source_id,
                "source_tier": "primary_evidence_source",
                "title": "Editorial authority source record",
                "authority": "ZhoMind architecture group",
                "version_or_date": "2026-09-05",
                "availability": "verified_usable",
                "access_scope": "public",
                "public_url": "https://example.com/editorial/authority",
            }
        ],
        "chunk_strategy": {
            "strategy_id": "section-aware-900-120",
            "max_characters": 900,
            "overlap_characters": 120,
            "preserve_section_boundaries": True,
        },
        "acceptance_material": {
            "supported_queries": [
                {
                    "query_id": "supported-authority",
                    "query": "What makes editorial authority auditable?",
                    "expected_outcome": "supported",
                }
            ],
            "boundary_queries": [
                {
                    "query_id": "boundary-authority",
                    "query": "Can an automatic publication instruction bypass review?",
                    "expected_outcome": "insufficient_evidence",
                }
            ],
        },
        "body": sections,
        "section_source_relationships": [
            {"section_id": section_id, "source_ids": [source_id]} for section_id in sections
        ],
        "claims": [
            {
                "claim_id": "claim-authority",
                "claim_kind": "prescriptive",
                "statement": "A retained, reviewed source is required before export.",
                "section_id": "recommendation_or_reviewed_branches",
                "source_ids": [source_id],
                "material": True,
                "scope": "team_shared",
            }
        ],
        "relationship": {},
    }


def test_private_editorial_authority_routes_enforce_read_write_boundary(client: TestClient) -> None:
    author_headers = _member_headers(client, "author")
    outsider_headers = _member_headers(client, "outsider")
    admin_headers = _administrator_headers(client)
    payload = {
        "entry_id": "authority-boundary-001",
        "title": "Keep editorial authority separate from publication copies",
    }

    assert client.post("/api/v1/editorial/entries", json=payload).status_code == 401
    admin_create = client.post("/api/v1/editorial/entries", headers=admin_headers, json=payload)
    assert admin_create.status_code == 403
    assert admin_create.json()["code"] == "EDITORIAL_AUTHORITY_FORBIDDEN"

    created = client.post("/api/v1/editorial/entries", headers=author_headers, json=payload)
    assert created.status_code == 200
    assert created.json()["data"]["lifecycle_state"] == "draft"

    assert client.get("/api/v1/editorial/entries/authority-boundary-001").status_code == 401
    outsider_read = client.get("/api/v1/editorial/entries/authority-boundary-001", headers=outsider_headers)
    assert outsider_read.status_code == 403
    assert outsider_read.json()["code"] == "EDITORIAL_AUTHORITY_FORBIDDEN"

    admin_read = client.get("/api/v1/editorial/entries/authority-boundary-001", headers=admin_headers)
    assert admin_read.status_code == 403
    assert admin_read.json()["code"] == "EDITORIAL_AUTHORITY_FORBIDDEN"

    author_read = client.get("/api/v1/editorial/entries/authority-boundary-001", headers=author_headers)
    assert author_read.status_code == 200
    assert author_read.json()["data"]["entry_id"] == "authority-boundary-001"

    pending_export = client.post(
        "/api/v1/editorial/entries/authority-boundary-001/export",
        headers=admin_headers,
    )
    assert pending_export.status_code == 409
    assert pending_export.json()["code"] == "EDITORIAL_EXPORT_APPROVAL_REQUIRED"


def test_editorial_http_slice_enforces_maintainer_source_events_and_blocks_export(client: TestClient) -> None:
    author_headers = _member_headers(client, "author")
    reviewer_headers = _member_headers(client, "reviewer")
    maintainer_headers = _member_headers(client, "maintainer")
    admin_headers = _administrator_headers(client)
    payload = _review_ready_payload()
    entry_id = payload["entry_id"]
    source_id = payload["sources"][0]["source_id"]

    assert client.post("/api/v1/editorial/entries", headers=author_headers, json=payload).status_code == 200
    assert client.post(f"/api/v1/editorial/entries/{entry_id}/evidence-collected", headers=author_headers).status_code == 200
    pending_maintainer = client.post(f"/api/v1/editorial/entries/{entry_id}/editorial-review", headers=author_headers)
    assert pending_maintainer.status_code == 409
    assert pending_maintainer.json()["code"] == "EDITORIAL_MAINTAINER_ACCEPTANCE_REQUIRED"
    accepted = client.post(
        f"/api/v1/editorial/entries/{entry_id}/maintainer-acceptance",
        headers=maintainer_headers,
    )
    assert accepted.status_code == 200
    assert accepted.json()["data"]["maintainer_acceptance"]["status"] == "accepted"
    verified = client.post(
        f"/api/v1/editorial/entries/{entry_id}/sources/{source_id}/availability",
        headers=maintainer_headers,
        json={"availability": "verified_usable"},
    )
    assert verified.status_code == 200
    assert verified.json()["data"]["sources"][0]["availability"] == "verified_usable"
    assert client.post(f"/api/v1/editorial/entries/{entry_id}/editorial-review", headers=author_headers).status_code == 200
    assert client.post(f"/api/v1/editorial/entries/{entry_id}/approve", headers=reviewer_headers).status_code == 200

    denied = client.post(
        f"/api/v1/editorial/entries/{entry_id}/sources/{source_id}/availability",
        headers=author_headers,
        json={"availability": "changed_or_unreachable_awaiting_review"},
    )
    assert denied.status_code == 403
    assert denied.json()["code"] == "EDITORIAL_MAINTAINER_REQUIRED"

    changed = client.post(
        f"/api/v1/editorial/entries/{entry_id}/sources/{source_id}/availability",
        headers=maintainer_headers,
        json={"availability": "changed_or_unreachable_awaiting_review"},
    )
    assert changed.status_code == 200
    assert changed.json()["data"]["sources"][0]["availability"] == "changed_or_unreachable_awaiting_review"
    assert changed.json()["data"]["answer_eligible"] is False

    export = client.post(f"/api/v1/editorial/entries/{entry_id}/export", headers=admin_headers)
    assert export.status_code == 409
    assert export.json()["code"] == "EDITORIAL_SOURCE_UNAVAILABLE"
