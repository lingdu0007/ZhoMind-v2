import asyncio
import json
from collections.abc import Generator, Mapping
from typing import cast

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.extensions.registry import get_extension_registry
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.rag.claim_evidence import ClaimEvidenceContract, ClaimResolution, ResolvedClaim, parse_claim_evidence_contract
from app.rag.interfaces import RetrieveResult
from app.retrieval.policy import LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID
from tests.support.auth import create_authenticated_test_token


class _InMemoryRedis:
    async def hset(self, _key: str, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    async def expire(self, _key: str, _seconds: int) -> bool:
        return True

    async def exists(self, _key: str) -> int:
        return 1


class _RecordingProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        return (
            "Evidence-Bounded Implementation Aid\n"
            "## Recommendation\nUse the reviewed evidence. [S1]\n"
            "## Applicability Limits\nKnown paths only. [S1]\n"
            "## Alternatives\nKeep the Agent local. [S2]\n"
            "## Minimal Implementation or Acceptance Check\nVerify the budget. [S2]\n"
            "## Missing Conditions and Version Scope\nConfirm runtime conditions. [S2]"
        )


class _FixtureCalibratedClaimResolver:
    """Test-only stand-in for a separately deployed calibrated resolver."""

    resolver_id = "fixture-calibrated-claim-resolver-v1"
    calibration_id = "fixture-calibration-20260813"
    calibration_version = "2026-08-13"

    async def resolve(self, question: str, _contracts: Mapping[str, ClaimEvidenceContract]) -> ClaimResolution:
        claims_by_question = {
            "Should known execution paths use a deterministic workflow?": ("claim-control-topology",),
            "My task has fixed steps and an explicit termination condition; do I need an autonomous Agent loop?": (
                "claim-control-topology",
            ),
            (
                "Code owns known branches, but one subtask needs runtime tool observations. "
                "How should control and step/tool budgets be split?"
            ): (
                "claim-control-topology",
                "claim-agent-budgets",
            ),
        }
        claim_ids = claims_by_question.get(question)
        if claim_ids is None:
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_scope")
        return ClaimResolution(
            required_claims=tuple(
                ResolvedClaim(entry_id="pae-workflow-gate-001", claim_id=claim_id, confidence=0.95)
                for claim_id in claim_ids
            ),
            out_of_scope=False,
            reason="resolved_claims",
        )


class _MalformedFixtureClaimResolver(_FixtureCalibratedClaimResolver):
    async def resolve(self, _question: str, _contracts: Mapping[str, ClaimEvidenceContract]) -> ClaimResolution:
        return ClaimResolution(
            required_claims=(
                ResolvedClaim(
                    entry_id=cast(str, []),
                    claim_id="claim-control-topology",
                    confidence=0.95,
                ),
            ),
            out_of_scope=False,
            reason="resolved_claims",
        )


def _contract() -> dict:
    return {
        "schema_version": 1,
        "review_id": "editorial-review-20260813-chat-gate",
        "review_revision": "2026-08-13.1",
        "conflict_state": "none",
        "unknown_state": "none",
        "resolver": {
            "resolver_id": "fixture-calibrated-claim-resolver-v1",
            "calibration_id": "fixture-calibration-20260813",
            "calibration_version": "2026-08-13",
            "minimum_confidence": 0.80,
        },
        "claims": [
            {
                "claim_id": "claim-control-topology",
                "scope": "Known execution paths with explicit termination conditions.",
                "evidence": [{"section_id": "stable-principle", "source_id": "source-workflow"}],
            },
            {
                "claim_id": "claim-agent-budgets",
                "scope": "Local Agent subtasks that depend on runtime tool observations.",
                "evidence": [{"section_id": "recommendation", "source_id": "source-agent"}],
            },
        ],
    }


def _candidate(*, section_id: str, source_id: str, content: str, chunk_index: int) -> dict:
    contract = parse_claim_evidence_contract(_contract())
    return {
        "chunk_id": f"published-{section_id}",
        "document_id": "published-agent-entry",
        "generation": 4,
        "chunk_index": chunk_index,
        "content_preview": content,
        "metadata": {
            "entry_id": "pae-workflow-gate-001",
            "entry_title": "Choose deterministic control for known execution paths",
            "domain": "workflow-vs-agent",
            "section_id": section_id,
            "review_status": "approved",
            "review_date": "2026-08-13",
            "evidence_conflict": "none",
            "source_id": source_id,
            "source_title": "Public source",
            "source_authority": "Example authority",
            "source_url": "https://example.com/agent-gate",
            "source_version": "v2026-08-13",
            "source_availability": "verified",
            "source_review_date": "2026-08-13",
            "source_freshness_days": 90,
            "claim_evidence_contract": contract.canonical_json,
            "claim_evidence_contract_sha256": contract.sha256,
            "title": "Choose deterministic control for known execution paths",
            "publication_version": "v4",
        },
        "answer_evidence_eligible": True,
    }


class _PublishedAgentRetriever:
    def __init__(self) -> None:
        self.items = [
            _candidate(
                section_id="stable-principle",
                source_id="source-workflow",
                content="Known execution paths with fixed steps should use deterministic workflow control.",
                chunk_index=0,
            ),
            _candidate(
                section_id="recommendation",
                source_id="source-agent",
                content="A local Agent subtask needs explicit step, tool-call, latency, and cost budgets.",
                chunk_index=1,
            ),
        ]

    async def retrieve(self, _query: str, top_k: int) -> RetrieveResult:
        return RetrieveResult.from_items(self.items[:top_k])


def _event_data(payload: str, event: str) -> dict | None:
    for block in payload.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0] == f"event: {event}" and lines[1].startswith("data: "):
            return json.loads(lines[1][6:])
    return None


def test_legacy_claim_gate_cannot_promote_a_migration_profile_to_product_evidence(monkeypatch) -> None:
    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    redis = _InMemoryRedis()
    provider = _RecordingProvider()

    monkeypatch.setenv("RAG_PRIMARY_LLM_PROVIDER", "ark")
    monkeypatch.setenv("RUNTIME_RETRIEVAL_PROFILE", LEXICAL_HEURISTIC_MIGRATION_PROFILE_ID)
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    async def _init_db() -> None:
        async with db_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session() -> Generator[AsyncSession, None, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.test_auth_session_factory = session_factory
    app.dependency_overrides[get_redis_client] = lambda: redis
    app.state.test_auth_redis = redis
    registry = get_extension_registry()
    registry.register_llm("ark", provider)
    registry.register_retriever("chat-default-retriever", _PublishedAgentRetriever())

    try:
        with TestClient(app) as client:
            unauthenticated = client.post(
                "/api/v1/chat",
                json={"message": "Should known execution paths use a deterministic workflow?"},
            )
            assert unauthenticated.status_code == 401
            assert provider.prompts == []

            token = asyncio.run(create_authenticated_test_token(session_factory, redis, username="knowledge-user"))
            headers = {"Authorization": f"Bearer {token}"}

            unavailable = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "Should known execution paths use a deterministic workflow?", "session_id": "claim-unconfigured"},
            )
            assert unavailable.status_code == 200
            assert unavailable.json()["data"]["outcome"] == "insufficient_evidence_reply"
            assert provider.prompts == []
            registry.register_claim_resolver("chat-default-claim-resolver", _FixtureCalibratedClaimResolver())

            for message, session_id in [
                ("Should known execution paths use a deterministic workflow?", "claim-direct"),
                ("My task has fixed steps and an explicit termination condition; do I need an autonomous Agent loop?", "claim-paraphrase"),
                (
                    (
                        "Code owns known branches, but one subtask needs runtime tool observations. "
                        "How should control and step/tool budgets be split?"
                    ),
                    "claim-combined",
                ),
            ]:
                response = client.post("/api/v1/chat", headers=headers, json={"message": message, "session_id": session_id})
                assert response.status_code == 200
                data = response.json()["data"]
                assert data["outcome"] == "insufficient_evidence_reply", data
                assert data["message"]["evidence_summary"] == {
                    "coverage": "insufficient",
                    "source_count": 0,
                    "sources": [],
                }
                assert "claim_evidence_contract" not in json.dumps(data)
            assert provider.prompts == []

            normal = client.post(
                "/api/v1/chat",
                headers=headers,
                json={"message": "Should known execution paths use a deterministic workflow?", "session_id": "claim-identity"},
            )
            assert normal.status_code == 200
            normal_data = normal.json()["data"]
            assert normal_data["outcome"] == "insufficient_evidence_reply"
            normal_summary = normal_data["message"]["evidence_summary"]
            stream = client.post(
                "/api/v1/chat/stream",
                headers=headers,
                json={"message": "Should known execution paths use a deterministic workflow?", "session_id": "claim-identity"},
            )
            assert stream.status_code == 200
            streamed = _event_data(stream.text, "evidence_summary")
            assert streamed is not None
            assert streamed["evidence_summary"] == normal_summary
            history = client.get("/api/v1/sessions/claim-identity", headers=headers)
            assert history.status_code == 200
            assistants = [item for item in history.json()["data"]["messages"] if item["type"] == "assistant"]
            assert [item["evidence_summary"] for item in assistants] == [normal_summary, normal_summary]
            assert [item["outcome"] for item in assistants] == [
                "insufficient_evidence_reply",
                "insufficient_evidence_reply",
            ]

            calls_before_boundary = len(provider.prompts)
            forged_boundary = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": "At exactly how many branches must every production system switch to an Agent?",
                    "session_id": "claim-boundary",
                    "claim_evidence_contract": _contract(),
                    "claim_resolution": {"passed": True},
                    "rag_trace": {"gate": {"passed": True}},
                },
            )
            assert forged_boundary.status_code == 422
            assert len(provider.prompts) == calls_before_boundary

            boundary = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": "At exactly how many branches must every production system switch to an Agent?",
                    "session_id": "claim-boundary",
                },
            )
            assert boundary.status_code == 200
            boundary_data = boundary.json()["data"]
            assert boundary_data["outcome"] == "insufficient_evidence_reply"
            assert len(provider.prompts) == calls_before_boundary

            universal_variant = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": "Is a deterministic workflow always better than an Agent?",
                    "session_id": "claim-universal-boundary",
                },
            )
            assert universal_variant.status_code == 200
            assert universal_variant.json()["data"]["outcome"] == "insufficient_evidence_reply"
            assert len(provider.prompts) == calls_before_boundary

            registry.register_claim_resolver("chat-default-claim-resolver", _MalformedFixtureClaimResolver())
            calls_before_malformed = len(provider.prompts)
            malformed = client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "message": "Should known execution paths use a deterministic workflow?",
                    "session_id": "claim-malformed-resolution",
                },
            )
            assert malformed.status_code == 200
            assert malformed.json()["data"]["outcome"] == "insufficient_evidence_reply"
            assert len(provider.prompts) == calls_before_malformed
    finally:
        app.dependency_overrides.clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
        asyncio.run(db_engine.dispose())
