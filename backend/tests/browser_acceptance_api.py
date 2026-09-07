from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime

import uvicorn
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.contracts.canonical import CanonicalEventType
from app.delivery_acceptance.schemas import CreateDeliveryAcceptanceRecordRequest, UpdateDeliveryAcceptanceStatusRequest
from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.documents import parsers
from app.editorial_authority.schemas import CreateEditorialEntryRequest
from app.editorial_authority.service import EditorialAuthorityService
from app.extensions.registry import get_extension_registry
from app.infra.db import SessionLocal, engine
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.model.chat import ChatMessage, ChatSession
from app.model.document import Document, DocumentChunk, DocumentJob
from app.model.system_settings import SystemSettingsState
from app.model.user import User
from app.rag.claim_evidence import ClaimEvidenceContract, ClaimResolution, ResolvedClaim, parse_claim_evidence_contract
from app.repository.user_repository import UserRepository
from app.service.member_admission_service import MemberAdmissionService
from app.settings import runtime as settings_runtime
from app.settings.generation_routes import GenerationRouteService
from app.settings.runtime import SystemSettingsRuntime
from app.settings.service import SystemSettingsDraftService
from tests.support.generation import generation_acceptance_payload

_BROWSER_AGENT_ENTRY_ID = "synthetic-workflow-001"
_BROWSER_AGENT_SOURCE_ID = "source-workflow"
_BROWSER_CLAIM_CONTRACT = parse_claim_evidence_contract(
    {
        "schema_version": 1,
        "review_id": "browser-editorial-review-20260812",
        "review_revision": "2026-08-12.1",
        "conflict_state": "none",
        "unknown_state": "none",
        "resolver": {
            "resolver_id": "browser-calibrated-claim-resolver-v1",
            "calibration_id": "browser-calibration-20260812",
            "calibration_version": "2026-08-12",
            "minimum_confidence": 0.80,
        },
        "claims": [
            {
                "claim_id": "claim-deterministic-workflow",
                "scope": "Known execution paths with explicit termination conditions.",
                "evidence": [
                    {
                        "section_id": "recommendation_or_reviewed_branches",
                        "source_id": _BROWSER_AGENT_SOURCE_ID,
                    }
                ],
            }
        ],
    }
)


class _InMemoryRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}
        self._values: dict[str, str] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self._hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return seconds > 0 and (key in self._hashes or key in self._values)

    async def exists(self, key: str) -> int:
        return int(key in self._hashes or key in self._values)

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def set(self, key: str, value: str) -> None:
        self._values[key] = value

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            deleted += int(key in self._values or key in self._hashes)
            self._values.pop(key, None)
            self._hashes.pop(key, None)
        return deleted

    async def scan_iter(self, *, match: str):
        prefix = match.removesuffix("*")
        for key in [*self._values, *self._hashes]:
            if key.startswith(prefix):
                yield key


class _DeterministicLlm:
    """Deterministic generation adapter for the disposable acceptance API.

    Environment-controlled behaviors keep browser acceptance journeys
    deterministic without mocking network traffic:
    - BROWSER_ACCEPTANCE_LLM_DELAY_MS: sleep before answering so in-flight
      streaming UI states stay observable.
    - BROWSER_ACCEPTANCE_FAIL_FIRST=1: raise on the first call so the
      fail-closed Generation Unavailable path is observable, then succeed on
      subsequent calls.
    """

    def __init__(self) -> None:
        self._delay_ms = int(os.getenv("BROWSER_ACCEPTANCE_LLM_DELAY_MS", "0") or "0")
        self._fail_first = os.getenv("BROWSER_ACCEPTANCE_FAIL_FIRST") == "1"
        self._calls = 0

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del system_prompt
        if prompt == "Connection validation. Reply with OK.":
            return "OK"
        self._calls += 1
        if self._fail_first and self._calls == 1:
            raise RuntimeError("browser acceptance first-call failure")
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000)
        envelope = json.loads(prompt)
        contract = envelope.get("response_contract")
        if isinstance(contract, dict):
            marker = f"[{contract['citation_markers'][0]}]"
            label = contract.get("required_label")
            heading = f"【{label}】\n\n" if label else ""
            return heading + "\n\n".join(
                f"## {section}\n已知路径应由 deterministic workflow 控制。{marker}"
                for section in contract["required_sections"]
            )
        return "部署前需要完成变更审批。"


class _BrowserClaimResolver:
    """Test-only calibrated resolver for the frozen Claim-Linked browser fixture."""

    resolver_id = "browser-calibrated-claim-resolver-v1"
    calibration_id = "browser-calibration-20260812"
    calibration_version = "2026-08-12"

    async def resolve(self, question: str, _contracts: Mapping[str, ClaimEvidenceContract]) -> ClaimResolution:
        if question != "什么时候使用 deterministic workflow？":
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_scope")
        return ClaimResolution(
            required_claims=(
                ResolvedClaim(
                    entry_id=_BROWSER_AGENT_ENTRY_ID,
                    claim_id="claim-deterministic-workflow",
                    confidence=0.95,
                ),
            ),
            out_of_scope=False,
            reason="resolved_claims",
        )


class _DeterministicSettingsRuntime(SystemSettingsRuntime):
    async def validate(self, *, settings: dict, provider_api_key: str | None) -> None:
        self._candidate_settings(settings=settings, provider_api_key=provider_api_key)

    async def apply(self, *, version: int, settings: dict, provider_api_key: str | None) -> None:
        await super().apply(version=version, settings=settings, provider_api_key=provider_api_key)


async def _create_schema() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def _seed_chat_history(session: AsyncSession) -> None:
    """Seed deterministic historical conversations used by browser journeys.

    - `lin` (a Knowledge User created at runtime) gets one session whose
      assistant messages carry sufficient / unavailable / insufficient evidence
      summaries, exercising historical read projections.
    - `operator` (the Bootstrap Administrator) gets one session with a partial
      RAG trace so unavailable retrieval-diagnostics fields stay observable.
    """
    now = datetime.now(UTC)
    session.add(ChatSession(id="session-evidence-history", user_id="lin", created_at=now, updated_at=now))
    session.add(
        ChatMessage(
            id="msg-history-user",
            session_id="session-evidence-history",
            user_id="lin",
            type="user",
            content="历史的部署问题",
            created_at=now,
        )
    )
    session.add(
        ChatMessage(
            id="msg-history-sufficient",
            session_id="session-evidence-history",
            user_id="lin",
            type="assistant",
            content="历史回答有可核对来源。",
            rag_trace={
                "outcome": "evidence_gated_answer",
                "gate": {"passed": True, "reason": "sufficient_evidence"},
                "evidence": [
                    {
                        "chunk_id": "chunk-history-7",
                        "metadata": {"filename": "deploy-runbook.md"},
                        "content_preview": "历史来源摘录。",
                    }
                ],
            },
            created_at=now,
        )
    )
    session.add(
        ChatMessage(
            id="msg-history-unavailable",
            session_id="session-evidence-history",
            user_id="lin",
            type="assistant",
            content="历史回答没有可用来源。",
            rag_trace=None,
            created_at=now,
        )
    )
    session.add(
        ChatMessage(
            id="msg-history-insufficient",
            session_id="session-evidence-history",
            user_id="lin",
            type="assistant",
            content="历史回答证据不足。",
            rag_trace={
                "outcome": "insufficient_evidence_reply",
                "gate": {"passed": False, "reason": "reject_insufficient_evidence"},
                "evidence": [],
            },
            created_at=now,
        )
    )
    session.add(ChatSession(id="session-admin-diagnostics", user_id="operator", created_at=now, updated_at=now))
    session.add(
        ChatMessage(
            id="msg-admin-diagnostics",
            session_id="session-admin-diagnostics",
            user_id="operator",
            type="assistant",
            content="这是一条缺少部分诊断字段的历史回答。",
            rag_trace={},
            created_at=now,
        )
    )


def _browser_editorial_entry() -> CreateEditorialEntryRequest:
    body = {
        "decision_query": "什么时候使用 deterministic workflow？",
        "recommendation_or_reviewed_branches": "已知路径应由 deterministic workflow 控制。",
        "applicability": "Applies to known execution paths with explicit termination conditions.",
        "non_applicability": "Does not authorize unbounded autonomous execution loops.",
        "alternatives": "Treat every execution path as equivalent.",
        "trade_offs": "Deterministic workflows constrain autonomy in exchange for replayable behavior.",
        "failure_modes": "Unknown termination conditions can invalidate the deterministic boundary.",
        "minimum_implementation_guidance": "Record the termination condition before enabling automation.",
        "minimum_validation_guidance": "Replay a fixed input through the configured workflow.",
        "minimum_diagnosis_guidance": "Inspect the retained execution boundary before changing the workflow.",
        "minimum_acceptance_guidance": "Verify one supported query and one boundary query.",
        "conflicts": "No current conflict is retained for this decision.",
        "unknowns": "Provider-specific orchestration behavior remains outside this entry.",
        "boundary_conditions": "The recommendation assumes an authorized team-shared corpus.",
    }
    return CreateEditorialEntryRequest(
        entry_id=_BROWSER_AGENT_ENTRY_ID,
        title="Prefer deterministic workflows",
        coverage_position="orchestration_retry_human_intervention_and_side_effects",
        assurance_level="claim_linked",
        approving_reviewer_username="browser-reviewer",
        accountable_maintainer_username="browser-maintainer",
        review_date="2026-08-12",
        applicable_versions=["framework-neutral", "Anthropic 2024-12-19"],
        applicability_conditions=[
            {
                "condition_id": "condition-known-execution-path",
                "field": "execution_path",
                "operator": "equals",
                "value": "known",
            }
        ],
        non_applicability_conditions=[
            {
                "condition_id": "condition-unbounded-loop",
                "field": "autonomy",
                "operator": "equals",
                "value": "unbounded",
            }
        ],
        freshness_triggers=[
            {
                "trigger_id": "freshness-workflow-source",
                "trigger_type": "source_release",
                "review_within_days": 7,
            }
        ],
        sources=[
            {
                "source_id": _BROWSER_AGENT_SOURCE_ID,
                "source_tier": "primary_evidence_source",
                "title": "Building effective agents",
                "authority": "Anthropic",
                "version_or_date": "2026-08-12",
                "availability": "verified_usable",
                "access_scope": "public",
                "public_url": "https://www.anthropic.com/engineering/building-effective-agents",
                "independent_public_verifiability": True,
            }
        ],
        chunk_strategy={
            "strategy_id": "section-aware-900-120",
            "max_characters": 900,
            "overlap_characters": 120,
            "preserve_section_boundaries": True,
        },
        acceptance_material={
            "supported_queries": [
                {
                    "query_id": "supported-deterministic-workflow",
                    "query": "什么时候使用 deterministic workflow？",
                    "expected_outcome": "supported",
                }
            ],
            "boundary_queries": [
                {
                    "query_id": "boundary-deterministic-workflow",
                    "query": "未验证来源能否直接作为 workflow 证据？",
                    "expected_outcome": "insufficient_evidence",
                }
            ],
        },
        body=body,
        section_source_relationships=[
            {
                "section_id": section_id,
                "source_ids": [_BROWSER_AGENT_SOURCE_ID],
            }
            for section_id in body
        ],
        claims=[
            {
                "claim_id": "claim-deterministic-workflow",
                "claim_kind": "prescriptive",
                "statement": "Known execution paths should use deterministic workflows.",
                "section_id": "recommendation_or_reviewed_branches",
                "source_ids": [_BROWSER_AGENT_SOURCE_ID],
                "material": True,
                "scope": "team_shared",
            }
        ],
        relationship={},
    )


async def _seed_browser_editorial_authority(
    session: AsyncSession,
    *,
    agent_chunk: DocumentChunk,
) -> None:
    author = User(username="browser-author", password_hash="browser-acceptance", role="user", is_active=True)
    reviewer = User(username="browser-reviewer", password_hash="browser-acceptance", role="user", is_active=True)
    maintainer = User(username="browser-maintainer", password_hash="browser-acceptance", role="user", is_active=True)
    session.add_all([author, reviewer, maintainer])
    await session.flush()

    authority = EditorialAuthorityService(session)
    draft = await authority.create_draft(_browser_editorial_entry(), author)
    await authority.collect_evidence(draft["entry_id"], author)
    await authority.accept_maintainer_responsibility(draft["entry_id"], maintainer)
    await authority.record_source_availability(
        draft["entry_id"],
        _BROWSER_AGENT_SOURCE_ID,
        "verified_usable",
        maintainer,
    )
    await authority.request_editorial_review(draft["entry_id"], author)
    approved = await authority.approve_current_revision(draft["entry_id"], reviewer)

    entry, events = await authority._entry_and_events(draft["entry_id"])
    reviewer_identity = approved["approval"]["reviewer_identity"]
    await authority._append_entry_event(
        entry,
        events,
        event_type=CanonicalEventType.STATE_CHANGED,
        from_state="editorial_review",
        to_state="candidate_build",
        action="candidate_build_admitted_for_browser_acceptance",
        revision_identity=approved["revision_identity"],
        actor_identity=reviewer_identity,
    )
    await session.commit()
    entry, events = await authority._entry_and_events(draft["entry_id"])
    await authority._append_entry_event(
        entry,
        events,
        event_type=CanonicalEventType.PUBLISHED,
        from_state="candidate_build",
        to_state="published",
        action="published_for_browser_acceptance",
        revision_identity=approved["revision_identity"],
        actor_identity=reviewer_identity,
    )
    await session.commit()

    current = await authority.get_retrieval_authority(draft["entry_id"])
    if current.get("answer_eligible") is not True:
        raise RuntimeError("browser acceptance fixture did not reach answer eligibility")
    relationships = current.get("section_source_relationships")
    if not isinstance(relationships, dict):
        raise RuntimeError("browser acceptance fixture has no section-source relationships")
    source_relationships = relationships.get("recommendation_or_reviewed_branches")
    if not isinstance(source_relationships, list):
        raise RuntimeError("browser acceptance fixture has no recommendation source relationship")
    agent_chunk.chunk_metadata = {
        **agent_chunk.chunk_metadata,
        "entry_identity": current["entry_identity"],
        "editorial_revision_identity": current["editorial_revision_identity"],
        "source_relationships": source_relationships,
        "assurance_level": current["assurance_level"],
        "applicability_conditions": current["applicability_conditions"],
        "non_applicability_conditions": current["non_applicability_conditions"],
        "freshness_triggers": current["freshness_triggers"],
    }
    await session.commit()

    if os.getenv("BROWSER_ACCEPTANCE_SOURCE_LOST") == "1":
        await authority.record_source_availability(
            draft["entry_id"],
            _BROWSER_AGENT_SOURCE_ID,
            "unavailable_for_new_evidence",
            maintainer,
        )


async def _seed_test_data() -> None:
    ready_documents = [
        ("browser-evidence", "browser-evidence.md", "部署前需要完成变更审批。"),
        ("browser-inspection", "browser-inspection.md", "已发布分块可用于检查部署审批记录。"),
        ("browser-single-delete", "browser-single-delete.md", "单个删除验收文档。"),
        ("browser-batch-first", "browser-batch-first.md", "第一份批量构建文档。"),
        ("browser-batch-second", "browser-batch-second.md", "第二份批量构建文档。"),
        ("browser-batch-partial-first", "browser-batch-partial-first.md", "批量删除部分失败的并发文档。"),
        ("browser-batch-partial-second", "browser-batch-partial-second.md", "批量删除成功文档。"),
    ]
    async with SessionLocal() as session:
        if os.getenv("BROWSER_ACCEPTANCE_SEED") != "minimal":
            for document_id, filename, content in ready_documents:
                session.add(
                    Document(
                        id=document_id,
                        filename=filename,
                        file_type="md",
                        file_size=len(content.encode("utf-8")),
                        source_content=content.encode("utf-8"),
                        status="ready",
                        chunk_strategy="general",
                        chunk_count=1,
                        published_generation=1,
                        next_generation=2,
                        latest_requested_generation=1,
                    )
                )
                session.add(
                    DocumentChunk(
                        id=f"chunk-{document_id}",
                        document_id=document_id,
                        generation=1,
                        chunk_index=0,
                        content=content,
                        keywords=["部署", "审批"],
                        generated_questions=[],
                        chunk_metadata={"filename": filename},
                    )
                )
                session.add(
                    DocumentJob(
                        id=f"job-{document_id}",
                        document_id=document_id,
                        build_generation=1,
                        requested_chunk_strategy="general",
                        status="succeeded",
                        stage="completed",
                        progress=100,
                        message="document build completed",
                    )
                )

            session.add(
                Document(
                    id="browser-agent-entry",
                    filename="synthetic-agent-entry.md",
                    file_type="md",
                    file_size=len("已知路径应由 deterministic workflow 控制。".encode()),
                    source_content=b"synthetic browser acceptance entry",
                    status="ready",
                    chunk_strategy="agent",
                    chunk_count=1,
                    published_generation=1,
                    next_generation=2,
                    latest_requested_generation=1,
                )
            )
            browser_agent_chunk = DocumentChunk(
                id="chunk-browser-agent-entry",
                document_id="browser-agent-entry",
                generation=1,
                chunk_index=0,
                content="已知路径应由 deterministic workflow 控制。",
                keywords=["deterministic", "workflow"],
                generated_questions=[],
                chunk_metadata={
                    "strategy": "agent",
                    "entry_id": _BROWSER_AGENT_ENTRY_ID,
                    "entry_title": "Prefer deterministic workflows",
                    "domain": "orchestration_retry_human_intervention_and_side_effects",
                    "section_id": "recommendation_or_reviewed_branches",
                    "review_status": "approved",
                    "review_date": "2026-08-12",
                    "assurance_level": "claim_linked",
                    "evidence_conflict": "none",
                    "applicable_versions": ["framework-neutral", "Anthropic 2024-12-19"],
                    "approved_summary": "已知路径应由 deterministic workflow 控制。",
                    "suggested_query": "什么时候使用 deterministic workflow？",
                    "sources": [
                        {
                            "source_id": _BROWSER_AGENT_SOURCE_ID,
                            "title": "Building effective agents",
                            "authority": "Anthropic",
                            "url": "https://www.anthropic.com/engineering/building-effective-agents",
                            "version": "2024-12-19",
                            "availability": "verified",
                            "review_date": "2026-08-12",
                            "freshness_days": 90,
                        }
                    ],
                    "source_id": _BROWSER_AGENT_SOURCE_ID,
                    "source_tier": "primary_evidence_source",
                    "source_title": "Building effective agents",
                    "source_authority": "Anthropic",
                    "source_url": "https://www.anthropic.com/engineering/building-effective-agents",
                    "source_version": "2024-12-19",
                    "source_availability": "verified",
                    "source_access_scope": "public",
                    "source_review_date": "2026-08-12",
                    "source_freshness_days": 90,
                    "claim_evidence_contract": _BROWSER_CLAIM_CONTRACT.canonical_json,
                    "claim_evidence_contract_sha256": _BROWSER_CLAIM_CONTRACT.sha256,
                    "title": "Prefer deterministic workflows",
                },
            )
            session.add(browser_agent_chunk)
            session.add(
                DocumentJob(
                    id="job-browser-agent-entry",
                    document_id="browser-agent-entry",
                    build_generation=1,
                    requested_chunk_strategy="agent",
                    status="succeeded",
                    stage="completed",
                    progress=100,
                    message="synthetic Agent entry published",
                )
            )
            await _seed_browser_editorial_authority(session, agent_chunk=browser_agent_chunk)
            session.add(
                Document(
                    id="browser-cancelable",
                    filename="browser-cancelable.md",
                    file_type="md",
                    file_size=0,
                    status="pending",
                    chunk_strategy="general",
                    next_generation=2,
                    latest_requested_generation=1,
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-cancelable",
                    document_id="browser-cancelable",
                    build_generation=1,
                    requested_chunk_strategy="general",
                    status="queued",
                    stage="queued",
                    progress=0,
                    message="queued for browser cancellation",
                )
            )

            session.add(
                Document(
                    id="browser-running",
                    filename="browser-running.md",
                    file_type="md",
                    file_size=len("正在生成可检索分块。".encode()),
                    source_content="正在生成可检索分块。".encode(),
                    status="processing",
                    chunk_strategy="general",
                    chunk_count=0,
                    next_generation=3,
                    latest_requested_generation=2,
                    active_build_generation=2,
                    active_build_job_id="job-browser-running",
                    active_build_heartbeat_at=datetime.now(UTC),
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-running",
                    document_id="browser-running",
                    build_generation=2,
                    requested_chunk_strategy="general",
                    status="running",
                    stage="chunking",
                    progress=56,
                    message="正在生成可检索分块",
                )
            )

            session.add(
                Document(
                    id="browser-candidate",
                    filename="browser-candidate.md",
                    file_type="md",
                    file_size=len("候选构建等待管理员发布。".encode()),
                    source_content="候选构建等待管理员发布。".encode(),
                    status="candidate",
                    chunk_strategy="general",
                    chunk_count=0,
                    published_generation=0,
                    candidate_generation=2,
                    candidate_chunk_strategy="general",
                    candidate_chunk_count=1,
                    next_generation=3,
                    latest_requested_generation=2,
                )
            )
            session.add(
                DocumentChunk(
                    id="chunk-browser-candidate",
                    document_id="browser-candidate",
                    generation=2,
                    chunk_index=0,
                    content="候选构建等待管理员发布。",
                    keywords=["候选", "发布"],
                    generated_questions=[],
                    chunk_metadata={"filename": "browser-candidate.md"},
                )
            )
            session.add(
                DocumentJob(
                    id="job-browser-candidate",
                    document_id="browser-candidate",
                    build_generation=2,
                    requested_chunk_strategy="general",
                    status="succeeded",
                    stage="completed",
                    progress=100,
                    message="candidate build completed; awaiting publication",
                )
            )

            await _seed_chat_history(session)
        await session.commit()

        await SystemSettingsDraftService(session).save(
            actor="browser-bootstrap",
            payload={
                "provider_type": "ark",
                "model": "Qwen/Qwen3-32B",
                "service_url": "https://provider.example.test/v1",
                "provider_api_key": "browser-acceptance-placeholder",
            },
        )
        if os.getenv("BROWSER_ACCEPTANCE_SETTINGS_FAILED") == "1":
            await session.merge(
                SystemSettingsState(
                    id=1,
                    latest_saved_version=1,
                    active_version=None,
                    application_state="failed",
                    application_version=None,
                    application_actor=None,
                    application_at=None,
                    application_message="runtime did not accept the saved configuration",
                )
            )
        await session.commit()


_route_provider_doubles = {}


async def _deterministic_route_providers(self, payload):
    del self
    return {
        item["provider"]: _route_provider_doubles.setdefault(
            (payload["route_identity"], item["provider"]), _DeterministicLlm(),
        )
        for item in payload["providers"]
    }


async def _seed_generation_route() -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        await MemberAdmissionService(session, redis=None).create_bootstrap_administrator(
            settings.bootstrap_admin_username, settings.bootstrap_admin_password,
        )
        administrator = await UserRepository(session).get_by_username(settings.bootstrap_admin_username)
        assert administrator is not None
        actor = f"member:{administrator.id}"
        routes = GenerationRouteService(session)
        route = await routes.save(actor=actor, payload={
            "data_scope": "team_shared_pilot", "max_attempts": 1, "total_timeout_seconds": 30,
            "providers": [{
                "provider": "browser-acceptance", "provider_type": "openai", "model": "browser-deterministic",
                "service_url": "https://provider.example.test/v1", "endpoint_class": "public_https",
                "data_scope": "team_shared_pilot", "timeout_seconds": 30, "provider_api_key": "browser-placeholder",
            }],
        })
        acceptance = DeliveryAcceptanceService(session)
        record = await acceptance.create(
            CreateDeliveryAcceptanceRecordRequest.model_validate(generation_acceptance_payload(route)), administrator,
        )
        await acceptance.update_status(record["record_id"], UpdateDeliveryAcceptanceStatusRequest.model_validate({
            "status": "active", "reason_code": "checks_verified",
            "verified_checks": [{"check_id": check["check_id"], "evidence_links": check["evidence_links"]} for check in record["checks"]],
        }), administrator)
        await routes.activate(actor=actor, payload={
            "route_identity": route["route_identity"], "expected_active_identity": None,
            "acceptance_record_identity": record["record_id"],
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated browser acceptance API environment.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_source_url = os.getenv("BROWSER_ACCEPTANCE_PUBLIC_SOURCE_URL", "").strip()
    if allowed_source_url:
        def probe_acceptance_source(url: str) -> None:
            if url != allowed_source_url:
                raise OSError("source URL is outside the browser acceptance allowlist")

        parsers._probe_public_source = probe_acceptance_source
    settings_runtime._runtime = _DeterministicSettingsRuntime()
    asyncio.run(_create_schema())
    asyncio.run(_seed_test_data())
    GenerationRouteService.providers = _deterministic_route_providers
    asyncio.run(_seed_generation_route())
    redis = _InMemoryRedis()
    app.dependency_overrides[get_redis_client] = lambda: redis
    registry = get_extension_registry()
    registry.register_claim_resolver("chat-default-claim-resolver", _BrowserClaimResolver())
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
