from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.maintenance.findings import load_finding
from app.maintenance.schemas import RoadmapInput, RoadmapReview
from app.model.answer_execution import AnswerExecutionEventModel, AnswerExecutionModel
from app.model.canonical import CanonicalRecordModel
from app.model.knowledge_feedback import KnowledgeFeedbackSignal, MaintenanceSignalLink
from app.model.user import User


def utcnow() -> datetime:
    return datetime.now(UTC)


def roadmap_projection(candidate: dict) -> dict:
    return {
        **candidate,
        "overdue_review": candidate["state"] == "deferred" and date.fromisoformat(candidate["review_date"]) <= utcnow().date(),
    }


async def qualify(session: AsyncSession, item: dict, payload: RoadmapInput, owner: User) -> dict:
    if item["severity"] in {"p0", "p1"}:
        raise AppError(status_code=409, code="ROADMAP_INTEGRITY_DEFERRAL_FORBIDDEN", message="P0/P1 must be contained and repaired")
    if item["state"] not in {"triaged", "in_progress"}:
        raise AppError(status_code=409, code="MAINTENANCE_TRANSITION_INVALID", message="triage before roadmap qualification")
    now = utcnow()
    if not now.date() < payload.review_date <= (now + timedelta(days=31)).date():
        raise AppError(status_code=422, code="ROADMAP_REVIEW_DATE_INVALID", message="monthly owner review required")
    signals = (
        await session.scalars(
            select(KnowledgeFeedbackSignal)
            .join(
                MaintenanceSignalLink,
                MaintenanceSignalLink.signal_id == KnowledgeFeedbackSignal.id,
            )
            .join(
                AnswerExecutionModel,
                AnswerExecutionModel.id == KnowledgeFeedbackSignal.normalized_metadata["answer_execution_id"].as_string(),
            )
            .join(
                AnswerExecutionEventModel,
                AnswerExecutionEventModel.execution_id == AnswerExecutionModel.id,
            )
            .where(
                MaintenanceSignalLink.item_id == item["id"],
                AnswerExecutionModel.user_id == KnowledgeFeedbackSignal.user_id,
                AnswerExecutionEventModel.event_type == "state_changed",
                AnswerExecutionEventModel.from_state == "running",
                AnswerExecutionEventModel.to_state == "completed",
                AnswerExecutionEventModel.occurred_at >= now - timedelta(days=30),
                AnswerExecutionEventModel.occurred_at <= now,
                KnowledgeFeedbackSignal.created_at <= now,
                KnowledgeFeedbackSignal.expires_at > now,
            )
        )
    ).all()
    executions = {
        value
        for signal in signals
        if isinstance((value := signal.normalized_metadata.get("answer_execution_id")), str) and value.startswith("answer_execution:")
    }
    findings = [await load_finding(session, item, identity) for identity in item.get("finding_identities", [])]
    fingerprints = {finding["verification_fingerprint"] for finding in findings}
    direct = item["classification"] == "scope-roadmap"
    if len(executions) < 3 and len(fingerprints) < 2 and not direct:
        raise AppError(
            status_code=409, code="ROADMAP_THRESHOLD_NOT_MET", message="three distinct executions or two independent findings required"
        )
    identity = f"maintenance_item:{uuid4().hex}"
    fields = {
        "schema": "knowledge_roadmap_candidate/v1",
        "item_identity": item["id"],
        "affected_scope": item["affected_scope"],
        "pattern": item.get("verified_pattern")
        or {
            "classification": item["classification"],
            "coverage_position": item["coverage_position"],
        },
        "scope_boundary": "production_rag_agent_engineering",
        "desired_outcome": payload.desired_outcome,
        "bounded_work_reason": payload.bounded_work_reason,
        "owner_identity": f"member:{owner.id.hex}",
        "review_date": payload.review_date.isoformat(),
        "qualification": {
            "distinct_executions_30_days": len(executions),
            "independent_findings": len(fingerprints),
            "direct_scope_deferral": direct,
        },
        "maintenance_decision_links": [item["id"]],
        "state": "deferred",
        "revision": 1,
    }
    session.add(
        CanonicalRecordModel(
            stable_id=identity,
            identity_kind="maintenance_item",
            identity_value=identity.split(":", 1)[1],
            state="deferred",
            record_class="immutable",
            payload=fields,
        )
    )
    return roadmap_projection({"id": identity, **fields})


async def review_candidate(session: AsyncSession, candidate: dict, payload: RoadmapReview, owner: User) -> dict:
    if candidate["state"] != "deferred":
        raise AppError(status_code=409, code="ROADMAP_REVIEW_CLOSED", message="candidate has a terminal disposition")
    now = utcnow().date()
    if payload.action == "renew" and (payload.review_date is None or not now < payload.review_date <= now + timedelta(days=31)):
        raise AppError(status_code=422, code="ROADMAP_REVIEW_DATE_INVALID", message="monthly owner review required")
    changes = {
        "state": {"renew": "deferred", "close": "closed", "start_wayfinder": "mapped"}[payload.action],
        "owner_identity": f"member:{owner.id.hex}",
        "rationale": payload.rationale,
        "review_date": payload.review_date.isoformat() if payload.action == "renew" and payload.review_date else None,
    }
    if payload.action == "start_wayfinder":
        identity = f"maintenance_item:{uuid4().hex}"
        markdown = (
            f"# Wayfinder: {candidate['desired_outcome']}\n\nStatus: open\n"
            f"Parent: {candidate['id']}\nOwner: member:{owner.id.hex}\n\n"
            f"## Scope\n\n{candidate['pattern']['coverage_position']}\n"
            f"Boundary: {candidate['scope_boundary']}\n\n"
            "## Current Frontier\n\n"
            "1. Type: grilling; Status: open; Define the bounded desired outcome and exclusions.\n"
            "2. Type: research; Status: open; Establish independently verifiable evidence and constraints.\n"
            "3. Type: task; Status: open; Decide scope before commissioning implementation.\n\n"
            "## Decisions So Far\n\nQualified as a separate effort; no product expansion is authorized.\n"
        )
        artifact = {
            "schema": "maintenance_wayfinder_map/v1",
            "candidate_identity": candidate["id"],
            "markdown": markdown,
        }
        session.add(
            CanonicalRecordModel(
                stable_id=identity,
                identity_kind="maintenance_item",
                identity_value=identity.split(":", 1)[1],
                state="open",
                record_class="authoritative",
                payload=artifact,
            )
        )
        changes["map_identity"] = identity
        changes["map_sha256"] = canonical_json_sha256(artifact)
    return changes
