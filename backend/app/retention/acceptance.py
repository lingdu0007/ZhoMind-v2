from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.delivery_acceptance.schemas import AcceptanceFailureKind
from app.model.canonical import CanonicalRecordModel
from app.retention.failures import read_failure_evidence
from app.retention.policy import RetentionDays, policy_projection, read_policy


async def retention_acceptance_blockers(session: AsyncSession, payload: dict, record_identity: str) -> list[dict]:
    scope = payload.get("affected_scope", {})
    deployment = scope.get("deployment_identity")
    if not deployment:
        return []
    retained = []
    for event_id, failure in await read_failure_evidence(session):
        if record_identity in failure.affected_acceptance_identities:
            retained.append({
                "check_id": "check:retention-boundary", "result": "failed", "reason": "retention_cleanup_failed",
                "failure_kind": AcceptanceFailureKind.SHARED_PRIVACY.value,
                "blocking_scope": {"scope": "deployment", "identity": deployment},
                "evidence_links": [f"evidence://retention/events/{event_id}"],
            })
            break
    current = await read_policy(session)
    policy_ids = {policy_projection(RetentionDays())["identity"]}
    records = (await session.scalars(select(CanonicalRecordModel).where(
        CanonicalRecordModel.identity_kind == "configuration",
    ))).all()
    policy_ids.update(record.stable_id for record in records if record.payload.get("schema") == "retention_policy/v1")
    bound = policy_ids.intersection(payload.get("product_identities", []))
    live_stage = payload.get("stage") in {"limited_team_pilot", "daily_use_release", "public_evidence_release"}
    if not bound and not live_stage:
        return retained
    reason = None
    if bound != {current["identity"]}:
        reason = "retention_policy_changed" if bound else "retention_policy_unbound"
    if reason is None and live_stage:
        from app.retention.cleanup import read_retention_status

        if (await read_retention_status(session))["privacy_blocked"]:
            reason = "retention_cleanup_unverified"
    if reason is None:
        return retained
    return retained + [{
        "check_id": "check:retention-boundary", "result": "failed", "reason": reason,
        "failure_kind": AcceptanceFailureKind.SHARED_PRIVACY.value,
        "blocking_scope": {"scope": "deployment", "identity": deployment},
    }]
