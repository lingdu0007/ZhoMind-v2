import logging
import re
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge_feedback.service import KnowledgeFeedbackService
from app.model.chat import ChatSession
from app.model.knowledge_feedback import KnowledgeFeedbackSignal
from app.model.operational_event import OperationalEvent
from app.operations.events import OperationalEventService
from app.repository.chat_repository import ChatRepository, orphaned_private_record_predicates
from app.retention.failures import read_failure_evidence
from app.retention.models import RetentionCleanupState
from app.retention.policy import DataClass, lock_registry, read_policy

DATA_CLASSES: tuple[DataClass, ...] = ("conversations", "operational_events", "feedback_signals")
logger = logging.getLogger(__name__)


async def expired_count(session: AsyncSession, data_class: DataClass, *, now: datetime, days: int) -> int:
    cutoff = now - timedelta(days=days)
    if data_class == "conversations":
        predicates = (
            (ChatSession, ChatSession.created_at <= cutoff),
            *orphaned_private_record_predicates(),
        )
    elif data_class == "operational_events":
        predicates = ((OperationalEvent, OperationalEvent.created_at <= cutoff),)
    else:
        predicates = ((KnowledgeFeedbackSignal, or_(
            KnowledgeFeedbackSignal.created_at <= cutoff, KnowledgeFeedbackSignal.expires_at <= now,
        )),)
    total = 0
    for model, predicate in predicates:
        total += int(await session.scalar(select(func.count()).select_from(model).where(predicate)) or 0)
    if data_class == "feedback_signals":
        total += len(await KnowledgeFeedbackService(session).unresolved_feedback_references())
    return total


async def run_retention_sweep(session_factory, *, now: datetime | None = None) -> dict:
    """Commit each class separately; durable pending states survive worker loss."""
    from app.delivery_acceptance.service import DeliveryAcceptanceService

    results = {}
    for data_class in DATA_CLASSES:
        checked_at = now or datetime.now(UTC)
        previous_attempt: int | None = None
        attempt: int | None = None
        failure_policy_identity: str | None = None
        pending_committed = False
        try:
            async with session_factory() as session:
                await lock_registry(session)
                checked_at = now or datetime.now(UTC)
                policy = await read_policy(session)
                failure_policy_identity = policy["identity"]
                state = await session.get(RetentionCleanupState, data_class)
                previous_checked_at = (
                    state.checked_at.replace(tzinfo=UTC) if state and state.checked_at.tzinfo is None
                    else state.checked_at if state else None
                )
                interrupted = state is not None and state.status == "pending" and (
                    previous_checked_at is None or previous_checked_at > checked_at
                    or previous_checked_at <= checked_at - timedelta(seconds=180)
                )
                if state is not None and (state.status == "failed" or interrupted):
                    await DeliveryAcceptanceService(session).suspend_retention_failure({
                        "data_class": data_class, **project_cleanup(state), "status": "failed",
                        "normalized_error": state.normalized_error or "RETENTION_CLEANUP_INTERRUPTED",
                    })
                if state is None:
                    state = RetentionCleanupState(data_class=data_class, attempt=0, policy_identity=policy["identity"])
                    session.add(state)
                previous_attempt = state.attempt
                known_failures = await read_failure_evidence(session)
                state.attempt = max(
                    state.attempt,
                    max((failure.invalidated_attempt for _, failure in known_failures if failure.data_class == data_class), default=0),
                ) + 1
                attempt = state.attempt
                state.policy_identity = policy["identity"]
                state.status = "pending"
                state.checked_at = checked_at
                state.deleted_count = 0
                state.remaining_expired = None
                state.normalized_error = None
                await session.commit()
                pending_committed = True

                await lock_registry(session)
                state = await session.get(RetentionCleanupState, data_class, populate_existing=True)
                if state is None or state.attempt != attempt:
                    continue
                policy = await read_policy(session)
                failure_policy_identity = policy["identity"]
                state.policy_identity = policy["identity"]
                days = policy["days"][data_class]
                before = await expired_count(session, data_class, now=checked_at, days=days)
                if data_class == "conversations":
                    await ChatRepository(session).purge_expired_sessions(now=checked_at, retention_days=days)
                elif data_class == "operational_events":
                    await OperationalEventService(session).purge_expired(now=checked_at, retention_days=days)
                else:
                    await KnowledgeFeedbackService(session).purge_expired(now=checked_at, retention_days=days)
                remaining = await expired_count(session, data_class, now=checked_at, days=days)
                state.deleted_count = max(0, before - remaining)
                state.remaining_expired = remaining
                state.status = "failed" if remaining else "verified"
                state.normalized_error = "PRIVACY_EXPIRY_SURVIVED" if remaining else None
                if remaining:
                    await DeliveryAcceptanceService(session).suspend_retention_failure({"data_class": data_class, **project_cleanup(state)})
                await session.commit()
                results[data_class] = project_cleanup(state)
        except Exception:
            # No exception string: database errors can embed private bound values.
            logger.error("RETENTION_CLEANUP_FAILED class=%s", data_class)
            results[data_class] = {"status": "failed", "normalized_error": "RETENTION_CLEANUP_FAILED"}
            try:
                async with session_factory() as session:
                    await lock_registry(session)
                    current = await session.get(RetentionCleanupState, data_class)
                    invalidated_attempt = attempt if pending_committed else previous_attempt
                    if invalidated_attempt is None:
                        invalidated_attempt = current.attempt if current else 0
                    await DeliveryAcceptanceService(session).suspend_retention_failure({
                        "data_class": data_class, "status": "failed", "attempt": attempt,
                        "invalidated_attempt": invalidated_attempt,
                        "policy_identity": failure_policy_identity, "checked_at": checked_at.isoformat(),
                        "deleted_count": 0, "remaining_expired": None, "pending_committed": pending_committed,
                        "normalized_error": "RETENTION_CLEANUP_FAILED",
                    })
                    await session.commit()
            except Exception:
                logger.error("RETENTION_FAILURE_EVIDENCE_UNAVAILABLE class=%s", data_class)
            try:
                async with session_factory() as session:
                    await lock_registry(session)
                    state = await session.get(RetentionCleanupState, data_class)
                    expected_attempt = attempt if pending_committed else previous_attempt
                    current_attempt = state.attempt if state else 0
                    observed_at = state.checked_at.replace(tzinfo=UTC) if state and state.checked_at.tzinfo is None else (
                        state.checked_at if state else None
                    )
                    owns_failure = (
                        current_attempt == expected_attempt if expected_attempt is not None
                        else observed_at is None or observed_at < checked_at
                    )
                    if owns_failure:
                        policy = await read_policy(session)
                        if state is None:
                            state = RetentionCleanupState(data_class=data_class)
                            session.add(state)
                        state.attempt = attempt if attempt is not None else current_attempt + 1
                        state.policy_identity = policy["identity"]
                        state.status = "failed"
                        state.checked_at = checked_at
                        state.deleted_count = 0
                        state.remaining_expired = None
                        state.normalized_error = "RETENTION_CLEANUP_FAILED"
                    await session.commit()
                    if state is not None:
                        results[data_class] = project_cleanup(state)
            except Exception:
                logger.error("RETENTION_STATUS_UNAVAILABLE class=%s", data_class)
    return results


def project_cleanup(state: RetentionCleanupState) -> dict:
    if (
        state.status not in {"pending", "verified", "failed"}
        or state.attempt < 1 or state.deleted_count < 0
        or (state.remaining_expired is not None and state.remaining_expired < 0)
        or not re.fullmatch(r"configuration:[0-9a-f]{64}", state.policy_identity)
        or state.normalized_error not in {None, "PRIVACY_EXPIRY_SURVIVED", "RETENTION_CLEANUP_FAILED"}
        or (state.status == "verified" and (state.remaining_expired != 0 or state.normalized_error is not None))
    ):
        return {"status": "unverified"}
    return {
        "status": state.status, "attempt": state.attempt, "policy_identity": state.policy_identity,
        "checked_at": state.checked_at.isoformat(), "deleted_count": state.deleted_count,
        "remaining_expired": state.remaining_expired, "normalized_error": state.normalized_error,
    }


async def read_retention_status(session: AsyncSession) -> dict:
    policy = await read_policy(session)
    failures = await read_failure_evidence(session)
    states = {state.data_class: state for state in (await session.scalars(select(RetentionCleanupState))).all()}
    cleanup = {}
    for data_class in DATA_CLASSES:
        state = states.get(data_class)
        if state is None:
            cleanup[data_class] = {"status": "unverified"}
            continue
        projection = project_cleanup(state)
        checked_at = state.checked_at.replace(tzinfo=UTC) if state.checked_at.tzinfo is None else state.checked_at
        now = datetime.now(UTC)
        if state.policy_identity != policy["identity"] or not now - timedelta(seconds=180) <= checked_at <= now:
            projection["status"] = "unverified"
        if any(
            failure.data_class == data_class and failure.invalidated_attempt >= state.attempt
            for _, failure in failures
        ):
            projection["status"] = "failed"
            projection["normalized_error"] = "RETENTION_CLEANUP_FAILED"
        cleanup[data_class] = projection
    return {
        "policy": policy, "cleanup": cleanup,
        "privacy_blocked": any(item["status"] != "verified" for item in cleanup.values()),
    }
