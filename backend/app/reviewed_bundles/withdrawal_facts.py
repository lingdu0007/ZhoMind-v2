from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel

WithdrawalReason = Literal["integrity_defect", "source_unavailable", "privacy_defect", "editorial_withdrawal"]
WithdrawalTrigger = Literal["integrity", "source", "editorial"]


class _WithdrawalScope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    scope: Literal["entry_version"]
    identity: Annotated[str, Field(pattern=r"^published_knowledge_version:.+$")]


class _WithdrawalFact(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_name: Literal["publication_withdrawal/v1"] = Field(alias="schema")
    event_identity: Annotated[str, Field(pattern=r"^event:.+$")]
    entry_identity: Annotated[str, Field(pattern=r"^entry:.+$")]
    publication_identity: Annotated[str, Field(pattern=r"^published_knowledge_version:.+$")]
    document_identity: Annotated[str, Field(min_length=1)]
    generation: Annotated[int, Field(gt=0)]
    configuration_identity: Annotated[str, Field(pattern=r"^configuration:.+$")]
    supersedes_version_id: Annotated[str, Field(pattern=r"^published_knowledge_version:.+$")] | None
    state: Literal["withdrawn"]
    reason_code: WithdrawalReason
    trigger: WithdrawalTrigger
    actor_identity: Annotated[str, Field(pattern=r"^member:.+$")]
    occurred_at: str
    affected_scope: _WithdrawalScope


class _ReconciliationFact(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_name: Literal["withdrawal_reconciliation/v1"] = Field(alias="schema")
    withdrawal_event_identity: str
    state: Literal["pending", "suspended", "completed"]
    reason_code: Literal[
        "derived_cleanup_pending", "history_redaction_failed", "derived_cleanup_failed", "derived_cleanup_completed",
    ]
    affected_scope: _WithdrawalScope
    allowed_next_action: Literal["none", "retry_reconciliation"]


def validate_withdrawal_reconciliation(event: CanonicalEventModel, withdrawal: dict) -> None:
    try:
        fact = _ReconciliationFact.model_validate(event.payload)
        if (
            event.aggregate_id != withdrawal["publication_identity"]
            or event.aggregate_kind != "published_knowledge_version" or event.event_type != "status_changed"
            or event.from_state != "withdrawn" or event.to_state != fact.state
            or event.recorded_by != withdrawal["actor_identity"]
            or fact.withdrawal_event_identity != withdrawal["event_identity"]
            or fact.affected_scope.model_dump() != withdrawal["affected_scope"]
            or fact.allowed_next_action != ("none" if fact.state == "completed" else "retry_reconciliation")
            or fact.reason_code not in {
                "pending": {"derived_cleanup_pending"},
                "suspended": {"history_redaction_failed", "derived_cleanup_failed"},
                "completed": {"derived_cleanup_completed"},
            }[fact.state]
        ):
            raise ValueError("reconciliation binding differs")
    except (ValidationError, ValueError, TypeError) as exc:
        raise AppError(
            status_code=409, code="WITHDRAWAL_RECONCILIATION_INVALID",
            message="withdrawal reconciliation cannot be verified",
        ) from exc


async def read_publication_withdrawals(
    session: AsyncSession, publication_identities: list[str] | None = None,
) -> list[CanonicalEventModel]:
    """Read authoritative withdrawal facts with one audit-binding check."""
    statement = select(CanonicalEventModel).where(
        CanonicalEventModel.aggregate_kind == "published_knowledge_version",
        CanonicalEventModel.event_type == "withdrawn",
    ).order_by(CanonicalEventModel.occurred_at, CanonicalEventModel.id)
    if publication_identities is not None:
        statement = statement.where(CanonicalEventModel.aggregate_id.in_(publication_identities))
    events = list((await session.scalars(statement)).all())
    for event in events:
        try:
            fact = _WithdrawalFact.model_validate(event.payload)
            occurred_at = datetime.fromisoformat(fact.occurred_at)
            event_time = event.occurred_at.replace(tzinfo=UTC) if event.occurred_at.tzinfo is None else event.occurred_at
            publication = await session.get(CanonicalRecordModel, fact.publication_identity)
            actor = await session.get(CanonicalRecordModel, fact.actor_identity)
            if (
                fact.publication_identity != event.aggregate_id
                or fact.event_identity != f"event:{event.id}" or fact.actor_identity != event.recorded_by
                or fact.affected_scope.identity != fact.publication_identity
                or occurred_at.tzinfo is None or occurred_at != event_time
                or event.from_state != "published" or event.to_state != "withdrawn"
                or actor is None or actor.identity_kind != "member"
                or publication is None or publication.identity_kind != "published_knowledge_version"
                or publication.record_class != "immutable" or publication.state != "published"
                or not isinstance(publication.payload, dict)
                or publication.payload.get("schema") != "published_knowledge_version/v1"
                or any(
                    type(publication.payload.get(field)) is not type(getattr(fact, field))
                    or publication.payload.get(field) != getattr(fact, field) for field in (
                        "entry_identity", "document_identity", "generation", "configuration_identity", "supersedes_version_id",
                    )
                )
            ):
                raise ValueError("withdrawal binding differs")
        except (ValidationError, ValueError, TypeError) as exc:
            raise AppError(
                status_code=409, code="WITHDRAWAL_FACT_INVALID",
                message="publication withdrawal audit cannot be verified",
            ) from exc
    return events
