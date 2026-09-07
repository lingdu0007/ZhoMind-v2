from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String, UniqueConstraint, event
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.model.base import Base


class AnswerExecutionModel(Base):
    """A private, conversation-retained canonical answer-execution request."""

    __tablename__ = "answer_executions"
    __table_args__ = (
        Index("ix_answer_executions_session_user_created", "session_id", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(192), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    initial_state: Mapped[str] = mapped_column(String(32), nullable=False)
    request: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class AnswerExecutionEventModel(Base):
    """An append-only event while its owning private conversation is retained."""

    __tablename__ = "answer_execution_events"
    __table_args__ = (
        UniqueConstraint(
            "execution_id",
            "sequence",
            name="uq_answer_execution_events_execution_sequence",
        ),
        Index("ix_answer_execution_events_execution_occurred", "execution_id", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    execution_id: Mapped[str] = mapped_column(String(192), nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


@event.listens_for(AnswerExecutionModel, "before_update")
def _reject_answer_execution_update(_mapper, _connection, _target: AnswerExecutionModel) -> None:
    raise ValueError("answer execution records are immutable while retained")


@event.listens_for(AnswerExecutionEventModel, "before_update")
def _reject_answer_execution_event_update(_mapper, _connection, _target: AnswerExecutionEventModel) -> None:
    raise ValueError("answer execution events are append-only while retained")


@event.listens_for(Session, "do_orm_execute")
def _reject_answer_execution_bulk_update(orm_execute_state) -> None:
    if not orm_execute_state.is_update:
        return
    mapper = orm_execute_state.bind_mapper
    if mapper is not None and mapper.class_ in {AnswerExecutionModel, AnswerExecutionEventModel}:
        raise ValueError("answer execution records and events cannot be updated")
