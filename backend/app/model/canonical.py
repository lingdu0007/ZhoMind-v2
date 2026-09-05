from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String, event
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


class CanonicalRecordModel(Base):
    __tablename__ = "canonical_records"
    __table_args__ = (
        Index("ix_canonical_records_kind_state", "identity_kind", "state"),
        Index("ix_canonical_records_legacy", "legacy_type", "legacy_id"),
    )

    stable_id: Mapped[str] = mapped_column(String(192), primary_key=True)
    identity_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    identity_value: Mapped[str] = mapped_column(String(160), nullable=False)
    state: Mapped[str] = mapped_column(String(96), nullable=False)
    record_class: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    legacy_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    legacy_id: Mapped[str | None] = mapped_column(String(192), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class CanonicalEventModel(Base):
    __tablename__ = "canonical_events"
    __table_args__ = (
        Index("ix_canonical_events_aggregate", "aggregate_id", "occurred_at"),
        Index("ix_canonical_events_type", "event_type", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    aggregate_id: Mapped[str] = mapped_column(String(192), nullable=False)
    aggregate_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(96), nullable=True)
    to_state: Mapped[str] = mapped_column(String(96), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    recorded_by: Mapped[str | None] = mapped_column(String(192), nullable=True)


@event.listens_for(CanonicalRecordModel, "before_update")
def _reject_canonical_record_update(_mapper, _connection, _target: CanonicalRecordModel) -> None:
    raise ValueError("canonical records are immutable; append a replacement record instead")


@event.listens_for(CanonicalRecordModel, "before_delete")
def _reject_canonical_record_delete(_mapper, _connection, _target: CanonicalRecordModel) -> None:
    raise ValueError("canonical records cannot be deleted")


@event.listens_for(CanonicalEventModel, "before_update")
def _reject_canonical_event_update(_mapper, _connection, _target: CanonicalEventModel) -> None:
    raise ValueError("canonical events are append-only")


@event.listens_for(CanonicalEventModel, "before_delete")
def _reject_canonical_event_delete(_mapper, _connection, _target: CanonicalEventModel) -> None:
    raise ValueError("canonical events are append-only")
