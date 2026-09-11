import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class KnowledgeFeedbackSignal(Base):
    __tablename__ = "knowledge_feedback_signals"
    __table_args__ = (
        UniqueConstraint("user_id", "answer_id", "scope_key", name="uq_feedback_user_answer_scope"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    answer_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    entry_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    scope_key: Mapped[str] = mapped_column(String(96), nullable=False)
    knowledge_edition: Mapped[str | None] = mapped_column(String(128), nullable=True)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class ReviewWorkItem(Base):
    __tablename__ = "knowledge_review_work_items"
    __table_args__ = (UniqueConstraint("dedupe_key", name="uq_knowledge_review_work_item_dedupe"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    kind: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(192), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    signal_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(16), index=True, nullable=False, default="pending")
    classification: Mapped[str | None] = mapped_column(String(16), nullable=True)
    normalized_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class MaintenanceSignalLink(Base):
    __tablename__ = "maintenance_signal_links"

    item_id: Mapped[str] = mapped_column(
        String(192), ForeignKey("canonical_records.stable_id"), primary_key=True,
    )
    signal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("knowledge_feedback_signals.id", ondelete="CASCADE"), primary_key=True,
        index=True,
    )
