from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


class RetentionCleanupState(Base):
    """One bounded non-content observation per independently cleaned data class."""

    __tablename__ = "retention_cleanup_states"

    data_class: Mapped[str] = mapped_column(String(32), primary_key=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    policy_identity: Mapped[str] = mapped_column(String(192), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    deleted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    remaining_expired: Mapped[int | None] = mapped_column(Integer, nullable=True)
    normalized_error: Mapped[str | None] = mapped_column(String(64), nullable=True)
