import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class OperationalEvent(Base):
    __tablename__ = "operational_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    request_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    route_outcome: Mapped[str] = mapped_column(String(128), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    gate_outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    provider_identity: Mapped[str | None] = mapped_column(String(128), nullable=True)
    normalized_error: Mapped[str | None] = mapped_column(String(128), nullable=True)
    candidate_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
