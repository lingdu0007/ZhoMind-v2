from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


class SystemSettingsState(Base):
    __tablename__ = "system_settings_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    latest_saved_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    application_state: Mapped[str] = mapped_column(String(24), nullable=False, default="draft_only")
    application_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    application_actor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    application_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    application_message: Mapped[str | None] = mapped_column(String(256), nullable=True)


class SystemSettingsDraft(Base):
    __tablename__ = "system_settings_drafts"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    settings: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    sealed_secrets: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    saved_by: Mapped[str] = mapped_column(String(64), nullable=False)
    saved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
