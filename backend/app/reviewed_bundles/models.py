from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


class CandidateBuildJob(Base):
    __tablename__ = "candidate_build_jobs"
    __table_args__ = (
        UniqueConstraint("entry_identity", "requested_generation", name="uq_candidate_build_jobs_entry_generation"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    bundle_id: Mapped[str] = mapped_column(String(192), index=True, nullable=False)
    bundle_item_id: Mapped[str] = mapped_column(String(192), index=True, nullable=False)
    entry_identity: Mapped[str] = mapped_column(String(192), index=True, nullable=False)
    document_identity: Mapped[str] = mapped_column(String(192), nullable=False)
    requested_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    editorial_source_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_strategy: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    embedding_configuration: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(48), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(48), nullable=False, default="queued")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    terminal_state: Mapped[str | None] = mapped_column(String(48), nullable=True)
    failure_reason: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    allowed_next_action: Mapped[str] = mapped_column(String(96), nullable=False, default="dispatch_candidate_build")
    candidate_id: Mapped[str | None] = mapped_column(String(192), nullable=True)
    derived_cleanup_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class CandidateBuildChunk(Base):
    __tablename__ = "candidate_build_chunks"
    __table_args__ = (
        UniqueConstraint("job_id", "attempt", "chunk_index", name="uq_candidate_build_chunks_job_attempt_index"),
        Index("ix_candidate_build_chunks_candidate", "candidate_id", "chunk_index"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    job_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    candidate_id: Mapped[str] = mapped_column(String(192), index=True, nullable=False)
    document_identity: Mapped[str] = mapped_column(String(192), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
