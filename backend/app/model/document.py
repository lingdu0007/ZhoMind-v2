import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Index, Integer, LargeBinary, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.model.base import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index(
            "ux_documents_filename_live",
            "filename",
            unique=True,
            sqlite_where=text("deleted_at IS NULL"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    source_content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    chunk_strategy: Mapped[str] = mapped_column(String(32), nullable=False, default="general")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    published_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dense_ready_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dense_ready_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    next_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    latest_requested_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_build_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active_build_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active_build_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class DocumentJob(Base):
    __tablename__ = "document_jobs"
    __table_args__ = (UniqueConstraint("document_id", "build_generation", name="uq_document_jobs_document_generation"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    document_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    build_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    requested_chunk_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "generation",
            "chunk_index",
            name="uq_document_chunks_document_generation_index",
        ),
        Index("ix_document_chunks_document_generation", "document_id", "generation"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    document_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    generated_questions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
