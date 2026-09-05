"""create reviewed release bundle Candidate Build tables

Revision ID: 20260905_0017
Revises: 20260905_0016
Create Date: 2026-09-05
"""

import sqlalchemy as sa

from alembic import op

revision = "20260905_0017"
down_revision = "20260905_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidate_build_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("bundle_id", sa.String(length=192), nullable=False),
        sa.Column("bundle_item_id", sa.String(length=192), nullable=False),
        sa.Column("entry_identity", sa.String(length=192), nullable=False),
        sa.Column("document_identity", sa.String(length=192), nullable=False),
        sa.Column("requested_generation", sa.Integer(), nullable=False),
        sa.Column("editorial_source_revision", sa.String(length=64), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("chunk_strategy", sa.JSON(), nullable=False),
        sa.Column("embedding_configuration", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=48), nullable=False),
        sa.Column("stage", sa.String(length=48), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("terminal_state", sa.String(length=48), nullable=True),
        sa.Column("failure_reason", sa.JSON(), nullable=True),
        sa.Column("allowed_next_action", sa.String(length=96), nullable=False),
        sa.Column("candidate_id", sa.String(length=192), nullable=True),
        sa.Column("derived_cleanup_pending", sa.Boolean(), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entry_identity", "requested_generation", name="uq_candidate_build_jobs_entry_generation"),
    )
    op.create_index("ix_candidate_build_jobs_bundle_id", "candidate_build_jobs", ["bundle_id"], unique=False)
    op.create_index("ix_candidate_build_jobs_bundle_item_id", "candidate_build_jobs", ["bundle_item_id"], unique=False)
    op.create_index("ix_candidate_build_jobs_entry_identity", "candidate_build_jobs", ["entry_identity"], unique=False)

    op.create_table(
        "candidate_build_chunks",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("candidate_id", sa.String(length=192), nullable=False),
        sa.Column("document_identity", sa.String(length=192), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "attempt", "chunk_index", name="uq_candidate_build_chunks_job_attempt_index"),
    )
    op.create_index("ix_candidate_build_chunks_job_id", "candidate_build_chunks", ["job_id"], unique=False)
    op.create_index("ix_candidate_build_chunks_candidate_id", "candidate_build_chunks", ["candidate_id"], unique=False)
    op.create_index(
        "ix_candidate_build_chunks_candidate",
        "candidate_build_chunks",
        ["candidate_id", "chunk_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_candidate_build_chunks_candidate", table_name="candidate_build_chunks")
    op.drop_index("ix_candidate_build_chunks_candidate_id", table_name="candidate_build_chunks")
    op.drop_index("ix_candidate_build_chunks_job_id", table_name="candidate_build_chunks")
    op.drop_table("candidate_build_chunks")
    op.drop_index("ix_candidate_build_jobs_entry_identity", table_name="candidate_build_jobs")
    op.drop_index("ix_candidate_build_jobs_bundle_item_id", table_name="candidate_build_jobs")
    op.drop_index("ix_candidate_build_jobs_bundle_id", table_name="candidate_build_jobs")
    op.drop_table("candidate_build_jobs")
