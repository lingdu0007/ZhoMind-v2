"""add Ticket 24 Candidate publication persistence

Revision ID: 20260908_t24_candidate_pub
Revises: 20260907_0020
Create Date: 2026-09-08

This branch is intentionally additive. Integrating parallel Ticket 22/23
migrations may require a merge migration or a revised down_revision.
"""

import sqlalchemy as sa

from alembic import op

revision = "20260908_t24_candidate_pub"
down_revision = "20260907_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "published_knowledge_versions",
        sa.Column("id", sa.String(length=192), nullable=False),
        sa.Column("candidate_id", sa.String(length=192), nullable=False),
        sa.Column("entry_identity", sa.String(length=192), nullable=False),
        sa.Column("document_identity", sa.String(length=192), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("frozen_input_sha256", sa.String(length=64), nullable=False),
        sa.Column("configuration_identity", sa.String(length=192), nullable=False),
        sa.Column("inspection_record_identity", sa.String(length=192), nullable=False),
        sa.Column("acceptance_record_identity", sa.String(length=192), nullable=False),
        sa.Column("supersedes_version_id", sa.String(length=192), nullable=True),
        sa.Column("published_by", sa.String(length=192), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("candidate_id", name="uq_published_knowledge_versions_candidate"),
    )
    op.create_index(
        "ix_published_knowledge_versions_entry",
        "published_knowledge_versions",
        ["entry_identity", "published_at"],
        unique=False,
    )
    op.create_table(
        "published_knowledge_pointers",
        sa.Column("entry_identity", sa.String(length=192), nullable=False),
        sa.Column("current_version_id", sa.String(length=192), nullable=False),
        sa.Column("document_identity", sa.String(length=192), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("entry_identity"),
        sa.UniqueConstraint("current_version_id"),
    )
    op.create_table(
        "candidate_publication_confirmations",
        sa.Column("id", sa.String(length=160), nullable=False),
        sa.Column("selection_sha256", sa.String(length=64), nullable=False),
        sa.Column("selected_items", sa.JSON(), nullable=False),
        sa.Column("actor_identity", sa.String(length=192), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("results", sa.JSON(), nullable=False),
        sa.Column("lease_owner", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("candidate_publication_confirmations")
    op.drop_table("published_knowledge_pointers")
    op.drop_index("ix_published_knowledge_versions_entry", table_name="published_knowledge_versions")
    op.drop_table("published_knowledge_versions")
