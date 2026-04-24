"""add document publication lifecycle fields

Revision ID: 20260424_0005
Revises: 20260423_0004
Create Date: 2026-04-24
"""
from alembic import op
import sqlalchemy as sa


revision = "20260424_0005"
down_revision = "20260423_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("published_generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "documents",
        sa.Column("next_generation", sa.Integer(), nullable=False, server_default=sa.text("1")),
    )
    op.add_column(
        "documents",
        sa.Column("latest_requested_generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("documents", sa.Column("active_build_generation", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("active_build_job_id", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("active_build_heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    op.add_column("document_jobs", sa.Column("build_generation", sa.Integer(), nullable=True))
    op.add_column("document_jobs", sa.Column("requested_chunk_strategy", sa.String(length=32), nullable=True))

    op.add_column(
        "document_chunks",
        sa.Column("generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )

    op.execute(
        """
        UPDATE documents
        SET published_generation = CASE
            WHEN EXISTS (
                SELECT 1
                FROM document_chunks AS dc
                WHERE dc.document_id = documents.id
            ) THEN 1
            ELSE 0
        END
        """
    )
    op.execute(
        """
        UPDATE documents
        SET next_generation = CASE
                WHEN published_generation = 1 THEN 2
                ELSE 1
            END,
            latest_requested_generation = published_generation
        """
    )
    op.execute("UPDATE document_chunks SET generation = 1")

    op.drop_index("ix_documents_filename", table_name="documents")
    op.create_index(
        "ux_documents_filename_live",
        "documents",
        ["filename"],
        unique=True,
        sqlite_where=sa.text("deleted_at IS NULL"),
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    with op.batch_alter_table("document_jobs", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_document_jobs_document_generation",
            ["document_id", "build_generation"],
        )

    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.create_unique_constraint(
            "uq_document_chunks_document_generation_index",
            ["document_id", "generation", "chunk_index"],
        )
        batch_op.create_index(
            "ix_document_chunks_document_generation",
            ["document_id", "generation"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.drop_index("ix_document_chunks_document_generation")
        batch_op.drop_constraint("uq_document_chunks_document_generation_index", type_="unique")
        batch_op.drop_column("generation")

    with op.batch_alter_table("document_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint("uq_document_jobs_document_generation", type_="unique")
        batch_op.drop_column("requested_chunk_strategy")
        batch_op.drop_column("build_generation")

    op.drop_index("ux_documents_filename_live", table_name="documents")

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("active_build_heartbeat_at")
        batch_op.drop_column("active_build_job_id")
        batch_op.drop_column("active_build_generation")
        batch_op.drop_column("latest_requested_generation")
        batch_op.drop_column("next_generation")
        batch_op.drop_column("published_generation")

    op.create_index("ix_documents_filename", "documents", ["filename"], unique=True)
