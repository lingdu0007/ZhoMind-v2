"""add dense retrieval fields

Revision ID: 20260425_0006
Revises: 20260424_0005
Create Date: 2026-04-25
"""
from __future__ import annotations

import hashlib

from alembic import op
import sqlalchemy as sa


revision = "20260425_0006"
down_revision = "20260424_0005"
branch_labels = None
depends_on = None


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _backfill_dense_retrieval_fields(bind) -> None:
    documents = sa.table(
        "documents",
        sa.column("id", sa.String(length=64)),
        sa.column("dense_ready_generation", sa.Integer()),
        sa.column("dense_ready_fingerprint", sa.String(length=128)),
    )
    document_chunks = sa.table(
        "document_chunks",
        sa.column("id", sa.String(length=64)),
        sa.column("content", sa.Text()),
        sa.column("content_sha256", sa.String(length=64)),
    )

    bind.execute(
        sa.update(documents).values(
            dense_ready_generation=0,
            dense_ready_fingerprint=None,
        )
    )

    chunk_rows = bind.execute(sa.select(document_chunks.c.id, document_chunks.c.content)).all()
    update_statement = (
        sa.update(document_chunks)
        .where(document_chunks.c.id == sa.bindparam("target_id"))
        .values(content_sha256=sa.bindparam("content_sha256"))
    )
    bind.execute(
        update_statement,
        [
            {
                "target_id": row.id,
                "content_sha256": _sha256_text(row.content or ""),
            }
            for row in chunk_rows
        ],
    )


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("dense_ready_generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "documents",
        sa.Column("dense_ready_fingerprint", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("content_sha256", sa.String(length=64), nullable=False, server_default=sa.text("''")),
    )

    _backfill_dense_retrieval_fields(op.get_bind())


def downgrade() -> None:
    with op.batch_alter_table("document_chunks", recreate="always") as batch_op:
        batch_op.drop_column("content_sha256")

    with op.batch_alter_table("documents", recreate="always") as batch_op:
        batch_op.drop_column("dense_ready_fingerprint")
        batch_op.drop_column("dense_ready_generation")
