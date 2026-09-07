"""allow entry-free closed insufficient-evidence feedback

Revision ID: 20260907_0020
Revises: 20260906_0019
Create Date: 2026-09-07
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260907_0020"
down_revision = "20260906_0019"
branch_labels = None
depends_on = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("knowledge_feedback_signals"):
        return

    with op.batch_alter_table("knowledge_feedback_signals") as batch:
        batch.add_column(sa.Column("scope_key", sa.String(length=96), nullable=True))

    op.execute(
        "UPDATE knowledge_feedback_signals "
        "SET scope_key = 'entry:' || entry_id "
        "WHERE scope_key IS NULL"
    )

    with op.batch_alter_table("knowledge_feedback_signals") as batch:
        batch.drop_constraint("uq_feedback_user_answer_entry", type_="unique")
        batch.alter_column(
            "entry_id",
            existing_type=sa.String(length=64),
            existing_nullable=False,
            nullable=True,
        )
        batch.alter_column(
            "knowledge_edition",
            existing_type=sa.String(length=128),
            existing_nullable=False,
            nullable=True,
        )
        batch.alter_column(
            "scope_key",
            existing_type=sa.String(length=96),
            existing_nullable=True,
            nullable=False,
        )
        batch.create_unique_constraint(
            "uq_feedback_user_answer_scope",
            ["user_id", "answer_id", "scope_key"],
        )


def downgrade() -> None:
    if not _has_table("knowledge_feedback_signals"):
        return

    op.execute(
        "DELETE FROM knowledge_review_work_items "
        "WHERE signal_id IN ("
        "SELECT id FROM knowledge_feedback_signals WHERE entry_id IS NULL"
        ")"
    )
    op.execute("DELETE FROM knowledge_feedback_signals WHERE entry_id IS NULL")

    with op.batch_alter_table("knowledge_feedback_signals") as batch:
        batch.drop_constraint("uq_feedback_user_answer_scope", type_="unique")
        batch.alter_column(
            "entry_id",
            existing_type=sa.String(length=64),
            existing_nullable=True,
            nullable=False,
        )
        batch.alter_column(
            "knowledge_edition",
            existing_type=sa.String(length=128),
            existing_nullable=True,
            nullable=False,
        )
        batch.drop_column("scope_key")
        batch.create_unique_constraint(
            "uq_feedback_user_answer_entry",
            ["user_id", "answer_id", "entry_id"],
        )
