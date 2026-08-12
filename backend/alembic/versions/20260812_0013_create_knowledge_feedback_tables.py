"""create knowledge feedback tables

Revision ID: 20260812_0013
Revises: 20260802_0012
Create Date: 2026-08-12
"""

import sqlalchemy as sa

from alembic import op

revision = "20260812_0013"
down_revision = "20260802_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_feedback_signals",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("answer_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("entry_id", sa.String(length=64), nullable=False),
        sa.Column("knowledge_edition", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("normalized_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "answer_id", "entry_id", name="uq_feedback_user_answer_entry"),
    )
    op.create_index("ix_knowledge_feedback_signals_answer_id", "knowledge_feedback_signals", ["answer_id"])
    op.create_index("ix_knowledge_feedback_signals_user_id", "knowledge_feedback_signals", ["user_id"])
    op.create_index("ix_knowledge_feedback_signals_entry_id", "knowledge_feedback_signals", ["entry_id"])
    op.create_index("ix_knowledge_feedback_signals_created_at", "knowledge_feedback_signals", ["created_at"])
    op.create_index("ix_knowledge_feedback_signals_expires_at", "knowledge_feedback_signals", ["expires_at"])

    op.create_table(
        "knowledge_review_work_items",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("dedupe_key", sa.String(length=192), nullable=False),
        sa.Column("subject_id", sa.String(length=64), nullable=False),
        sa.Column("signal_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("classification", sa.String(length=16), nullable=True),
        sa.Column("normalized_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("dedupe_key", name="uq_knowledge_review_work_item_dedupe"),
    )
    op.create_index("ix_knowledge_review_work_items_kind", "knowledge_review_work_items", ["kind"])
    op.create_index("ix_knowledge_review_work_items_subject_id", "knowledge_review_work_items", ["subject_id"])
    op.create_index("ix_knowledge_review_work_items_signal_id", "knowledge_review_work_items", ["signal_id"])
    op.create_index("ix_knowledge_review_work_items_status", "knowledge_review_work_items", ["status"])
    op.create_index("ix_knowledge_review_work_items_created_at", "knowledge_review_work_items", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_review_work_items_created_at", table_name="knowledge_review_work_items")
    op.drop_index("ix_knowledge_review_work_items_status", table_name="knowledge_review_work_items")
    op.drop_index("ix_knowledge_review_work_items_signal_id", table_name="knowledge_review_work_items")
    op.drop_index("ix_knowledge_review_work_items_subject_id", table_name="knowledge_review_work_items")
    op.drop_index("ix_knowledge_review_work_items_kind", table_name="knowledge_review_work_items")
    op.drop_table("knowledge_review_work_items")

    op.drop_index("ix_knowledge_feedback_signals_expires_at", table_name="knowledge_feedback_signals")
    op.drop_index("ix_knowledge_feedback_signals_created_at", table_name="knowledge_feedback_signals")
    op.drop_index("ix_knowledge_feedback_signals_entry_id", table_name="knowledge_feedback_signals")
    op.drop_index("ix_knowledge_feedback_signals_user_id", table_name="knowledge_feedback_signals")
    op.drop_index("ix_knowledge_feedback_signals_answer_id", table_name="knowledge_feedback_signals")
    op.drop_table("knowledge_feedback_signals")
