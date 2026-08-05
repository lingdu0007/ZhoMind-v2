"""add conversation retention created at

Revision ID: 20260802_0011
Revises: 20260801_0010
Create Date: 2026-08-02
"""

import sqlalchemy as sa

from alembic import op

revision = "20260802_0011"
down_revision = "20260801_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("created_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE chat_sessions SET created_at = updated_at WHERE created_at IS NULL")
    with op.batch_alter_table("chat_sessions") as batch_op:
        batch_op.alter_column("created_at", nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("chat_sessions") as batch_op:
        batch_op.drop_column("created_at")
