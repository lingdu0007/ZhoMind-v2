"""add system settings application lifecycle

Revision ID: 20260731_0008
Revises: 20260731_0007
Create Date: 2026-07-31
"""

from alembic import op
import sqlalchemy as sa


revision = "20260731_0008"
down_revision = "20260731_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("system_settings_state") as batch_op:
        batch_op.add_column(sa.Column("application_state", sa.String(length=24), nullable=False, server_default="draft_only"))
        batch_op.add_column(sa.Column("application_version", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("application_actor", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("application_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("application_message", sa.String(length=256), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("system_settings_state") as batch_op:
        batch_op.drop_column("application_message")
        batch_op.drop_column("application_at")
        batch_op.drop_column("application_actor")
        batch_op.drop_column("application_version")
        batch_op.drop_column("application_state")
