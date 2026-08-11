"""create system settings drafts

Revision ID: 20260731_0007
Revises: 20260425_0006
Create Date: 2026-07-31
"""
import sqlalchemy as sa

from alembic import op

revision = "20260731_0007"
down_revision = "20260425_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_settings_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("latest_saved_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("active_version", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "system_settings_drafts",
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("settings", sa.JSON(), nullable=False),
        sa.Column("sealed_secrets", sa.JSON(), nullable=False),
        sa.Column("saved_by", sa.String(length=64), nullable=False),
        sa.Column("saved_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("version"),
    )


def downgrade() -> None:
    op.drop_table("system_settings_drafts")
    op.drop_table("system_settings_state")
