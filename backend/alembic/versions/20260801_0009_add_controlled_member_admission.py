"""add controlled member admission

Revision ID: 20260801_0009
Revises: 20260731_0008
Create Date: 2026-08-01
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260801_0009"
down_revision = "20260731_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.alter_column("users", "is_active", server_default=None)
    op.add_column(
        "users", sa.Column("is_bootstrap_administrator", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.alter_column("users", "is_bootstrap_administrator", server_default=None)
    op.create_table(
        "team_invitations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.UniqueConstraint("code_hash"),
    )
    op.create_index("ix_team_invitations_created_by_user_id", "team_invitations", ["created_by_user_id"])
    op.create_index("ix_team_invitations_expires_at", "team_invitations", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_team_invitations_expires_at", table_name="team_invitations")
    op.drop_index("ix_team_invitations_created_by_user_id", table_name="team_invitations")
    op.drop_table("team_invitations")
    op.drop_column("users", "is_bootstrap_administrator")
    op.drop_column("users", "is_active")
