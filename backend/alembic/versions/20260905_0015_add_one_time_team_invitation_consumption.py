"""enforce one-time invitation consumption and one bootstrap administrator

Revision ID: 20260905_0015
Revises: 20260904_0014
Create Date: 2026-09-05
"""

import sqlalchemy as sa

from alembic import op

revision = "20260905_0015"
down_revision = "20260904_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "uq_users_bootstrap_administrator",
        "users",
        ["is_bootstrap_administrator"],
        unique=True,
        postgresql_where=sa.text("is_bootstrap_administrator = true"),
        sqlite_where=sa.text("is_bootstrap_administrator = 1"),
    )
    with op.batch_alter_table("team_invitations") as batch:
        batch.add_column(sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("consumed_by_user_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_team_invitations_consumed_by_user_id_users",
            "users",
            ["consumed_by_user_id"],
            ["id"],
        )
        batch.create_index("ix_team_invitations_consumed_by_user_id", ["consumed_by_user_id"])
    op.execute(
        sa.text(
            """
            UPDATE team_invitations
            SET consumed_at = COALESCE(created_at, CURRENT_TIMESTAMP)
            WHERE consumed_at IS NULL
              AND revoked_at IS NULL
              AND expires_at > CURRENT_TIMESTAMP
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("team_invitations") as batch:
        batch.drop_index("ix_team_invitations_consumed_by_user_id")
        batch.drop_constraint("fk_team_invitations_consumed_by_user_id_users", type_="foreignkey")
        batch.drop_column("consumed_by_user_id")
        batch.drop_column("consumed_at")
    op.drop_index("uq_users_bootstrap_administrator", table_name="users")
