"""Retain protected credentials and an explicit approved-route pointer."""

import sqlalchemy as sa

from alembic import op

revision = "20260908_0021"
down_revision = "20260907_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("operational_events"):
        op.add_column("operational_events", sa.Column("generation_route", sa.JSON(none_as_null=True), nullable=True))
    op.create_table(
        "generation_route_secrets",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("ciphertext", sa.String(2048), nullable=False),
    )
    state = op.create_table(
        "generation_route_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("draft_identity", sa.String(192), nullable=True),
        sa.Column("active_identity", sa.String(192), nullable=True),
    )
    op.bulk_insert(state, [{"id": 1, "version": 0, "draft_identity": None, "active_identity": None}])


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("operational_events"):
        op.drop_column("operational_events", "generation_route")
    op.drop_table("generation_route_state")
    op.drop_table("generation_route_secrets")
