"""Add bounded non-content operational dimensions."""

import sqlalchemy as sa

from alembic import op

revision = "20260912_0028"
down_revision = "20260911_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("operational_events"):
        op.add_column("operational_events", sa.Column("dimensions", sa.JSON(none_as_null=True), nullable=True))


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("operational_events"):
        op.drop_column("operational_events", "dimensions")
