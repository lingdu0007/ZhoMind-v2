"""Add independently observable non-content retention cleanup state."""

import sqlalchemy as sa

from alembic import op

revision = "20260909_0026"
down_revision = "20260908_merge_t22_t24"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retention_cleanup_states",
        sa.Column("data_class", sa.String(32), primary_key=True),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("policy_identity", sa.String(192), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_count", sa.Integer(), nullable=False),
        sa.Column("remaining_expired", sa.Integer(), nullable=True),
        sa.Column("normalized_error", sa.String(64), nullable=True),
    )
    if "knowledge_review_work_items" in sa.inspect(op.get_bind()).get_table_names():
        items = sa.table(
            "knowledge_review_work_items", sa.column("kind", sa.String()), sa.column("normalized_metadata", sa.JSON()),
        )
        op.execute(items.update().where(items.c.kind == "feedback_signal").values(normalized_metadata={}))


def downgrade() -> None:
    op.drop_table("retention_cleanup_states")
