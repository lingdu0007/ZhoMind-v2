"""Separate expiring feedback references from immutable maintenance authority."""

import sqlalchemy as sa

from alembic import op

revision = "20260911_0027"
down_revision = "20260909_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "maintenance_signal_links",
        sa.Column("item_id", sa.String(192), sa.ForeignKey("canonical_records.stable_id"), primary_key=True),
        sa.Column("signal_id", sa.String(64), sa.ForeignKey("knowledge_feedback_signals.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_index("ix_maintenance_signal_links_signal_id", "maintenance_signal_links", ["signal_id"])


def downgrade() -> None:
    op.drop_index("ix_maintenance_signal_links_signal_id", table_name="maintenance_signal_links")
    op.drop_table("maintenance_signal_links")
