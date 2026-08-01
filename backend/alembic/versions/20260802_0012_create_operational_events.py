"""create operational events

Revision ID: 20260802_0012
Revises: 20260802_0011
Create Date: 2026-08-02
"""

from alembic import op
import sqlalchemy as sa


revision = "20260802_0012"
down_revision = "20260802_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operational_events",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("route_outcome", sa.String(length=128), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("gate_outcome", sa.String(length=16), nullable=True),
        sa.Column("provider_identity", sa.String(length=128), nullable=True),
        sa.Column("normalized_error", sa.String(length=128), nullable=True),
        sa.Column("candidate_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_operational_events_request_id", "operational_events", ["request_id"])
    op.create_index("ix_operational_events_created_at", "operational_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_operational_events_created_at", table_name="operational_events")
    op.drop_index("ix_operational_events_request_id", table_name="operational_events")
    op.drop_table("operational_events")
