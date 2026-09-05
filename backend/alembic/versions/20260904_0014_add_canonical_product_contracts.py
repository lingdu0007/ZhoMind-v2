"""add additive canonical product contract records

Revision ID: 20260904_0014
Revises: 20260812_0013
Create Date: 2026-09-04

The tables are deliberately separate from legacy runtime tables. Existing rows
remain readable through compatibility projections until later tickets migrate
their writers and readers.
"""

import sqlalchemy as sa

from alembic import op

revision = "20260904_0014"
down_revision = "20260812_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "canonical_records",
        sa.Column("stable_id", sa.String(length=192), primary_key=True, nullable=False),
        sa.Column("identity_kind", sa.String(length=48), nullable=False),
        sa.Column("identity_value", sa.String(length=160), nullable=False),
        sa.Column("state", sa.String(length=96), nullable=False),
        sa.Column("record_class", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("legacy_type", sa.String(length=48), nullable=True),
        sa.Column("legacy_id", sa.String(length=192), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_canonical_records_kind_state",
        "canonical_records",
        ["identity_kind", "state"],
    )
    op.create_index(
        "ix_canonical_records_legacy",
        "canonical_records",
        ["legacy_type", "legacy_id"],
    )

    op.create_table(
        "canonical_events",
        sa.Column("id", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("aggregate_id", sa.String(length=192), nullable=False),
        sa.Column("aggregate_kind", sa.String(length=48), nullable=False),
        sa.Column("event_type", sa.String(length=48), nullable=False),
        sa.Column("from_state", sa.String(length=96), nullable=True),
        sa.Column("to_state", sa.String(length=96), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_by", sa.String(length=192), nullable=True),
    )
    op.create_index(
        "ix_canonical_events_aggregate",
        "canonical_events",
        ["aggregate_id", "occurred_at"],
    )
    op.create_index(
        "ix_canonical_events_type",
        "canonical_events",
        ["event_type", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_canonical_events_type", table_name="canonical_events")
    op.drop_index("ix_canonical_events_aggregate", table_name="canonical_events")
    op.drop_table("canonical_events")
    op.drop_index("ix_canonical_records_legacy", table_name="canonical_records")
    op.drop_index("ix_canonical_records_kind_state", table_name="canonical_records")
    op.drop_table("canonical_records")
