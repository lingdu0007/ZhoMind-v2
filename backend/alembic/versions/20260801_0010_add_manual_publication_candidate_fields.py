"""add manual publication candidate fields

Revision ID: 20260801_0010
Revises: 20260801_0009
Create Date: 2026-08-01
"""
import sqlalchemy as sa

from alembic import op

revision = "20260801_0010"
down_revision = "20260801_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("candidate_generation", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("candidate_chunk_strategy", sa.String(length=32), nullable=True))
    op.add_column("documents", sa.Column("candidate_chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")))
    op.add_column("documents", sa.Column("candidate_dense_ready_generation", sa.Integer(), nullable=False, server_default=sa.text("0")))
    op.add_column("documents", sa.Column("candidate_dense_ready_fingerprint", sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "candidate_dense_ready_fingerprint")
    op.drop_column("documents", "candidate_dense_ready_generation")
    op.drop_column("documents", "candidate_chunk_count")
    op.drop_column("documents", "candidate_chunk_strategy")
    op.drop_column("documents", "candidate_generation")
