"""add document source content

Revision ID: 20260423_0004
Revises: 20260416_0003
Create Date: 2026-04-23
"""
from alembic import op
import sqlalchemy as sa


revision = "20260423_0004"
down_revision = "20260416_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_content", sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "source_content")
