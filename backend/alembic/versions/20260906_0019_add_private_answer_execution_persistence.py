"""persist private closed answer executions

Revision ID: 20260906_0019
Revises: 20260906_0018
Create Date: 2026-09-06
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260906_0019"
down_revision = "20260906_0018"
branch_labels = None
depends_on = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def _is_postgresql() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    return any(column["name"] == column_name for column in sa.inspect(op.get_bind()).get_columns(table_name))


def _has_index(table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in sa.inspect(op.get_bind()).get_indexes(table_name))


def upgrade() -> None:
    op.create_table(
        "answer_executions",
        sa.Column("id", sa.String(length=192), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("initial_state", sa.String(length=32), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_answer_executions_session_user_created",
        "answer_executions",
        ["session_id", "user_id", "created_at"],
        unique=False,
    )
    op.create_index("ix_answer_executions_session_id", "answer_executions", ["session_id"], unique=False)
    op.create_index("ix_answer_executions_user_id", "answer_executions", ["user_id"], unique=False)

    op.create_table(
        "answer_execution_events",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("execution_id", sa.String(length=192), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=48), nullable=False),
        sa.Column("from_state", sa.String(length=32), nullable=True),
        sa.Column("to_state", sa.String(length=32), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "execution_id",
            "sequence",
            name="uq_answer_execution_events_execution_sequence",
        ),
    )
    op.create_index(
        "ix_answer_execution_events_execution_occurred",
        "answer_execution_events",
        ["execution_id", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_answer_execution_events_execution_id",
        "answer_execution_events",
        ["execution_id"],
        unique=False,
    )

    if _has_table("chat_messages"):
        if not _has_column("chat_messages", "answer_execution_id"):
            with op.batch_alter_table("chat_messages") as batch:
                batch.add_column(sa.Column("answer_execution_id", sa.String(length=192), nullable=True))
        if not _has_index("chat_messages", "ix_chat_messages_answer_execution_id"):
            op.create_index(
                "ix_chat_messages_answer_execution_id",
                "chat_messages",
                ["answer_execution_id"],
                unique=False,
            )

    if _is_sqlite():
        op.execute(
            """
            CREATE TRIGGER answer_executions_immutable_update
            BEFORE UPDATE ON answer_executions
            BEGIN
                SELECT RAISE(ABORT, 'answer execution records are immutable while retained');
            END;
            """
        )
        op.execute(
            """
            CREATE TRIGGER answer_execution_events_immutable_update
            BEFORE UPDATE ON answer_execution_events
            BEGIN
                SELECT RAISE(ABORT, 'answer execution events are append-only while retained');
            END;
            """
        )
    elif _is_postgresql():
        op.execute(
            """
            CREATE OR REPLACE FUNCTION answer_executions_reject_update()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'answer execution records are immutable while retained';
            END;
            $$;
            """
        )
        op.execute(
            """
            CREATE OR REPLACE FUNCTION answer_execution_events_reject_update()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'answer execution events are append-only while retained';
            END;
            $$;
            """
        )
        op.execute(
            """
            CREATE TRIGGER answer_executions_immutable_update
            BEFORE UPDATE ON answer_executions
            FOR EACH ROW EXECUTE FUNCTION answer_executions_reject_update();
            """
        )
        op.execute(
            """
            CREATE TRIGGER answer_execution_events_immutable_update
            BEFORE UPDATE ON answer_execution_events
            FOR EACH ROW EXECUTE FUNCTION answer_execution_events_reject_update();
            """
        )


def downgrade() -> None:
    if _is_sqlite():
        op.execute("DROP TRIGGER IF EXISTS answer_execution_events_immutable_update")
        op.execute("DROP TRIGGER IF EXISTS answer_executions_immutable_update")
    elif _is_postgresql():
        op.execute("DROP TRIGGER IF EXISTS answer_execution_events_immutable_update ON answer_execution_events")
        op.execute("DROP TRIGGER IF EXISTS answer_executions_immutable_update ON answer_executions")
        op.execute("DROP FUNCTION IF EXISTS answer_execution_events_reject_update()")
        op.execute("DROP FUNCTION IF EXISTS answer_executions_reject_update()")

    if _has_table("chat_messages"):
        if _has_index("chat_messages", "ix_chat_messages_answer_execution_id"):
            op.drop_index("ix_chat_messages_answer_execution_id", table_name="chat_messages")
        if _has_column("chat_messages", "answer_execution_id"):
            with op.batch_alter_table("chat_messages") as batch:
                batch.drop_column("answer_execution_id")

    op.drop_index("ix_answer_execution_events_execution_id", table_name="answer_execution_events")
    op.drop_index(
        "ix_answer_execution_events_execution_occurred",
        table_name="answer_execution_events",
    )
    op.drop_table("answer_execution_events")

    op.drop_index("ix_answer_executions_user_id", table_name="answer_executions")
    op.drop_index("ix_answer_executions_session_id", table_name="answer_executions")
    op.drop_index(
        "ix_answer_executions_session_user_created",
        table_name="answer_executions",
    )
    op.drop_table("answer_executions")
