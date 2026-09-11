import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config

from alembic import command
from app.common.config import get_settings


@pytest.mark.parametrize("prior_head", ["20260908_0021", "20260908_t24_candidate_pub"])
def test_answer_execution_tail_migration_adds_private_append_only_persistence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    prior_head: str,
) -> None:
    database_path = tmp_path / "answer-execution-tail.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE chat_sessions (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        );
        CREATE TABLE chat_messages (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            type VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            rag_trace JSON,
            created_at DATETIME NOT NULL
        );
        CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
        INSERT INTO alembic_version VALUES ('20260906_0018');
        """
    )
    connection.close()

    backend_directory = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{database_path}")
    get_settings.cache_clear()
    config = Config(str(backend_directory / "alembic.ini"))
    config.set_main_option("script_location", str(backend_directory / "alembic"))
    try:
        command.upgrade(config, prior_head)
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()

    migrated = sqlite3.connect(database_path)
    try:
        execution_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(answer_executions)")
        }
        event_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(answer_execution_events)")
        }
        message_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(chat_messages)")
        }
        execution_indexes = {
            row[1] for row in migrated.execute("PRAGMA index_list(answer_executions)")
        }
        event_indexes = {
            row[1] for row in migrated.execute("PRAGMA index_list(answer_execution_events)")
        }
        message_indexes = {
            row[1] for row in migrated.execute("PRAGMA index_list(chat_messages)")
        }
        tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        triggers = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'answer_execution%immutable_update'"
            )
        }

        migrated.execute(
            """
            INSERT INTO answer_executions (
                id, session_id, user_id, initial_state, request, created_at
            ) VALUES (
                'answer_execution:migration-test',
                'migration-session',
                'knowledge-user',
                'admitted',
                '{}',
                '2026-09-06 00:00:00'
            )
            """
        )
        migrated.execute(
            """
            INSERT INTO answer_execution_events (
                id, execution_id, sequence, event_type, from_state, to_state, payload, occurred_at
            ) VALUES (
                'migration-event',
                'answer_execution:migration-test',
                1,
                'created',
                NULL,
                'admitted',
                '{}',
                '2026-09-06 00:00:00'
            )
            """
        )
        migrated.commit()
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                """
                INSERT INTO answer_execution_events (
                    id, execution_id, sequence, event_type, from_state, to_state, payload, occurred_at
                ) VALUES (
                    'migration-duplicate-sequence',
                    'answer_execution:migration-test',
                    1,
                    'state_changed',
                    'admitted',
                    'queued',
                    '{}',
                    '2026-09-06 00:00:01'
                )
                """
            )
        migrated.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated.execute(
                "UPDATE answer_executions SET initial_state = 'queued' "
                "WHERE id = 'answer_execution:migration-test'"
            )
        migrated.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable|append-only"):
            migrated.execute(
                "UPDATE answer_execution_events SET event_type = 'tampered' "
                "WHERE id = 'migration-event'"
            )
        migrated.rollback()
        migrated.execute("DELETE FROM answer_execution_events WHERE id = 'migration-event'")
        migrated.execute("DELETE FROM answer_executions WHERE id = 'answer_execution:migration-test'")
        migrated.commit()
        revision = migrated.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        migrated.close()

    assert {
        "id",
        "session_id",
        "user_id",
        "initial_state",
        "request",
        "created_at",
    }.issubset(execution_columns)
    assert {
        "id",
        "execution_id",
        "sequence",
        "event_type",
        "from_state",
        "to_state",
        "payload",
        "occurred_at",
    }.issubset(event_columns)
    assert "answer_execution_id" in message_columns
    assert "ix_answer_executions_session_user_created" in execution_indexes
    assert "ix_answer_execution_events_execution_occurred" in event_indexes
    assert "ix_chat_messages_answer_execution_id" in message_indexes
    assert triggers == {
        "answer_execution_events_immutable_update",
        "answer_executions_immutable_update",
    }
    assert "knowledge_feedback_signals" not in tables
    assert {
        "generation_route_secrets",
        "generation_route_state",
        "published_knowledge_versions",
        "published_knowledge_pointers",
        "candidate_publication_confirmations",
    }.issubset(tables)
    assert revision == ("20260911_0027",)


def test_answer_execution_migration_defines_postgresql_immutable_update_guards() -> None:
    backend_directory = Path(__file__).resolve().parents[2]
    migration = (
        backend_directory
        / "alembic"
        / "versions"
        / "20260906_0019_add_private_answer_execution_persistence.py"
    ).read_text(encoding="utf-8")

    assert "def _is_postgresql()" in migration
    assert "CREATE OR REPLACE FUNCTION answer_executions_reject_update()" in migration
    assert "CREATE OR REPLACE FUNCTION answer_execution_events_reject_update()" in migration
    assert "CREATE TRIGGER answer_executions_immutable_update" in migration
    assert "CREATE TRIGGER answer_execution_events_immutable_update" in migration
    assert "DROP FUNCTION IF EXISTS answer_executions_reject_update()" in migration
    assert "DROP FUNCTION IF EXISTS answer_execution_events_reject_update()" in migration
