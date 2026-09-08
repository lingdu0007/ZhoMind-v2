import sqlite3
from pathlib import Path

from alembic.config import Config

from alembic import command
from app.common.config import get_settings


def test_ticket14_tail_migration_fails_closed_for_legacy_active_invitations(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "ticket14-tail.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE users (
            id UUID NOT NULL PRIMARY KEY,
            username VARCHAR(64) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(16) NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            is_bootstrap_administrator BOOLEAN NOT NULL DEFAULT 0,
            created_at DATETIME,
            updated_at DATETIME
        );
        CREATE TABLE team_invitations (
            id UUID NOT NULL PRIMARY KEY,
            code_hash VARCHAR(64) NOT NULL UNIQUE,
            created_by_user_id UUID NOT NULL,
            expires_at DATETIME NOT NULL,
            revoked_at DATETIME,
            created_at DATETIME,
            FOREIGN KEY(created_by_user_id) REFERENCES users(id)
        );
        CREATE TABLE canonical_records (
            stable_id VARCHAR(192) NOT NULL PRIMARY KEY,
            identity_kind VARCHAR(48) NOT NULL,
            identity_value VARCHAR(160) NOT NULL,
            state VARCHAR(96) NOT NULL,
            record_class VARCHAR(32) NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            payload JSON NOT NULL,
            legacy_type VARCHAR(48),
            legacy_id VARCHAR(192),
            created_at DATETIME NOT NULL
        );
        CREATE TABLE canonical_events (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            aggregate_id VARCHAR(192) NOT NULL,
            aggregate_kind VARCHAR(48) NOT NULL,
            event_type VARCHAR(48) NOT NULL,
            from_state VARCHAR(96),
            to_state VARCHAR(96) NOT NULL,
            payload JSON NOT NULL,
            occurred_at DATETIME NOT NULL,
            recorded_by VARCHAR(192)
        );
        CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
        INSERT INTO users VALUES (
            '00000000000000000000000000000001',
            'legacy-admin',
            'hash',
            'admin',
            1,
            0,
            '2026-01-02 03:04:05',
            '2026-01-02 03:04:05'
        );
        INSERT INTO team_invitations VALUES (
            '00000000000000000000000000000011',
            'active-legacy',
            '00000000000000000000000000000001',
            '2099-01-02 03:04:05',
            NULL,
            '2026-01-02 03:04:05'
        );
        INSERT INTO team_invitations VALUES (
            '00000000000000000000000000000012',
            'revoked-legacy',
            '00000000000000000000000000000001',
            '2099-01-02 03:04:05',
            '2026-01-03 03:04:05',
            '2026-01-02 03:04:05'
        );
        INSERT INTO team_invitations VALUES (
            '00000000000000000000000000000013',
            'expired-legacy',
            '00000000000000000000000000000001',
            '2000-01-02 03:04:05',
            NULL,
            '2026-01-02 03:04:05'
        );
        INSERT INTO alembic_version VALUES ('20260904_0014');
        """
    )
    connection.close()

    backend_directory = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{database_path}")
    get_settings.cache_clear()
    config = Config(str(backend_directory / "alembic.ini"))
    config.set_main_option("script_location", str(backend_directory / "alembic"))
    try:
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()

    migrated = sqlite3.connect(database_path)
    try:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(team_invitations)")}
        user_indexes = {row[1] for row in migrated.execute("PRAGMA index_list(users)")}
        invitations = {
            row[0]: row[1:]
            for row in migrated.execute(
                "SELECT code_hash, consumed_at, consumed_by_user_id FROM team_invitations ORDER BY code_hash"
            )
        }
        tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        revision = migrated.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        migrated.close()

    assert {"consumed_at", "consumed_by_user_id"}.issubset(columns)
    assert "uq_users_bootstrap_administrator" in user_indexes
    assert invitations["active-legacy"] == ("2026-01-02 03:04:05", None)
    assert invitations["expired-legacy"] == (None, None)
    assert invitations["revoked-legacy"] == (None, None)
    assert "answer_executions" in tables
    assert "answer_execution_events" in tables
    assert "chat_messages" not in tables
    assert revision == ("20260908_merge_t22_t24",)
