import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config

from alembic import command
from app.common.config import get_settings


def test_canonical_immutability_tail_migration_rejects_database_rewrites(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "canonical-immutability.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
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
        INSERT INTO alembic_version VALUES ('20260905_0015');
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
        immutable_triggers = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'canonical_%_immutable_%'"
            )
        }
        revision = migrated.execute("SELECT version_num FROM alembic_version").fetchone()
        migrated.execute(
            """
            INSERT INTO canonical_records (
                stable_id, identity_kind, identity_value, state, record_class,
                schema_version, payload, created_at
            ) VALUES (
                'delivery_acceptance_record:migration-test',
                'delivery_acceptance_record',
                'migration-test',
                'active',
                'canonical',
                1,
                '{}',
                '2026-09-05 00:00:00'
            )
            """
        )
        migrated.execute(
            """
            INSERT INTO canonical_events (
                id, aggregate_id, aggregate_kind, event_type, to_state,
                payload, occurred_at
            ) VALUES (
                'migration-test-event',
                'delivery_acceptance_record:migration-test',
                'delivery_acceptance_record',
                'created',
                'active',
                '{}',
                '2026-09-05 00:00:00'
            )
            """
        )
        migrated.commit()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated.execute(
                "UPDATE canonical_records SET state = 'superseded' "
                "WHERE stable_id = 'delivery_acceptance_record:migration-test'"
            )
        migrated.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated.execute("DELETE FROM canonical_events WHERE id = 'migration-test-event'")
        migrated.rollback()
    finally:
        migrated.close()

    assert immutable_triggers == {
        "canonical_events_immutable_delete",
        "canonical_events_immutable_update",
        "canonical_records_immutable_delete",
        "canonical_records_immutable_update",
    }
    assert revision == ("20260905_0016",)
