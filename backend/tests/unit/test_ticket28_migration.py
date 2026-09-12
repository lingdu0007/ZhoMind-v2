import sqlite3
from pathlib import Path

from alembic.config import Config

from alembic import command
from app.common.config import get_settings


def test_upgrade_preserves_existing_non_content_events_and_adds_nullable_dimensions(tmp_path, monkeypatch):
    path = tmp_path / "operational-upgrade.db"
    backend = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    get_settings.cache_clear()
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    try:
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TABLE operational_events (
                    id VARCHAR(64) PRIMARY KEY, request_id VARCHAR(64) NOT NULL,
                    route_outcome VARCHAR(128) NOT NULL, duration_ms INTEGER NOT NULL,
                    gate_outcome VARCHAR(16), provider_identity VARCHAR(128),
                    normalized_error VARCHAR(128), candidate_count INTEGER,
                    generation_route JSON, created_at DATETIME NOT NULL
                );
                CREATE INDEX ix_operational_events_request_id ON operational_events(request_id);
                CREATE INDEX ix_operational_events_created_at ON operational_events(created_at);
            """)
            assert "dimensions" not in {row[1] for row in connection.execute("PRAGMA table_info(operational_events)")}
            connection.execute("""
                INSERT INTO operational_events (id, request_id, route_outcome, duration_ms, created_at)
                VALUES ('retained-event', '52c05c25-2bcf-4463-909f-c009d98348de',
                        'POST /api/v1/chat:success', 123, '2026-09-12 00:00:00')
            """)
        command.stamp(config, "20260911_0027")
        command.upgrade(config, "head")
        with sqlite3.connect(path) as connection:
            assert connection.execute(
                "SELECT request_id, duration_ms, dimensions FROM operational_events WHERE id = 'retained-event'",
            ).fetchone() == ("52c05c25-2bcf-4463-909f-c009d98348de", 123, None)
            assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == ("20260912_0028",)
        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()
