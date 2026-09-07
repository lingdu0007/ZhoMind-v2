import sqlite3

from alembic.config import Config

from alembic import command
from app.common.config import get_settings


def test_tail_upgrade_preserves_legacy_settings_without_granting_route_authority(tmp_path, monkeypatch):
    path = tmp_path / "route-migration.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
            INSERT INTO alembic_version VALUES ('20260907_0020');
            CREATE TABLE system_settings_state (id INTEGER PRIMARY KEY, active_version INTEGER);
            INSERT INTO system_settings_state VALUES (1, 7);
        """)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    get_settings.cache_clear()
    try:
        command.upgrade(Config("alembic.ini"), "head")
    finally:
        get_settings.cache_clear()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT active_identity FROM generation_route_state WHERE id=1").fetchone() == (None,)
        assert db.execute("SELECT active_version FROM system_settings_state WHERE id=1").fetchone() == (7,)
        assert db.execute("SELECT COUNT(*) FROM generation_route_secrets").fetchone() == (0,)
