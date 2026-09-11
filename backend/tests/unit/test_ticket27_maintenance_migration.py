import sqlite3

from alembic.config import Config

from alembic import command
from app.common.config import get_settings


def test_upgrade_adds_deletable_links_without_rewriting_authority(tmp_path, monkeypatch):
    path = tmp_path / "maintenance.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
            INSERT INTO alembic_version VALUES ('20260909_0026');
            CREATE TABLE canonical_records (stable_id VARCHAR(192) PRIMARY KEY, payload JSON);
            INSERT INTO canonical_records VALUES ('maintenance_item:retained', '{"original":true}');
            CREATE TABLE knowledge_feedback_signals (id VARCHAR(64) PRIMARY KEY);
            INSERT INTO knowledge_feedback_signals VALUES ('raw-signal');
        """)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    get_settings.cache_clear()
    try:
        command.upgrade(Config("alembic.ini"), "head")
    finally:
        get_settings.cache_clear()
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        assert db.execute("SELECT COUNT(*) FROM maintenance_signal_links").fetchone() == (0,)
        db.execute("INSERT INTO maintenance_signal_links VALUES ('maintenance_item:retained', 'raw-signal')")
        db.execute("DELETE FROM knowledge_feedback_signals WHERE id='raw-signal'")
        assert db.execute("SELECT COUNT(*) FROM maintenance_signal_links").fetchone() == (0,)
        assert db.execute("SELECT payload FROM canonical_records").fetchone() == ('{"original":true}',)
