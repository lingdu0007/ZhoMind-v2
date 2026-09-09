import sqlite3

from alembic.config import Config

from alembic import command
from app.common.config import get_settings


def test_retention_upgrade_adds_only_non_content_cleanup_state(tmp_path, monkeypatch):
    path = tmp_path / "retention-migration.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
            INSERT INTO alembic_version VALUES ('20260908_merge_t22_t24');
            CREATE TABLE chat_sessions (id VARCHAR(64) PRIMARY KEY);
            INSERT INTO chat_sessions VALUES ('retained-private-session');
            CREATE TABLE knowledge_review_work_items (kind VARCHAR(32), normalized_metadata JSON);
            INSERT INTO knowledge_review_work_items VALUES ('feedback_signal', '{"note":"old-private-copy"}');
        """)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{path}")
    get_settings.cache_clear()
    try:
        command.upgrade(Config("alembic.ini"), "head")
    finally:
        get_settings.cache_clear()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM retention_cleanup_states").fetchone() == (0,)
        assert db.execute("SELECT id FROM chat_sessions").fetchone() == ("retained-private-session",)
        assert db.execute("SELECT normalized_metadata FROM knowledge_review_work_items").fetchone() == ("{}",)
        columns = {row[1] for row in db.execute("PRAGMA table_info(retention_cleanup_states)")}
        assert columns == {
            "data_class", "attempt", "policy_identity", "status", "checked_at", "deleted_count",
            "remaining_expired", "normalized_error",
        }
