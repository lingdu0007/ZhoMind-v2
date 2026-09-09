import json
import sqlite3
from pathlib import Path

import pytest
from alembic.config import Config

from alembic import command
from app.common.canonical_json import canonical_json_sha256
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
        candidate_job_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(candidate_build_jobs)")
        }
        candidate_chunk_columns = {
            row[1] for row in migrated.execute("PRAGMA table_info(candidate_build_chunks)")
        }
        candidate_job_indexes = {
            row[1] for row in migrated.execute("PRAGMA index_list(candidate_build_jobs)")
        }
        candidate_chunk_indexes = {
            row[1] for row in migrated.execute("PRAGMA index_list(candidate_build_chunks)")
        }
        publication_tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ("
                "'published_knowledge_versions', "
                "'published_knowledge_pointers', "
                "'candidate_publication_confirmations'"
                ")"
            )
        }
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
        migrated.execute(
            """
            INSERT INTO candidate_build_jobs (
                id, bundle_id, bundle_item_id, entry_identity, document_identity,
                requested_generation, editorial_source_revision, input_sha256, frozen_input_sha256,
                chunk_strategy, embedding_configuration, status, stage, progress,
                attempt, allowed_next_action, derived_cleanup_pending, created_at,
                updated_at
            ) VALUES (
                'candidate-migration-job',
                'bundle:migration-test',
                'bundle_item:migration-test',
                'entry:migration-test',
                'runtime-document:migration-test',
                1,
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                '{}',
                '{}',
                'queued',
                'queued',
                0,
                1,
                'dispatch_candidate_build',
                0,
                '2026-09-05 00:00:00',
                '2026-09-05 00:00:00'
            )
            """
        )
        migrated.execute(
            """
            INSERT INTO candidate_build_chunks (
                id, job_id, candidate_id, document_identity, generation, attempt,
                chunk_index, content, content_sha256, metadata, created_at
            ) VALUES (
                'candidate-migration-chunk',
                'candidate-migration-job',
                'candidate:migration-test',
                'runtime-document:migration-test',
                1,
                1,
                0,
                'immutable candidate chunk',
                'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
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
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                """
                INSERT INTO candidate_build_jobs (
                    id, bundle_id, bundle_item_id, entry_identity, document_identity,
                    requested_generation, editorial_source_revision, input_sha256, frozen_input_sha256,
                    chunk_strategy, embedding_configuration, status, stage, progress,
                    attempt, allowed_next_action, derived_cleanup_pending, created_at,
                    updated_at
                ) VALUES (
                    'candidate-migration-job-duplicate',
                    'bundle:migration-test-duplicate',
                    'bundle_item:migration-test-duplicate',
                    'entry:migration-test',
                    'runtime-document:migration-test',
                    1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                    '{}',
                    '{}',
                    'queued',
                    'queued',
                    0,
                    1,
                    'dispatch_candidate_build',
                    0,
                    '2026-09-05 00:00:00',
                    '2026-09-05 00:00:00'
                )
                """
            )
        migrated.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                """
                INSERT INTO candidate_build_chunks (
                    id, job_id, candidate_id, document_identity, generation, attempt,
                    chunk_index, content, content_sha256, metadata, created_at
                ) VALUES (
                    'candidate-migration-chunk-duplicate',
                    'candidate-migration-job',
                    'candidate:migration-test',
                    'runtime-document:migration-test',
                    1,
                    1,
                    0,
                    'immutable candidate chunk duplicate',
                    'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
                    '{}',
                    '2026-09-05 00:00:00'
                )
                """
            )
        migrated.rollback()
    finally:
        migrated.close()

    assert immutable_triggers == {
        "canonical_events_immutable_delete",
        "canonical_events_immutable_update",
        "canonical_records_immutable_delete",
        "canonical_records_immutable_update",
    }
    assert {
        "id",
        "bundle_id",
        "bundle_item_id",
        "entry_identity",
        "requested_generation",
        "frozen_input_sha256",
        "status",
        "stage",
        "attempt",
        "dispatched_at",
        "lease_owner",
        "lease_expires_at",
    }.issubset(candidate_job_columns)
    assert {
        "id",
        "job_id",
        "candidate_id",
        "generation",
        "attempt",
        "chunk_index",
        "content",
        "content_sha256",
        "metadata",
    }.issubset(candidate_chunk_columns)
    assert {
        "ix_candidate_build_jobs_bundle_id",
        "ix_candidate_build_jobs_bundle_item_id",
        "ix_candidate_build_jobs_entry_identity",
    }.issubset(candidate_job_indexes)
    assert {
        "ix_candidate_build_chunks_job_id",
        "ix_candidate_build_chunks_candidate_id",
        "ix_candidate_build_chunks_candidate",
    }.issubset(candidate_chunk_indexes)
    assert publication_tables == {
        "candidate_publication_confirmations",
        "published_knowledge_pointers",
        "published_knowledge_versions",
    }
    assert revision == ("20260909_0026",)


def test_frozen_candidate_input_hash_migration_backfills_existing_immutable_input(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "frozen-candidate-input.db"
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
        INSERT INTO alembic_version VALUES ('20260905_0015');
        """
    )
    connection.close()
    backend_directory = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{database_path}")
    get_settings.cache_clear()
    config = Config(str(backend_directory / "alembic.ini"))
    config.set_main_option("script_location", str(backend_directory / "alembic"))
    frozen_input = {
        "schema": "candidate_build_input/v1",
        "bundle_id": "bundle:migration-input",
        "bundle_sha256": "a" * 64,
        "bundle_item_id": "bundle_item:migration-input",
        "bundle_item_sha256": "b" * 64,
        "entry_identity": "entry:migration-input",
        "document_identity": "runtime-document:migration-input",
        "requested_generation": 1,
        "editorial_source_revision": "c" * 64,
        "input_sha256": "d" * 64,
        "chunk_strategy": {
            "strategy_id": "section-aware-900-120",
            "max_characters": 900,
            "overlap_characters": 120,
            "preserve_section_boundaries": True,
        },
        "embedding_configuration": {
            "schema": "candidate_embedding_configuration/v1",
            "active": False,
            "embedding_model": None,
            "dense_embedding_dim": 0,
            "fingerprint": "e" * 64,
        },
    }
    try:
        command.upgrade(config, "20260905_0017")
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                """
                INSERT INTO canonical_records (
                    stable_id, identity_kind, identity_value, state, record_class,
                    schema_version, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "build_generation:legacy-candidate-input",
                    "build_generation",
                    "legacy-candidate-input",
                    "queued",
                    "immutable",
                    1,
                    json.dumps(frozen_input, ensure_ascii=True, sort_keys=True),
                    "2026-09-06 00:00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO candidate_build_jobs (
                    id, bundle_id, bundle_item_id, entry_identity, document_identity,
                    requested_generation, editorial_source_revision, input_sha256,
                    chunk_strategy, embedding_configuration, status, stage, progress,
                    attempt, allowed_next_action, derived_cleanup_pending, created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-candidate-input",
                    frozen_input["bundle_id"],
                    frozen_input["bundle_item_id"],
                    frozen_input["entry_identity"],
                    frozen_input["document_identity"],
                    frozen_input["requested_generation"],
                    frozen_input["editorial_source_revision"],
                    frozen_input["input_sha256"],
                    json.dumps(frozen_input["chunk_strategy"], sort_keys=True),
                    json.dumps(frozen_input["embedding_configuration"], sort_keys=True),
                    "queued",
                    "queued",
                    0,
                    1,
                    "dispatch_candidate_build",
                    0,
                    "2026-09-06 00:00:00",
                    "2026-09-06 00:00:00",
                ),
            )
            connection.commit()
        finally:
            connection.close()

        command.upgrade(config, "head")
    finally:
        get_settings.cache_clear()

    migrated = sqlite3.connect(database_path)
    try:
        stored_hash = migrated.execute(
            "SELECT frozen_input_sha256 FROM candidate_build_jobs WHERE id = ?",
            ("legacy-candidate-input",),
        ).fetchone()
        stored_input = migrated.execute(
            "SELECT payload FROM canonical_records WHERE stable_id = ?",
            ("build_generation:legacy-candidate-input",),
        ).fetchone()
        revision = migrated.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        migrated.close()

    assert stored_hash == (canonical_json_sha256(frozen_input),)
    assert stored_input == (json.dumps(frozen_input, ensure_ascii=True, sort_keys=True),)
    assert revision == ("20260909_0026",)
