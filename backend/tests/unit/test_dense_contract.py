from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect

from app.common.config import Settings
from app.model.document import Document, DocumentChunk


def _load_dense_contract_module():
    try:
        return importlib.import_module("app.rag.dense_contract")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing dense contract module: {exc}")


def _load_dense_retrieval_migration_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260425_0006_add_dense_retrieval_fields.py"
    )
    assert module_path.exists(), f"missing migration file: {module_path}"

    spec = importlib.util.spec_from_file_location("migration_20260425_0006", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dense_ready_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "EMBEDDING_API_KEY": "emb-key",
        "EMBEDDING_BASE_URL": "https://emb.example.com/v1/",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "DENSE_EMBEDDING_DIM": 1536,
        "MILVUS_URI": "http://milvus.example.com:19530",
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"EMBEDDING_API_KEY": ""}, "missing api key"),
        ({"EMBEDDING_BASE_URL": ""}, "missing base url"),
        ({"EMBEDDING_MODEL": ""}, "missing embedding model"),
        ({"DENSE_EMBEDDING_DIM": 0}, "non-positive dimension"),
        ({"DENSE_EMBEDDING_DIM": -1}, "negative dimension"),
        ({"MILVUS_URI": ""}, "missing milvus uri"),
    ],
)
def test_dense_mode_active_requires_all_dense_inputs(
    overrides: dict[str, object],
    reason: str,
) -> None:
    dense_contract = _load_dense_contract_module()

    settings = _dense_ready_settings(**overrides)

    assert dense_contract.dense_mode_active(settings) is False, reason
    assert dense_contract.DenseEmbeddingContract.from_settings(settings).active is False


def test_dense_mode_active_when_required_inputs_present() -> None:
    dense_contract = _load_dense_contract_module()

    settings = _dense_ready_settings()

    assert dense_contract.dense_mode_active(settings) is True

    contract = dense_contract.DenseEmbeddingContract.from_settings(settings)
    assert contract.active is True
    assert contract.base_url == "https://emb.example.com/v1"
    assert contract.model == "text-embedding-3-large"
    assert contract.dimension == 1536


def test_embedding_contract_fingerprint_depends_only_on_base_url_model_and_dimension() -> None:
    dense_contract = _load_dense_contract_module()

    baseline = _dense_ready_settings()
    rotated_credentials = _dense_ready_settings(
        EMBEDDING_API_KEY="rotated-key",
        MILVUS_URI="http://other-milvus.example.com:19530",
        MILVUS_TOKEN="secret-token",
    )
    changed_contract = _dense_ready_settings(EMBEDDING_MODEL="text-embedding-3-small")

    baseline_fingerprint = dense_contract.build_embedding_contract_fingerprint(baseline)
    rotated_fingerprint = dense_contract.build_embedding_contract_fingerprint(rotated_credentials)
    changed_fingerprint = dense_contract.build_embedding_contract_fingerprint(changed_contract)

    assert baseline_fingerprint == rotated_fingerprint
    assert baseline_fingerprint != changed_fingerprint


def test_milvus_collection_name_is_deterministic_for_fingerprint() -> None:
    dense_contract = _load_dense_contract_module()

    fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    collection_name = dense_contract.build_milvus_collection_name(fingerprint)

    assert collection_name == dense_contract.build_milvus_collection_name(fingerprint)
    assert collection_name.startswith("document_chunks_")
    assert collection_name == f"document_chunks_{fingerprint}"


def test_dense_schema_fields_exist_on_document_models() -> None:
    document_mapper = inspect(Document)
    assert "dense_ready_generation" in document_mapper.columns
    assert "dense_ready_fingerprint" in document_mapper.columns

    chunk_mapper = inspect(DocumentChunk)
    assert "content_sha256" in chunk_mapper.columns


def test_dense_retrieval_migration_backfills_legacy_rows() -> None:
    migration = _load_dense_retrieval_migration_module()
    assert hasattr(migration, "_backfill_dense_retrieval_fields")

    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    documents = sa.Table(
        "documents",
        metadata,
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("dense_ready_generation", sa.Integer(), nullable=True),
        sa.Column("dense_ready_fingerprint", sa.String(length=128), nullable=True),
    )
    document_chunks = sa.Table(
        "document_chunks",
        metadata,
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            documents.insert(),
            [
                {
                    "id": "doc-1",
                    "dense_ready_generation": None,
                    "dense_ready_fingerprint": "legacy-fingerprint",
                }
            ],
        )
        conn.execute(
            document_chunks.insert(),
            [
                {
                    "id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Alpha chunk",
                    "content_sha256": None,
                },
                {
                    "id": "chunk-2",
                    "document_id": "doc-1",
                    "content": "Beta chunk",
                    "content_sha256": None,
                },
            ],
        )

        migration._backfill_dense_retrieval_fields(conn)

        documents_row = conn.execute(
            sa.select(
                documents.c.dense_ready_generation,
                documents.c.dense_ready_fingerprint,
            )
        ).one()
        chunk_rows = conn.execute(
            sa.select(document_chunks.c.id, document_chunks.c.content_sha256).order_by(document_chunks.c.id.asc())
        ).all()

    assert documents_row.dense_ready_generation == 0
    assert documents_row.dense_ready_fingerprint is None
    assert chunk_rows == [
        ("chunk-1", hashlib.sha256("Alpha chunk".encode("utf-8")).hexdigest()),
        ("chunk-2", hashlib.sha256("Beta chunk".encode("utf-8")).hexdigest()),
    ]


def test_dense_retrieval_migration_handles_empty_document_chunks() -> None:
    migration = _load_dense_retrieval_migration_module()

    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    documents = sa.Table(
        "documents",
        metadata,
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("dense_ready_generation", sa.Integer(), nullable=True),
        sa.Column("dense_ready_fingerprint", sa.String(length=128), nullable=True),
    )
    document_chunks = sa.Table(
        "document_chunks",
        metadata,
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            documents.insert(),
            [
                {
                    "id": "doc-1",
                    "dense_ready_generation": 9,
                    "dense_ready_fingerprint": "stale-fingerprint",
                }
            ],
        )

        migration._backfill_dense_retrieval_fields(conn)

        documents_row = conn.execute(
            sa.select(
                documents.c.dense_ready_generation,
                documents.c.dense_ready_fingerprint,
            )
        ).one()
        chunk_count = conn.execute(sa.select(sa.func.count()).select_from(document_chunks)).scalar_one()

    assert documents_row.dense_ready_generation == 0
    assert documents_row.dense_ready_fingerprint is None
    assert chunk_count == 0


def test_document_chunk_defaults_content_sha256_from_content() -> None:
    chunk = DocumentChunk(
        document_id="doc-1",
        generation=1,
        chunk_index=0,
        content="abc",
    )

    assert chunk.content_sha256 == hashlib.sha256("abc".encode("utf-8")).hexdigest()
