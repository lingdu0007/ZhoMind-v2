from __future__ import annotations

import asyncio
import gc
import importlib.util
import os
from pathlib import Path
import tempfile
from types import MethodType
import warnings

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.exceptions import AppError
from app.documents.build_service import DocumentBuildService
from app.documents.job_dispatcher import DocumentJobDispatcher
from app.documents.types import ChunkRecord
from app.model.base import Base
from app.model.document import Document, DocumentChunk, DocumentJob


class _SessionSpy:
    def __init__(self) -> None:
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


def _load_publication_lifecycle_migration_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260424_0005_add_document_publication_lifecycle_fields.py"
    )
    spec = importlib.util.spec_from_file_location("migration_20260424_0005", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_lifecycle_fields_exist_on_document_models() -> None:
    document_mapper = inspect(Document)
    assert "published_generation" in document_mapper.columns
    assert "next_generation" in document_mapper.columns
    assert "latest_requested_generation" in document_mapper.columns
    assert "active_build_generation" in document_mapper.columns
    assert "active_build_job_id" in document_mapper.columns
    assert "active_build_heartbeat_at" in document_mapper.columns
    assert "deleted_at" in document_mapper.columns

    job_mapper = inspect(DocumentJob)
    assert "build_generation" in job_mapper.columns
    assert "requested_chunk_strategy" in job_mapper.columns

    chunk_mapper = inspect(DocumentChunk)
    assert "generation" in chunk_mapper.columns


def test_publication_lifecycle_metadata_defines_live_row_generation_rules() -> None:
    document_indexes = {index.name: index for index in Document.__table__.indexes}
    filename_index = document_indexes["ux_documents_filename_live"]
    assert filename_index.unique is True
    assert tuple(filename_index.columns.keys()) == ("filename",)
    assert str(filename_index.dialect_options["sqlite"]["where"]) == "deleted_at IS NULL"
    assert str(filename_index.dialect_options["postgresql"]["where"]) == "deleted_at IS NULL"

    job_constraints = {constraint.name: constraint for constraint in DocumentJob.__table__.constraints}
    job_generation_constraint = job_constraints["uq_document_jobs_document_generation"]
    assert tuple(job_generation_constraint.columns.keys()) == ("document_id", "build_generation")

    chunk_constraints = {constraint.name: constraint for constraint in DocumentChunk.__table__.constraints}
    chunk_generation_constraint = chunk_constraints["uq_document_chunks_document_generation_index"]
    assert tuple(chunk_generation_constraint.columns.keys()) == ("document_id", "generation", "chunk_index")

    chunk_indexes = {index.name: index for index in DocumentChunk.__table__.indexes}
    generation_index = chunk_indexes["ix_document_chunks_document_generation"]
    assert generation_index.unique is False
    assert tuple(generation_index.columns.keys()) == ("document_id", "generation")


def test_publication_lifecycle_migration_dedupes_legacy_duplicate_chunk_rows() -> None:
    migration = _load_publication_lifecycle_migration_module()
    assert hasattr(migration, "_dedupe_legacy_document_chunk_rows")

    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    document_chunks = sa.Table(
        "document_chunks",
        metadata,
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("generated_questions", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            document_chunks.insert(),
            [
                {
                    "id": "b-dup",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "content": "duplicate",
                    "keywords": [],
                    "generated_questions": [],
                    "metadata": {},
                    "generation": 0,
                },
                {
                    "id": "a-keep",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "content": "keep",
                    "keywords": [],
                    "generated_questions": [],
                    "metadata": {},
                    "generation": 0,
                },
                {
                    "id": "c-keep",
                    "document_id": "doc-1",
                    "chunk_index": 1,
                    "content": "neighbor",
                    "keywords": [],
                    "generated_questions": [],
                    "metadata": {},
                    "generation": 0,
                },
                {
                    "id": "d-keep",
                    "document_id": "doc-2",
                    "chunk_index": 0,
                    "content": "other-doc",
                    "keywords": [],
                    "generated_questions": [],
                    "metadata": {},
                    "generation": 0,
                },
            ],
        )

        migration._dedupe_legacy_document_chunk_rows(conn)

        remaining_rows = conn.execute(
            sa.select(document_chunks.c.id, document_chunks.c.document_id, document_chunks.c.chunk_index).order_by(
                document_chunks.c.document_id,
                document_chunks.c.chunk_index,
                document_chunks.c.id,
            )
        ).all()

    assert [tuple(row) for row in remaining_rows] == [
        ("a-keep", "doc-1", 0),
        ("c-keep", "doc-1", 1),
        ("d-keep", "doc-2", 0),
    ]


async def test_process_job_success_updates_document_and_job_status() -> None:
    session = _SessionSpy()
    service = DocumentBuildService(session)  # type: ignore[arg-type]

    document = Document(
        filename="guide.txt",
        file_type="txt",
        file_size=1200,
        status="processing",
        chunk_strategy="general",
    )
    document.id = "doc-1"

    job = DocumentJob(
        document_id=document.id,
        status="queued",
        stage="queued",
        progress=0,
        message="queued for rebuild",
    )
    job.id = "job-1"

    replaced: list[ChunkRecord] = []

    async def _fake_get_document(self: DocumentBuildService, document_id: str) -> Document:
        assert document_id == document.id
        return document

    async def _fake_get_job(self: DocumentBuildService, job_id: str) -> DocumentJob:
        assert job_id == job.id
        return job

    async def _fake_replace_chunks(self: DocumentBuildService, *, document_id: str, chunks: list[ChunkRecord]) -> None:
        assert document_id == document.id
        # Spec alignment: there is no separate indexing stage.
        assert job.stage == "chunking"
        replaced.extend(chunks)

    service._get_document = MethodType(_fake_get_document, service)
    service._get_job = MethodType(_fake_get_job, service)
    service._replace_document_chunks = MethodType(_fake_replace_chunks, service)

    await service.process_job(
        document_id=document.id,
        job_id=job.id,
        content=("A" * 1200).encode("utf-8"),
    )

    assert document.status == "ready"
    assert document.chunk_count == 2

    assert job.status == "succeeded"
    assert job.stage == "completed"
    assert job.progress == 100
    assert job.message == "document build completed"

    assert [chunk.chunk_index for chunk in replaced] == [0, 1]
    assert session.commit_calls == 1


async def test_process_job_missing_source_persists_app_error_message() -> None:
    session = _SessionSpy()
    service = DocumentBuildService(session)  # type: ignore[arg-type]

    document = Document(
        filename="legacy.txt",
        file_type="txt",
        file_size=12,
        source_content=None,
        status="pending",
        chunk_strategy="general",
    )
    document.id = "doc-missing"

    job = DocumentJob(
        document_id=document.id,
        status="queued",
        stage="queued",
        progress=0,
        message="queued for rebuild",
    )
    job.id = "job-missing"

    async def _fake_get_document(self: DocumentBuildService, document_id: str) -> Document:
        assert document_id == document.id
        return document

    async def _fake_get_job(self: DocumentBuildService, job_id: str) -> DocumentJob:
        assert job_id == job.id
        return job

    service._get_document = MethodType(_fake_get_document, service)
    service._get_job = MethodType(_fake_get_job, service)

    with pytest.raises(AppError) as exc_info:
        await service.process_job(document_id=document.id, job_id=job.id)

    assert exc_info.value.code == "DOC_SOURCE_CONTENT_MISSING"
    assert document.status == "failed"
    assert job.status == "failed"
    assert job.stage == "failed"
    assert job.progress == 10
    assert "DOC_SOURCE_CONTENT_MISSING" in job.message
    assert "document source content is missing" in job.message
    assert session.commit_calls == 1


async def test_get_document_defers_source_content_outside_build_load() -> None:
    db_fd, db_path = tempfile.mkstemp(prefix="build-service-", suffix=".db")
    os.close(db_fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_factory() as session:
            document = Document(
                filename="deferred.txt",
                file_type="txt",
                file_size=5,
                source_content=b"hello",
                status="ready",
                chunk_strategy="general",
            )
            session.add(document)
            await session.commit()

        async with session_factory() as session:
            ordinary_result = await session.execute(select(Document).where(Document.filename == "deferred.txt"))
            ordinary_document = ordinary_result.scalar_one()
            assert "source_content" in inspect(ordinary_document).unloaded

            service = DocumentBuildService(session)
            build_document = await service._get_document(ordinary_document.id)

            assert build_document.id == ordinary_document.id
            assert "source_content" not in inspect(build_document).unloaded
            assert build_document.source_content == b"hello"
    finally:
        await engine.dispose()
        os.remove(db_path)


async def test_dispatcher_enqueue_accepts_generic_contract() -> None:
    dispatcher = DocumentJobDispatcher()

    finished = asyncio.Event()
    calls: list[str] = []

    async def _job_coro() -> None:
        calls.append("ran")
        await asyncio.sleep(0)
        finished.set()

    task_id = await dispatcher.enqueue("job-1", _job_coro())

    assert task_id == "job-1"
    await asyncio.wait_for(finished.wait(), timeout=1)
    await asyncio.sleep(0)
    assert calls == ["ran"]

    assert await dispatcher.cancel("job-1") is False


async def test_dispatcher_duplicate_enqueue_does_not_close_or_run_new_awaitable() -> None:
    dispatcher = DocumentJobDispatcher()
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    first_done = asyncio.Event()

    async def _first_job() -> None:
        first_started.set()
        await first_release.wait()
        first_done.set()

    class _TrackedAwaitable:
        def __init__(self) -> None:
            self.closed = False
            self.awaited = False

        def __await__(self):
            self.awaited = True
            if False:
                yield None
            return None

        def close(self) -> None:
            self.closed = True

    tracked = _TrackedAwaitable()

    assert await dispatcher.enqueue("job-dup", _first_job()) == "job-dup"
    await asyncio.wait_for(first_started.wait(), timeout=1)

    assert await dispatcher.enqueue("job-dup", tracked) == "job-dup"
    assert tracked.closed is False
    assert tracked.awaited is False

    first_release.set()
    await asyncio.wait_for(first_done.wait(), timeout=1)
    await asyncio.sleep(0)


async def test_dispatcher_duplicate_enqueue_plain_coroutine_avoids_unconsumed_warning() -> None:
    dispatcher = DocumentJobDispatcher()
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    first_done = asyncio.Event()

    async def _first_job() -> None:
        first_started.set()
        await first_release.wait()
        first_done.set()

    async def _duplicate_job() -> None:
        await asyncio.sleep(0)

    first_coro = _first_job()
    assert await dispatcher.enqueue("job-dup-coro", first_coro) == "job-dup-coro"
    await asyncio.wait_for(first_started.wait(), timeout=1)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        duplicate_coro = _duplicate_job()
        assert await dispatcher.enqueue("job-dup-coro", duplicate_coro) == "job-dup-coro"

        del duplicate_coro
        gc.collect()
        await asyncio.sleep(0)

    never_awaited = [
        item
        for item in captured
        if issubclass(item.category, RuntimeWarning) and "was never awaited" in str(item.message)
    ]
    assert never_awaited == []

    assert await dispatcher.enqueue("job-dup-coro", first_coro) == "job-dup-coro"

    first_release.set()
    await asyncio.wait_for(first_done.wait(), timeout=1)
    await asyncio.sleep(0)


async def test_dispatcher_expected_app_error_does_not_escape_as_unretrieved_task_exception() -> None:
    dispatcher = DocumentJobDispatcher()
    loop = asyncio.get_running_loop()
    captured: list[dict[str, object]] = []
    original_handler = loop.get_exception_handler()

    def _capture_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        captured.append(context)

    finished = asyncio.Event()

    async def _failing_job() -> None:
        try:
            raise AppError(
                status_code=409,
                code="DOC_SOURCE_CONTENT_MISSING",
                message="document source content is missing",
            )
        finally:
            finished.set()

    loop.set_exception_handler(_capture_exception)
    try:
        assert await dispatcher.enqueue("job-fail", _failing_job()) == "job-fail"
        await asyncio.wait_for(finished.wait(), timeout=1)
        await asyncio.sleep(0)
        assert await dispatcher.cancel("job-fail") is False

        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(original_handler)

    unretrieved = [
        item
        for item in captured
        if item.get("message") == "Task exception was never retrieved"
    ]
    assert unretrieved == []
