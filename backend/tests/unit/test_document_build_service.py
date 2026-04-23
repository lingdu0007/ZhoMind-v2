from __future__ import annotations

import asyncio
from types import MethodType

from app.documents.build_service import DocumentBuildService
from app.documents.job_dispatcher import DocumentJobDispatcher
from app.documents.types import ChunkRecord
from app.model.document import Document, DocumentJob


class _SessionSpy:
    def __init__(self) -> None:
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


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
