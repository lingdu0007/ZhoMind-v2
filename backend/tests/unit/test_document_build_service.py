from __future__ import annotations

import asyncio
import gc
from types import MethodType
import warnings

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
