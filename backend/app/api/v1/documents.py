import asyncio
from collections.abc import Awaitable
from concurrent.futures import Future
from pathlib import Path
from contextlib import suppress
import threading
from typing import TypeVar

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.common.deps import require_admin
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.documents.build_service import DocumentBuildService
from app.documents.job_dispatcher import DocumentJobDispatcher
from app.documents.schemas import BatchBuildRequest, BatchDeleteRequest, BuildDocumentRequest
from app.extensions.registry import get_task_backend
from app.infra.db import get_db_session
from app.model.document import Document, DocumentChunk, DocumentJob

router = APIRouter(prefix="/documents", tags=["documents"])
_job_dispatcher = DocumentJobDispatcher()
_T = TypeVar("_T")


class _DispatcherLoop:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    def _ensure_running(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._thread = threading.Thread(target=self._run, name="document-job-loop", daemon=True)
                self._thread.start()
        self._ready.wait(timeout=2)
        if self._loop is None:
            raise RuntimeError("document job loop unavailable")
        return self._loop

    def submit(self, coro: Awaitable[_T]) -> _T:
        loop = self._ensure_running()
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


_dispatcher_loop = _DispatcherLoop()


def _ok(data: dict) -> dict:
    payload = ok_response(data=data, request_id=get_request_id())
    payload.update(data)
    return payload


def _validate_pagination(page: int, page_size: int) -> tuple[int, int]:
    if page < 1 or page_size < 1:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="page and page_size must be positive")
    return page, min(page_size, 200)


def _serialize_document(document: Document) -> dict:
    return {
        "document_id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "file_size": document.file_size,
        "status": document.status,
        "chunk_strategy": document.chunk_strategy,
        "chunk_count": document.chunk_count,
        "uploaded_at": document.uploaded_at.isoformat(),
    }


def _serialize_job(job: DocumentJob) -> dict:
    return {
        "job_id": job.id,
        "document_id": job.document_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
        "updated_at": job.updated_at.isoformat(),
    }


def _serialize_chunk(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "keywords": chunk.keywords,
        "generated_questions": chunk.generated_questions,
        "metadata": chunk.chunk_metadata,
    }


def _validate_supported_upload_file_type(file_type: str) -> None:
    allowed = set(get_settings().document_allowed_extensions)
    if file_type not in allowed:
        raise AppError(
            status_code=415,
            code="DOC_FILE_TYPE_NOT_SUPPORTED",
            message="document file type not supported",
            detail={"file_type": file_type},
        )


async def _get_document_or_404(session: AsyncSession, document_id: str) -> Document:
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="document not found")
    return document


async def _create_job(
    session: AsyncSession,
    document_id: str,
    *,
    status: str,
    stage: str,
    progress: int,
    message: str,
) -> DocumentJob:
    job = DocumentJob(
        document_id=document_id,
        status=status,
        stage=stage,
        progress=progress,
        message=message,
    )
    session.add(job)
    await session.flush()
    return job


async def _enqueue_document_task(name: str, payload: dict) -> str:
    backend = get_task_backend("inmemory")
    return await backend.enqueue(name=name, payload=payload)


def _build_document_runner(
    *,
    bind_url: str,
    document_id: str,
    job_id: str,
    content: bytes | None = None,
    gate: threading.Event | None = None,
) -> Awaitable[None]:
    async def _runner() -> None:
        if gate is not None:
            while not gate.is_set():
                await asyncio.sleep(0.005)

        engine = create_async_engine(bind_url)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with session_factory() as background_session:
                service = DocumentBuildService(background_session)
                await service.process_job(document_id=document_id, job_id=job_id, content=content)
        finally:
            await engine.dispose()

    return _runner()


def _close_runner(runner: Awaitable[None]) -> None:
    close = getattr(runner, "close", None)
    if callable(close):
        close()


def _enqueue_document_runner(job_id: str, runner: Awaitable[None]) -> None:
    try:
        _dispatcher_loop.submit(_job_dispatcher.enqueue(job_id, runner))
    except Exception:
        _close_runner(runner)
        raise


def _require_bind_url(session: AsyncSession) -> str:
    bind = session.bind
    if bind is None:
        raise AppError(status_code=500, code="INTERNAL_ERROR", message="database binding unavailable")
    return bind.url.render_as_string(hide_password=False)


async def _enqueue_document_build(
    session: AsyncSession,
    *,
    document_id: str,
    job_id: str,
    content: bytes | None = None,
) -> None:
    bind_url = _require_bind_url(session)

    await _enqueue_document_task("build_document", {"document_id": document_id, "job_id": job_id})
    runner = _build_document_runner(
        bind_url=bind_url,
        document_id=document_id,
        job_id=job_id,
        content=content,
    )
    _enqueue_document_runner(job_id, runner)


async def _best_effort_cancel_enqueued(job_id: str) -> None:
    with suppress(Exception):
        await get_task_backend("inmemory").cancel(job_id)
    with suppress(Exception):
        _dispatcher_loop.submit(_job_dispatcher.cancel(job_id))


def _enqueue_failed_error() -> AppError:
    return AppError(status_code=500, code="INTERNAL_ERROR", message="failed to enqueue document build")


async def _compensate_upload_enqueue_failure(session: AsyncSession, *, document_id: str, job_id: str) -> None:
    await session.rollback()
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    await session.execute(delete(DocumentJob).where(DocumentJob.id == job_id))
    await session.execute(delete(Document).where(Document.id == document_id))
    await session.commit()
    await _best_effort_cancel_enqueued(job_id)


async def _compensate_rebuild_enqueue_failure(
    session: AsyncSession,
    *,
    document_id: str,
    job_id: str,
    previous_status: str,
    previous_strategy: str,
) -> None:
    await session.rollback()

    document_result = await session.execute(select(Document).where(Document.id == document_id))
    document = document_result.scalar_one_or_none()
    if document is not None:
        document.status = previous_status
        document.chunk_strategy = previous_strategy

    job_result = await session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
    job = job_result.scalar_one_or_none()
    if job is not None and job.status in {"queued", "running"}:
        job.status = "failed"
        job.stage = "failed"
        job.progress = min(job.progress, 99)
        job.message = "failed to enqueue document build"

    await session.commit()
    await _best_effort_cancel_enqueued(job_id)


async def _compensate_batch_enqueue_failure(
    session: AsyncSession,
    *,
    targets: list[tuple[str, str, str, str]],
) -> None:
    await session.rollback()

    for document_id, job_id, previous_status, previous_strategy in targets:
        document_result = await session.execute(select(Document).where(Document.id == document_id))
        document = document_result.scalar_one_or_none()
        if document is not None:
            document.status = previous_status
            document.chunk_strategy = previous_strategy

        job_result = await session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
        job = job_result.scalar_one_or_none()
        if job is not None:
            if job.status != "canceled":
                job.status = "failed"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = "failed to enqueue document build"

    await session.commit()


@router.get("")
async def list_documents(
    page: int = Query(1),
    page_size: int = Query(20),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    page, page_size = _validate_pagination(page, page_size)

    total = await session.scalar(select(func.count()).select_from(Document))
    result = await session.execute(
        select(Document).order_by(Document.uploaded_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    items = [_serialize_document(doc) for doc in result.scalars().all()]
    return _ok(
        {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total or 0,
            },
        }
    )


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    filename = (file.filename or "").strip()
    if not filename:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="file is required")

    existing = await session.execute(select(Document).where(Document.filename == filename))
    if existing.scalar_one_or_none() is not None:
        raise AppError(status_code=409, code="RESOURCE_CONFLICT", message="filename already exists")

    file_type = Path(filename).suffix.lower().lstrip(".") or "unknown"
    _validate_supported_upload_file_type(file_type)
    content = await file.read()
    document = Document(
        filename=filename,
        file_type=file_type,
        file_size=len(content),
        source_content=content,
        status="pending",
        chunk_strategy="general",
    )
    session.add(document)
    await session.flush()

    job = await _create_job(
        session,
        document_id=document.id,
        status="queued",
        stage="queued",
        progress=0,
        message="queued for build",
    )
    await session.commit()
    try:
        await _enqueue_document_build(session, document_id=document.id, job_id=job.id, content=content)
    except Exception as exc:
        with suppress(Exception):
            await _compensate_upload_enqueue_failure(session, document_id=document.id, job_id=job.id)
        raise _enqueue_failed_error() from exc

    return _ok({
        "document_id": document.id,
        "job_id": job.id,
    })


@router.post("/{document_id}/build")
async def build_document(
    document_id: str,
    payload: BuildDocumentRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    document = await _get_document_or_404(session, document_id)
    previous_status = document.status
    previous_strategy = document.chunk_strategy
    document.status = "pending"
    document.chunk_strategy = payload.chunk_strategy

    job = await _create_job(
        session,
        document_id=document.id,
        status="queued",
        stage="queued",
        progress=0,
        message="queued for rebuild",
    )
    await session.commit()
    try:
        await _enqueue_document_build(session, document_id=document.id, job_id=job.id)
    except Exception as exc:
        with suppress(Exception):
            await _compensate_rebuild_enqueue_failure(
                session,
                document_id=document.id,
                job_id=job.id,
                previous_status=previous_status,
                previous_strategy=previous_strategy,
            )
        raise _enqueue_failed_error() from exc
    return _ok(_serialize_job(job))


@router.post("/batch-build")
async def batch_build_documents(
    payload: BatchBuildRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    document_ids = [document_id for document_id in dict.fromkeys(payload.document_ids) if document_id]
    if not document_ids:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="document_ids is required")

    items: list[dict] = []
    queued_jobs: list[tuple[str, str, str, str]] = []
    for document_id in document_ids:
        document = await _get_document_or_404(session, document_id)
        previous_status = document.status
        previous_strategy = document.chunk_strategy
        document.status = "pending"
        document.chunk_strategy = payload.chunk_strategy

        job = await _create_job(
            session,
            document_id=document.id,
            status="queued",
            stage="queued",
            progress=0,
            message="queued for rebuild",
        )
        items.append(_serialize_job(job))
        queued_jobs.append((document.id, job.id, previous_status, previous_strategy))

    await session.commit()
    gate = threading.Event()
    bind_url = _require_bind_url(session)
    enqueued_job_ids: list[str] = []
    runner_plans: list[tuple[str, Awaitable[None]]] = []
    submitted_runner_count = 0
    try:
        for document_id, job_id, _, _ in queued_jobs:
            await _enqueue_document_task("build_document", {"document_id": document_id, "job_id": job_id})
            enqueued_job_ids.append(job_id)

            runner_plans.append(
                (
                    job_id,
                    _build_document_runner(
                        bind_url=bind_url,
                        document_id=document_id,
                        job_id=job_id,
                        gate=gate,
                    ),
                )
            )

        for job_id, runner in runner_plans:
            _enqueue_document_runner(job_id, runner)
            submitted_runner_count += 1
    except Exception as exc:
        for _, pending_runner in runner_plans[submitted_runner_count:]:
            with suppress(Exception):
                _close_runner(pending_runner)
        with suppress(Exception):
            for enqueued_job_id in enqueued_job_ids:
                await _best_effort_cancel_enqueued(enqueued_job_id)
        with suppress(Exception):
            await _compensate_batch_enqueue_failure(session, targets=queued_jobs)
        raise _enqueue_failed_error() from exc

    gate.set()
    return _ok({"items": items})


@router.post("/batch-delete")
async def batch_delete_documents(
    payload: BatchDeleteRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    document_ids = [document_id for document_id in dict.fromkeys(payload.document_ids) if document_id]
    if not document_ids:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="document_ids is required")

    success_ids: list[str] = []
    failed_items: list[dict] = []

    for document_id in document_ids:
        result = await session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document is None:
            failed_items.append({"document_id": document_id, "message": "document not found"})
            continue

        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        await session.execute(delete(DocumentJob).where(DocumentJob.document_id == document_id))
        await session.delete(document)
        success_ids.append(document_id)

    await session.commit()
    return _ok({"success_ids": success_ids, "failed_items": failed_items})


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    page: int = Query(1),
    page_size: int = Query(10),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    page, page_size = _validate_pagination(page, page_size)

    document = await _get_document_or_404(session, document_id)
    if document.status != "ready":
        raise AppError(
            status_code=409,
            code="DOC_CHUNK_RESULT_NOT_READY",
            message="chunk result is not ready",
            detail={"document_id": document_id, "status": document.status},
        )

    total = await session.scalar(select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document_id))
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [_serialize_chunk(item) for item in result.scalars().all()]
    return _ok(
        {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total or 0,
            },
        }
    )


@router.delete("/{filename}")
async def delete_document(
    filename: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await session.execute(select(Document).where(Document.filename == filename))
    document = result.scalar_one_or_none()
    if document is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="document not found")

    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    await session.execute(delete(DocumentJob).where(DocumentJob.document_id == document.id))
    await session.delete(document)
    await session.commit()

    return _ok({"success_ids": [document.id], "failed_items": []})


@router.get("/jobs")
async def list_jobs(
    page: int = Query(1),
    page_size: int = Query(20),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    page, page_size = _validate_pagination(page, page_size)

    total = await session.scalar(select(func.count()).select_from(DocumentJob))
    result = await session.execute(
        select(DocumentJob).order_by(DocumentJob.updated_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    jobs = result.scalars().all()
    items = [_serialize_job(item) for item in jobs]
    return _ok(
        {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total or 0,
            },
        }
    )


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="job not found")

    return _ok(_serialize_job(job))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    result = await session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="job not found")

    if job.status in {"queued", "running"}:
        document = await _get_document_or_404(session, job.document_id)
        document.status = "pending"
        job.status = "canceled"
        job.stage = "failed"
        job.progress = min(job.progress, 99)
        if not job.message:
            job.message = "job canceled by user"

    backend = get_task_backend("inmemory")
    await backend.cancel(job.id)
    _dispatcher_loop.submit(_job_dispatcher.cancel(job.id))
    await session.commit()

    return _ok(_serialize_job(job))
