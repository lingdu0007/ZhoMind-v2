import asyncio
import threading
from collections.abc import Awaitable, Coroutine
from concurrent.futures import Future
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from redis.asyncio import Redis
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import undefer

from app.common.deps import require_admin
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.documents.build_service import DocumentBuildService
from app.documents.dense_maintenance_service import DenseMaintenanceService
from app.documents.job_dispatcher import DocumentJobDispatcher
from app.documents.operator_service import DocumentsOperatorService
from app.documents.parsers import parse_document
from app.documents.schemas import BatchBuildRequest, BatchDeleteRequest, BuildDocumentRequest, ChunkStrategy, DenseMaintenanceRequest
from app.extensions.registry import get_task_backend
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.model.chat import ChatMessage
from app.model.document import Document, DocumentChunk, DocumentJob
from app.operations.limits import MAX_PUBLISHED_SOURCES, MAX_UPLOAD_BYTES
from app.rag.answer_evidence import evidence_snapshot_id
from app.repository.chat_repository import ChatRepository
from app.service.answer_execution_store import AnswerExecutionStore

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

    def submit(self, coro: Coroutine[Any, Any, _T]) -> _T:
        loop = self._ensure_running()
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


_dispatcher_loop = _DispatcherLoop()


def _ok(data: dict) -> dict:
    payload = ok_response(data=data, request_id=get_request_id())
    payload.update(data)
    return payload


def _get_active_dispatcher_tasks() -> int:
    return _dispatcher_loop.submit(_job_dispatcher.active_count())


async def _collect_operator_status(*, session: AsyncSession, redis: Redis) -> dict:
    operator_service = DocumentsOperatorService(redis=redis)
    return await operator_service.collect_status(
        session=session,
        active_dispatcher_tasks=_get_active_dispatcher_tasks(),
    )


def _ready_for_dense_maintenance(status: dict) -> bool:
    return (
        status["drain_enabled"]
        and status["queued_jobs"] == 0
        and status["running_jobs"] == 0
        and status["active_dispatcher_tasks"] == 0
    )


async def _ensure_dense_maintenance_ready(*, session: AsyncSession, redis: Redis) -> None:
    status = await _collect_operator_status(session=session, redis=redis)
    if not status["drain_enabled"]:
        raise AppError(
            status_code=409,
            code="DOC_MIGRATION_DRAIN_INACTIVE",
            message="migration drain is not active",
        )
    if not _ready_for_dense_maintenance(status):
        raise AppError(
            status_code=409,
            code="DOC_MIGRATION_NOT_READY",
            message="migration drain is not quiescent",
        )


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
        "published_generation": document.published_generation,
        "candidate_generation": document.candidate_generation,
        "candidate_chunk_count": document.candidate_chunk_count,
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
    allowed = {"txt", "md", "pdf"}
    if file_type not in allowed:
        raise AppError(
            status_code=415,
            code="DOC_FILE_TYPE_NOT_SUPPORTED",
            message="document file type not supported",
            detail={"file_type": file_type},
        )


def _validate_upload_content(filename: str, file_type: str, content: bytes) -> None:
    if len(content) > MAX_UPLOAD_BYTES:
        raise AppError(
            status_code=413,
            code="DOC_FILE_TOO_LARGE",
            message="document file exceeds the 25 MiB limit",
            detail={"file_type": file_type},
        )
    parse_document(filename, content)


async def _get_document_or_404(session: AsyncSession, document_id: str) -> Document:
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.deleted_at.is_(None),
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="document not found")
    return document


def _reject_reviewed_bundle_runtime_mutation(document: Document) -> None:
    if document.file_type != "reviewed_release_bundle":
        return
    raise AppError(
        status_code=409,
        code="REVIEWED_BUNDLE_RUNTIME_MUTATION_REJECTED",
        message=(
            "Reviewed Bundle runtime projections can change only through Candidate "
            "inspection, acceptance, and explicit batch publication"
        ),
        detail={"document_id": document.id},
    )


async def _ensure_published_source_capacity(session: AsyncSession, *, document: Document) -> None:
    if document.published_generation > 0:
        return
    await session.execute(
        select(Document.id)
        .where(Document.deleted_at.is_(None), Document.published_generation > 0)
        .with_for_update()
    )
    published_sources = await session.scalar(
        select(func.count())
        .select_from(Document)
        .where(Document.deleted_at.is_(None), Document.published_generation > 0)
    )
    if int(published_sources or 0) >= MAX_PUBLISHED_SOURCES:
        raise AppError(
            status_code=409,
            code="PUBLISHED_SOURCE_LIMIT_REACHED",
            message="the first-release published source limit has been reached",
        )


async def _ensure_agent_candidate_publishable(session: AsyncSession, *, document: Document, generation: int) -> None:
    if document.candidate_chunk_strategy != "agent":
        return
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id, DocumentChunk.generation == generation)
        .order_by(DocumentChunk.chunk_index.asc())
        .limit(1)
    )
    chunk = result.scalar_one_or_none()
    metadata = chunk.chunk_metadata if chunk is not None and isinstance(chunk.chunk_metadata, dict) else {}
    raw_sources = metadata.get("sources")
    sources: list[object] = raw_sources if isinstance(raw_sources, list) else []
    unavailable_sources = [
        source.get("url")
        for source in sources
        if isinstance(source, dict) and source.get("availability") != "verified"
    ]
    if metadata.get("review_status") != "approved" or unavailable_sources:
        raise AppError(
            status_code=409,
            code="AGENT_ENTRY_NOT_APPROVED",
            message="Agent entry requires approved review status and verified sources before publication",
            detail={"entry_id": metadata.get("entry_id")},
        )


async def _ensure_document_mutations_allowed(redis: Redis) -> None:
    operator_service = DocumentsOperatorService(redis=redis)
    drain_enabled, _ = await operator_service.read_drain_state()
    if drain_enabled:
        raise AppError(
            status_code=503,
            code="DOC_MIGRATION_DRAIN_ACTIVE",
            message="document mutations are temporarily disabled for migration drain",
        )


async def _create_job(
    session: AsyncSession,
    document_id: str,
    *,
    status: str,
    stage: str,
    progress: int,
    message: str,
    build_generation: int | None = None,
    requested_chunk_strategy: str | None = None,
) -> DocumentJob:
    job = DocumentJob(
        document_id=document_id,
        build_generation=build_generation,
        requested_chunk_strategy=requested_chunk_strategy,
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


async def _compensate_replacement_upload_enqueue_failure(
    session: AsyncSession,
    *,
    document_id: str,
    job_id: str,
    previous_state: dict,
) -> None:
    await session.rollback()
    result = await session.execute(select(Document).options(undefer(Document.source_content)).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is not None:
        document.file_type = previous_state["file_type"]
        document.file_size = previous_state["file_size"]
        document.source_content = previous_state["source_content"]
        document.status = previous_state["status"]
        document.next_generation = previous_state["next_generation"]
        document.latest_requested_generation = previous_state["latest_requested_generation"]
    await session.execute(delete(DocumentJob).where(DocumentJob.id == job_id))
    await session.commit()
    await _best_effort_cancel_enqueued(job_id)


async def _compensate_rebuild_enqueue_failure(
    session: AsyncSession,
    *,
    document_id: str,
    job_id: str,
    previous_status: str,
) -> None:
    await session.rollback()

    document_result = await session.execute(select(Document).where(Document.id == document_id))
    document = document_result.scalar_one_or_none()
    if document is not None:
        document.status = previous_status

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
    targets: list[tuple[str, str, str]],
) -> None:
    await session.rollback()

    for document_id, job_id, previous_status in targets:
        document_result = await session.execute(select(Document).where(Document.id == document_id))
        document = document_result.scalar_one_or_none()
        if document is not None:
            document.status = previous_status

        job_result = await session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
        job = job_result.scalar_one_or_none()
        if job is not None:
            if job.status != "canceled":
                job.status = "failed"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = "failed to enqueue document build"

    await session.commit()


def _redact_frozen_answer_evidence_set(value: object, *, document_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    items = value.get("items")
    if not isinstance(items, list):
        return False
    changed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("document_id") != document_id:
            continue
        evidence.pop("content_preview", None)
        evidence.pop("content", None)
        evidence["withdrawn"] = True
        item["withdrawn"] = True
        changed = True
    return changed


def _redact_nested_frozen_answer_evidence(value: object, *, document_id: str) -> bool:
    if isinstance(value, dict):
        changed = False
        for key, nested in value.items():
            if key == "answer_evidence_set":
                changed = _redact_frozen_answer_evidence_set(nested, document_id=document_id) or changed
            else:
                changed = _redact_nested_frozen_answer_evidence(nested, document_id=document_id) or changed
        return changed
    if isinstance(value, list):
        return any(_redact_nested_frozen_answer_evidence(item, document_id=document_id) for item in value)
    return False


async def _tombstone_document(session: AsyncSession, *, document: Document) -> None:
    await session.execute(
        select(Document.id)
        .where(Document.id == document.id)
        .with_for_update()
    )
    document.deleted_at = datetime.now(UTC)
    document.status = "pending"
    document.latest_requested_generation = document.published_generation
    document.active_build_generation = None
    document.active_build_job_id = None
    document.active_build_heartbeat_at = None

    jobs_result = await session.execute(select(DocumentJob).where(DocumentJob.document_id == document.id))
    for job in jobs_result.scalars().all():
        if job.status in {"queued", "running"}:
            job.status = "canceled"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = "job canceled because document was deleted"

    messages_result = await session.execute(select(ChatMessage).where(ChatMessage.type == "assistant"))
    for message in messages_result.scalars().all():
        trace = deepcopy(message.rag_trace) if isinstance(message.rag_trace, dict) else None
        evidence = trace.get("evidence") if isinstance(trace, dict) else None
        changed = False
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, dict) or item.get("document_id") != document.id:
                    continue
                metadata = item.get("metadata")
                excerpt = item.get("content_preview") or item.get("content")
                if isinstance(metadata, dict) and isinstance(metadata.get("entry_id"), str) and isinstance(excerpt, str):
                    item["snapshot_id"] = evidence_snapshot_id(
                        title=str(metadata.get("title") or ""),
                        publication_version=str(metadata.get("publication_version") or f"v{item.get('generation', 1)}"),
                        excerpt=excerpt,
                        citation_metadata=metadata,
                    )
                item.pop("content_preview", None)
                item.pop("content", None)
                item["withdrawn"] = True
                changed = True
        if isinstance(trace, dict):
            changed = _redact_nested_frozen_answer_evidence(trace, document_id=document.id) or changed
        if changed:
            message.rag_trace = trace

    await AnswerExecutionStore(session, ChatRepository(session)).redact_document_evidence(document_id=document.id)


@router.get("")
async def list_documents(
    page: int = Query(1),
    page_size: int = Query(20),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    page, page_size = _validate_pagination(page, page_size)

    total = await session.scalar(
        select(func.count()).select_from(Document).where(Document.deleted_at.is_(None))
    )
    result = await session.execute(
        select(Document)
        .where(Document.deleted_at.is_(None))
        .order_by(Document.uploaded_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
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


@router.post("/ops/migration-drain")
async def migration_drain(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    operator_service = DocumentsOperatorService(redis=redis)
    await operator_service.enable_drain()
    return _ok(
        await operator_service.collect_status(
            session=session,
            active_dispatcher_tasks=_get_active_dispatcher_tasks(),
        )
    )


@router.get("/ops/migration-status")
async def migration_status(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    return _ok(await _collect_operator_status(session=session, redis=redis))


@router.get("/ops/dense-status")
async def dense_status(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    status = await _collect_operator_status(session=session, redis=redis)
    status["ready_for_dense_maintenance"] = _ready_for_dense_maintenance(status)
    status.update(asdict(await DenseMaintenanceService().collect_status(session=session)))
    return _ok(status)


@router.post("/ops/migration-reconcile")
async def migration_reconcile(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    operator_service = DocumentsOperatorService(redis=redis)

    async def _cancel_dispatcher_job(job_id: str) -> None:
        _dispatcher_loop.submit(_job_dispatcher.cancel(job_id))

    reconciled_job_ids = await operator_service.reconcile_queued_jobs(
        session=session,
        cancel_dispatcher_job=_cancel_dispatcher_job,
    )
    status = await operator_service.collect_status(
        session=session,
        active_dispatcher_tasks=_get_active_dispatcher_tasks(),
    )
    status["reconciled_job_ids"] = reconciled_job_ids
    return _ok(status)


@router.post("/ops/dense-backfill")
async def dense_backfill(
    request: DenseMaintenanceRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_dense_maintenance_ready(session=session, redis=redis)
    result = await DenseMaintenanceService().backfill_published_documents(
        session=session,
        limit=request.limit,
    )
    return _ok(asdict(result))


@router.post("/ops/dense-reconcile")
async def dense_reconcile(
    request: DenseMaintenanceRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_dense_maintenance_ready(session=session, redis=redis)
    result = await DenseMaintenanceService().reconcile_current_fingerprint_documents(
        session=session,
        limit=request.limit,
    )
    return _ok(asdict(result))


@router.post("/ops/migration-resume")
async def migration_resume(
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    operator_service = DocumentsOperatorService(redis=redis)
    await operator_service.clear_drain()
    return _ok(
        await operator_service.collect_status(
            session=session,
            active_dispatcher_tasks=_get_active_dispatcher_tasks(),
        )
    )


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: ChunkStrategy = Form("general"),
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_document_mutations_allowed(redis)
    filename = (file.filename or "").strip()
    if not filename:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="file is required")

    existing = await session.execute(
        select(Document).options(undefer(Document.source_content)).where(
            Document.filename == filename,
            Document.deleted_at.is_(None),
        )
    )
    file_type = Path(filename).suffix.lower().lstrip(".") or "unknown"
    _validate_supported_upload_file_type(file_type)
    content = await file.read()
    _validate_upload_content(filename, file_type, content)
    document = existing.scalar_one_or_none()
    created_document = document is None
    previous_state: dict | None = None
    if document is None:
        document = Document(
            filename=filename,
            file_type=file_type,
            file_size=len(content),
            source_content=content,
            status="pending",
            chunk_strategy=chunk_strategy,
        )
        session.add(document)
        await session.flush()
        queued_message = "queued for build"
    else:
        previous_state = {
            "file_type": document.file_type,
            "file_size": document.file_size,
            "source_content": document.source_content,
            "status": document.status,
            "next_generation": document.next_generation,
            "latest_requested_generation": document.latest_requested_generation,
        }
        document.file_type = file_type
        document.file_size = len(content)
        document.source_content = content
        document.status = "pending"
        queued_message = "queued for replacement build"

    job = await _create_job(
        session,
        document_id=document.id,
        build_generation=document.next_generation,
        requested_chunk_strategy=chunk_strategy,
        status="queued",
        stage="queued",
        progress=0,
        message=queued_message,
    )
    document.latest_requested_generation = document.next_generation
    document.next_generation += 1
    await session.commit()
    try:
        await _enqueue_document_build(session, document_id=document.id, job_id=job.id, content=content)
    except Exception as exc:
        with suppress(Exception):
            if created_document:
                await _compensate_upload_enqueue_failure(session, document_id=document.id, job_id=job.id)
            else:
                assert previous_state is not None
                await _compensate_replacement_upload_enqueue_failure(
                    session,
                    document_id=document.id,
                    job_id=job.id,
                    previous_state=previous_state,
                )
        raise _enqueue_failed_error() from exc

    return _ok({
        "document_id": document.id,
        "job_id": job.id,
    })


@router.post("/{document_id}/publish")
async def publish_document(
    document_id: str,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    del session, redis
    raise AppError(
        status_code=410,
        code="LEGACY_PUBLICATION_BYPASS_REJECTED",
        message=(
            "Reviewed Bundle runtime projections must be published through "
            "Candidate inspection, acceptance, and explicit batch confirmation"
        ),
        detail={"document_id": document_id},
    )


@router.post("/{document_id}/build")
async def build_document(
    document_id: str,
    payload: BuildDocumentRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_document_mutations_allowed(redis)
    document = await _get_document_or_404(session, document_id)
    _reject_reviewed_bundle_runtime_mutation(document)
    previous_status = document.status
    document.status = "pending"
    build_generation = document.next_generation

    job = await _create_job(
        session,
        document_id=document.id,
        build_generation=build_generation,
        requested_chunk_strategy=payload.chunk_strategy,
        status="queued",
        stage="queued",
        progress=0,
        message="queued for rebuild",
    )
    document.latest_requested_generation = build_generation
    document.next_generation += 1
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
            )
        raise _enqueue_failed_error() from exc
    return _ok(_serialize_job(job))


@router.post("/batch-build")
async def batch_build_documents(
    payload: BatchBuildRequest,
    _: object = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_document_mutations_allowed(redis)
    document_ids = [document_id for document_id in dict.fromkeys(payload.document_ids) if document_id]
    if not document_ids:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="document_ids is required")

    items: list[dict] = []
    queued_jobs: list[tuple[str, str, str]] = []
    for document_id in document_ids:
        document = await _get_document_or_404(session, document_id)
        _reject_reviewed_bundle_runtime_mutation(document)
        previous_status = document.status
        document.status = "pending"
        build_generation = document.next_generation

        job = await _create_job(
            session,
            document_id=document.id,
            build_generation=build_generation,
            requested_chunk_strategy=payload.chunk_strategy,
            status="queued",
            stage="queued",
            progress=0,
            message="queued for rebuild",
        )
        document.latest_requested_generation = build_generation
        document.next_generation += 1
        items.append(_serialize_job(job))
        queued_jobs.append((document.id, job.id, previous_status))

    await session.commit()
    gate = threading.Event()
    bind_url = _require_bind_url(session)
    enqueued_job_ids: list[str] = []
    runner_plans: list[tuple[str, Awaitable[None]]] = []
    submitted_runner_count = 0
    try:
        for document_id, job_id, _ in queued_jobs:
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
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_document_mutations_allowed(redis)
    document_ids = [document_id for document_id in dict.fromkeys(payload.document_ids) if document_id]
    if not document_ids:
        raise AppError(status_code=400, code="VALIDATION_ERROR", message="document_ids is required")

    success_ids: list[str] = []
    failed_items: list[dict] = []

    for document_id in document_ids:
        result = await session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.deleted_at.is_(None),
            )
        )
        document = result.scalar_one_or_none()
        if document is None:
            failed_items.append({"document_id": document_id, "message": "document not found"})
            continue

        await _tombstone_document(session, document=document)
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
    generation = document.candidate_generation or document.published_generation
    if generation == 0:
        raise AppError(
            status_code=409,
            code="DOC_CHUNK_RESULT_NOT_READY",
            message="chunk result is not ready",
            detail={"document_id": document_id, "status": document.status},
        )

    total = await session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.generation == generation,
        )
    )
    result = await session.execute(
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.generation == generation,
        )
        .order_by(DocumentChunk.chunk_index.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [_serialize_chunk(item) for item in result.scalars().all()]
    return _ok(
        {
            "items": items,
            "generation": generation,
            "generation_state": "candidate" if document.candidate_generation else "published",
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
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_document_mutations_allowed(redis)
    result = await session.execute(
        select(Document).where(
            Document.filename == filename,
            Document.deleted_at.is_(None),
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="document not found")

    await _tombstone_document(session, document=document)
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
    redis: Redis = Depends(get_redis_client),
) -> dict:
    await _ensure_document_mutations_allowed(redis)
    result = await session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="job not found")

    backend = get_task_backend("inmemory")
    with suppress(Exception):
        await backend.cancel(job.id)
    with suppress(Exception):
        _dispatcher_loop.submit(_job_dispatcher.cancel(job.id))
    await session.refresh(job)

    if job.status in {"queued", "running"}:
        document_result = await session.execute(select(Document).where(Document.id == job.document_id))
        document = document_result.scalar_one_or_none()
        if document is not None and document.deleted_at is None:
            document.status = "ready" if document.published_generation > 0 else "pending"
            if job.build_generation is not None and document.latest_requested_generation == job.build_generation:
                document.latest_requested_generation = document.published_generation
            if (
                job.build_generation is not None
                and document.active_build_generation == job.build_generation
                and document.active_build_job_id == job.id
            ):
                document.active_build_generation = None
                document.active_build_job_id = None
                document.active_build_heartbeat_at = None
        job.status = "canceled"
        job.stage = "failed"
        job.progress = min(job.progress, 99)
        if not job.message:
            job.message = "job canceled by user"

    if job.status == "canceled":
        document = await session.get(Document, job.document_id)
        if document is not None and document.deleted_at is None:
            if job.build_generation is not None and document.latest_requested_generation == job.build_generation:
                document.latest_requested_generation = document.published_generation
            if document.published_generation > 0:
                document.status = "ready"
            if (
                job.build_generation is not None
                and document.active_build_generation == job.build_generation
                and document.active_build_job_id == job.id
            ):
                document.active_build_generation = None
                document.active_build_job_id = None
                document.active_build_heartbeat_at = None

    await session.commit()

    return _ok(_serialize_job(job))
