from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.deps import require_admin
from app.common.exceptions import AppError
from app.common.request_id import get_request_id
from app.common.responses import ok_response
from app.documents.schemas import BatchBuildRequest, BatchDeleteRequest, BuildDocumentRequest
from app.extensions.registry import get_task_backend
from app.infra.db import get_db_session
from app.model.document import Document, DocumentChunk, DocumentJob

router = APIRouter(prefix="/documents", tags=["documents"])


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


def _mock_chunks(seed_text: str) -> list[dict]:
    lines = [line.strip() for line in seed_text.splitlines() if line.strip()]
    if not lines:
        lines = [seed_text.strip() or "empty document"]
    chunks = lines[:3]
    return [
        {
            "chunk_index": idx,
            "content": item[:280],
            "keywords": [],
            "generated_questions": [],
            "metadata": {"source": "phase2-mock", "length": len(item[:280])},
        }
        for idx, item in enumerate(chunks)
    ]


async def _get_document_or_404(session: AsyncSession, document_id: str) -> Document:
    result = await session.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if document is None:
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="document not found")
    return document


async def _replace_document_chunks(session: AsyncSession, document_id: str, chunks: list[dict]) -> None:
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    for chunk in chunks:
        session.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                keywords=chunk["keywords"],
                generated_questions=chunk["generated_questions"],
                chunk_metadata=chunk["metadata"],
            )
        )


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


async def _sync_job_if_ready(session: AsyncSession, job: DocumentJob) -> None:
    if job.status != "queued":
        return

    backend = get_task_backend("inmemory")
    status = await backend.get_status(job.id)
    if status.get("status") != "queued":
        return

    document = await _get_document_or_404(session, job.document_id)
    chunks = _mock_chunks(f"{document.filename}\n{document.chunk_strategy}")
    await _replace_document_chunks(session=session, document_id=document.id, chunks=chunks)
    document.chunk_count = len(chunks)
    document.status = "ready"

    job.status = "succeeded"
    job.stage = "completed"
    job.progress = 100
    if not job.message:
        job.message = "document build completed"

    await backend.cancel(job.id)


async def _sync_jobs_page(session: AsyncSession, jobs: list[DocumentJob]) -> None:
    for item in jobs:
        await _sync_job_if_ready(session, item)


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

    content = await file.read()
    file_type = Path(filename).suffix.lower().lstrip(".") or "unknown"
    document = Document(
        filename=filename,
        file_type=file_type,
        file_size=len(content),
        status="ready",
        chunk_strategy="general",
    )
    session.add(document)
    await session.flush()

    chunks = _mock_chunks(content.decode("utf-8", errors="ignore") or filename)
    await _replace_document_chunks(session=session, document_id=document.id, chunks=chunks)
    document.chunk_count = len(chunks)

    job = await _create_job(
        session,
        document_id=document.id,
        status="succeeded",
        stage="completed",
        progress=100,
        message="upload parsed and indexed",
    )
    await session.commit()

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
    document.status = "processing"
    document.chunk_strategy = payload.chunk_strategy

    job = await _create_job(
        session,
        document_id=document.id,
        status="queued",
        stage="queued",
        progress=0,
        message="queued for rebuild",
    )
    await _enqueue_document_task("build_document", {"document_id": document.id, "job_id": job.id})
    await session.commit()
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
    for document_id in document_ids:
        document = await _get_document_or_404(session, document_id)
        document.status = "processing"
        document.chunk_strategy = payload.chunk_strategy

        job = await _create_job(
            session,
            document_id=document.id,
            status="queued",
            stage="queued",
            progress=0,
            message="queued for rebuild",
        )
        await _enqueue_document_task("build_document", {"document_id": document.id, "job_id": job.id})
        items.append(_serialize_job(job))

    await session.commit()
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
    await _sync_jobs_page(session, jobs)
    await session.commit()
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

    await _sync_job_if_ready(session, job)
    await session.commit()
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
        job.status = "canceled"
        job.stage = "failed"
        job.progress = min(job.progress, 99)
        if not job.message:
            job.message = "job canceled by user"

    backend = get_task_backend("inmemory")
    await backend.cancel(job.id)
    await session.commit()

    return _ok(_serialize_job(job))
