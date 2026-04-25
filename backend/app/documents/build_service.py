from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.common.exceptions import AppError
from app.documents.chunker import chunk_document
from app.documents.dense_index_service import DenseIndexResult, DenseIndexService
from app.documents.parsers import parse_document
from app.documents.types import ChunkRecord, ParsedDocument
from app.model.document import Document, DocumentChunk, DocumentJob

_TERMINAL_JOB_STATUSES = {"succeeded", "failed", "canceled"}
_STARTABLE_JOB_STATUSES = ("queued", "running")
_CLAIM_RETRY_DELAY_SECONDS = 0.01
_CLAIM_LEASE_TIMEOUT = timedelta(seconds=30)


class DocumentBuildService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        parser: Callable[[str, bytes], ParsedDocument] = parse_document,
        chunker: Callable[[ParsedDocument], list[ChunkRecord]] | None = None,
        dense_index_service: DenseIndexService | None = None,
    ) -> None:
        self.session = session
        self._parser = parser
        self._chunker = chunker
        self._dense_index_service = dense_index_service or DenseIndexService()
        self._pending_dense_index_result: DenseIndexResult | None = None

    async def process_job(self, *, document_id: str, job_id: str, content: bytes | None = None) -> None:
        job = await self._get_job(job_id)
        if job.status in _TERMINAL_JOB_STATUSES:
            return

        document = await self._get_document(document_id)
        generation = self._require_build_generation(job)

        if document.deleted_at is not None:
            await self._terminalize_non_owner(document=document, job=job)
            return

        claimed = await self._wait_for_claim(document=document, job=job)
        if not claimed:
            return

        if not await self._start_claimed_job(document=document, job=job):
            await self._terminalize_non_owner(document=document, job=job)
            return

        try:
            self._pending_dense_index_result = None
            parsed = self._parser(document.filename, self._resolve_content(document, content))
            if not await self._refresh_and_verify_owner(document=document, job=job):
                await self._terminalize_non_owner(document=document, job=job)
                return

            job.stage = "chunking"
            job.progress = 60
            job.message = "chunking document"
            if not await self._refresh_heartbeat_if_owned(document=document, job=job):
                await self._terminalize_non_owner(document=document, job=job)
                return

            requested_strategy = job.requested_chunk_strategy or document.chunk_strategy
            chunker = self._chunker or (
                lambda parsed_document: chunk_document(parsed_document, strategy=requested_strategy)
            )
            chunks = chunker(parsed)
            if not await self._refresh_and_verify_owner(document=document, job=job):
                await self._terminalize_non_owner(document=document, job=job)
                return

            job.progress = 90
            job.message = "writing candidate document chunks"
            if not await self._refresh_heartbeat_if_owned(document=document, job=job):
                await self._terminalize_non_owner(document=document, job=job)
                return

            wrote_candidates = await self._write_candidate_chunks(
                document=document,
                job=job,
                generation=generation,
                chunks=chunks,
            )
            if not wrote_candidates:
                await self._terminalize_non_owner(document=document, job=job)
                return
            if not await self._refresh_and_verify_owner(document=document, job=job):
                await self._terminalize_non_owner(document=document, job=job)
                return
            dense_result = await self._dense_index_service.index_candidate_generation(
                document_id=document.id,
                generation=generation,
                chunks=self._build_dense_candidate_chunks(document_id=document.id, generation=generation, chunks=chunks),
            )
            self._pending_dense_index_result = dense_result

            if not await self._refresh_heartbeat_if_owned(document=document, job=job):
                await self._terminalize_non_owner(document=document, job=job)
                return
            await self._publish_generation(document=document, job=job, generation=generation, chunks=chunks)
        except asyncio.CancelledError:
            await self._cancel_claimed_job(document=document, job=job)
            raise
        except AppError as exc:
            await self._fail_claimed_job(document=document, job=job, message=self._format_app_error(exc))
            raise
        except Exception as exc:
            await self._fail_claimed_job(document=document, job=job, message=str(exc) or "document build failed")
            raise
        finally:
            self._pending_dense_index_result = None

    async def _claim_generation(self, *, document: Document, job: DocumentJob) -> bool:
        generation = self._require_build_generation(job)
        now = datetime.now(timezone.utc)
        lease_cutoff = now - _CLAIM_LEASE_TIMEOUT
        claim_result = await self.session.execute(
            update(Document)
            .where(
                Document.id == document.id,
                Document.deleted_at.is_(None),
                Document.latest_requested_generation == generation,
                or_(
                    Document.active_build_generation.is_(None),
                    Document.active_build_job_id == job.id,
                    and_(
                        Document.active_build_heartbeat_at < lease_cutoff,
                        Document.active_build_generation <= generation,
                    ),
                ),
            )
            .values(
                active_build_generation=generation,
                active_build_job_id=job.id,
                active_build_heartbeat_at=now,
            )
        )
        await self.session.commit()
        await self.session.refresh(document)
        await self.session.refresh(job)
        return claim_result.rowcount == 1 and document.active_build_generation == generation and document.active_build_job_id == job.id

    async def _wait_for_claim(self, *, document: Document, job: DocumentJob) -> bool:
        while True:
            claimed = await self._claim_generation(document=document, job=job)
            if claimed:
                return True
            if await self._should_retry_claim(document=document, job=job):
                await self._sleep_before_claim_retry()
                continue
            await self._terminalize_non_owner(document=document, job=job)
            return False

    async def _should_retry_claim(self, *, document: Document, job: DocumentJob) -> bool:
        await self.session.refresh(document)
        await self.session.refresh(job)
        generation = job.build_generation
        if generation is None:
            return False
        if document.deleted_at is not None or job.status in _TERMINAL_JOB_STATUSES:
            return False
        if document.latest_requested_generation != generation:
            return False
        if document.active_build_job_id == job.id:
            return True
        if document.active_build_generation is None:
            return True
        return document.active_build_generation < generation

    async def _sleep_before_claim_retry(self) -> None:
        await asyncio.sleep(_CLAIM_RETRY_DELAY_SECONDS)

    async def _start_claimed_job(self, *, document: Document, job: DocumentJob) -> bool:
        generation = job.build_generation
        if generation is None:
            return False

        document_result = await self.session.execute(
            update(Document)
            .where(
                Document.id == document.id,
                Document.deleted_at.is_(None),
                Document.latest_requested_generation == generation,
                Document.active_build_generation == generation,
                Document.active_build_job_id == job.id,
            )
            .values(status="processing")
        )
        if document_result.rowcount != 1:
            await self.session.rollback()
            await self.session.refresh(document)
            await self.session.refresh(job)
            return False

        job_result = await self.session.execute(
            update(DocumentJob)
            .where(
                DocumentJob.id == job.id,
                DocumentJob.status.in_(_STARTABLE_JOB_STATUSES),
            )
            .values(
                status="running",
                stage="parsing",
                progress=10,
                message="parsing document",
            )
        )
        if job_result.rowcount != 1:
            await self.session.rollback()
            await self.session.refresh(document)
            await self.session.refresh(job)
            return False

        await self.session.commit()
        await self.session.refresh(document)
        await self.session.refresh(job)
        return True

    async def _refresh_heartbeat_if_owned(self, *, document: Document, job: DocumentJob) -> bool:
        if not await self._touch_heartbeat_if_owned(document=document, job=job):
            await self.session.rollback()
            await self.session.refresh(document)
            await self.session.refresh(job)
            return False

        await self.session.commit()
        await self.session.refresh(document)
        await self.session.refresh(job)
        return True

    async def _touch_heartbeat_if_owned(self, *, document: Document, job: DocumentJob) -> bool:
        generation = job.build_generation
        if generation is None:
            return False

        heartbeat_result = await self.session.execute(
            update(Document)
            .where(
                Document.id == document.id,
                Document.deleted_at.is_(None),
                Document.latest_requested_generation == generation,
                Document.active_build_generation == generation,
                Document.active_build_job_id == job.id,
            )
            .values(active_build_heartbeat_at=datetime.now(timezone.utc))
        )
        return heartbeat_result.rowcount == 1

    async def _refresh_and_verify_owner(self, *, document: Document, job: DocumentJob) -> bool:
        await self.session.refresh(document)
        await self.session.refresh(job)
        generation = job.build_generation
        if generation is None:
            return False
        if document.deleted_at is not None or job.status in _TERMINAL_JOB_STATUSES:
            return False
        return (
            document.latest_requested_generation == generation
            and document.active_build_generation == generation
            and document.active_build_job_id == job.id
        )

    async def _write_candidate_chunks(
        self,
        *,
        document: Document,
        job: DocumentJob,
        generation: int,
        chunks: list[ChunkRecord],
    ) -> bool:
        await self.session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document.id,
                DocumentChunk.generation == generation,
            )
        )
        for chunk in chunks:
            self.session.add(
                DocumentChunk(
                    document_id=document.id,
                    generation=generation,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    keywords=[],
                    generated_questions=[],
                    chunk_metadata=chunk.metadata,
                )
            )
        await self.session.flush()
        if not await self._touch_heartbeat_if_owned(document=document, job=job):
            await self.session.rollback()
            await self.session.refresh(document)
            await self.session.refresh(job)
            return False
        await self.session.commit()
        await self.session.refresh(document)
        await self.session.refresh(job)
        return True

    async def _publish_generation(
        self,
        *,
        document: Document,
        job: DocumentJob,
        generation: int,
        chunks: list[ChunkRecord],
    ) -> None:
        dense_ready_generation, dense_ready_fingerprint = self._resolve_dense_readiness(generation=generation)
        publish_result = await self.session.execute(
            update(Document)
            .where(
                Document.id == document.id,
                Document.deleted_at.is_(None),
                Document.latest_requested_generation == generation,
                Document.active_build_generation == generation,
                Document.active_build_job_id == job.id,
            )
            .values(
                published_generation=generation,
                dense_ready_generation=dense_ready_generation,
                dense_ready_fingerprint=dense_ready_fingerprint,
                chunk_strategy=job.requested_chunk_strategy or document.chunk_strategy,
                chunk_count=len(chunks),
                status="ready",
                active_build_generation=None,
                active_build_job_id=None,
                active_build_heartbeat_at=None,
            )
        )
        if publish_result.rowcount != 1:
            await self.session.rollback()
            await self._terminalize_non_owner(document=document, job=job)
            return

        job.status = "succeeded"
        job.stage = "completed"
        job.progress = 100
        job.message = "document build completed"
        await self.session.commit()

    async def _terminalize_non_owner(self, *, document: Document, job: DocumentJob) -> None:
        await self.session.refresh(document)
        await self.session.refresh(job)
        cleanup_error = await self._delete_candidate_generation_assets(
            document_id=document.id,
            generation=job.build_generation,
        )
        self._release_claim_if_owned(document=document, job=job)

        if document.deleted_at is not None:
            if job.status not in _TERMINAL_JOB_STATUSES:
                job.status = "canceled"
                job.stage = "failed"
                job.progress = min(job.progress, 99)
                job.message = "job canceled because document was deleted"
        elif job.status == "canceled":
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            if not job.message:
                job.message = "job canceled by user"
            document.status = "pending"
        elif job.status not in _TERMINAL_JOB_STATUSES:
            job.status = "canceled"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = "job superseded by newer build request"
            document.status = self._derive_non_publish_status(document)

        job.message = self._append_cleanup_warning(job.message, cleanup_error)
        await self.session.commit()

    async def _cancel_claimed_job(self, *, document: Document, job: DocumentJob) -> None:
        with suppress(Exception):
            await self.session.rollback()
        await self._terminalize_non_owner(document=document, job=job)

    async def _fail_claimed_job(self, *, document: Document, job: DocumentJob, message: str) -> None:
        if not await self._refresh_and_verify_owner(document=document, job=job):
            await self._terminalize_non_owner(document=document, job=job)
            return

        cleanup_error = await self._delete_candidate_generation_assets(
            document_id=document.id,
            generation=job.build_generation,
        )
        self._release_claim_if_owned(document=document, job=job)

        if document.deleted_at is not None:
            job.status = "canceled"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = "job canceled because document was deleted"
        else:
            document.status = self._derive_non_publish_status(document)
            job.status = "failed"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = message

        job.message = self._append_cleanup_warning(job.message, cleanup_error)
        await self.session.commit()

    async def _delete_candidate_generation(self, *, document_id: str, generation: int | None) -> None:
        if generation is None:
            return
        await self.session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.generation == generation,
            )
        )

    async def _delete_candidate_generation_assets(self, *, document_id: str, generation: int | None) -> Exception | None:
        await self._delete_candidate_generation(document_id=document_id, generation=generation)
        try:
            await self._dense_index_service.delete_candidate_generation(document_id=document_id, generation=generation)
        except Exception as exc:
            return exc
        return None

    @staticmethod
    def _build_dense_candidate_chunks(
        *,
        document_id: str,
        generation: int,
        chunks: list[ChunkRecord],
    ) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                document_id=document_id,
                generation=generation,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_sha256=sha256(chunk.content.encode("utf-8")).hexdigest(),
                keywords=[],
                generated_questions=[],
                chunk_metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

    def _resolve_dense_readiness(self, *, generation: int) -> tuple[int, str | None]:
        dense_result = self._pending_dense_index_result
        if dense_result is not None and dense_result.active and dense_result.fingerprint:
            return generation, dense_result.fingerprint
        return 0, None

    @staticmethod
    def _release_claim_if_owned(*, document: Document, job: DocumentJob) -> None:
        if job.build_generation is None:
            return
        if document.active_build_generation == job.build_generation and document.active_build_job_id == job.id:
            document.active_build_generation = None
            document.active_build_job_id = None
            document.active_build_heartbeat_at = None

    @staticmethod
    def _derive_non_publish_status(document: Document) -> str:
        if document.published_generation > 0 and document.latest_requested_generation > document.published_generation:
            return "pending"
        if document.published_generation > 0:
            return "ready"
        return "failed"

    @staticmethod
    def _require_build_generation(job: DocumentJob) -> int:
        if job.build_generation is None:
            raise AppError(
                status_code=409,
                code="DOC_BUILD_GENERATION_MISSING",
                message="document build generation is missing",
                detail={"job_id": job.id},
            )
        return job.build_generation

    async def _get_document(self, document_id: str) -> Document:
        result = await self.session.execute(
            select(Document).options(undefer(Document.source_content)).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="document not found")
        return document

    async def _get_job(self, job_id: str) -> DocumentJob:
        result = await self.session.execute(select(DocumentJob).where(DocumentJob.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="job not found")
        return job

    def _resolve_content(self, document: Document, content: bytes | None) -> bytes:
        if content is not None:
            return content
        if document.source_content is not None:
            return document.source_content
        raise AppError(
            status_code=409,
            code="DOC_SOURCE_CONTENT_MISSING",
            message="document source content is missing",
            detail={"document_id": document.id},
        )

    @staticmethod
    def _format_app_error(exc: AppError) -> str:
        if exc.code and exc.message:
            return f"{exc.code}: {exc.message}"
        return exc.message or exc.code or "document build failed"

    @staticmethod
    def _append_cleanup_warning(message: str, cleanup_error: Exception | None) -> str:
        if cleanup_error is None:
            return message
        warning = f"dense cleanup warning: {cleanup_error}"
        if not message:
            return warning
        return f"{message} ({warning})"
