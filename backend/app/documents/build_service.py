from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.common.exceptions import AppError
from app.documents.chunker import chunk_document
from app.documents.parsers import parse_document
from app.documents.types import ChunkRecord, ParsedDocument
from app.model.document import Document, DocumentChunk, DocumentJob


class DocumentBuildService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        parser: Callable[[str, bytes], ParsedDocument] = parse_document,
        chunker: Callable[[ParsedDocument], list[ChunkRecord]] | None = None,
    ) -> None:
        self.session = session
        self._parser = parser
        self._chunker = chunker

    async def process_job(self, *, document_id: str, job_id: str, content: bytes | None = None) -> None:
        document = await self._get_document(document_id)
        job = await self._get_job(job_id)

        document.status = "processing"
        job.status = "running"
        job.stage = "parsing"
        job.progress = 10
        job.message = "parsing document"

        try:
            parsed = self._parser(document.filename, self._resolve_content(document, content))

            job.stage = "chunking"
            job.progress = 60
            job.message = "chunking document"

            chunker = self._chunker or (lambda parsed_document: chunk_document(parsed_document, strategy=document.chunk_strategy))
            chunks = chunker(parsed)

            job.progress = 90
            job.message = "replacing document chunks"

            await self._replace_document_chunks(document_id=document.id, chunks=chunks)

            document.chunk_count = len(chunks)
            document.status = "ready"

            job.status = "succeeded"
            job.stage = "completed"
            job.progress = 100
            job.message = "document build completed"
            await self.session.commit()
        except AppError as exc:
            document.status = "failed"
            job.status = "failed"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = self._format_app_error(exc)
            await self.session.commit()
            raise
        except Exception as exc:
            document.status = "failed"
            job.status = "failed"
            job.stage = "failed"
            job.progress = min(job.progress, 99)
            job.message = str(exc) or "document build failed"
            await self.session.commit()
            raise

    async def _replace_document_chunks(self, *, document_id: str, chunks: list[ChunkRecord]) -> None:
        await self.session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))

        for chunk in chunks:
            self.session.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    keywords=[],
                    generated_questions=[],
                    chunk_metadata=chunk.metadata,
                )
            )

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
