from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import Settings, get_settings
from app.common.exceptions import AppError
from app.documents.dense_index_service import DenseIndexService
from app.model.document import Document, DocumentChunk
from app.rag.dense_contract import build_embedding_contract_fingerprint, dense_mode_active


@dataclass(frozen=True)
class DenseMaintenanceStatus:
    current_embedding_contract_fingerprint: str
    dense_mode_active: bool
    published_live_documents: int
    published_live_dense_ready_documents: int
    published_live_not_dense_ready_documents: int
    published_live_stale_generation_documents: int
    tombstoned_current_fingerprint_documents: int


@dataclass(frozen=True)
class DenseMaintenanceDocumentResult:
    document_id: str
    outcome: str
    reason: str | None = None


@dataclass(frozen=True)
class DenseMaintenanceBackfillResult:
    current_embedding_contract_fingerprint: str
    dense_mode_active: bool
    processed_documents: int
    indexed_documents: int
    skipped_documents: int
    failed_documents: int
    documents: list[DenseMaintenanceDocumentResult]


@dataclass(frozen=True)
class DenseMaintenanceReconcileResult:
    current_embedding_contract_fingerprint: str
    dense_mode_active: bool
    processed_documents: int
    reconciled_documents: int
    failed_documents: int
    documents: list[DenseMaintenanceDocumentResult]


class DenseMaintenanceService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        dense_index_service: DenseIndexService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._dense_index_service = dense_index_service or DenseIndexService(settings=self._settings)

    async def collect_status(self, *, session: AsyncSession) -> DenseMaintenanceStatus:
        fingerprint = self._current_fingerprint()
        mode_active = dense_mode_active(self._settings)
        published_live = self._published_live_predicates()
        current_ready = self._current_ready_predicates(fingerprint=fingerprint)

        published_live_documents = await self._count(session, *published_live)
        published_live_dense_ready_documents = await self._count(session, *published_live, *current_ready)
        published_live_stale_generation_documents = await self._count(
            session,
            *published_live,
            Document.dense_ready_fingerprint == fingerprint,
            Document.dense_ready_generation != Document.published_generation,
        )
        published_live_not_dense_ready_documents = published_live_documents - published_live_dense_ready_documents
        tombstoned_current_fingerprint_documents = await self._count(
            session,
            Document.deleted_at.is_not(None),
            Document.dense_ready_fingerprint == fingerprint,
        )

        return DenseMaintenanceStatus(
            current_embedding_contract_fingerprint=fingerprint,
            dense_mode_active=mode_active,
            published_live_documents=published_live_documents,
            published_live_dense_ready_documents=published_live_dense_ready_documents,
            published_live_not_dense_ready_documents=published_live_not_dense_ready_documents,
            published_live_stale_generation_documents=published_live_stale_generation_documents,
            tombstoned_current_fingerprint_documents=tombstoned_current_fingerprint_documents,
        )

    async def backfill_published_documents(
        self,
        *,
        session: AsyncSession,
        limit: int,
    ) -> DenseMaintenanceBackfillResult:
        fingerprint = self._require_dense_mode_active()
        result_rows = await session.execute(
            select(
                Document.id,
                Document.published_generation,
                Document.active_build_generation,
                Document.dense_ready_generation,
                Document.dense_ready_fingerprint,
            )
            .where(
                *self._published_live_predicates(),
                or_(
                    Document.dense_ready_generation != Document.published_generation,
                    Document.dense_ready_fingerprint.is_(None),
                    Document.dense_ready_fingerprint != fingerprint,
                ),
            )
            .order_by(Document.uploaded_at.asc(), Document.id.asc())
            .limit(limit)
        )
        documents: list[DenseMaintenanceDocumentResult] = []
        indexed_documents = 0
        skipped_documents = 0
        failed_documents = 0

        for document_id, published_generation, active_build_generation, dense_ready_generation, dense_ready_fingerprint in (
            result_rows.all()
        ):
            if active_build_generation is not None:
                documents.append(
                    DenseMaintenanceDocumentResult(
                        document_id=document_id,
                        outcome="skipped",
                        reason="active_build_in_progress",
                    )
                )
                skipped_documents += 1
                continue

            if dense_ready_fingerprint == fingerprint and dense_ready_generation != published_generation:
                documents.append(
                    DenseMaintenanceDocumentResult(
                        document_id=document_id,
                        outcome="skipped",
                        reason="stale_current_fingerprint",
                    )
                )
                skipped_documents += 1
                continue

            chunks_result = await session.execute(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.generation == published_generation,
                )
                .order_by(DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
            )
            chunks = list(chunks_result.scalars().all())

            try:
                index_result = await self._dense_index_service.index_candidate_generation(
                    document_id=document_id,
                    generation=published_generation,
                    chunks=chunks,
                )
                document = await session.get(Document, document_id)
                if document is None:
                    raise RuntimeError("document disappeared during dense backfill")
                document.dense_ready_generation = published_generation
                document.dense_ready_fingerprint = index_result.fingerprint
                await session.commit()
                documents.append(DenseMaintenanceDocumentResult(document_id=document_id, outcome="indexed"))
                indexed_documents += 1
            except Exception as exc:
                await session.rollback()
                documents.append(
                    DenseMaintenanceDocumentResult(
                        document_id=document_id,
                        outcome="failed",
                        reason=self._failure_reason(exc),
                    )
                )
                failed_documents += 1

        return DenseMaintenanceBackfillResult(
            current_embedding_contract_fingerprint=fingerprint,
            dense_mode_active=True,
            processed_documents=len(documents),
            indexed_documents=indexed_documents,
            skipped_documents=skipped_documents,
            failed_documents=failed_documents,
            documents=documents,
        )

    async def reconcile_current_fingerprint_documents(
        self,
        *,
        session: AsyncSession,
        limit: int,
    ) -> DenseMaintenanceReconcileResult:
        fingerprint = self._require_dense_mode_active()
        result_rows = await session.execute(
            select(Document.id)
            .where(
                Document.dense_ready_fingerprint == fingerprint,
                (
                    Document.deleted_at.is_not(None)
                    | (
                        Document.deleted_at.is_(None)
                        & (
                            (Document.published_generation == 0)
                            | (Document.dense_ready_generation != Document.published_generation)
                        )
                    )
                ),
            )
            .order_by(Document.uploaded_at.asc(), Document.id.asc())
            .limit(limit)
        )

        documents: list[DenseMaintenanceDocumentResult] = []
        reconciled_documents = 0
        failed_documents = 0

        for (document_id,) in result_rows.all():
            try:
                await self._dense_index_service.delete_document_current_fingerprint(document_id=document_id)
                document = await session.get(Document, document_id)
                if document is None:
                    raise RuntimeError("document disappeared during dense reconcile")
                document.dense_ready_generation = 0
                document.dense_ready_fingerprint = None
                await session.commit()
                documents.append(DenseMaintenanceDocumentResult(document_id=document_id, outcome="reconciled"))
                reconciled_documents += 1
            except Exception as exc:
                await session.rollback()
                documents.append(
                    DenseMaintenanceDocumentResult(
                        document_id=document_id,
                        outcome="failed",
                        reason=self._failure_reason(exc),
                    )
                )
                failed_documents += 1

        return DenseMaintenanceReconcileResult(
            current_embedding_contract_fingerprint=fingerprint,
            dense_mode_active=True,
            processed_documents=len(documents),
            reconciled_documents=reconciled_documents,
            failed_documents=failed_documents,
            documents=documents,
        )

    async def _count(self, session: AsyncSession, *predicates) -> int:
        result = await session.execute(select(func.count()).select_from(Document).where(*predicates))
        return int(result.scalar_one())

    def _current_fingerprint(self) -> str:
        return build_embedding_contract_fingerprint(self._settings)

    def _require_dense_mode_active(self) -> str:
        fingerprint = self._current_fingerprint()
        if dense_mode_active(self._settings):
            return fingerprint
        raise AppError(
            status_code=409,
            code="DOC_DENSE_MODE_INACTIVE",
            message="dense mode is inactive",
        )

    @staticmethod
    def _published_live_predicates():
        return (
            Document.deleted_at.is_(None),
            Document.published_generation > 0,
        )

    @staticmethod
    def _current_ready_predicates(*, fingerprint: str):
        return (
            Document.dense_ready_generation == Document.published_generation,
            Document.dense_ready_fingerprint == fingerprint,
        )

    @staticmethod
    def _failure_reason(exc: Exception) -> str:
        if isinstance(exc, AppError):
            return exc.code
        return type(exc).__name__
