from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import Settings, get_settings
from app.common.exceptions import AppError
from app.documents.dense_index_service import DenseIndexCancelledBeforeWrite, DenseIndexService
from app.documents.runtime_dense_obligations import register_runtime_dense_target, settle_runtime_dense_target
from app.infra.milvus_document_index import MilvusUpsertCancelledAfterDrain
from app.model.canonical import CanonicalRecordModel
from app.model.document import Document, DocumentChunk
from app.rag.dense_contract import build_embedding_contract_fingerprint, dense_mode_active
from app.repository.chat_repository import ChatRepository
from app.reviewed_bundles.models import PublishedKnowledgePointer, PublishedKnowledgeVersion
from app.reviewed_bundles.publication import CandidatePublicationService
from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier
from app.reviewed_bundles.withdrawal_facts import read_publication_withdrawals


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

        for document_id, published_generation in result_rows.all():
            target = None
            writer_invoked = False
            document, versions, skip_reason = await self._lock_backfill_snapshot(
                session, document_id, published_generation, fingerprint,
            )
            if skip_reason:
                documents.append(DenseMaintenanceDocumentResult(
                    document_id=document_id, outcome="skipped", reason=skip_reason,
                ))
                skipped_documents += 1
                await session.commit()
                continue
            try:
                if versions:
                    registered_versions = versions
                    target = register_runtime_dense_target(
                        session, publication_identity=versions[0], document_identity=document_id,
                        generation=published_generation, embedding_fingerprint=fingerprint,
                    )
                    # Preserve the address before any external write can take effect.
                    await session.commit()
                    document, versions, skip_reason = await self._lock_backfill_snapshot(
                        session, document_id, published_generation, fingerprint,
                    )
                    if versions != registered_versions and skip_reason is None:
                        skip_reason = "publication_changed"
                    if skip_reason:
                        documents.append(DenseMaintenanceDocumentResult(
                            document_id=document_id, outcome="skipped", reason=skip_reason,
                        ))
                        skipped_documents += 1
                        await settle_runtime_dense_target(session, target)
                        await session.commit()
                        continue
                assert document is not None
                dense_ready_generation = document.dense_ready_generation
                dense_ready_fingerprint = document.dense_ready_fingerprint
                chunks_result = await session.execute(
                    select(DocumentChunk)
                    .where(
                        DocumentChunk.document_id == document_id,
                        DocumentChunk.generation == published_generation,
                    )
                    .order_by(DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
                )
                chunks = list(chunks_result.scalars().all())
                if dense_ready_fingerprint is not None and dense_ready_fingerprint != fingerprint:
                    await self._dense_index_service.delete_candidate_generation(
                        document_id=document_id, generation=dense_ready_generation,
                        embedding_fingerprint=dense_ready_fingerprint,
                    )
                writer_invoked = True
                index_result = await self._dense_index_service.index_candidate_generation(
                    document_id=document_id,
                    generation=published_generation,
                    chunks=chunks,
                    **({"embedding_fingerprint": fingerprint} if versions else {}),
                )
                if target is not None:
                    await settle_runtime_dense_target(session, target)
                document.dense_ready_generation = published_generation
                document.dense_ready_fingerprint = index_result.fingerprint
                await session.commit()
                documents.append(DenseMaintenanceDocumentResult(document_id=document_id, outcome="indexed"))
                indexed_documents += 1
            except (MilvusUpsertCancelledAfterDrain, DenseIndexCancelledBeforeWrite):
                await session.rollback()
                if target is not None:
                    await settle_runtime_dense_target(session, target)
                    await session.commit()
                raise
            except asyncio.CancelledError:
                await session.rollback()
                if target is not None and not writer_invoked:
                    await settle_runtime_dense_target(session, target, write_never_started=True)
                    await session.commit()
                raise
            except Exception as exc:
                await session.rollback()
                if target is not None:
                    await settle_runtime_dense_target(session, target, write_never_started=not writer_invoked)
                    await session.commit()
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

    async def _lock_backfill_snapshot(
        self, session: AsyncSession, document_id: str, generation: int, fingerprint: str,
    ) -> tuple[Document | None, list[str], str | None]:
        await ChatRepository(session).acquire_private_conversation_write_fence()
        document = await session.scalar(select(Document).where(
            Document.id == document_id,
        ).with_for_update().execution_options(populate_existing=True))
        if document is None or document.deleted_at is not None or document.published_generation != generation:
            return document, [], "publication_changed"
        versions = list((await session.scalars(select(PublishedKnowledgeVersion.id).where(
            PublishedKnowledgeVersion.document_identity == document_id,
            PublishedKnowledgeVersion.generation == generation,
        ))).all())
        canonical_versions = list((await session.scalars(select(CanonicalRecordModel.stable_id).where(
            CanonicalRecordModel.identity_kind == "published_knowledge_version",
            CanonicalRecordModel.payload["document_identity"].as_string() == document_id,
            CanonicalRecordModel.payload["generation"].as_integer() == generation,
        ))).all())
        pointers = (await session.scalars(select(PublishedKnowledgePointer).where(
            PublishedKnowledgePointer.document_identity == document_id,
        ))).all()
        claimed_versions = set(versions) | set(canonical_versions) | {pointer.current_version_id for pointer in pointers}
        if await read_publication_withdrawals(session, list(claimed_versions)):
            return document, list(claimed_versions), "publication_withdrawn"
        if claimed_versions and (
            len(versions) != 1 or versions != canonical_versions or len(pointers) != 1
            or pointers[0].current_version_id != versions[0] or pointers[0].generation != generation
        ):
            raise AppError(
                status_code=409, code="DENSE_PUBLICATION_BINDING_INVALID",
                message="dense backfill publication binding cannot be verified",
            )
        if versions:
            version = await session.get(PublishedKnowledgeVersion, versions[0])
            assert version is not None
            await CandidatePublicationService(
                session, editorial_export_verifier=CanonicalEditorialExportVerifier(session),
            ).get_publication(version.candidate_id)
        if document.active_build_generation is not None:
            return document, versions, "active_build_in_progress"
        if document.dense_ready_fingerprint == fingerprint:
            reason = "already_dense_ready" if document.dense_ready_generation == generation else "stale_current_fingerprint"
            return document, versions, reason
        return document, versions, None

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
