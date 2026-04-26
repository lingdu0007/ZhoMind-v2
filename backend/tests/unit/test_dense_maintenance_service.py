from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import Settings
from app.common.exceptions import AppError
from app.documents.dense_index_service import DenseIndexResult
from app.documents.dense_maintenance_service import DenseMaintenanceService
from app.model.base import Base
from app.model.document import Document, DocumentChunk
from app.rag.dense_contract import build_embedding_contract_fingerprint


class _DenseIndexSpy:
    def __init__(
        self,
        *,
        index_error_document_ids: set[str] | None = None,
        delete_error_document_ids: set[str] | None = None,
        assert_not_ready: callable | None = None,
        result_fingerprint: str = "ignored-by-maintenance-spy",
    ) -> None:
        self.index_error_document_ids = index_error_document_ids or set()
        self.delete_error_document_ids = delete_error_document_ids or set()
        self.assert_not_ready = assert_not_ready
        self.result_fingerprint = result_fingerprint
        self.index_calls: list[tuple[str, int, list[DocumentChunk]]] = []
        self.delete_calls: list[str] = []

    async def index_candidate_generation(
        self,
        *,
        document_id: str,
        generation: int,
        chunks: list[DocumentChunk],
    ) -> DenseIndexResult:
        self.index_calls.append((document_id, generation, chunks))
        if self.assert_not_ready is not None:
            await self.assert_not_ready(document_id)
        if document_id in self.index_error_document_ids:
            raise RuntimeError(f"index failure for {document_id}")
        return DenseIndexResult(active=True, fingerprint=self.result_fingerprint)

    async def delete_document_current_fingerprint(self, *, document_id: str) -> None:
        self.delete_calls.append(document_id)
        if document_id in self.delete_error_document_ids:
            raise RuntimeError(f"delete failure for {document_id}")


def _dense_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "EMBEDDING_API_KEY": "emb-key",
        "EMBEDDING_BASE_URL": "https://emb.example.com/v1",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "DENSE_EMBEDDING_DIM": 3,
        "MILVUS_URI": "http://milvus.example.com:19530",
    }
    values.update(overrides)
    return Settings(**values)


async def _make_session(tmp_path, name: str) -> tuple[AsyncSession, async_sessionmaker[AsyncSession], object]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/{name}.db")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = session_factory()
    return session, session_factory, engine


def _document(
    document_id: str,
    *,
    uploaded_at: datetime,
    published_generation: int,
    dense_ready_generation: int = 0,
    dense_ready_fingerprint: str | None = None,
    active_build_generation: int | None = None,
    deleted_at: datetime | None = None,
) -> Document:
    return Document(
        id=document_id,
        filename=f"{document_id}.txt",
        file_type="text/plain",
        file_size=1,
        status="ready",
        chunk_strategy="general",
        chunk_count=0,
        published_generation=published_generation,
        dense_ready_generation=dense_ready_generation,
        dense_ready_fingerprint=dense_ready_fingerprint,
        next_generation=max(1, published_generation + 1),
        latest_requested_generation=published_generation,
        active_build_generation=active_build_generation,
        deleted_at=deleted_at,
        uploaded_at=uploaded_at,
    )


def _chunk(document_id: str, generation: int, chunk_index: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        id=f"{document_id}:{generation}:{chunk_index}",
        document_id=document_id,
        generation=generation,
        chunk_index=chunk_index,
        content=content,
        chunk_metadata={"chunk_index": chunk_index},
    )


@pytest.mark.asyncio
async def test_collect_status_counts_current_fingerprint_documents(tmp_path) -> None:
    session, _session_factory, engine = await _make_session(tmp_path, "dense-maintenance-status")
    settings = _dense_settings()
    current_fingerprint = build_embedding_contract_fingerprint(settings)
    other_fingerprint = build_embedding_contract_fingerprint(_dense_settings(EMBEDDING_MODEL="text-embedding-3-small"))
    uploaded_at = datetime(2026, 4, 1, tzinfo=timezone.utc)

    try:
        session.add_all(
            [
                _document(
                    "doc-ready-current",
                    uploaded_at=uploaded_at,
                    published_generation=2,
                    dense_ready_generation=2,
                    dense_ready_fingerprint=current_fingerprint,
                ),
                _document("doc-not-ready", uploaded_at=uploaded_at + timedelta(seconds=1), published_generation=1),
                _document(
                    "doc-stale-current",
                    uploaded_at=uploaded_at + timedelta(seconds=2),
                    published_generation=3,
                    dense_ready_generation=2,
                    dense_ready_fingerprint=current_fingerprint,
                ),
                _document(
                    "doc-tombstoned-current",
                    uploaded_at=uploaded_at + timedelta(seconds=3),
                    published_generation=4,
                    dense_ready_generation=4,
                    dense_ready_fingerprint=current_fingerprint,
                    deleted_at=uploaded_at + timedelta(days=1),
                ),
                _document(
                    "doc-ready-other",
                    uploaded_at=uploaded_at + timedelta(seconds=4),
                    published_generation=5,
                    dense_ready_generation=5,
                    dense_ready_fingerprint=other_fingerprint,
                ),
                _document(
                    "doc-unpublished-current",
                    uploaded_at=uploaded_at + timedelta(seconds=5),
                    published_generation=0,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=current_fingerprint,
                ),
            ]
        )
        await session.commit()

        service = DenseMaintenanceService(settings=settings)

        status = await service.collect_status(session=session)

        assert status.current_embedding_contract_fingerprint == current_fingerprint
        assert status.dense_mode_active is True
        assert status.published_live_documents == 4
        assert status.published_live_dense_ready_documents == 1
        assert status.published_live_not_dense_ready_documents == 3
        assert status.published_live_stale_generation_documents == 1
        assert status.tombstoned_current_fingerprint_documents == 1
    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_backfill_skips_stale_current_fingerprint_docs_and_marks_ready_only_after_upsert(tmp_path) -> None:
    session, _session_factory, engine = await _make_session(tmp_path, "dense-maintenance-backfill")
    settings = _dense_settings()
    current_fingerprint = build_embedding_contract_fingerprint(settings)
    collaborator_fingerprint = "fingerprint-from-collaborator"
    uploaded_at = datetime(2026, 4, 1, tzinfo=timezone.utc)

    async def _assert_document_not_ready(document_id: str) -> None:
        document = await session.get(Document, document_id)
        assert document is not None
        assert document.dense_ready_generation == 0
        assert document.dense_ready_fingerprint is None

    dense_index_service = _DenseIndexSpy(
        assert_not_ready=_assert_document_not_ready,
        result_fingerprint=collaborator_fingerprint,
    )
    service = DenseMaintenanceService(settings=settings, dense_index_service=dense_index_service)

    try:
        session.add_all(
            [
                _document("doc-b", uploaded_at=uploaded_at + timedelta(seconds=1), published_generation=1),
                _document(
                    "doc-stale",
                    uploaded_at=uploaded_at + timedelta(seconds=2),
                    published_generation=2,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=current_fingerprint,
                ),
                _document(
                    "doc-active-build",
                    uploaded_at=uploaded_at + timedelta(seconds=3),
                    published_generation=3,
                    active_build_generation=4,
                ),
                _document("doc-a", uploaded_at=uploaded_at, published_generation=2),
            ]
        )
        session.add_all(
            [
                _chunk("doc-a", 2, 0, "doc-a generation-2 chunk-0"),
                _chunk("doc-a", 2, 1, "doc-a generation-2 chunk-1"),
                _chunk("doc-a", 1, 0, "doc-a stale chunk"),
                _chunk("doc-b", 1, 0, "doc-b generation-1 chunk-0"),
                _chunk("doc-stale", 2, 0, "doc-stale generation-2 chunk-0"),
                _chunk("doc-active-build", 3, 0, "doc-active-build generation-3 chunk-0"),
            ]
        )
        await session.commit()

        result = await service.backfill_published_documents(session=session, limit=10)

        assert [call[:2] for call in dense_index_service.index_calls] == [("doc-a", 2), ("doc-b", 1)]
        assert [chunk.generation for chunk in dense_index_service.index_calls[0][2]] == [2, 2]
        assert [chunk.document_id for chunk in dense_index_service.index_calls[0][2]] == ["doc-a", "doc-a"]
        assert [item.document_id for item in result.documents] == ["doc-a", "doc-b", "doc-stale", "doc-active-build"]
        assert [(item.outcome, item.reason) for item in result.documents] == [
            ("indexed", None),
            ("indexed", None),
            ("skipped", "stale_current_fingerprint"),
            ("skipped", "active_build_in_progress"),
        ]
        assert result.indexed_documents == 2
        assert result.failed_documents == 0
        assert result.skipped_documents == 2

        doc_a = await session.get(Document, "doc-a")
        doc_b = await session.get(Document, "doc-b")
        doc_stale = await session.get(Document, "doc-stale")
        doc_active = await session.get(Document, "doc-active-build")
        assert (
            doc_a is not None
            and doc_a.dense_ready_generation == 2
            and doc_a.dense_ready_fingerprint == collaborator_fingerprint
        )
        assert (
            doc_b is not None
            and doc_b.dense_ready_generation == 1
            and doc_b.dense_ready_fingerprint == collaborator_fingerprint
        )
        assert doc_stale is not None and doc_stale.dense_ready_generation == 1
        assert doc_stale.dense_ready_fingerprint == current_fingerprint
        assert doc_active is not None and doc_active.dense_ready_generation == 0
        assert doc_active.dense_ready_fingerprint is None
    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_reconcile_deletes_document_vectors_and_clears_dense_readiness(tmp_path) -> None:
    session, _session_factory, engine = await _make_session(tmp_path, "dense-maintenance-reconcile")
    settings = _dense_settings()
    current_fingerprint = build_embedding_contract_fingerprint(settings)
    other_fingerprint = build_embedding_contract_fingerprint(_dense_settings(EMBEDDING_MODEL="text-embedding-3-small"))
    uploaded_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    dense_index_service = _DenseIndexSpy()
    service = DenseMaintenanceService(settings=settings, dense_index_service=dense_index_service)

    try:
        session.add_all(
            [
                _document(
                    "doc-stale-live",
                    uploaded_at=uploaded_at,
                    published_generation=3,
                    dense_ready_generation=2,
                    dense_ready_fingerprint=current_fingerprint,
                ),
                _document(
                    "doc-tombstoned-current",
                    uploaded_at=uploaded_at + timedelta(seconds=1),
                    published_generation=4,
                    dense_ready_generation=4,
                    dense_ready_fingerprint=current_fingerprint,
                    deleted_at=uploaded_at + timedelta(days=1),
                ),
                _document(
                    "doc-unpublished-current",
                    uploaded_at=uploaded_at + timedelta(seconds=2),
                    published_generation=0,
                    dense_ready_generation=1,
                    dense_ready_fingerprint=current_fingerprint,
                ),
                _document(
                    "doc-other-fingerprint",
                    uploaded_at=uploaded_at + timedelta(seconds=3),
                    published_generation=5,
                    dense_ready_generation=5,
                    dense_ready_fingerprint=other_fingerprint,
                ),
            ]
        )
        await session.commit()

        result = await service.reconcile_current_fingerprint_documents(session=session, limit=10)

        assert dense_index_service.delete_calls == [
            "doc-stale-live",
            "doc-tombstoned-current",
            "doc-unpublished-current",
        ]
        assert [item.document_id for item in result.documents] == [
            "doc-stale-live",
            "doc-tombstoned-current",
            "doc-unpublished-current",
        ]
        assert all(item.outcome == "reconciled" for item in result.documents)
        assert result.reconciled_documents == 3
        assert result.failed_documents == 0

        for document_id in ("doc-stale-live", "doc-tombstoned-current", "doc-unpublished-current"):
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.dense_ready_generation == 0
            assert document.dense_ready_fingerprint is None

        document = await session.get(Document, "doc-other-fingerprint")
        assert document is not None
        assert document.dense_ready_generation == 5
        assert document.dense_ready_fingerprint == other_fingerprint
    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_dense_maintenance_service_raises_when_dense_mode_is_inactive(tmp_path) -> None:
    session, _session_factory, engine = await _make_session(tmp_path, "dense-maintenance-inactive")
    settings = _dense_settings(EMBEDDING_API_KEY="")
    service = DenseMaintenanceService(settings=settings, dense_index_service=_DenseIndexSpy())

    try:
        with pytest.raises(AppError) as backfill_exc:
            await service.backfill_published_documents(session=session, limit=1)
        assert backfill_exc.value.status_code == 409
        assert backfill_exc.value.code == "DOC_DENSE_MODE_INACTIVE"

        with pytest.raises(AppError) as reconcile_exc:
            await service.reconcile_current_fingerprint_documents(session=session, limit=1)
        assert reconcile_exc.value.status_code == 409
        assert reconcile_exc.value.code == "DOC_DENSE_MODE_INACTIVE"
    finally:
        await session.close()
        await engine.dispose()
