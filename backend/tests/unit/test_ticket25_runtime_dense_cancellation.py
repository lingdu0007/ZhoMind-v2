import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from app.infra.milvus_document_index import MilvusDocumentIndex


@pytest.mark.asyncio
async def test_candidate_finalization_allows_authority_first_withdrawal_to_finish(db_session, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.common.exceptions import AppError
    from app.reviewed_bundles.build_service import CandidateBuildService
    from app.reviewed_bundles.models import CandidateBuildJob
    from app.reviewed_bundles.service import ReviewedReleaseBundleService
    from tests.unit.test_reviewed_release_bundle_intake import (
        _ApprovedExportVerifier,
        _bundle_manifest,
        _DenseIndexSpy,
        _test_member_identity,
    )

    authority_wait = asyncio.Event()
    withdrawal_done = asyncio.Event()
    job_locked = False
    lock_cycle = False

    class Verifier(_ApprovedExportVerifier):
        @asynccontextmanager
        async def verify_for_candidate_finalization(self, artifact, artifact_sha256):
            authority_wait.set()
            await withdrawal_done.wait()
            if lock_cycle:
                raise AppError(status_code=409, code="TEST_ROW_LOCK_CYCLE", message="authority/job lock cycle")
            yield await self.verify(artifact, artifact_sha256)

    verifier = Verifier()
    _, actor = await _test_member_identity(db_session, username="ticket25-lock-order")
    intake = ReviewedReleaseBundleService(db_session, editorial_export_verifier=verifier)
    imported = await intake.import_bundle(_bundle_manifest(), actor_identity=actor)
    job_id = imported["items"][0]["job_id"]
    builder = CandidateBuildService(db_session, editorial_export_verifier=verifier, dense_index_service=_DenseIndexSpy())
    await builder.dispatch_job(job_id, actor_identity=actor)
    original_refresh = db_session.refresh
    original_commit = db_session.commit
    original_rollback = db_session.rollback

    # SQLite has no row locks. Model the competing transaction's wait-for edge
    # at the session adapter, while exercising the real builder and persisted job.
    async def refresh(instance, *args, **kwargs):
        nonlocal job_locked
        if isinstance(instance, CandidateBuildJob) and kwargs.get("with_for_update"):
            job_locked = True
        return await original_refresh(instance, *args, **kwargs)

    async def commit():
        nonlocal job_locked
        await original_commit()
        job_locked = False

    async def rollback():
        nonlocal job_locked
        await original_rollback()
        job_locked = False

    monkeypatch.setattr(db_session, "refresh", refresh)
    monkeypatch.setattr(db_session, "commit", commit)
    monkeypatch.setattr(db_session, "rollback", rollback)
    factory = async_sessionmaker(bind=db_session.bind, expire_on_commit=False)

    async def withdrawal():
        nonlocal lock_cycle
        await authority_wait.wait()
        try:
            lock_cycle = job_locked
            if not lock_cycle:
                async with factory() as session:
                    job = await session.get(CandidateBuildJob, job_id)
                    job.status = "superseded"
                    job.terminal_state = "superseded"
                    job.lease_owner = None
                    job.allowed_next_action = "none"
                    await session.commit()
        finally:
            withdrawal_done.set()

    worker = asyncio.create_task(builder.process_job(job_id))
    withdrawing = asyncio.create_task(withdrawal())
    try:
        result, _ = await asyncio.wait_for(asyncio.gather(worker, withdrawing), timeout=5)
        assert result["status"] == "superseded", result
        assert result["allowed_next_action"] == "none"
    finally:
        for task in (worker, withdrawing):
            if not task.done():
                task.cancel()
        await asyncio.gather(worker, withdrawing, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_upsert_does_not_release_its_caller_before_thread_exit():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    rows = []

    class Client:
        def upsert(self, collection_name, payload):
            started.set()
            try:
                assert release.wait(5), "test did not release the upsert"
                rows.extend(payload)
            finally:
                finished.set()

        def has_collection(self, collection_name):
            return True

        def delete(self, *args):
            rows.clear()

    index = MilvusDocumentIndex(client=Client())
    write = asyncio.create_task(index.upsert_generation(collection_name="ticket25", rows=[{
        "document_id": "runtime-document:ticket25", "generation": 1, "chunk_index": 0,
        "content_sha256": "a" * 64, "vector": [1.0],
    }]))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        write.cancel()
        cancellation_turn = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(lambda: cancellation_turn.set_result(write.done()))
        assert await cancellation_turn is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await write
        assert finished.is_set()
        await index.delete_generation(collection_name="ticket25", document_id="runtime-document:ticket25", generation=1)
        assert rows == []
    finally:
        release.set()
        await asyncio.gather(write, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 5)


@pytest.mark.asyncio
async def test_candidate_cancelled_during_embedding_proves_no_vector_write(db_session):
    from app.documents.dense_index_service import DenseIndexService
    from app.reviewed_bundles.build_service import CandidateBuildService
    from app.reviewed_bundles.inputs import load_frozen_candidate_build_input
    from app.reviewed_bundles.models import CandidateBuildJob
    from app.reviewed_bundles.service import ReviewedReleaseBundleService
    from app.reviewed_bundles.writer_exit import require_candidate_writers_settled
    from tests.unit.test_dense_maintenance_service import _dense_settings
    from tests.unit.test_reviewed_release_bundle_intake import _ApprovedExportVerifier, _bundle_manifest, _test_member_identity

    entered = asyncio.Event()
    upserts = []

    class Embeddings:
        async def embed(self, texts):
            entered.set()
            await asyncio.Event().wait()
            return []

    class Index:
        async def ensure_collection(self, **kwargs):
            pass

        async def upsert_generation(self, **kwargs):
            upserts.append(kwargs)

        async def delete_generation(self, **kwargs):
            pass

    settings = _dense_settings()
    _, actor = await _test_member_identity(db_session, username="ticket25-embedding-cancel")
    intake = ReviewedReleaseBundleService(db_session, settings=settings, editorial_export_verifier=_ApprovedExportVerifier())
    imported = await intake.import_bundle(_bundle_manifest(), actor_identity=actor)
    job_id = imported["items"][0]["job_id"]
    builder = CandidateBuildService(
        db_session, settings=settings, editorial_export_verifier=_ApprovedExportVerifier(),
        dense_index_service=DenseIndexService(settings=settings, embedding_provider=Embeddings(), document_index=Index()),
    )
    await builder.dispatch_job(job_id, actor_identity=actor)
    worker = asyncio.create_task(builder.process_job(job_id))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert upserts == []
        job = await db_session.get(CandidateBuildJob, job_id)
        frozen = await load_frozen_candidate_build_input(db_session, job_id)
        await require_candidate_writers_settled(db_session, job, frozen)
    finally:
        if not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
