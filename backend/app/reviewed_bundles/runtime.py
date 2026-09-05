from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Awaitable, Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.exceptions import AppError
from app.documents.dense_index_service import DenseIndexService
from app.documents.job_dispatcher import DocumentJobDispatcher
from app.extensions.registry import get_task_backend
from app.reviewed_bundles.build_service import CandidateBuildService
from app.reviewed_bundles.dispatch_authority import has_current_dispatch_authorization
from app.reviewed_bundles.models import CandidateBuildJob
from app.reviewed_bundles.recovery import CandidateBuildRecoveryService
from app.reviewed_bundles.verifier import CanonicalEditorialExportVerifier

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
                self._thread = threading.Thread(target=self._run, name="candidate-build-loop", daemon=True)
                self._thread.start()
        self._ready.wait(timeout=2)
        if self._loop is None:
            raise RuntimeError("Candidate Build dispatcher loop unavailable")
        return self._loop

    def submit(self, coro: Coroutine[Any, Any, _T]) -> _T:
        loop = self._ensure_running()
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


class CandidateBuildRuntime:
    """Dispatch Candidate work in a separate worker without owning publication."""

    def __init__(self) -> None:
        self._dispatcher = DocumentJobDispatcher()
        self._loop = _DispatcherLoop()
        self._runtime_id = uuid.uuid4().hex

    async def enqueue(self, session: AsyncSession, job_id: str) -> None:
        job = await session.get(CandidateBuildJob, job_id)
        if job is None:
            raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="Candidate Build job not found")
        await session.refresh(job, with_for_update=True)
        if job.status != "queued":
            raise AppError(
                status_code=409,
                code="CANDIDATE_ENQUEUE_NOT_ALLOWED",
                message="only a queued Candidate Build can be enqueued",
                detail={"job_id": job.id, "status": job.status},
            )
        if not await has_current_dispatch_authorization(session, job):
            raise AppError(
                status_code=409,
                code="CANDIDATE_DISPATCH_REQUIRED",
                message="Candidate Build requires explicit administrator dispatch before enqueue",
                detail={"job_id": job.id},
            )
        bind_url = self._require_bind_url(session)
        await get_task_backend("inmemory").enqueue(name="build_reviewed_candidate", payload={"job_id": job_id})
        runner = self._build_runner(bind_url=bind_url, job_id=job_id, lease_owner=self._runtime_id)
        try:
            self._loop.submit(self._dispatcher.enqueue(job_id, runner))
        except Exception:
            close = getattr(runner, "close", None)
            if callable(close):
                close()
            await get_task_backend("inmemory").cancel(job_id)
            raise

    async def cancel(self, job_id: str) -> bool:
        await get_task_backend("inmemory").cancel(job_id)
        return self._loop.submit(self._dispatcher.cancel(job_id))

    async def recover(self, session_factory) -> dict[str, list[str]]:
        async with session_factory() as session:
            return await CandidateBuildRecoveryService(
                session,
                dense_index_service=DenseIndexService(),
            ).recover(
                enqueue=lambda job_id: self.enqueue(session, job_id),
                recovery_owner=self._runtime_id,
            )

    @staticmethod
    def _require_bind_url(session: AsyncSession) -> str:
        bind = session.bind
        if bind is None:
            raise RuntimeError("Candidate Build database binding unavailable")
        return bind.url.render_as_string(hide_password=False)

    @staticmethod
    def _build_runner(*, bind_url: str, job_id: str, lease_owner: str) -> Awaitable[None]:
        async def runner() -> None:
            engine = create_async_engine(bind_url)
            session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
            try:
                async with session_factory() as session:
                    await CandidateBuildService(
                        session,
                        editorial_export_verifier=CanonicalEditorialExportVerifier(session),
                        lease_owner=lease_owner,
                    ).process_job(job_id)
            finally:
                await engine.dispose()

        return runner()


candidate_build_runtime = CandidateBuildRuntime()
