from __future__ import annotations

import asyncio
from contextlib import suppress

from app.documents.build_service import DocumentBuildService


class DocumentJobDispatcher:
    def __init__(self, service: DocumentBuildService) -> None:
        self._service = service
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, *, document_id: str, job_id: str, content: bytes | None = None) -> str:
        async with self._lock:
            task = self._tasks.get(job_id)
            if task is not None and not task.done():
                return job_id

            self._tasks[job_id] = asyncio.create_task(
                self._run(document_id=document_id, job_id=job_id, content=content),
                name=f"document-job-{job_id}",
            )
            return job_id

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(job_id)

        if task is None:
            return False

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    async def _run(self, *, document_id: str, job_id: str, content: bytes | None) -> None:
        try:
            await self._service.process_job(document_id=document_id, job_id=job_id, content=content)
        finally:
            async with self._lock:
                task = self._tasks.get(job_id)
                if task is asyncio.current_task():
                    self._tasks.pop(job_id, None)
