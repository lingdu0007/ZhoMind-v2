from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress


class DocumentJobDispatcher:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, job_id: str, coro: Awaitable[None]) -> str:
        async with self._lock:
            task = self._tasks.get(job_id)
            if task is not None and not task.done():
                return job_id

            self._tasks[job_id] = asyncio.create_task(
                self._run(job_id=job_id, coro=coro),
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

    async def _run(self, *, job_id: str, coro: Awaitable[None]) -> None:
        try:
            await coro
        finally:
            async with self._lock:
                task = self._tasks.get(job_id)
                if task is asyncio.current_task():
                    self._tasks.pop(job_id, None)
