from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
import inspect

from app.common.exceptions import AppError


class DocumentJobDispatcher:
    def __init__(self) -> None:
        self._tasks: dict[str, tuple[asyncio.Task[None], Awaitable[None]]] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, job_id: str, coro: Awaitable[None]) -> str:
        async with self._lock:
            current = self._tasks.get(job_id)
            if current is not None and not current[0].done():
                _, running_coro = current
                if inspect.iscoroutine(coro) and coro is not running_coro:
                    coro.close()
                return job_id

            task = asyncio.create_task(
                self._run(job_id=job_id, coro=coro),
                name=f"document-job-{job_id}",
            )
            self._tasks[job_id] = (task, coro)
            return job_id

    async def cancel(self, job_id: str) -> bool:
        async with self._lock:
            current = self._tasks.get(job_id)

        if current is None:
            return False
        task, _ = current

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    async def _run(self, *, job_id: str, coro: Awaitable[None]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except AppError:
            # The build service already persisted terminal job/document state.
            return
        finally:
            async with self._lock:
                current = self._tasks.get(job_id)
                if current is not None and current[0] is asyncio.current_task():
                    self._tasks.pop(job_id, None)
