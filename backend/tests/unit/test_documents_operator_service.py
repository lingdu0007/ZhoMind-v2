from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.documents.job_dispatcher import DocumentJobDispatcher
from app.documents.operator_service import (
    DRAIN_MODE_KEY,
    DRAIN_STARTED_AT_KEY,
    DocumentsOperatorService,
)
from app.model.base import Base
from app.model.document import DocumentJob


class _RedisSpy:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def set(self, key: str, value: str) -> bool:
        self._values[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += 1 if self._values.pop(key, None) is not None else 0
        return removed


@pytest.mark.asyncio
async def test_enable_drain_sets_mode_and_preserves_existing_started_at():
    redis = _RedisSpy()
    service = DocumentsOperatorService(redis=redis)
    first_started_at = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    second_started_at = datetime(2026, 4, 24, 13, 0, tzinfo=timezone.utc)

    await service.enable_drain(now=first_started_at)
    await service.enable_drain(now=second_started_at)

    assert await redis.get(DRAIN_MODE_KEY) == "1"
    assert await redis.get(DRAIN_STARTED_AT_KEY) == first_started_at.isoformat()


@pytest.mark.asyncio
async def test_collect_status_derives_ready_for_migration_from_db_and_dispatcher(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ops-status.db")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    redis = _RedisSpy()
    service = DocumentsOperatorService(redis=redis)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        session.add_all(
            [
                DocumentJob(document_id="doc-q", status="queued", stage="queued", progress=0, message="queued"),
                DocumentJob(document_id="doc-r", status="running", stage="parsing", progress=10, message="running"),
            ]
        )
        await session.commit()

        await service.enable_drain(now=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc))
        status = await service.collect_status(session=session, active_dispatcher_tasks=1)

    assert status["drain_enabled"] is True
    assert status["queued_jobs"] == 1
    assert status["running_jobs"] == 1
    assert status["active_dispatcher_tasks"] == 1
    assert status["ready_for_migration"] is False

    await engine.dispose()


@pytest.mark.asyncio
async def test_dispatcher_active_count_only_counts_live_tasks():
    dispatcher = DocumentJobDispatcher()
    release = asyncio.Event()
    started = asyncio.Event()

    async def _job() -> None:
        started.set()
        await release.wait()

    await dispatcher.enqueue("job-1", _job())
    await asyncio.wait_for(started.wait(), timeout=1)
    assert await dispatcher.active_count() == 1

    release.set()
    await asyncio.sleep(0)
    assert await dispatcher.active_count() == 0
