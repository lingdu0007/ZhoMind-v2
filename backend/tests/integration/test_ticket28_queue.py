import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.exceptions import AppError
from app.model.base import Base
from app.repository.chat_repository import ChatRepository
from app.service.answer_execution_store import AnswerExecutionStore
from app.service.chat_service import ChatService


@pytest.mark.asyncio
async def test_waiting_chat_reloads_as_queued_then_completes_without_inventing_an_outcome(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'queue.db'}")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    entered, release = asyncio.Event(), asyncio.Event()
    retained = []

    async def admitted(handle):
        retained.append(handle)

    async def wait_for_capacity():
        entered.set()
        await release.wait()

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with factory() as session:
            task = asyncio.create_task(ChatService(session).run_chat(
                user_id="queue-owner", question="hello", session_id="queue-session",
                on_admitted=admitted, await_capacity=wait_for_capacity,
            ))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                async with factory() as reader:
                    loaded = await AnswerExecutionStore(reader, ChatRepository(reader)).load(
                        execution_id=retained[0].execution_id, user_id="queue-owner",
                    )
                    assert loaded.projection["state"] == "queued"
                    assert loaded.projection.get("outcome") is None
                release.set()
                result = await asyncio.wait_for(task, timeout=5)
                assert result["message"]["answer_execution"]["state"] == "completed"
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_queue_timeout_retains_failed_execution_without_insufficiency(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'timeout.db'}")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    retained = []

    async def admitted(handle):
        retained.append(handle)

    async def timeout():
        raise AppError(status_code=503, code="CHAT_QUEUE_TIMEOUT", message="queue wait expired")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with factory() as session:
            with pytest.raises(AppError) as failure:
                await ChatService(session).run_chat(
                    user_id="queue-owner", question="hello", session_id="queue-session",
                    on_admitted=admitted, await_capacity=timeout,
                )
            assert failure.value.code == "CHAT_QUEUE_TIMEOUT"
        async with factory() as reader:
            loaded = await AnswerExecutionStore(reader, ChatRepository(reader)).load(
                execution_id=retained[0].execution_id, user_id="queue-owner",
            )
            assert loaded.projection["state"] == "failed"
            assert loaded.projection.get("outcome") is None
            assert loaded.projection["failure_code"] == "CHAT_QUEUE_TIMEOUT"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", ["store", "lifespan"])
async def test_restart_terminalizes_retained_queue_once_without_reexecuting(tmp_path, monkeypatch, restart):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'restart.db'}")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with factory() as session:
            repo = ChatRepository(session)
            await repo.get_or_create_session_for_admission(session_id="restart-session", user_id="owner")
            store = AnswerExecutionStore(session, repo)
            resolution = await store.resolve_query_conditions(
                user_id="owner", session_id="restart-session", question="hello",
                explicit_conditions=None, inherit_conditions=False,
            )
            handle = await store.admit(
                request_id="restart-request", user_id="owner", session_id="restart-session",
                question="hello", resolution=resolution, queued=True,
            )
            await session.commit()
        if restart == "lifespan":
            from app.main import app, lifespan

            monkeypatch.setattr(app.state, "settings_session_factory", factory)
            async with lifespan(app):
                pass
        async with factory() as session:
            store = AnswerExecutionStore(session, ChatRepository(session))
            if restart == "store":
                assert await store.recover_interrupted() == 1
            await session.commit()
            assert await store.recover_interrupted() == 0
            loaded = await store.load(execution_id=handle.execution_id, user_id="owner")
            assert loaded.projection["state"] == "failed"
            assert loaded.projection["failure_code"] == "ANSWER_EXECUTION_INTERRUPTED"
            assert loaded.projection.get("outcome") is None
    finally:
        await engine.dispose()
