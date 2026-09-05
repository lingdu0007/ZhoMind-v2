import asyncio
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import main
from app.infra.db import get_db_session
from app.model.base import Base


def test_candidate_recovery_does_not_silently_ignore_a_runtime_database_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def recover(_: object) -> None:
        raise SQLAlchemyError("candidate recovery transaction failed")

    monkeypatch.setattr(main.candidate_build_runtime, "recover", recover)

    with pytest.raises(SQLAlchemyError, match="candidate recovery transaction failed"):
        asyncio.run(main._recover_candidate_builds(object()))


def test_candidate_recovery_only_ignores_pre_migration_missing_candidate_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    async def recover(_: object) -> None:
        raise OperationalError(
            "SELECT * FROM candidate_build_jobs",
            {},
            Exception("no such table: candidate_build_jobs"),
        )

    monkeypatch.setattr(main.candidate_build_runtime, "recover", recover)

    asyncio.run(main._recover_candidate_builds(object()))


def test_lifespan_recovery_uses_the_database_dependency_override_before_the_default_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lifespan-recovery.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    seen_override_session = False
    original_factory = main.app.state.settings_session_factory

    async def initialize() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    async def recover(factory) -> None:
        nonlocal seen_override_session
        async with factory() as session:
            assert session.bind is not None
            assert "lifespan-recovery.db" in str(session.bind.url)
            seen_override_session = True

    def forbidden_default_factory():
        raise AssertionError("lifespan selected the default session factory instead of the dependency override")

    asyncio.run(initialize())
    monkeypatch.setattr(main.candidate_build_runtime, "recover", recover)
    main.app.dependency_overrides[get_db_session] = override_get_db_session
    main.app.state.settings_session_factory = forbidden_default_factory
    try:
        with TestClient(main.app):
            pass
    finally:
        main.app.dependency_overrides.clear()
        main.app.state.settings_session_factory = original_factory
        asyncio.run(engine.dispose())

    assert seen_override_session is True
