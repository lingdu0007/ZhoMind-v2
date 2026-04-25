import asyncio
import os

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.extensions.registry import get_extension_registry
from app.infra.milvus import get_milvus_provider
from app.model.base import Base


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()
    await engine.dispose()


@pytest.fixture(autouse=True)
def isolate_dense_runtime_environment(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    allow_real_dense_env = os.getenv("RUN_MILVUS_E2E") and request.node.nodeid.startswith(
        "tests/integration/test_milvus_dense_e2e.py"
    )
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    get_milvus_provider.cache_clear()
    if not allow_real_dense_env:
        monkeypatch.setenv("EMBEDDING_API_KEY", "")
        monkeypatch.setenv("EMBEDDING_BASE_URL", "")
        monkeypatch.setenv("EMBEDDING_MODEL", "")
        monkeypatch.setenv("DENSE_EMBEDDING_DIM", "0")
        monkeypatch.setenv("MILVUS_URI", "")
        monkeypatch.setenv("MILVUS_TOKEN", "")
    try:
        yield
    finally:
        get_milvus_provider.cache_clear()
        get_extension_registry.cache_clear()
        get_settings.cache_clear()
