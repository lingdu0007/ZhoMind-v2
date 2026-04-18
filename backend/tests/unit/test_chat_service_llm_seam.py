import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.extensions.registry import get_extension_registry
from app.model.base import Base
from app.service.chat_service import ChatService


class _StubLlm:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return "基于证据的回答"


def _build_service() -> ChatService:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _init() -> AsyncSession:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return factory()

    session = asyncio.run(_init())
    return ChatService(session)


def test_assistant_reply_uses_llm_when_gate_passed() -> None:
    service = _build_service()
    answer = asyncio.run(
        service._assistant_reply(
            question="测试问题",
            retrieved=[{"content_preview": "证据A"}],
            gate_passed=True,
            llm=_StubLlm(),
        )
    )
    assert answer == "基于证据的回答"


def test_assistant_reply_rejects_when_gate_failed() -> None:
    service = _build_service()
    answer = asyncio.run(
        service._assistant_reply(
            question="测试问题",
            retrieved=[],
            gate_passed=False,
            llm=_StubLlm(),
        )
    )
    assert "未检索到足够相关的知识片段" in answer


def test_resolve_llm_prefers_configured_provider(monkeypatch) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("RAG_DEFAULT_LLM_PROVIDER", "cfg-llm")

    service = _build_service()

    class _ConfiguredLlm:
        async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
            return "ok"

    registry = get_extension_registry()
    registry.register_llm("cfg-llm", _ConfiguredLlm())

    provider, provider_name = service._resolve_llm()
    assert provider_name == "cfg-llm"
    assert provider is registry.get_llm("cfg-llm")
