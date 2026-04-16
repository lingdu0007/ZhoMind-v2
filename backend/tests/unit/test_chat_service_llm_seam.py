import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
