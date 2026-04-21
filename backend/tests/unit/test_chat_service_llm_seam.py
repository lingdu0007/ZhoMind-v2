import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.extensions.registry import get_extension_registry
from app.model.base import Base
from app.service.chat_service import ChatService


class _StubLlm:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return "基于证据的回答"


@pytest.fixture
def service() -> ChatService:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _init() -> AsyncSession:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return factory()

    session = asyncio.run(_init())
    try:
        yield ChatService(session)
    finally:
        asyncio.run(session.close())
        asyncio.run(engine.dispose())


def test_assistant_reply_uses_llm_when_gate_passed(service: ChatService) -> None:
    answer = asyncio.run(
        service._assistant_reply(
            question="测试问题",
            retrieved=[{"content_preview": "证据A"}],
            gate_passed=True,
            gate_reason="passed",
            llm=_StubLlm(),
        )
    )
    assert answer == "基于证据的回答"


def test_assistant_reply_rejects_when_gate_failed(service: ChatService) -> None:
    answer = asyncio.run(
        service._assistant_reply(
            question="测试问题",
            retrieved=[],
            gate_passed=False,
            gate_reason="reject_insufficient_evidence",
            llm=_StubLlm(),
        )
    )
    assert "未检索到足够相关的知识片段" in answer


def test_runtime_trace_keeps_required_keys_and_adds_metadata(service: ChatService) -> None:
    trace = service._runtime_trace(
        {
            "request_id": "rid-1",
            "session_id": "s-1",
            "graph_alias": "default_v1",
            "gate": {"passed": False, "reason": "reject_insufficient_evidence"},
            "steps": [{"step": "normalize", "detail": {"ok": True}}],
            "tool_budget": {"max_calls": 2, "max_parallel": 2, "max_latency_ms": 5000},
            "tool_errors": [],
        }
    )
    assert set(["request_id", "session_id", "graph_alias", "gate", "steps", "step_names"]).issubset(
        trace.keys()
    )
    assert trace["step_names"] == ["normalize"]
    assert trace["tool_budget"]["max_calls"] == 2


def test_resolve_llm_prefers_configured_provider(monkeypatch, service: ChatService) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("RAG_DEFAULT_LLM_PROVIDER", "cfg-llm")

    class _ConfiguredLlm:
        async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
            return "ok"

    registry = get_extension_registry()
    registry.register_llm("cfg-llm", _ConfiguredLlm())

    provider, provider_name = service._resolve_llm()
    assert provider_name == "cfg-llm"
    assert provider is registry.get_llm("cfg-llm")
