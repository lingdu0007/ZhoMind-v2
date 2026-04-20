import asyncio

from app.extensions.ark_llm_provider import ArkLlmProvider
from app.rag.runtime.provider_adapters import (
    JudgeAdapter,
    LlmAdapter,
    RerankerAdapter,
    RetrieverAdapter,
)



class _RetrieverOk:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        return [{"chunk_id": "c1", "content": f"{query}-{top_k}"}]


class _RetrieverBoom:
    async def retrieve(self, query: str, top_k: int) -> list[dict]:
        raise RuntimeError("retriever exploded")


def test_retriever_primary_success() -> None:
    adapter = RetrieverAdapter(_RetrieverOk(), provider_name="retriever-main")

    items, detail = asyncio.run(adapter.retrieve("hello", top_k=2))

    assert items == [{"chunk_id": "c1", "content": "hello-2"}]
    assert detail == {
        "provider": "retriever-main",
        "fallback_used": False,
        "error": None,
    }


def test_retriever_fallback_on_exception_has_normalized_error() -> None:
    adapter = RetrieverAdapter(_RetrieverBoom(), provider_name="retriever-main")

    items, detail = asyncio.run(adapter.retrieve("hello", top_k=2))

    assert items == []
    assert detail["provider"] == "retriever-main"
    assert detail["fallback_used"] is True
    assert detail["error"] is not None
    assert detail["error"]["code"] == "PROVIDER_EXEC_FAILED"
    assert detail["error"]["message"] == "retriever exploded"
    assert detail["error"]["type"] == "RuntimeError"


def test_llm_missing_provider_fallback_empty_string() -> None:
    adapter = LlmAdapter(None, provider_name="missing-llm")

    text, detail = asyncio.run(adapter.complete("prompt"))

    assert text == ""
    assert detail == {
        "provider": "missing-llm",
        "fallback_used": True,
        "error": None,
    }


class _RerankerOk:
    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        return [{"chunk_id": "r1", "content": f"{query}-{len(items)}"}]


class _RerankerBoom:
    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        raise RuntimeError("reranker exploded")


class _JudgeOk:
    async def judge(self, query: str, context: list[dict]) -> bool:
        return False


class _JudgeBoom:
    async def judge(self, query: str, context: list[dict]) -> bool:
        raise RuntimeError("judge exploded")


class _LlmBoom:
    async def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        raise RuntimeError("llm exploded")


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, headers: dict, content: str):
        return _FakeResponse(self._payload)


def test_ark_llm_provider_parses_chat_completions_payload(monkeypatch) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": "  这是模型回答  ",
                }
            }
        ]
    }

    monkeypatch.setattr(
        "app.extensions.ark_llm_provider.AsyncClient",
        lambda timeout: _FakeAsyncClient(payload),
    )

    provider = ArkLlmProvider(
        api_key="ark-key",
        model="ark-model",
        base_url="https://ark.example.com/api/v3",
    )

    text = asyncio.run(provider.complete("hello"))
    assert text == "这是模型回答"


def test_reranker_primary_success() -> None:
    adapter = RerankerAdapter(_RerankerOk(), provider_name="reranker-main")

    items, detail = asyncio.run(
        adapter.rerank("hello", [{"chunk_id": "c1", "content": "a"}])
    )

    assert items == [{"chunk_id": "r1", "content": "hello-1"}]
    assert detail == {
        "provider": "reranker-main",
        "fallback_used": False,
        "error": None,
    }


def test_reranker_fallback_on_exception_has_normalized_error() -> None:
    adapter = RerankerAdapter(_RerankerBoom(), provider_name="reranker-main")
    original_items = [{"chunk_id": "c1", "content": "a"}]

    items, detail = asyncio.run(adapter.rerank("hello", original_items))

    assert items == original_items
    assert detail["provider"] == "reranker-main"
    assert detail["fallback_used"] is True
    assert detail["error"] is not None
    assert detail["error"]["code"] == "PROVIDER_EXEC_FAILED"
    assert detail["error"]["message"] == "reranker exploded"
    assert detail["error"]["type"] == "RuntimeError"


def test_judge_primary_success() -> None:
    adapter = JudgeAdapter(_JudgeOk(), provider_name="judge-main")

    accepted, detail = asyncio.run(
        adapter.judge("hello", [{"chunk_id": "c1", "content": "a"}])
    )

    assert accepted is False
    assert detail == {
        "provider": "judge-main",
        "fallback_used": False,
        "error": None,
    }


def test_judge_fallback_on_exception_uses_context_length() -> None:
    adapter = JudgeAdapter(_JudgeBoom(), provider_name="judge-main")

    accepted_with_context, detail_with_context = asyncio.run(
        adapter.judge("hello", [{"chunk_id": "c1", "content": "a"}])
    )
    accepted_without_context, detail_without_context = asyncio.run(
        adapter.judge("hello", [])
    )

    assert accepted_with_context is True
    assert accepted_without_context is False
    assert detail_with_context["provider"] == "judge-main"
    assert detail_with_context["fallback_used"] is True
    assert detail_with_context["error"] is not None
    assert detail_with_context["error"]["code"] == "PROVIDER_EXEC_FAILED"
    assert detail_with_context["error"]["message"] == "judge exploded"
    assert detail_with_context["error"]["type"] == "RuntimeError"
    assert detail_without_context == detail_with_context


def test_llm_fallback_on_exception_has_normalized_error() -> None:
    adapter = LlmAdapter(_LlmBoom(), provider_name="llm-main")

    text, detail = asyncio.run(adapter.complete("prompt"))

    assert text == ""
    assert detail["provider"] == "llm-main"
    assert detail["fallback_used"] is True
    assert detail["error"] is not None
    assert detail["error"]["code"] == "PROVIDER_EXEC_FAILED"
    assert detail["error"]["message"] == "llm exploded"
    assert detail["error"]["type"] == "RuntimeError"
