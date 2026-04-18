import asyncio

from app.rag.runtime.provider_adapters import LlmAdapter, RetrieverAdapter


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
