from typing import Protocol


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[dict]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, items: list[dict]) -> list[dict]: ...


class RelevanceJudge(Protocol):
    async def judge(self, query: str, context: list[dict]) -> bool: ...


class LlmProvider(Protocol):
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
