from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrieveResult:
    items: list[dict[str, Any]]
    strategy: str = "sparse_only"
    dense_candidate_count: int = 0
    dense_hydrated_count: int = 0
    lexical_candidate_count: int = 0
    merged_count: int = 0
    dense_query_failed: bool = False
    lexical_scope: str = "full_published_live"
    fallback_used: bool = False
    provider_error: dict[str, Any] | None = None

    @classmethod
    def from_items(
        cls,
        items: list[dict[str, Any]],
        *,
        strategy: str = "sparse_only",
        lexical_scope: str = "full_published_live",
    ) -> "RetrieveResult":
        dense_count = sum(1 for item in items if item.get("retrieval_source") == "dense")
        lexical_count = len(items) - dense_count
        return cls(
            items=items,
            strategy=strategy,
            dense_candidate_count=dense_count,
            dense_hydrated_count=dense_count,
            lexical_candidate_count=lexical_count,
            merged_count=len(items),
            dense_query_failed=False,
            lexical_scope=lexical_scope,
            fallback_used=False,
            provider_error=None,
        )


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int) -> RetrieveResult | list[dict[str, Any]]: ...


class Reranker(Protocol):
    async def rerank(self, query: str, items: list[dict]) -> list[dict]: ...


class RelevanceJudge(Protocol):
    async def judge(self, query: str, context: list[dict]) -> bool: ...


class LlmProvider(Protocol):
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
