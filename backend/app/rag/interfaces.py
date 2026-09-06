from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict


class ProviderExecError(TypedDict):
    code: str
    message: str
    type: str


@dataclass(frozen=True)
class GenerationCompletion:
    """One provider attempt's text and content-free input observation."""

    text: str
    generation_envelope: Mapping[str, Any] | None = None


class GenerationAttemptError(RuntimeError):
    """A failed provider attempt that still has a wire-envelope observation."""

    def __init__(self, message: str, *, generation_envelope: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.generation_envelope = generation_envelope


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
    provider_error: ProviderExecError | None = None
    embedding_provider_ms: float = 0.0
    profile_identity: str | None = None
    candidate_pool_scope: str | None = None
    candidate_exclusions: list[dict[str, str]] = field(default_factory=list)

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
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str | GenerationCompletion: ...


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
