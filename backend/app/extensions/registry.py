from dataclasses import dataclass, field
from functools import lru_cache

from app.rag.interfaces import EmbeddingProvider, LlmProvider, RelevanceJudge, Reranker, Retriever
from app.tasks.interfaces import InMemoryTaskBackend, TaskBackend, create_inmemory_task_backend


@dataclass
class ExtensionRegistry:
    llm_providers: dict[str, LlmProvider] = field(default_factory=dict)
    embedding_providers: dict[str, EmbeddingProvider] = field(default_factory=dict)
    rerank_providers: dict[str, Reranker] = field(default_factory=dict)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    judges: dict[str, RelevanceJudge] = field(default_factory=dict)
    task_backends: dict[str, TaskBackend] = field(default_factory=dict)
    capabilities: dict[str, dict[str, dict]] = field(
        default_factory=lambda: {
            "llm": {},
            "retriever": {},
            "reranker": {},
            "judge": {},
            "tool": {},
        }
    )

    def register_llm(self, name: str, provider: LlmProvider) -> None:
        self.llm_providers[name] = provider

    def get_llm(self, name: str) -> LlmProvider | None:
        return self.llm_providers.get(name)

    def register_embedding(self, name: str, provider: EmbeddingProvider) -> None:
        self.embedding_providers[name] = provider

    def get_embedding(self, name: str) -> EmbeddingProvider | None:
        return self.embedding_providers.get(name)

    def register_rerank(self, name: str, provider: Reranker) -> None:
        self.rerank_providers[name] = provider

    def get_rerank(self, name: str) -> Reranker | None:
        return self.rerank_providers.get(name)

    def register_retriever(self, name: str, provider: Retriever) -> None:
        self.retrievers[name] = provider

    def get_retriever(self, name: str) -> Retriever | None:
        return self.retrievers.get(name)

    def register_judge(self, name: str, provider: RelevanceJudge) -> None:
        self.judges[name] = provider

    def get_judge(self, name: str) -> RelevanceJudge | None:
        return self.judges.get(name)

    def register_task_backend(self, name: str, backend: TaskBackend) -> None:
        self.task_backends[name] = backend

    def get_task_backend(self, name: str) -> TaskBackend | None:
        return self.task_backends.get(name)

    def register_capability(self, kind: str, name: str, capability: dict) -> None:
        bucket = self.capabilities.setdefault(kind, {})
        bucket[name] = capability

    def get_capability(self, kind: str, name: str) -> dict | None:
        return self.capabilities.get(kind, {}).get(name)

    def choose_provider(self, kind: str, candidates: list[str], required: dict) -> str | None:
        for name in candidates:
            capability = self.get_capability(kind, name) or {}
            if all(capability.get(k) == v for k, v in required.items()):
                return name
        return None


@lru_cache
def get_extension_registry() -> ExtensionRegistry:
    registry = ExtensionRegistry()
    registry.register_task_backend("inmemory", create_inmemory_task_backend())
    return registry


def get_task_backend(name: str = "inmemory") -> TaskBackend:
    backend = get_extension_registry().get_task_backend(name)
    if backend is None:
        return InMemoryTaskBackend()
    return backend
