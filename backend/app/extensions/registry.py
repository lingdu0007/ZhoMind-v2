from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.extensions.generation_factory import build_generation_provider
from app.extensions.langchain_embedding_providers import OpenAIEmbeddingProvider
from app.rag.claim_evidence import ClaimResolver
from app.rag.dense_contract import build_embedding_contract_fingerprint
from app.rag.interfaces import EmbeddingProvider, LlmProvider, RelevanceJudge, Reranker, Retriever
from app.rag.semantic_claim_resolver import SemanticClaimResolverArtifactError, load_semantic_claim_resolver_file
from app.settings.runtime import get_runtime_settings
from app.tasks.interfaces import InMemoryTaskBackend, TaskBackend, create_inmemory_task_backend


@dataclass
class ExtensionRegistry:
    llm_providers: dict[str, LlmProvider] = field(default_factory=dict)
    embedding_providers: dict[str, EmbeddingProvider] = field(default_factory=dict)
    rerank_providers: dict[str, Reranker] = field(default_factory=dict)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    judges: dict[str, RelevanceJudge] = field(default_factory=dict)
    claim_resolvers: dict[str, ClaimResolver] = field(default_factory=dict)
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

    def register_claim_resolver(self, name: str, resolver: ClaimResolver) -> None:
        self.claim_resolvers[name] = resolver

    def get_claim_resolver(self, name: str) -> ClaimResolver | None:
        return self.claim_resolvers.get(name)

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
    settings = get_runtime_settings()
    # Agent evidence stays fail-closed until trusted process bootstrap registers
    # an independently calibrated ClaimResolver.
    if settings.runtime_generation_settings_managed:
        generation_provider = build_generation_provider(settings)
        if generation_provider is not None:
            registry.register_llm(settings.rag_primary_llm_provider, generation_provider)
            registry.register_llm("chat-default-llm", generation_provider)
    else:
        for provider_type in ("ark", "openai", "anthropic"):
            candidate = settings.model_copy(update={"rag_primary_llm_provider": provider_type})
            provider = build_generation_provider(candidate)
            if provider is not None:
                registry.register_llm(provider_type, provider)
        if "ark" in registry.llm_providers:
            registry.register_llm("chat-default-llm", registry.llm_providers["ark"])

    if (
        settings.embedding_api_key_configured
        and settings.embedding_base_url_normalized
        and settings.embedding_model_normalized
        and settings.dense_embedding_dim > 0
    ):
        registry.register_embedding(
            "embedding-default",
            OpenAIEmbeddingProvider(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url_normalized,
                model=settings.embedding_model_normalized,
                dimensions=settings.dense_embedding_dim,
            ),
        )

    resolver_path = settings.claim_resolver_profile_path.strip()
    resolver_sha256 = settings.claim_resolver_profile_sha256.strip()
    if bool(resolver_path) != bool(resolver_sha256):
        raise SemanticClaimResolverArtifactError(
            "CLAIM_RESOLVER_PROFILE_PATH and CLAIM_RESOLVER_PROFILE_SHA256 must be configured together"
        )
    if resolver_path:
        embedding_provider = registry.get_embedding("embedding-default")
        if embedding_provider is None:
            raise SemanticClaimResolverArtifactError("claim resolver requires the active embedding provider")
        resolver = load_semantic_claim_resolver_file(
            Path(resolver_path),
            expected_sha256=resolver_sha256,
            embedding_provider=embedding_provider,
            active_embedding_model=settings.embedding_model_normalized,
            active_embedding_dimension=settings.dense_embedding_dim,
            active_embedding_contract_fingerprint=build_embedding_contract_fingerprint(settings),
        )
        registry.register_claim_resolver("chat-default-claim-resolver", resolver)

    registry.register_task_backend("inmemory", create_inmemory_task_backend())
    return registry


def get_task_backend(name: str = "inmemory") -> TaskBackend:
    backend = get_extension_registry().get_task_backend(name)
    if backend is None:
        return InMemoryTaskBackend()
    return backend
