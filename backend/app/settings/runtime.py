from __future__ import annotations

from app.common.config import Settings, get_settings


class RuntimeApplicationError(Exception):
    """A bounded error returned when a saved version cannot become active."""


class SystemSettingsRuntime:
    def __init__(self) -> None:
        self._active_settings: Settings | None = None
        self._active_version: int | None = None

    def current_settings(self) -> Settings:
        return self._active_settings or get_settings()

    def validate(self, settings: dict) -> None:
        unsupported: dict[str, str] = {}
        if settings.get("model_provider") != "ark":
            unsupported["model_provider"] = "requires an Ark runtime lifecycle"
        if settings.get("retrieval_strategy") != "migration":
            unsupported["retrieval_strategy"] = "requires a migration retrieval lifecycle"
        if settings.get("index_name") != "zhomind_docs":
            unsupported["index_name"] = "requires a document reindex lifecycle"
        if settings.get("runtime_timeout_ms") != 8000:
            unsupported["runtime_timeout_ms"] = "requires a provider timeout lifecycle"
        if unsupported:
            error = RuntimeApplicationError("saved settings include unsupported runtime fields")
            error.fields = unsupported
            raise error

    def has_active_version(self, version: int) -> bool:
        return self._active_version == version

    async def apply(self, *, version: int, settings: dict, provider_api_key: str | None) -> None:
        self.validate(settings)
        base = get_settings()
        updates = {
            "llm_model": settings["llm_model"],
            "embedding_model": settings["embedding_model"],
            "milvus_uri": settings["milvus_uri"],
            "runtime_retrieval_top_k": settings["retrieval_top_k"],
            "runtime_score_threshold": settings["score_threshold"],
        }
        if provider_api_key is not None:
            updates["ark_api_key"] = provider_api_key

        try:
            next_settings = base.model_copy(update=updates)
            self._active_settings = next_settings
            # New requests construct providers and Milvus clients from this snapshot.
            from app.extensions.registry import get_extension_registry
            from app.infra.milvus import get_milvus_provider

            get_extension_registry.cache_clear()
            get_milvus_provider.cache_clear()
            self._active_version = version
        except Exception as exc:  # pragma: no cover - defensive boundary around third-party caches
            self._active_settings = None
            self._active_version = None
            raise RuntimeApplicationError("runtime did not accept the saved configuration") from exc

    def reset(self) -> None:
        self._active_settings = None
        self._active_version = None


_runtime = SystemSettingsRuntime()


def get_system_settings_runtime() -> SystemSettingsRuntime:
    return _runtime


def get_runtime_settings() -> Settings:
    return _runtime.current_settings()
