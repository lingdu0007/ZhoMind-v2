from __future__ import annotations

from app.common.config import Settings, get_settings
from app.extensions.generation_factory import build_generation_provider


class RuntimeApplicationError(Exception):
    """A bounded error returned when a saved version cannot become active."""


class SystemSettingsRuntime:
    def __init__(self) -> None:
        self._active_settings: Settings | None = None
        self._active_version: int | None = None

    def current_settings(self) -> Settings:
        return self._active_settings or get_settings()

    async def validate(self, *, settings: dict, provider_api_key: str | None) -> None:
        candidate = self._candidate_settings(settings=settings, provider_api_key=provider_api_key)
        provider = build_generation_provider(candidate)
        if provider is None:
            raise RuntimeApplicationError("runtime rejected the saved configuration")
        try:
            response = await provider.complete("Connection validation. Reply with OK.")
        except Exception as exc:
            raise RuntimeApplicationError("runtime rejected the saved configuration") from exc
        if not response:
            raise RuntimeApplicationError("runtime rejected the saved configuration")

    def has_active_version(self, version: int) -> bool:
        return self._active_version == version

    async def apply(self, *, version: int, settings: dict, provider_api_key: str | None) -> None:
        next_settings = self._candidate_settings(settings=settings, provider_api_key=provider_api_key)
        try:
            self._active_settings = next_settings
            # New requests construct providers and Milvus clients from this snapshot.
            from app.extensions.registry import get_extension_registry
            from app.infra.milvus import get_milvus_provider

            get_extension_registry.cache_clear()
            get_milvus_provider.cache_clear()
            self._active_version = version
        except Exception as exc:  # pragma: no cover - defensive boundary around third-party caches
            raise RuntimeApplicationError("runtime did not accept the saved configuration") from exc

    @staticmethod
    def _candidate_settings(*, settings: dict, provider_api_key: str | None) -> Settings:
        if not provider_api_key:
            raise RuntimeApplicationError("runtime rejected the saved configuration")
        provider_type = settings.get("provider_type")
        model = settings.get("model")
        service_url = settings.get("service_url")
        updates = {
            "rag_primary_llm_provider": provider_type,
            "runtime_generation_settings_managed": True,
        }
        if provider_type == "ark":
            updates.update({"ark_api_key": provider_api_key, "llm_model": model, "llm_base_url": service_url})
        elif provider_type == "openai":
            updates.update({"openai_api_key": provider_api_key, "openai_model": model, "openai_base_url": service_url})
        elif provider_type == "anthropic":
            updates.update({"anthropic_api_key": provider_api_key, "anthropic_model": model, "anthropic_base_url": service_url})
        else:
            raise RuntimeApplicationError("runtime rejected the saved configuration")
        return get_settings().model_copy(update=updates)

    def reset(self) -> None:
        self._active_settings = None
        self._active_version = None


_runtime = SystemSettingsRuntime()


def get_system_settings_runtime() -> SystemSettingsRuntime:
    return _runtime


def get_runtime_settings() -> Settings:
    return _runtime.current_settings()
