from app.common.config import get_settings
from app.extensions.registry import ExtensionRegistry, get_extension_registry


def test_capability_registry_roundtrip() -> None:
    registry = ExtensionRegistry()
    registry.register_capability("llm", "chat-default-llm", {"supports_stream": True, "cost_tier": "medium"})
    cap = registry.get_capability("llm", "chat-default-llm")
    assert cap is not None
    assert cap["supports_stream"] is True


def test_choose_provider_by_capability() -> None:
    registry = ExtensionRegistry()
    registry.register_capability("llm", "provider-a", {"supports_stream": False})
    registry.register_capability("llm", "provider-b", {"supports_stream": True})
    assert registry.choose_provider("llm", ["provider-a", "provider-b"], {"supports_stream": True}) == "provider-b"


def test_registry_registers_multi_llm_providers(monkeypatch) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()

    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    monkeypatch.setenv("BASE_URL", "https://ark.example.com/v3")
    monkeypatch.setenv("MODEL", "ark-model")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    registry = get_extension_registry()

    assert registry.get_llm("ark") is not None
    assert registry.get_llm("openai") is not None
    assert registry.get_llm("anthropic") is not None

    get_settings.cache_clear()
    get_extension_registry.cache_clear()
