from app.extensions.registry import ExtensionRegistry


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
