import asyncio

from app.settings.runtime import SystemSettingsRuntime


class _ValidatedProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del system_prompt
        self.prompts.append(prompt)
        return "OK"


def _provider_settings(*, model: str) -> dict[str, str]:
    return {
        "provider_type": "openai",
        "model": model,
        "service_url": "https://provider.example.test/v1",
    }


def test_runtime_validates_connectivity_without_disclosing_the_provider_key(monkeypatch) -> None:
    runtime = SystemSettingsRuntime()
    provider = _ValidatedProvider()
    monkeypatch.setattr("app.settings.runtime.build_generation_provider", lambda _: provider)

    asyncio.run(runtime.validate(settings=_provider_settings(model="gpt-4o-mini"), provider_api_key="test-secret"))

    assert provider.prompts == ["Connection validation. Reply with OK."]


def test_provider_replacement_leaves_existing_request_snapshot_unchanged() -> None:
    runtime = SystemSettingsRuntime()

    asyncio.run(runtime.apply(version=1, settings=_provider_settings(model="model-v1"), provider_api_key="key-v1"))
    in_flight_snapshot = runtime.current_settings()
    asyncio.run(runtime.apply(version=2, settings=_provider_settings(model="model-v2"), provider_api_key="key-v2"))

    assert in_flight_snapshot.openai_model == "model-v1"
    assert runtime.current_settings().openai_model == "model-v2"
    assert runtime.has_active_version(2) is True
