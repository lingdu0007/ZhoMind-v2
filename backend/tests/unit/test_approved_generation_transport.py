from types import SimpleNamespace

import httpx
import pytest

from app.extensions.generation_factory import build_generation_provider


@pytest.mark.asyncio
async def test_declared_endpoint_cannot_redirect_or_retry_and_clients_close(monkeypatch):
    observed = []

    class Model:
        def __init__(self, **kwargs):
            observed.append(kwargs)

        async def ainvoke(self, messages):
            return SimpleNamespace(content="OK", additional_kwargs={}, response_metadata={})

    monkeypatch.setattr("app.extensions.generation_factory.ChatOpenAI", Model)
    provider = build_generation_provider(SimpleNamespace(
        rag_primary_llm_provider="openai", openai_api_key="fixture-only",
        openai_model="model-v1", openai_base_url="https://approved.example.test/v1",
    ), timeout_seconds=3)
    assert provider is not None
    await provider.complete("Connection validation. Reply with OK.")
    assert observed[0]["max_retries"] == 0
    assert observed[0]["timeout"] == 3
    assert isinstance(observed[0]["http_async_client"], httpx.AsyncClient)
    assert not observed[0]["http_async_client"].follow_redirects
    assert not observed[0]["http_client"].follow_redirects
    assert observed[0]["http_async_client"].is_closed
    assert observed[0]["http_client"].is_closed


def test_local_validation_mode_can_never_construct_a_real_generation_provider(monkeypatch):
    from app.common.config import get_settings

    monkeypatch.setattr(get_settings(), "generation_validation_mode", "local_development", raising=False)
    assert build_generation_provider(SimpleNamespace(
        rag_primary_llm_provider="openai", openai_api_key="fixture-only",
        openai_model="model-v1", openai_base_url="https://approved.example.test/v1",
    )) is None
