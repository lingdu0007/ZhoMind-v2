import asyncio
import json
from types import SimpleNamespace

import pytest

from app.extensions.langchain_chat_providers import (
    AnthropicChatProvider,
    OpenAICompatibleChatProvider,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, text: str) -> None:
        self._text = text

    async def ainvoke(self, messages):
        return _FakeMessage(self._text)


def test_openai_compatible_provider_returns_trimmed_text() -> None:
    provider = OpenAICompatibleChatProvider(model=_FakeModel("  hello  "), provider_name="ark")
    completion = asyncio.run(provider.complete("prompt"))
    assert completion.text == "hello"
    assert completion.generation_envelope is not None
    assert completion.generation_envelope["source_count"] == 0


def test_anthropic_provider_returns_trimmed_text() -> None:
    provider = AnthropicChatProvider(model=_FakeModel("  world  "), provider_name="anthropic")
    completion = asyncio.run(provider.complete("prompt"))
    assert completion.text == "world"
    assert completion.generation_envelope is not None


def test_langchain_observation_tracks_final_messages_without_prompt_text() -> None:
    provider = OpenAICompatibleChatProvider(model=_FakeModel("answer"), provider_name="ark")

    completion = asyncio.run(provider.complete("user envelope", system_prompt="policy"))

    assert completion.generation_envelope is not None
    assert completion.generation_envelope["identity"] == __import__("hashlib").sha256(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "user envelope"},
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("provider_type", [OpenAICompatibleChatProvider, AnthropicChatProvider])
@pytest.mark.parametrize("status,reason", [(401, "authorization_failed"), (403, "authorization_failed"),
                                         (429, "rate_limit"), (503, "temporary_service_error"), (400, "service_error")])
def test_provider_normalizes_structured_status_without_retainable_exception_text(provider_type, status, reason):
    from app.rag.interfaces import GenerationAttemptError

    class Failure(Exception):
        status_code = status

    class Model:
        async def ainvoke(self, messages):
            raise Failure("private credential timeout 503")

    with pytest.raises(GenerationAttemptError) as raised:
        asyncio.run(provider_type(model=Model(), provider_name="declared").complete("payload"))
    assert raised.value.reason == reason
    assert "private" not in str(raised.value)


@pytest.mark.parametrize("provider_type", [OpenAICompatibleChatProvider, AnthropicChatProvider])
def test_provider_preserves_explicit_safety_refusal_as_nonadvancing(provider_type):
    from app.rag.interfaces import GenerationAttemptError

    class Model:
        async def ainvoke(self, messages):
            return SimpleNamespace(content="", additional_kwargs={"refusal": "private refusal"},
                                   response_metadata={"stop_reason": "refusal", "finish_reason": "content_filter"})

    with pytest.raises(GenerationAttemptError) as raised:
        asyncio.run(provider_type(model=Model(), provider_name="declared").complete("payload"))
    assert raised.value.reason == "safety_refusal"
    assert "private" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_type", [OpenAICompatibleChatProvider, AnthropicChatProvider])
async def test_unclassified_sdk_application_error_cannot_be_hidden_by_fallback(provider_type):
    from app.extensions.provider_router import ProviderRouter
    from tests.support.generation import approved_test_route

    calls = []

    class Broken:
        async def ainvoke(self, messages):
            calls.append("primary")
            raise ValueError("private parse failure")

    class Fallback:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("fallback")
            return "must not run"

    result = await ProviderRouter(
        providers={"primary": provider_type(model=Broken(), provider_name="primary"), "fallback": Fallback()},
        approved_route=approved_test_route("primary", "fallback"),
    ).complete(primary="primary", fallbacks=[], prompt="payload")
    assert calls == ["primary"]
    assert result["route_reason"] == "application_failure"
    assert "private" not in str(result)
