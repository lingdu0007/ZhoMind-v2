import asyncio
import json

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
    text = asyncio.run(provider.complete("prompt"))
    assert text == "hello"
    assert provider.last_generation_envelope is not None
    assert provider.last_generation_envelope["source_count"] == 0


def test_anthropic_provider_returns_trimmed_text() -> None:
    provider = AnthropicChatProvider(model=_FakeModel("  world  "), provider_name="anthropic")
    text = asyncio.run(provider.complete("prompt"))
    assert text == "world"
    assert provider.last_generation_envelope is not None


def test_langchain_observation_tracks_final_messages_without_prompt_text() -> None:
    provider = OpenAICompatibleChatProvider(model=_FakeModel("answer"), provider_name="ark")

    asyncio.run(provider.complete("user envelope", system_prompt="policy"))

    assert provider.last_generation_envelope is not None
    assert provider.last_generation_envelope["identity"] == __import__("hashlib").sha256(
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
