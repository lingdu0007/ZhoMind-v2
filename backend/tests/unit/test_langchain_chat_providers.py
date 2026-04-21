import asyncio

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


def test_anthropic_provider_returns_trimmed_text() -> None:
    provider = AnthropicChatProvider(model=_FakeModel("  world  "), provider_name="anthropic")
    text = asyncio.run(provider.complete("prompt"))
    assert text == "world"
