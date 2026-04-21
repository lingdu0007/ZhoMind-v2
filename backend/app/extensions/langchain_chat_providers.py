from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.interfaces import LlmProvider


@dataclass
class OpenAICompatibleChatProvider(LlmProvider):
    model: object
    provider_name: str

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = await self.model.ainvoke(messages)
        return str(getattr(response, "content", "") or "").strip()


@dataclass
class AnthropicChatProvider(LlmProvider):
    model: object
    provider_name: str

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        response = await self.model.ainvoke(messages)
        return str(getattr(response, "content", "") or "").strip()
