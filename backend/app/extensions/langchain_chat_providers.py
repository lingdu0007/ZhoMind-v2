from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.generation_observation import provider_visible_snapshot_ids, wire_generation_envelope_observation
from app.rag.interfaces import LlmProvider


@dataclass
class OpenAICompatibleChatProvider(LlmProvider):
    model: object
    provider_name: str

    def __post_init__(self) -> None:
        self.last_generation_envelope: dict | None = None

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        wire_messages = [
            {"role": "system" if isinstance(message, SystemMessage) else "user", "content": message.content}
            for message in messages
        ]
        self.last_generation_envelope = wire_generation_envelope_observation(
            wire_payload=json.dumps(
                {"messages": wire_messages}, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            snapshot_ids=provider_visible_snapshot_ids(prompt),
        )
        response = await self.model.ainvoke(messages)
        return str(getattr(response, "content", "") or "").strip()


@dataclass
class AnthropicChatProvider(LlmProvider):
    model: object
    provider_name: str

    def __post_init__(self) -> None:
        self.last_generation_envelope: dict | None = None

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        wire_messages = [
            {"role": "system" if isinstance(message, SystemMessage) else "user", "content": message.content}
            for message in messages
        ]
        self.last_generation_envelope = wire_generation_envelope_observation(
            wire_payload=json.dumps(
                {"messages": wire_messages}, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            snapshot_ids=provider_visible_snapshot_ids(prompt),
        )
        response = await self.model.ainvoke(messages)
        return str(getattr(response, "content", "") or "").strip()
