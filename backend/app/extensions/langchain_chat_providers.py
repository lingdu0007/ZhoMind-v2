from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.generation_observation import provider_visible_snapshot_ids, wire_generation_envelope_observation
from app.rag.interfaces import GenerationAttemptError, GenerationCompletion, LlmProvider


@dataclass
class OpenAICompatibleChatProvider(LlmProvider):
    model: object
    provider_name: str

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        wire_messages = [
            {"role": "system" if isinstance(message, SystemMessage) else "user", "content": message.content}
            for message in messages
        ]
        observation = wire_generation_envelope_observation(
            wire_payload=json.dumps(
                {"messages": wire_messages}, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            snapshot_ids=provider_visible_snapshot_ids(prompt),
        )
        try:
            response = await self.model.ainvoke(messages)
        except Exception as exc:
            raise GenerationAttemptError(str(exc), generation_envelope=observation) from exc
        return GenerationCompletion(text=str(getattr(response, "content", "") or "").strip(), generation_envelope=observation)


@dataclass
class AnthropicChatProvider(LlmProvider):
    model: object
    provider_name: str

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        wire_messages = [
            {"role": "system" if isinstance(message, SystemMessage) else "user", "content": message.content}
            for message in messages
        ]
        observation = wire_generation_envelope_observation(
            wire_payload=json.dumps(
                {"messages": wire_messages}, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            snapshot_ids=provider_visible_snapshot_ids(prompt),
        )
        try:
            response = await self.model.ainvoke(messages)
        except Exception as exc:
            raise GenerationAttemptError(str(exc), generation_envelope=observation) from exc
        return GenerationCompletion(text=str(getattr(response, "content", "") or "").strip(), generation_envelope=observation)
