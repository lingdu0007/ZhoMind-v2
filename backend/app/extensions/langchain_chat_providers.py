from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
import openai
from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.generation_observation import provider_visible_snapshot_ids, wire_generation_envelope_observation
from app.rag.interfaces import GenerationAttemptError, GenerationCompletion, LlmProvider


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, openai.APITimeoutError, anthropic.APITimeoutError)):
        return "timeout"
    if isinstance(exc, (ConnectionError, openai.APIConnectionError, anthropic.APIConnectionError)):
        return "connection_failure"
    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return "authorization_failed"
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("code") in {"content_filter", "policy_violation", "safety_refusal"}:
            return "safety_refusal"
    if status == 429:
        return "rate_limit"
    if status in {500, 502, 503, 504, 529}:
        return "temporary_service_error"
    return "service_error" if isinstance(status, int) and 400 <= status <= 599 else "application_failure"


def _check_refusal(response: object, observation: dict) -> None:
    extra = getattr(response, "additional_kwargs", {}) or {}
    metadata = getattr(response, "response_metadata", {}) or {}
    if ((isinstance(extra, dict) and extra.get("refusal"))
            or (isinstance(metadata, dict) and (
                metadata.get("finish_reason") == "content_filter" or metadata.get("stop_reason") == "refusal"
            ))):
        raise GenerationAttemptError("generation refused", reason="safety_refusal", generation_envelope=observation)


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
            raise GenerationAttemptError("generation failed", reason=_failure_reason(exc), generation_envelope=observation) from exc
        _check_refusal(response, observation)
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
            raise GenerationAttemptError("generation failed", reason=_failure_reason(exc), generation_envelope=observation) from exc
        _check_refusal(response, observation)
        return GenerationCompletion(text=str(getattr(response, "content", "") or "").strip(), generation_envelope=observation)
