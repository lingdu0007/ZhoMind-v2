from __future__ import annotations

import json

from httpx import AsyncClient, HTTPError, Timeout

from app.rag.generation_observation import provider_visible_snapshot_ids, wire_generation_envelope_observation
from app.rag.interfaces import GenerationAttemptError, GenerationCompletion, LlmProvider


class ArkLlmProvider(LlmProvider):
    name = "ark-chat-completions"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
        if not self.api_key or not self.model or not self.base_url:
            return GenerationCompletion(text="")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
        wire_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        observation = wire_generation_envelope_observation(
            wire_payload=wire_body,
            snapshot_ids=provider_visible_snapshot_ids(prompt),
        )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with AsyncClient(timeout=Timeout(self.timeout_seconds)) as client:
                response = await client.post(url, headers=headers, content=wire_body)
                response.raise_for_status()
                data = response.json()
        except (HTTPError, ValueError) as exc:
            raise GenerationAttemptError(str(exc), generation_envelope=observation) from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return GenerationCompletion(text="", generation_envelope=observation)

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        return GenerationCompletion(text=str(content or "").strip(), generation_envelope=observation)
