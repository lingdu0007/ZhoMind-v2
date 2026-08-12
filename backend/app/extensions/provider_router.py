from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.rag.generation_observation import observed_generation_envelope


@dataclass
class ProviderRouter:
    providers: dict[str, Any]

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        text = str(exc).lower()
        return any(code in text for code in ["429", "500", "502", "503", "504", "timeout"])

    async def complete(
        self,
        *,
        primary: str,
        fallbacks: list[str],
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict:
        order: list[str] = []
        for name in [primary, *fallbacks]:
            if name and name not in order:
                order.append(name)

        attempts: list[dict] = []
        text = ""
        final_provider = primary
        generation_envelope: dict | None = None

        for idx, provider_name in enumerate(order, start=1):
            provider = self.providers.get(provider_name)
            if provider is None:
                attempts.append(
                    {
                        "provider": provider_name,
                        "attempt": idx,
                        "latency_ms": 0,
                        "error_code": "PROVIDER_NOT_CONFIGURED",
                    }
                )
                continue

            started = time.perf_counter()
            try:
                if hasattr(provider, "last_generation_envelope"):
                    provider.last_generation_envelope = None
                # The provider owns the final wire/message construction. Do
                # not recreate it here from the generic protocol arguments.
                text = await provider.complete(prompt=prompt, system_prompt=system_prompt)
                observed = observed_generation_envelope(getattr(provider, "last_generation_envelope", None))
                if observed is not None:
                    generation_envelope = observed
                latency_ms = int((time.perf_counter() - started) * 1000)
                final_provider = provider_name
                attempts.append(
                    {
                        "provider": provider_name,
                        "attempt": idx,
                        "latency_ms": latency_ms,
                        "error_code": None,
                        "generation_envelope": observed,
                    }
                )
                if text:
                    break
            except Exception as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                attempts.append(
                    {
                        "provider": provider_name,
                        "attempt": idx,
                        "latency_ms": latency_ms,
                        "error_code": type(exc).__name__,
                        "generation_envelope": observed_generation_envelope(
                            getattr(provider, "last_generation_envelope", None)
                        ),
                    }
                )
                observed = observed_generation_envelope(getattr(provider, "last_generation_envelope", None))
                if observed is not None:
                    generation_envelope = observed
                final_provider = provider_name
                if not self._is_retryable(exc):
                    break

        hops = max(0, len(attempts) - 1)
        return {
            "text": text,
            "final_provider": final_provider,
            "provider_attempts": attempts,
            "fallback_hops": hops,
            "generation_envelope": generation_envelope,
        }
