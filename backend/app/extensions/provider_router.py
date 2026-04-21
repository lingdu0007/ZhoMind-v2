from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass
class ProviderRouter:
    providers: dict[str, Any]

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        text = str(exc).lower()
        return any(code in text for code in ["429", "500", "502", "503", "504", "timeout"])

    async def complete(self, *, primary: str, fallbacks: list[str], prompt: str, system_prompt: str | None = None) -> dict:
        order: list[str] = []
        for name in [primary, *fallbacks]:
            if name and name not in order:
                order.append(name)

        attempts: list[dict] = []
        text = ""
        final_provider = primary

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
                text = await provider.complete(prompt=prompt, system_prompt=system_prompt)
                latency_ms = int((time.perf_counter() - started) * 1000)
                final_provider = provider_name
                attempts.append(
                    {
                        "provider": provider_name,
                        "attempt": idx,
                        "latency_ms": latency_ms,
                        "error_code": None,
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
                    }
                )
                final_provider = provider_name
                if not self._is_retryable(exc):
                    break

        hops = max(0, len([x for x in attempts if x["error_code"]]))
        return {
            "text": text,
            "final_provider": final_provider,
            "provider_attempts": attempts,
            "fallback_hops": hops,
        }
