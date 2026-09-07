from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.common.canonical_json import canonical_json_sha256
from app.common.generation_audit import publish_generation_audit
from app.rag.generation_observation import observed_generation_envelope, provider_visible_snapshot_ids
from app.rag.interfaces import GenerationAttemptError, GenerationCompletion

ADVANCE_REASONS = frozenset({
    "connection_failure", "timeout", "rate_limit", "temporary_service_error",
    "service_error", "answer_structure_invalid", "citation_invalid",
})
STOP_REASONS = frozenset({
    "insufficient_evidence", "user_cancellation", "authorization_failed",
    "data_scope_mismatch", "safety_refusal", "policy_refusal", "unapproved_provider",
    "application_failure", "privacy_refusal",
})


def normalized_generation_error(exc: Exception) -> str:
    if isinstance(exc, GenerationAttemptError):
        return exc.reason if exc.reason in ADVANCE_REASONS | STOP_REASONS else "application_failure"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection_failure"
    return "application_failure"


@dataclass(frozen=True)
class ApprovedRouteProvider:
    approval_identity: str
    provider: str
    model: str
    data_scope: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._/-]{0,191}", value)
               for value in (self.approval_identity, self.provider, self.model, self.data_scope)):
            raise ValueError("invalid approved provider identity")
        if isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 60:
            raise ValueError("invalid provider timeout")


@dataclass(frozen=True)
class ApprovedGenerationRoute:
    identity: str
    data_scope: str
    providers: tuple[ApprovedRouteProvider, ...]
    max_attempts: int
    total_timeout_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", tuple(self.providers))
        if not self.identity.startswith("provider_route:") or not self.data_scope:
            raise ValueError("invalid route identity or scope")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= len(self.providers) <= 4:
            raise ValueError("invalid route attempt bound")
        if len({item.provider for item in self.providers}) != len(self.providers):
            raise ValueError("duplicate provider in route")
        if (isinstance(self.total_timeout_seconds, bool) or not math.isfinite(self.total_timeout_seconds)
                or not 0 < self.total_timeout_seconds <= 60):
            raise ValueError("invalid total route budget")


@dataclass
class ProviderRouter:
    providers: dict[str, Any]
    approved_route: ApprovedGenerationRoute | None = None

    async def complete(
        self,
        *,
        primary: str,
        fallbacks: list[str],
        prompt: str,
        system_prompt: str | None = None,
        data_scope: str = "team_shared_pilot",
        validate_answer: Callable[[str], str | None] | None = None,
    ) -> dict:
        route = self.approved_route
        denial = None
        if route is None or any(item.provider not in self.providers for item in route.providers):
            denial = "unapproved_provider"
        elif data_scope != route.data_scope or any(item.data_scope != data_scope for item in route.providers):
            denial = "data_scope_mismatch"
        if denial or route is None:
            publish_generation_audit({"route_identity": route.identity if route else None,
                                      "route_reason": denial, "provider_attempts": []})
            return {
                "text": "",
                "final_provider": None,
                "provider_attempts": [],
                "fallback_hops": 0,
                "generation_envelope": None,
                "provider_failure": True,
                "route_reason": denial,
            }
        order = [item.provider for item in route.providers[:route.max_attempts]]
        payload_sha256 = canonical_json_sha256({"prompt": prompt, "system_prompt": system_prompt})
        snapshot_ids = list(provider_visible_snapshot_ids(prompt))
        snapshot_sha256 = canonical_json_sha256(snapshot_ids)

        attempts: list[dict] = []
        text = ""
        final_provider = order[0]
        generation_envelope: dict | None = None
        generation_envelope_invalid = False
        completed_provider_call = False
        deadline = time.perf_counter() + route.total_timeout_seconds
        reason = "timeout"
        try:
            for idx, provider_name in enumerate(order, start=1):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    reason = "timeout"
                    break
                final_provider = provider_name
                started = time.perf_counter()
                attempt: dict = {
                    "provider": provider_name,
                    "approval_identity": route.providers[idx - 1].approval_identity,
                    "payload_sha256": payload_sha256,
                    "snapshot_sha256": snapshot_sha256,
                    "attempt": idx,
                    "latency_ms": 0,
                    "error_code": None,
                }
                attempts.append(attempt)
                raw_envelope = None
                reason = None
                try:
                    async with asyncio.timeout(min(remaining, route.providers[idx - 1].timeout_seconds)):
                        completion = await self.providers[provider_name].complete(prompt=prompt, system_prompt=system_prompt)
                    text = completion.text if isinstance(completion, GenerationCompletion) else str(completion or "")
                    raw_envelope = completion.generation_envelope if isinstance(completion, GenerationCompletion) else None
                    completed_provider_call = True
                except asyncio.CancelledError:
                    reason = "user_cancellation"
                    attempt["error_code"] = reason
                    raise
                except Exception as exc:
                    raw_envelope = getattr(exc, "generation_envelope", None)
                    reason = normalized_generation_error(exc)
                finally:
                    attempt["latency_ms"] = int((time.perf_counter() - started) * 1000)
                observed = observed_generation_envelope(raw_envelope)
                invalid = raw_envelope is not None and (observed is None or observed["snapshot_ids"] != snapshot_ids)
                generation_envelope = observed
                generation_envelope_invalid = generation_envelope_invalid or invalid
                attempt.update(generation_envelope=observed, generation_envelope_invalid=invalid)
                if invalid:
                    reason = "application_failure"
                elif reason is None:
                    reason = validate_answer(text) if validate_answer else (None if text else "answer_structure_invalid")
                    if reason is not None and reason not in ADVANCE_REASONS | STOP_REASONS:
                        reason = "application_failure"
                attempt["error_code"] = reason
                if reason is not None:
                    text = ""
                if text or reason not in ADVANCE_REASONS:
                    break
        except Exception:
            text = ""
            reason = "application_failure"
            if attempts:
                attempts[-1]["error_code"] = reason
            raise
        finally:
            publish_generation_audit({
                "route_identity": route.identity, "route_reason": "succeeded" if text else reason,
                "provider_attempts": [
                    {key: value for key, value in item.items() if key not in {"generation_envelope", "generation_envelope_invalid"}}
                    for item in attempts
                ],
            })

        hops = max(0, len(attempts) - 1)
        return {
            "text": text,
            "final_provider": final_provider,
            "provider_attempts": attempts,
            "fallback_hops": hops,
            "generation_envelope": generation_envelope,
            "generation_envelope_invalid": generation_envelope_invalid,
            "provider_failure": not completed_provider_call,
            "route_identity": route.identity,
            "route_reason": "succeeded" if text else reason,
        }
