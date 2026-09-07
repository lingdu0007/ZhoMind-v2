import asyncio
import json

from app.extensions.provider_router import ProviderRouter
from app.rag.generation_observation import provider_visible_snapshot_ids, wire_generation_envelope_observation
from app.rag.interfaces import GenerationAttemptError, GenerationCompletion
from tests.support.generation import approved_test_route


class _OkProvider:
    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return GenerationCompletion(
            text=self.text,
            generation_envelope=wire_generation_envelope_observation(
            wire_payload=json.dumps(
                {"messages": ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                    {"role": "user", "content": prompt}
                ]},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            snapshot_ids=provider_visible_snapshot_ids(prompt),
            ),
        )


class _RetryableFailProvider:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise TimeoutError("upstream timeout")


class _HardFailProvider:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise GenerationAttemptError(
            "bad request",
            reason="authorization_failed",
            generation_envelope=wire_generation_envelope_observation(
                wire_payload=json.dumps({"messages": [{"role": "user", "content": prompt}]}, separators=(",", ":"))
                .encode("utf-8"),
                snapshot_ids=provider_visible_snapshot_ids(prompt),
            ),
        )


def test_router_failover_on_retryable_error() -> None:
    router = ProviderRouter(
        approved_route=approved_test_route("ark", "openai"),
        providers={
            "ark": _RetryableFailProvider(),
            "openai": _OkProvider("fallback-answer"),
        }
    )

    result = asyncio.run(router.complete(primary="ark", fallbacks=["openai"], prompt="hi"))

    assert result["text"] == "fallback-answer"
    assert result["final_provider"] == "openai"
    assert len(result["provider_attempts"]) == 2
    assert result["fallback_hops"] == 1


def test_router_stops_on_non_retryable_error() -> None:
    router = ProviderRouter(
        approved_route=approved_test_route("ark", "openai"),
        providers={
            "ark": _HardFailProvider(),
            "openai": _OkProvider("should-not-run"),
        }
    )

    result = asyncio.run(router.complete(primary="ark", fallbacks=["openai"], prompt="hi"))

    assert result["text"] == ""
    assert result["final_provider"] == "ark"
    assert len(result["provider_attempts"]) == 1


def test_router_observes_snapshot_ids_from_actual_provider_envelope() -> None:
    router = ProviderRouter(providers={"ark": _OkProvider("answer")}, approved_route=approved_test_route("ark"))
    prompt = json.dumps(
        {
            "user_question": "question",
            "evidence_sources": [
                {
                    "entry_id": "entry-1",
                    "entry_title": "Decision",
                    "domain": "operations",
                    "section_id": "recommendation",
                    "source_title": "Source",
                    "source_authority": "Authority",
                    "source_url": "https://example.com/source",
                    "source_version": "2026",
                    "review_date": "2026-08-12",
                    "publication_version": "v1",
                    "excerpt": "Observed evidence",
                }
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    result = asyncio.run(
        router.complete(
            primary="ark",
            fallbacks=[],
            prompt=prompt,
            system_prompt="policy",
        )
    )

    envelope = result["generation_envelope"]
    assert envelope["snapshot_ids"]
    assert len(envelope["identity"]) == 64


def test_router_does_not_observe_an_envelope_without_a_provider_call() -> None:
    result = asyncio.run(ProviderRouter(providers={}).complete(primary="ark", fallbacks=[], prompt="{}"))

    assert result["generation_envelope"] is None
    assert result["provider_failure"] is True


def test_router_observes_an_envelope_when_provider_call_fails() -> None:
    result = asyncio.run(
        ProviderRouter(providers={"ark": _HardFailProvider()}, approved_route=approved_test_route("ark")).complete(
            primary="ark",
            fallbacks=[],
            prompt='{"evidence_sources":[]}',
            system_prompt="policy",
        )
    )

    assert result["generation_envelope"] == {
        "identity": result["generation_envelope"]["identity"],
        "snapshot_ids": [],
        "source_count": 0,
    }
    assert len(result["generation_envelope"]["identity"]) == 64


def test_router_uses_provider_observation_instead_of_rebuilding_generic_arguments() -> None:
    router = ProviderRouter(providers={"ark": _OkProvider("answer")}, approved_route=approved_test_route("ark"))

    without_system_prompt = asyncio.run(
        router.complete(primary="ark", fallbacks=[], prompt='{"evidence_sources":[]}')
    )
    with_empty_system_prompt = asyncio.run(
        router.complete(primary="ark", fallbacks=[], prompt='{"evidence_sources":[]}', system_prompt="")
    )

    assert without_system_prompt["generation_envelope"]["identity"] == with_empty_system_prompt["generation_envelope"]["identity"]


def test_router_keeps_concurrent_provider_observations_request_scoped() -> None:
    class _ConcurrentProvider:
        async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
            await asyncio.sleep(0)
            return GenerationCompletion(
                text=prompt,
                generation_envelope=wire_generation_envelope_observation(
                    wire_payload=prompt.encode("utf-8"),
                    snapshot_ids=provider_visible_snapshot_ids(prompt),
                ),
            )

    first_prompt = '{"evidence_sources":[]}'
    second_prompt = '{"evidence_sources":[{"title":"t","publication_version":"v1","excerpt":"e"}]}'
    router = ProviderRouter(providers={"ark": _ConcurrentProvider()}, approved_route=approved_test_route("ark"))
    async def run_concurrently() -> tuple[dict, dict]:
        return await asyncio.gather(
            router.complete(primary="ark", fallbacks=[], prompt=first_prompt),
            router.complete(primary="ark", fallbacks=[], prompt=second_prompt),
        )

    first, second = asyncio.run(run_concurrently())

    assert first["generation_envelope"]["source_count"] == 0
    assert second["generation_envelope"]["source_count"] == 1
    assert first["generation_envelope"]["identity"] != second["generation_envelope"]["identity"]


def test_router_does_not_retain_observation_from_failed_attempt_when_fallback_has_none() -> None:
    class _RetryableObservedFailure:
        async def complete(self, prompt: str, *, system_prompt: str | None = None) -> GenerationCompletion:
            raise GenerationAttemptError(
                "upstream timeout",
                reason="timeout",
                generation_envelope=wire_generation_envelope_observation(
                    wire_payload=b"failed-attempt",
                    snapshot_ids=(),
                ),
            )

    class _FallbackWithoutObservation:
        async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
            return "fallback"

    result = asyncio.run(
        ProviderRouter(
            providers={"primary": _RetryableObservedFailure(), "fallback": _FallbackWithoutObservation()},
            approved_route=approved_test_route("primary", "fallback"),
        ).complete(
            primary="primary", fallbacks=["fallback"], prompt="question"
        )
    )

    assert result["final_provider"] == "fallback"
    assert result["generation_envelope"] is None
