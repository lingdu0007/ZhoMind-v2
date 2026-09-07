import pytest

from app.extensions.provider_router import ProviderRouter


def approved_route(*names):
    from app.extensions.provider_router import ApprovedGenerationRoute, ApprovedRouteProvider

    return ApprovedGenerationRoute(
        identity="provider_route:route-v1",
        data_scope="team_shared_pilot",
        providers=tuple(
            ApprovedRouteProvider(f"approval:{name}-v1", name, "model-v1", "team_shared_pilot", 1.0)
            for name in names
        ),
        max_attempts=len(names),
        total_timeout_seconds=2.0,
    )


@pytest.mark.asyncio
async def test_available_provider_without_an_approved_route_receives_no_payload() -> None:
    calls = []

    class AvailableProvider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append(prompt)
            return "unapproved answer"

    result = await ProviderRouter(providers={"available": AvailableProvider()}).complete(
        primary="available", fallbacks=[], prompt="private question",
    )

    assert calls == []
    assert result["text"] == ""
    assert result["route_reason"] == "unapproved_provider"


@pytest.mark.asyncio
async def test_route_budget_cancels_and_awaits_timed_out_attempt_without_starting_fallback() -> None:
    import asyncio
    from dataclasses import replace

    stopped = asyncio.Event()
    calls = []

    class Slow:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("primary")
            try:
                await asyncio.sleep(0.1)
                return "late answer"
            finally:
                stopped.set()

    class Fallback:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("fallback")
            return "not allowed after budget"

    route = replace(approved_route("primary", "fallback"), total_timeout_seconds=0.01)
    result = await ProviderRouter(
        providers={"primary": Slow(), "fallback": Fallback()}, approved_route=route,
    ).complete(primary="ignored", fallbacks=[], prompt="payload")

    assert stopped.is_set()
    assert calls == ["primary"]
    assert result["text"] == ""
    assert result["route_reason"] == "timeout"


@pytest.mark.asyncio
async def test_approved_route_uses_its_declared_order_not_caller_fallbacks() -> None:
    from app.extensions.provider_router import ApprovedGenerationRoute, ApprovedRouteProvider

    calls = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append((prompt, system_prompt))
            return "contract-valid"

    route = ApprovedGenerationRoute(
        identity="provider_route:route-v1",
        data_scope="team_shared_pilot",
        providers=(ApprovedRouteProvider("approval:primary-v1", "primary", "model-v1", "team_shared_pilot", 1.0),),
        max_attempts=1,
        total_timeout_seconds=2.0,
    )
    result = await ProviderRouter(providers={"primary": Provider()}, approved_route=route).complete(
        primary="injected", fallbacks=["undeclared"], prompt="immutable payload", system_prompt="policy",
    )

    assert calls == [("immutable payload", "policy")]
    assert result["text"] == "contract-valid"
    assert result["route_identity"] == "provider_route:route-v1"
    assert result["provider_attempts"][0]["approval_identity"] == "approval:primary-v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "connection_failure", "timeout", "rate_limit", "temporary_service_error",
    "service_error", "answer_structure_invalid", "citation_invalid",
])
async def test_only_normalized_advance_reasons_reuse_the_immutable_payload(reason) -> None:
    from app.rag.interfaces import GenerationAttemptError

    calls = []

    class Primary:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append((prompt, system_prompt))
            raise GenerationAttemptError("private error 429", reason=reason)

    class Fallback:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append((prompt, system_prompt))
            return "valid"

    result = await ProviderRouter(
        providers={"primary": Primary(), "fallback": Fallback()},
        approved_route=approved_route("primary", "fallback"),
    ).complete(primary="ignored", fallbacks=[], prompt='{"private":"payload"}', system_prompt="policy")

    assert calls == [('{"private":"payload"}', "policy")] * 2
    assert result["text"] == "valid"
    assert result["provider_attempts"][0]["error_code"] == reason
    assert result["provider_attempts"][0]["payload_sha256"] == result["provider_attempts"][1]["payload_sha256"]
    assert result["provider_attempts"][0]["snapshot_sha256"] == result["provider_attempts"][1]["snapshot_sha256"]
    assert "private" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [
    "insufficient_evidence", "user_cancellation", "authorization_failed",
    "data_scope_mismatch", "safety_refusal", "policy_refusal", "unapproved_provider",
])
async def test_protected_reason_never_advances_even_when_error_text_says_timeout(reason) -> None:
    from app.rag.interfaces import GenerationAttemptError

    calls = []

    class Primary:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("primary")
            raise GenerationAttemptError("503 timeout with private content", reason=reason)

    class Fallback:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("fallback")
            return "must not run"

    result = await ProviderRouter(
        providers={"primary": Primary(), "fallback": Fallback()},
        approved_route=approved_route("primary", "fallback"),
    ).complete(primary="ignored", fallbacks=[], prompt="payload")

    assert calls == ["primary"]
    assert result["route_reason"] == reason
    assert result["provider_attempts"][0]["error_code"] == reason
    assert result["text"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["missing_primary", "provider_scope", "request_scope"])
async def test_route_rejects_unproven_provider_or_data_scope_before_any_call(mismatch) -> None:
    from dataclasses import replace

    calls = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append(prompt)
            return "not authorized"

    route = approved_route("primary", "fallback")
    providers = {"primary": Provider(), "fallback": Provider()}
    if mismatch == "missing_primary":
        del providers["primary"]
    elif mismatch == "provider_scope":
        route = replace(route, providers=(replace(route.providers[0], data_scope="foreign"), route.providers[1]))
    result = await ProviderRouter(providers=providers, approved_route=route).complete(
        primary="primary", fallbacks=[], prompt="payload",
        data_scope="foreign" if mismatch == "request_scope" else "team_shared_pilot",
    )

    assert calls == []
    assert result["route_reason"] == ("unapproved_provider" if mismatch == "missing_primary" else "data_scope_mismatch")


@pytest.mark.parametrize("changes", [
    {"max_attempts": 0}, {"max_attempts": True}, {"max_attempts": 9},
    {"total_timeout_seconds": float("nan")}, {"total_timeout_seconds": float("inf")},
    {"total_timeout_seconds": -1}, {"providers": ()},
])
def test_route_rejects_invalid_declared_bounds(changes) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(approved_route("primary"), **changes)


@pytest.mark.asyncio
async def test_invalid_answer_advances_but_invalid_input_observation_never_does() -> None:
    from app.rag.interfaces import GenerationCompletion

    calls = []

    class Primary:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("primary")
            return "invalid citation"

    class Fallback:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("fallback")
            return "valid citation"

    router = ProviderRouter(
        providers={"primary": Primary(), "fallback": Fallback()},
        approved_route=approved_route("primary", "fallback"),
    )
    result = await router.complete(
        primary="primary", fallbacks=[], prompt="payload",
        validate_answer=lambda text: None if text == "valid citation" else "citation_invalid",
    )
    assert calls == ["primary", "fallback"]
    assert result["text"] == "valid citation"
    assert result["provider_attempts"][0]["error_code"] == "citation_invalid"

    class Corrupt:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("corrupt")
            return GenerationCompletion(text="invalid citation", generation_envelope={"snapshot_ids": ["forged"]})

    calls.clear()
    router.providers["primary"] = Corrupt()
    result = await router.complete(
        primary="primary", fallbacks=[], prompt="payload",
        validate_answer=lambda _: "citation_invalid",
    )
    assert calls == ["corrupt"]
    assert result["generation_envelope_invalid"] is True
    assert result["text"] == ""


@pytest.mark.asyncio
async def test_valid_but_mismatched_attempt_observation_cannot_be_hidden_by_fallback():
    from app.rag.generation_observation import wire_generation_envelope_observation
    from app.rag.interfaces import GenerationCompletion

    calls = []

    class Primary:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("primary")
            return GenerationCompletion("invalid", wire_generation_envelope_observation(
                wire_payload=b"wrong payload", snapshot_ids=("f" * 64,),
            ))

    class Fallback:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append("fallback")
            return "valid"

    result = await ProviderRouter(
        providers={"primary": Primary(), "fallback": Fallback()}, approved_route=approved_route("primary", "fallback"),
    ).complete(primary="primary", fallbacks=[], prompt='{"evidence_sources":[]}',
               validate_answer=lambda answer: None if answer == "valid" else "citation_invalid")
    assert calls == ["primary"]
    assert result["generation_envelope_invalid"] is True
    assert result["text"] == ""


def test_environment_credentials_do_not_discover_or_register_generation_providers(monkeypatch):
    from app.common.config import get_settings
    from app.extensions.registry import get_extension_registry

    settings = get_settings().model_copy(update={
        "runtime_generation_settings_managed": False, "openai_api_key": "fixture-only",
        "openai_model": "fixture-model", "claim_resolver_profile_path": "", "claim_resolver_profile_sha256": "",
    })
    monkeypatch.setattr("app.extensions.registry.get_runtime_settings", lambda: settings)
    get_extension_registry.cache_clear()
    try:
        assert get_extension_registry().llm_providers == {}
    finally:
        get_extension_registry.cache_clear()


@pytest.mark.asyncio
async def test_task_cancellation_never_starts_another_provider():
    import asyncio

    started = asyncio.Event()
    stopped = asyncio.Event()
    calls = []

    class Provider:
        async def complete(self, prompt, *, system_prompt=None):
            calls.append(prompt)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    router = ProviderRouter(
        providers={"primary": Provider(), "fallback": Provider()},
        approved_route=approved_route("primary", "fallback"),
    )
    task = asyncio.create_task(router.complete(primary="ignored", fallbacks=[], prompt="frozen"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert calls == ["frozen"]


@pytest.mark.asyncio
async def test_empty_completion_advances_only_with_a_normalized_structure_failure():
    class Provider:
        def __init__(self, text):
            self.text = text

        async def complete(self, prompt, *, system_prompt=None):
            return self.text

    result = await ProviderRouter(
        providers={"primary": Provider(""), "fallback": Provider("valid")},
        approved_route=approved_route("primary", "fallback"),
    ).complete(primary="primary", fallbacks=[], prompt="payload")
    assert result["text"] == "valid"
    assert result["provider_attempts"][0]["error_code"] == "answer_structure_invalid"
