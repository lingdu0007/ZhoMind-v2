import asyncio

from app.extensions.provider_router import ProviderRouter


class _OkProvider:
    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return self.text


class _RetryableFailProvider:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise TimeoutError("upstream timeout")


class _HardFailProvider:
    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise ValueError("bad request")


def test_router_failover_on_retryable_error() -> None:
    router = ProviderRouter(
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
        providers={
            "ark": _HardFailProvider(),
            "openai": _OkProvider("should-not-run"),
        }
    )

    result = asyncio.run(router.complete(primary="ark", fallbacks=["openai"], prompt="hi"))

    assert result["text"] == ""
    assert result["final_provider"] == "ark"
    assert len(result["provider_attempts"]) == 1
