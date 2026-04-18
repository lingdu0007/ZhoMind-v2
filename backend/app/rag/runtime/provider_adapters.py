from typing import TypedDict

from app.rag.interfaces import LlmProvider, RelevanceJudge, Reranker, Retriever


class ProviderExecError(TypedDict):
    code: str
    message: str
    type: str


class ProviderExecDetail(TypedDict):
    provider: str
    fallback_used: bool
    error: ProviderExecError | None


def _normalize_provider_error(exc: Exception) -> ProviderExecError:
    return {
        "code": "PROVIDER_EXEC_FAILED",
        "message": str(exc),
        "type": type(exc).__name__,
    }


def _build_detail(
    *,
    provider: str,
    fallback_used: bool,
    error: ProviderExecError | None,
) -> ProviderExecDetail:
    return {
        "provider": provider,
        "fallback_used": fallback_used,
        "error": error,
    }


class RetrieverAdapter:
    def __init__(self, provider: Retriever | None, *, provider_name: str = "unconfigured") -> None:
        self.provider = provider
        self.provider_name = provider_name

    async def retrieve(self, query: str, top_k: int) -> tuple[list[dict], ProviderExecDetail]:
        if self.provider is None:
            return [], _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=None,
            )

        try:
            result = await self.provider.retrieve(query, top_k=top_k)
            return result, _build_detail(
                provider=self.provider_name,
                fallback_used=False,
                error=None,
            )
        except Exception as exc:
            return [], _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=_normalize_provider_error(exc),
            )


class RerankerAdapter:
    def __init__(self, provider: Reranker | None, *, provider_name: str = "unconfigured") -> None:
        self.provider = provider
        self.provider_name = provider_name

    async def rerank(self, query: str, items: list[dict]) -> tuple[list[dict], ProviderExecDetail]:
        if self.provider is None:
            return items, _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=None,
            )

        try:
            result = await self.provider.rerank(query, items)
            return result, _build_detail(
                provider=self.provider_name,
                fallback_used=False,
                error=None,
            )
        except Exception as exc:
            return items, _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=_normalize_provider_error(exc),
            )


class JudgeAdapter:
    def __init__(self, provider: RelevanceJudge | None, *, provider_name: str = "unconfigured") -> None:
        self.provider = provider
        self.provider_name = provider_name

    async def judge(self, query: str, context: list[dict]) -> tuple[bool, ProviderExecDetail]:
        fallback_result = len(context) > 0
        if self.provider is None:
            return fallback_result, _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=None,
            )

        try:
            result = await self.provider.judge(query, context)
            return result, _build_detail(
                provider=self.provider_name,
                fallback_used=False,
                error=None,
            )
        except Exception as exc:
            return fallback_result, _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=_normalize_provider_error(exc),
            )


class LlmAdapter:
    def __init__(self, provider: LlmProvider | None, *, provider_name: str = "unconfigured") -> None:
        self.provider = provider
        self.provider_name = provider_name

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[str, ProviderExecDetail]:
        if self.provider is None:
            return "", _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=None,
            )

        try:
            result = await self.provider.complete(prompt, system_prompt=system_prompt)
            return result, _build_detail(
                provider=self.provider_name,
                fallback_used=False,
                error=None,
            )
        except Exception as exc:
            return "", _build_detail(
                provider=self.provider_name,
                fallback_used=True,
                error=_normalize_provider_error(exc),
            )
