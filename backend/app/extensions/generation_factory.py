from dataclasses import dataclass
from functools import cached_property
from typing import Any

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import Field

from app.common.config import get_settings
from app.extensions.langchain_chat_providers import AnthropicChatProvider, OpenAICompatibleChatProvider
from app.rag.interfaces import GenerationCompletion, LlmProvider


class _BoundAnthropic(ChatAnthropic):
    """The pinned LangChain adapter exposes its SDK client through this property."""

    http_async_client: Any = Field(exclude=True)

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        return anthropic.AsyncClient(**self._client_params, http_client=self.http_async_client)


@dataclass
class _AttemptScopedProvider:
    settings: Any
    timeout_seconds: float

    async def complete(self, prompt: str, *, system_prompt: str | None = None) -> str | GenerationCompletion:
        with httpx.Client(follow_redirects=False, timeout=self.timeout_seconds) as sync_client:
            async with httpx.AsyncClient(follow_redirects=False, timeout=self.timeout_seconds) as async_client:
                provider = _build_generation_provider(
                    self.settings, timeout_seconds=self.timeout_seconds,
                    sync_client=sync_client, async_client=async_client,
                )
                if provider is None:
                    raise ValueError("unsupported approved provider")
                return await provider.complete(prompt, system_prompt=system_prompt)


def build_generation_provider(settings: Any, *, timeout_seconds: float = 30) -> LlmProvider | None:
    if get_settings().generation_validation_mode != "controlled_live":
        return None
    if settings.rag_primary_llm_provider not in {"ark", "openai", "anthropic"}:
        return None
    return _AttemptScopedProvider(settings, timeout_seconds)


def _build_generation_provider(
    settings: Any, *, timeout_seconds: float, sync_client: httpx.Client, async_client: httpx.AsyncClient,
) -> LlmProvider | None:
    provider_type = settings.rag_primary_llm_provider
    if provider_type == "ark" and settings.ark_api_key and settings.llm_base_url and settings.llm_model:
        return OpenAICompatibleChatProvider(
            model=ChatOpenAI(
                api_key=settings.ark_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=0.2,
                max_retries=0,
                timeout=timeout_seconds,
                http_client=sync_client,
                http_async_client=async_client,
            ),
            provider_name="ark",
        )
    if provider_type == "openai" and settings.openai_api_key and settings.openai_model:
        openai_kwargs = {
            "api_key": settings.openai_api_key,
            "model": settings.openai_model,
            "temperature": 0.2,
            "max_retries": 0,
            "timeout": timeout_seconds,
            "http_client": sync_client,
            "http_async_client": async_client,
        }
        if settings.openai_base_url:
            openai_kwargs["base_url"] = settings.openai_base_url
        return OpenAICompatibleChatProvider(model=ChatOpenAI(**openai_kwargs), provider_name="openai")
    if provider_type == "anthropic" and settings.anthropic_api_key and settings.anthropic_model:
        anthropic_kwargs = {
            "api_key": settings.anthropic_api_key,
            "model": settings.anthropic_model,
            "temperature": 0.2,
            "max_retries": 0,
            "timeout": timeout_seconds,
            "http_async_client": async_client,
        }
        if settings.anthropic_base_url:
            anthropic_kwargs["base_url"] = settings.anthropic_base_url
        return AnthropicChatProvider(model=_BoundAnthropic(**anthropic_kwargs), provider_name="anthropic")
    return None
