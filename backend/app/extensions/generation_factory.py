from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from app.extensions.langchain_chat_providers import AnthropicChatProvider, OpenAICompatibleChatProvider
from app.rag.interfaces import LlmProvider


def build_generation_provider(settings: Any) -> LlmProvider | None:
    provider_type = settings.rag_primary_llm_provider
    if provider_type == "ark" and settings.ark_api_key and settings.llm_base_url and settings.llm_model:
        return OpenAICompatibleChatProvider(
            model=ChatOpenAI(
                api_key=settings.ark_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=0.2,
            ),
            provider_name="ark",
        )
    if provider_type == "openai" and settings.openai_api_key and settings.openai_model:
        openai_kwargs = {
            "api_key": settings.openai_api_key,
            "model": settings.openai_model,
            "temperature": 0.2,
        }
        if settings.openai_base_url:
            openai_kwargs["base_url"] = settings.openai_base_url
        return OpenAICompatibleChatProvider(model=ChatOpenAI(**openai_kwargs), provider_name="openai")
    if provider_type == "anthropic" and settings.anthropic_api_key and settings.anthropic_model:
        anthropic_kwargs = {
            "api_key": settings.anthropic_api_key,
            "model": settings.anthropic_model,
            "temperature": 0.2,
        }
        if settings.anthropic_base_url:
            anthropic_kwargs["base_url"] = settings.anthropic_base_url
        return AnthropicChatProvider(model=ChatAnthropic(**anthropic_kwargs), provider_name="anthropic")
    return None
