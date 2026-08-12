from __future__ import annotations

from typing import cast

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr


class OpenAIEmbeddingProvider:
    def __init__(self, *, api_key: str, base_url: str, model: str, dimensions: int) -> None:
        # langchain-openai accepts a plain string at runtime and wraps it in a
        # SecretStr; its stubs annotate `SecretStr | None`, so cast narrows the
        # type without changing the runtime argument.
        self._embeddings = OpenAIEmbeddings(
            api_key=cast(SecretStr, api_key),
            base_url=base_url,
            model=model,
            dimensions=dimensions,
            tiktoken_enabled=False,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)
