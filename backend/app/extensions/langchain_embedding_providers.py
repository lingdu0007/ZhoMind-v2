from __future__ import annotations

from langchain_openai import OpenAIEmbeddings


class OpenAIEmbeddingProvider:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._embeddings = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            tiktoken_enabled=False,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)
