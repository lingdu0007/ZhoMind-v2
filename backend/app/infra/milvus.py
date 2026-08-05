from functools import lru_cache

from pymilvus import MilvusClient

from app.settings.runtime import get_runtime_settings


class MilvusProvider:
    def __init__(self, uri: str, token: str | None = None) -> None:
        self.uri = uri
        self.token = token
        self._client: MilvusClient | None = None

    def get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self.uri, token=self.token or "")
        return self._client


@lru_cache
def get_milvus_provider() -> MilvusProvider:
    settings = get_runtime_settings()
    return MilvusProvider(uri=settings.milvus_uri, token=settings.milvus_token)


def get_milvus_client() -> MilvusClient:
    return get_milvus_provider().get_client()
