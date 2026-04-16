from functools import lru_cache

from pymilvus import MilvusClient

from app.common.config import get_settings


class MilvusProvider:
    def __init__(self, uri: str, token: str | None = None) -> None:
        self.uri = uri
        self.token = token
        self._client: MilvusClient | None = None

    def get_client(self) -> MilvusClient:
        if self._client is None:
            kwargs = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            self._client = MilvusClient(**kwargs)
        return self._client


@lru_cache
def get_milvus_provider() -> MilvusProvider:
    settings = get_settings()
    return MilvusProvider(uri=settings.milvus_uri, token=settings.milvus_token)


def get_milvus_client() -> MilvusClient:
    return get_milvus_provider().get_client()
