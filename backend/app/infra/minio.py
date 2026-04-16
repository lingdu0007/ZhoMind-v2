from functools import lru_cache

from minio import Minio

from app.common.config import get_settings


class MinioProvider:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool) -> None:
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self._client: Minio | None = None

    def get_client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self._client


@lru_cache
def get_minio_provider() -> MinioProvider:
    settings = get_settings()
    return MinioProvider(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def get_minio_client() -> Minio:
    return get_minio_provider().get_client()
