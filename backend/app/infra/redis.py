from functools import lru_cache

from redis.asyncio import Redis

from app.common.config import get_settings


class RedisProvider:
    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Redis | None = None

    def get_client(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(self.url, decode_responses=True)
        return self._client


@lru_cache
def get_redis_provider() -> RedisProvider:
    settings = get_settings()
    return RedisProvider(settings.redis_url)


def get_redis_client() -> Redis:
    return get_redis_provider().get_client()
