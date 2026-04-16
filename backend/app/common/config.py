from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "zhomind-backend"
    app_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"

    database_url: str = Field(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/zhomind_app",
        alias="DATABASE_URL",
    )
    jwt_secret: str = Field("change-me", alias="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(120, alias="JWT_EXPIRE_MINUTES")
    admin_invite_code: str = Field("", alias="ADMIN_INVITE_CODE")
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    milvus_uri: str = Field("http://localhost:19530", alias="MILVUS_URI")
    milvus_token: str | None = Field(default=None, alias="MILVUS_TOKEN")
    minio_endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field("minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field("minioadmin", alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(False, alias="MINIO_SECURE")


@lru_cache
def get_settings() -> Settings:
    return Settings()
