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

    # LLM / ARK
    ark_api_key: str = Field("", alias="ARK_API_KEY")
    llm_model: str = Field("", alias="MODEL")
    grade_model: str = Field("", alias="GRADE_MODEL")
    fast_model: str = Field("", alias="FAST_MODEL")
    llm_base_url: str = Field("", alias="BASE_URL")

    # Embedding / Rerank (reserved for provider wiring)
    embedding_api_key: str = Field("", alias="EMBEDDING_API_KEY")
    embedding_base_url: str = Field("", alias="EMBEDDING_BASE_URL")
    embedding_model: str = Field("", alias="EMBEDDING_MODEL")
    dense_embedding_dim: int = Field(1024, alias="DENSE_EMBEDDING_DIM")

    rerank_model: str = Field("", alias="RERANK_MODEL")
    rerank_binding_host: str = Field("", alias="RERANK_BINDING_HOST")
    rerank_api_key: str = Field("", alias="RERANK_API_KEY")

    milvus_uri: str = Field("http://localhost:19530", alias="MILVUS_URI")
    milvus_token: str | None = Field(default=None, alias="MILVUS_TOKEN")
    minio_endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field("minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field("minioadmin", alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(False, alias="MINIO_SECURE")

    bm25_state_path: str = Field("", alias="BM25_STATE_PATH")

    rag_graph_alias: str = Field("default_v1", alias="RAG_GRAPH_ALIAS")
    rag_enable_tools: bool = Field(False, alias="RAG_ENABLE_TOOLS")
    rag_tool_max_calls: int = Field(3, alias="RAG_TOOL_MAX_CALLS")
    rag_tool_max_parallel: int = Field(2, alias="RAG_TOOL_MAX_PARALLEL")
    rag_tool_timeout_ms: int = Field(8000, alias="RAG_TOOL_TIMEOUT_MS")
    rag_default_llm_provider: str = Field("chat-default-llm", alias="RAG_DEFAULT_LLM_PROVIDER")
    rag_disable_gate: bool = Field(False, alias="RAG_DISABLE_GATE")


@lru_cache
def get_settings() -> Settings:
    return Settings()


