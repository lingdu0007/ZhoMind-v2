import asyncio
import secrets
from collections.abc import Generator

import pytest
from alembic import command
from alembic.config import Config
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.common.config import get_settings
from app.infra.db import get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base


class _InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, mapping: dict[str, str]) -> None:
        self.hashes[key] = {str(name): str(value) for name, value in mapping.items()}

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self.hashes and seconds > 0

    async def exists(self, key: str) -> int:
        return int(key in self.hashes)


@pytest.fixture
def client(tmp_path, monkeypatch) -> Generator[TestClient, None, None]:
    db_path = tmp_path / "system-settings-flow.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    fake_redis = _InMemoryRedis()

    async def _init_db() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(_init_db())

    async def override_get_db_session():
        async with session_factory() as session:
            yield session

    settings = get_settings()
    monkeypatch.setattr(settings, "admin_invite_code", "test-admin-code")
    monkeypatch.setattr(settings, "system_settings_draft_enabled", True)
    monkeypatch.setattr(settings, "system_settings_encryption_key", Fernet.generate_key().decode("ascii"))

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    with TestClient(app) as test_client:
        test_client.app.state.settings_session_factory = session_factory
        yield test_client
    app.dependency_overrides.clear()
    asyncio.run(engine.dispose())


def _register(client: TestClient, *, username: str, role: str = "user") -> str:
    payload = {"username": username, "password": "test-password", "role": role}
    if role == "admin":
        payload["admin_code"] = "test-admin-code"
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 200
    return response.json()["data"]["access_token"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _valid_draft(**overrides: object) -> dict[str, object]:
    draft: dict[str, object] = {
        "model_provider": "ark",
        "llm_model": "Qwen/Qwen3-32B",
        "embedding_model": "BAAI/bge-m3",
        "retrieval_strategy": "migration",
        "retrieval_top_k": 8,
        "score_threshold": 0.3,
        "milvus_uri": "http://milvus.internal:19530",
        "index_name": "zhomind_docs",
        "runtime_timeout_ms": 8000,
    }
    draft.update(overrides)
    return draft


def test_system_settings_draft_is_admin_only_and_masks_first_read(client: TestClient) -> None:
    user_token = _register(client, username="knowledge-user")
    admin_token = _register(client, username="settings-admin", role="admin")

    forbidden = client.get("/api/v1/settings/draft", headers=_headers(user_token))
    assert forbidden.status_code == 403
    assert forbidden.json()["code"] == "AUTH_FORBIDDEN"

    response = client.get("/api/v1/settings/draft", headers=_headers(admin_token))
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["saved_version"] is None
    assert data["active_version"] is None
    assert data["draft"]["provider_api_key"] == {"configured": False}
    assert data["last_modified"] is None


def test_system_settings_save_validates_versions_and_never_returns_secret(client: TestClient) -> None:
    admin_token = _register(client, username="settings-admin", role="admin")
    secret_value = secrets.token_urlsafe(32)

    invalid = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(retrieval_strategy="hybrid_rrf"),
    )
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "VALIDATION_ERROR"
    assert invalid.json()["detail"]["fields"] == {"retrieval_strategy": "unsupported retrieval strategy"}

    saved = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(provider_api_key=secret_value),
    )
    assert saved.status_code == 200
    data = saved.json()["data"]
    assert data["saved_version"] == 1
    assert data["active_version"] is None
    assert data["draft"]["provider_api_key"] == {"configured": True}
    assert data["last_modified"]["actor"] == "settings-admin"
    assert data["last_modified"]["at"]
    assert secret_value not in saved.text

    loaded = client.get("/api/v1/settings/draft", headers=_headers(admin_token))
    assert loaded.status_code == 200
    assert loaded.json()["data"]["saved_version"] == 1
    assert loaded.json()["data"]["draft"]["provider_api_key"] == {"configured": True}
    assert secret_value not in loaded.text

    saved_again = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(llm_model="Qwen/Qwen3-14B"),
    )
    assert saved_again.status_code == 200
    assert saved_again.json()["data"]["saved_version"] == 2
    assert saved_again.json()["data"]["active_version"] is None
    assert saved_again.json()["data"]["draft"]["provider_api_key"] == {"configured": True}

    async def _stored_secret_values() -> list[dict]:
        from app.model.system_settings import SystemSettingsDraft

        async with client.app.state.settings_session_factory() as session:
            result = await session.execute(select(SystemSettingsDraft.sealed_secrets))
            return list(result.scalars())

    sealed_values = asyncio.run(_stored_secret_values())
    assert all(secret_value not in str(value) for value in sealed_values)


def test_system_settings_never_echoes_an_invalid_secret_value(client: TestClient) -> None:
    admin_token = _register(client, username="settings-admin", role="admin")
    secret_value = secrets.token_urlsafe(32)

    response = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(provider_api_key={"replacement": secret_value}),
    )

    assert response.status_code == 400
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert response.json()["detail"]["fields"] == {"provider_api_key": "must be a string when replacing a secret"}
    assert secret_value not in response.text


def test_system_settings_never_echoes_a_secret_from_missing_or_unknown_fields(client: TestClient) -> None:
    admin_token = _register(client, username="settings-admin", role="admin")
    secret_value = secrets.token_urlsafe(32)

    missing_fields = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json={"provider_api_key": secret_value},
    )
    assert missing_fields.status_code == 400
    assert missing_fields.json()["code"] == "VALIDATION_ERROR"
    assert missing_fields.json()["detail"]["fields"]["model_provider"] == "is required"
    assert secret_value not in missing_fields.text

    unknown_field = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(unrecognized_setting=secret_value),
    )
    assert unknown_field.status_code == 400
    assert unknown_field.json()["code"] == "VALIDATION_ERROR"
    assert unknown_field.json()["detail"]["fields"] == {"unrecognized_setting": "unsupported setting"}
    assert secret_value not in unknown_field.text


def test_system_settings_draft_gate_stays_closed_when_disabled(client: TestClient, monkeypatch) -> None:
    admin_token = _register(client, username="settings-admin", role="admin")
    monkeypatch.setattr(get_settings(), "system_settings_draft_enabled", False)

    response = client.get("/api/v1/settings/draft", headers=_headers(admin_token))
    assert response.status_code == 404
    assert response.json()["code"] == "SETTINGS_DRAFT_DISABLED"


def test_system_settings_migration_upgrades_and_downgrades(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "system-settings-migration.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    with monkeypatch.context() as settings_env:
        settings_env.setenv("DATABASE_URL", database_url)
        get_settings.cache_clear()

        config = Config("alembic.ini")
        command.upgrade(config, "20260731_0007")

        sync_engine = create_engine(f"sqlite:///{db_path}")
        assert {"system_settings_drafts", "system_settings_state"}.issubset(inspect(sync_engine).get_table_names())

        command.downgrade(config, "20260425_0006")
        assert "system_settings_drafts" not in inspect(sync_engine).get_table_names()
        assert "system_settings_state" not in inspect(sync_engine).get_table_names()
        sync_engine.dispose()
    get_settings.cache_clear()
