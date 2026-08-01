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
from app.infra.db import SessionLocal, get_db_session
from app.infra.redis import get_redis_client
from app.main import app
from app.model.base import Base
from app.settings.runtime import RuntimeApplicationError, get_system_settings_runtime
from app.settings.service import SystemSettingsDraftService
from tests.support.auth import create_authenticated_test_token


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
    monkeypatch.setattr(settings, "system_settings_draft_enabled", True)
    monkeypatch.setattr(settings, "system_settings_application_enabled", True)
    monkeypatch.setattr(settings, "system_settings_encryption_key", Fernet.generate_key().decode("ascii"))

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    app.state.settings_session_factory = session_factory
    app.state.test_auth_session_factory = session_factory
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.settings_session_factory = SessionLocal
    delattr(app.state, "test_auth_session_factory")
    get_system_settings_runtime().reset()
    asyncio.run(engine.dispose())


def _register(client: TestClient, *, username: str, role: str = "user") -> str:
    return asyncio.run(
        create_authenticated_test_token(
            client.app.state.test_auth_session_factory,
            client.app.dependency_overrides[get_redis_client](),
            username=username,
            role=role,
        )
    )


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


def test_system_settings_application_requires_current_admin_saved_version_and_preserves_active_version(client: TestClient) -> None:
    user_token = _register(client, username="knowledge-user")
    admin_token = _register(client, username="settings-admin", role="admin")

    forbidden = client.post("/api/v1/settings/apply", headers=_headers(user_token), json={"version": 1})
    assert forbidden.status_code == 403
    assert forbidden.json()["code"] == "AUTH_FORBIDDEN"

    missing = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": 1})
    assert missing.status_code == 404
    assert missing.json()["code"] == "SETTINGS_VERSION_NOT_FOUND"

    invalid = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": True})
    assert invalid.status_code == 400
    assert invalid.json()["code"] == "SETTINGS_VERSION_INVALID"

    saved_v1 = client.put("/api/v1/settings/draft", headers=_headers(admin_token), json=_valid_draft())
    assert saved_v1.status_code == 200
    assert saved_v1.json()["data"]["application_state"] == "saved"
    assert saved_v1.json()["data"]["active_version"] is None

    applying = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": 1})
    assert applying.status_code == 200
    applying_data = applying.json()["data"]
    assert applying_data["application_state"] == "applying"
    assert applying_data["active_version"] is None
    assert applying_data["application"]["version"] == 1
    assert applying_data["application"]["actor"] == "settings-admin"
    assert applying_data["application"]["at"]

    active = client.get("/api/v1/settings/draft", headers=_headers(admin_token))
    assert active.status_code == 200
    active_data = active.json()["data"]
    assert active_data["application_state"] == "active"
    assert active_data["active_version"] == 1
    assert active_data["application"] == {
        "version": 1,
        "actor": "settings-admin",
        "at": active_data["application"]["at"],
        "message": "settings version is active",
    }

    get_system_settings_runtime().reset()
    restored = client.get("/api/v1/settings/draft", headers=_headers(admin_token)).json()["data"]
    assert restored["active_version"] == 1
    assert get_system_settings_runtime().current_settings().llm_model == "Qwen/Qwen3-32B"
    assert get_system_settings_runtime().current_settings().runtime_retrieval_top_k == 8

    saved_v2 = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(llm_model="Qwen/Qwen3-14B"),
    )
    assert saved_v2.status_code == 200
    assert saved_v2.json()["data"]["saved_version"] == 2
    assert saved_v2.json()["data"]["active_version"] == 1
    assert saved_v2.json()["data"]["application_state"] == "saved"

    stale = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": 1})
    assert stale.status_code == 409
    assert stale.json()["code"] == "SETTINGS_VERSION_STALE"

    active_after_stale = client.get("/api/v1/settings/draft", headers=_headers(admin_token))
    assert active_after_stale.json()["data"]["active_version"] == 1
    assert active_after_stale.json()["data"]["saved_version"] == 2
    assert active_after_stale.json()["data"]["application_state"] == "saved"


def test_system_settings_application_rejects_unsupported_versions_and_records_a_safe_failure(client: TestClient, monkeypatch) -> None:
    admin_token = _register(client, username="settings-admin", role="admin")
    secret_value = secrets.token_urlsafe(32)

    saved_v1 = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(provider_api_key=secret_value),
    )
    assert saved_v1.status_code == 200
    active_v1 = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": 1})
    assert active_v1.status_code == 200
    assert client.get("/api/v1/settings/draft", headers=_headers(admin_token)).json()["data"]["active_version"] == 1

    unsupported = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(model_provider="openai"),
    )
    assert unsupported.status_code == 200
    rejected = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": 2})
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "SETTINGS_VERSION_UNSUPPORTED"
    assert rejected.json()["detail"]["fields"] == {"model_provider": "requires an Ark runtime lifecycle"}
    assert secret_value not in rejected.text

    async def fail_runtime_application(**_: object) -> None:
        raise RuntimeApplicationError("runtime rejected the saved configuration")

    monkeypatch.setattr(get_system_settings_runtime(), "apply", fail_runtime_application)
    saved_v3 = client.put(
        "/api/v1/settings/draft",
        headers=_headers(admin_token),
        json=_valid_draft(llm_model="Qwen/Qwen3-14B"),
    )
    assert saved_v3.status_code == 200
    applying = client.post("/api/v1/settings/apply", headers=_headers(admin_token), json={"version": 3})
    assert applying.status_code == 200
    assert applying.json()["data"]["application_state"] == "applying"

    failed = client.get("/api/v1/settings/draft", headers=_headers(admin_token))
    failed_data = failed.json()["data"]
    assert failed_data["saved_version"] == 3
    assert failed_data["active_version"] == 1
    assert failed_data["application_state"] == "failed"
    assert failed_data["application"]["version"] == 3
    assert failed_data["application"]["actor"] == "settings-admin"
    assert failed_data["application"]["message"] == "runtime rejected the saved configuration"
    assert secret_value not in failed.text

    get_system_settings_runtime().reset()

    async def _restore_rejected_active_runtime() -> None:
        async with client.app.state.settings_session_factory() as session:
            await SystemSettingsDraftService(session).restore_active_application()

    asyncio.run(_restore_rejected_active_runtime())
    restored_failure = client.get("/api/v1/settings/draft", headers=_headers(admin_token)).json()["data"]
    assert restored_failure["active_version"] is None
    assert restored_failure["application_state"] == "failed"


def test_system_settings_migration_upgrades_and_downgrades(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "system-settings-migration.db"
    database_url = f"sqlite+aiosqlite:///{db_path}"
    with monkeypatch.context() as settings_env:
        settings_env.setenv("DATABASE_URL", database_url)
        get_settings.cache_clear()

        config = Config("alembic.ini")
        command.upgrade(config, "20260731_0008")

        sync_engine = create_engine(f"sqlite:///{db_path}")
        assert {"system_settings_drafts", "system_settings_state"}.issubset(inspect(sync_engine).get_table_names())
        state_columns = {column["name"] for column in inspect(sync_engine).get_columns("system_settings_state")}
        assert {"application_state", "application_version", "application_actor", "application_at", "application_message"}.issubset(
            state_columns
        )

        command.downgrade(config, "20260425_0006")
        assert "system_settings_drafts" not in inspect(sync_engine).get_table_names()
        assert "system_settings_state" not in inspect(sync_engine).get_table_names()
        sync_engine.dispose()
    get_settings.cache_clear()
