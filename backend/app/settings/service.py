import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.common.exceptions import AppError
from app.model.system_settings import SystemSettingsDraft, SystemSettingsState

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_SUPPORTED_PROVIDERS = {"ark", "openai", "anthropic"}
_SUPPORTED_RETRIEVAL_STRATEGIES = {"migration"}
_DEFAULT_DRAFT = {
    "model_provider": "ark",
    "llm_model": "",
    "embedding_model": "",
    "retrieval_strategy": "migration",
    "retrieval_top_k": 8,
    "score_threshold": 0.3,
    "milvus_uri": "",
    "index_name": "zhomind_docs",
    "runtime_timeout_ms": 8000,
}


class SystemSettingsDraftService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def read(self) -> dict:
        state = await self.session.get(SystemSettingsState, 1)
        draft = await self.session.scalar(
            select(SystemSettingsDraft).order_by(SystemSettingsDraft.version.desc()).limit(1)
        )
        return self._project(draft=draft, active_version=state.active_version if state else None)

    async def save(self, *, actor: str, payload: object) -> dict:
        normalized, provider_api_key = self._validate(payload)
        state = await self.session.get(SystemSettingsState, 1)
        if state is None:
            state = SystemSettingsState(id=1, latest_saved_version=0, active_version=None)
            self.session.add(state)
            await self.session.flush()

        previous = await self.session.scalar(
            select(SystemSettingsDraft).order_by(SystemSettingsDraft.version.desc()).limit(1)
        )
        sealed_secrets = dict(previous.sealed_secrets) if previous else {}
        if provider_api_key is not None:
            sealed_secrets["provider_api_key"] = self._seal_secret(provider_api_key)

        state.latest_saved_version += 1
        draft = SystemSettingsDraft(
            version=state.latest_saved_version,
            settings=normalized,
            sealed_secrets=sealed_secrets,
            saved_by=actor,
            saved_at=datetime.now(timezone.utc),
        )
        self.session.add(draft)
        await self.session.commit()
        return self._project(draft=draft, active_version=state.active_version)

    def _validate(self, payload: object) -> tuple[dict, str | None]:
        fields: dict[str, str] = {}
        if not isinstance(payload, dict):
            raise AppError(
                status_code=400,
                code="VALIDATION_ERROR",
                message="system settings draft is invalid",
                detail={"fields": {"draft": "must be an object"}},
            )

        allowed_fields = {*_DEFAULT_DRAFT, "provider_api_key"}
        for field in payload:
            if field not in allowed_fields:
                fields[str(field)] = "unsupported setting"
        for field in _DEFAULT_DRAFT:
            if field not in payload:
                fields[field] = "is required"
        if fields:
            raise AppError(
                status_code=400,
                code="VALIDATION_ERROR",
                message="system settings draft is invalid",
                detail={"fields": fields},
            )

        raw = payload

        model_provider = str(raw["model_provider"]).strip().lower()
        if model_provider not in _SUPPORTED_PROVIDERS:
            fields["model_provider"] = "unsupported model provider"

        llm_model = str(raw["llm_model"]).strip()
        if not _IDENTIFIER_PATTERN.fullmatch(llm_model):
            fields["llm_model"] = "model identifier is unsafe or unsupported"

        embedding_model = str(raw["embedding_model"]).strip()
        if not _IDENTIFIER_PATTERN.fullmatch(embedding_model):
            fields["embedding_model"] = "model identifier is unsafe or unsupported"

        retrieval_strategy = str(raw["retrieval_strategy"]).strip()
        if retrieval_strategy not in _SUPPORTED_RETRIEVAL_STRATEGIES:
            fields["retrieval_strategy"] = "unsupported retrieval strategy"

        retrieval_top_k = raw["retrieval_top_k"]
        if isinstance(retrieval_top_k, bool) or not isinstance(retrieval_top_k, int) or not 1 <= retrieval_top_k <= 20:
            fields["retrieval_top_k"] = "must be between 1 and 20"

        score_threshold = raw["score_threshold"]
        if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)) or not 0 <= score_threshold <= 1:
            fields["score_threshold"] = "must be between 0 and 1"

        milvus_uri = str(raw["milvus_uri"]).strip()
        if not self._is_safe_milvus_uri(milvus_uri):
            fields["milvus_uri"] = "must be an http or https URI without credentials"

        index_name = str(raw["index_name"]).strip()
        if not _IDENTIFIER_PATTERN.fullmatch(index_name):
            fields["index_name"] = "index name is unsafe or unsupported"

        runtime_timeout_ms = raw["runtime_timeout_ms"]
        if isinstance(runtime_timeout_ms, bool) or not isinstance(runtime_timeout_ms, int) or not 1000 <= runtime_timeout_ms <= 60000:
            fields["runtime_timeout_ms"] = "must be between 1000 and 60000"

        raw_provider_api_key = raw.get("provider_api_key")
        provider_api_key: str | None = None
        if raw_provider_api_key is not None:
            if not isinstance(raw_provider_api_key, str):
                fields["provider_api_key"] = "must be a string when replacing a secret"
            else:
                provider_api_key = raw_provider_api_key.strip()
                if not provider_api_key:
                    fields["provider_api_key"] = "must not be blank when replacing a secret"
                elif len(provider_api_key) > 1024:
                    fields["provider_api_key"] = "must not exceed 1024 characters"
                elif self._secret_box() is None:
                    fields["provider_api_key"] = "secure secret storage is unavailable"

        if fields:
            raise AppError(
                status_code=400,
                code="VALIDATION_ERROR",
                message="system settings draft is invalid",
                detail={"fields": fields},
            )

        return (
            {
                "model_provider": model_provider,
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_top_k": retrieval_top_k,
                "score_threshold": float(score_threshold),
                "milvus_uri": milvus_uri,
                "index_name": index_name,
                "runtime_timeout_ms": runtime_timeout_ms,
            },
            provider_api_key,
        )

    @staticmethod
    def _is_safe_milvus_uri(value: str) -> bool:
        try:
            parsed = urlsplit(value)
        except ValueError:
            return False
        return bool(
            value
            and parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )

    @staticmethod
    def _secret_box() -> Fernet | None:
        encryption_key = get_settings().system_settings_encryption_key.strip()
        if not encryption_key:
            return None
        try:
            return Fernet(encryption_key.encode("ascii"))
        except (TypeError, ValueError):
            return None

    def _seal_secret(self, value: str) -> str:
        secret_box = self._secret_box()
        if secret_box is None:
            raise AppError(
                status_code=400,
                code="VALIDATION_ERROR",
                message="system settings draft is invalid",
                detail={"fields": {"provider_api_key": "secure secret storage is unavailable"}},
            )
        return secret_box.encrypt(value.encode("utf-8")).decode("ascii")

    @staticmethod
    def _project(*, draft: SystemSettingsDraft | None, active_version: int | None) -> dict:
        if draft is None:
            return {
                "draft": {**_DEFAULT_DRAFT, "provider_api_key": {"configured": False}},
                "saved_version": None,
                "active_version": active_version,
                "last_modified": None,
                "application_state": "draft_only",
            }
        return {
            "draft": {
                **draft.settings,
                "provider_api_key": {"configured": bool(draft.sealed_secrets.get("provider_api_key"))},
            },
            "saved_version": draft.version,
            "active_version": active_version,
            "last_modified": {"actor": draft.saved_by, "at": draft.saved_at.isoformat()},
            "application_state": "draft_only",
        }
