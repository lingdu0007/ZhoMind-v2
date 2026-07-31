import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.config import get_settings
from app.common.exceptions import AppError
from app.model.system_settings import SystemSettingsDraft, SystemSettingsState
from app.settings.runtime import RuntimeApplicationError, SystemSettingsRuntime, get_system_settings_runtime

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
    def __init__(self, session: AsyncSession, *, runtime: SystemSettingsRuntime | None = None) -> None:
        self.session = session
        self.runtime = runtime or get_system_settings_runtime()

    async def read(self) -> dict:
        state = await self.session.get(SystemSettingsState, 1)
        if state and state.active_version is not None and not self.runtime.has_active_version(state.active_version):
            await self.restore_active_application()
            state = await self.session.get(SystemSettingsState, 1)
        draft = await self.session.scalar(
            select(SystemSettingsDraft).order_by(SystemSettingsDraft.version.desc()).limit(1)
        )
        return self._project(draft=draft, state=state)

    async def save(self, *, actor: str, payload: object) -> dict:
        normalized, provider_api_key = self._validate(payload)
        state = await self.session.get(SystemSettingsState, 1)
        if state is None:
            state = SystemSettingsState(id=1, latest_saved_version=0, active_version=None)
            self.session.add(state)
            await self.session.flush()
        if state.application_state == "applying":
            raise AppError(
                status_code=409,
                code="SETTINGS_APPLICATION_IN_PROGRESS",
                message="a settings application is already in progress",
            )

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
        state.application_state = "saved"
        state.application_version = None
        state.application_actor = None
        state.application_at = None
        state.application_message = None
        await self.session.commit()
        return self._project(draft=draft, state=state)

    async def begin_application(self, *, actor: str, version: object) -> dict:
        requested_version = self._parse_version(version)
        state = await self.session.get(SystemSettingsState, 1)
        draft = await self.session.get(SystemSettingsDraft, requested_version)
        if draft is None:
            raise AppError(status_code=404, code="SETTINGS_VERSION_NOT_FOUND", message="saved settings version was not found")
        if state is None or requested_version != state.latest_saved_version:
            raise AppError(status_code=409, code="SETTINGS_VERSION_STALE", message="saved settings version is stale")
        if state.application_state == "applying":
            raise AppError(
                status_code=409,
                code="SETTINGS_APPLICATION_IN_PROGRESS",
                message="a settings application is already in progress",
            )

        self._validate_saved_version(draft)
        try:
            self.runtime.validate(draft.settings)
        except RuntimeApplicationError as exc:
            raise AppError(
                status_code=409,
                code="SETTINGS_VERSION_UNSUPPORTED",
                message="saved settings version is unsupported by the running system",
                detail={"fields": getattr(exc, "fields", {"settings": "cannot be applied by the running system"})},
            ) from exc

        now = datetime.now(timezone.utc)
        state.application_state = "applying"
        state.application_version = requested_version
        state.application_actor = actor
        state.application_at = now
        state.application_message = "settings version is applying"
        await self.session.commit()
        return self._project(draft=draft, state=state)

    async def complete_application(self, *, actor: str, version: int) -> None:
        state = await self.session.get(SystemSettingsState, 1)
        draft = await self.session.get(SystemSettingsDraft, version)
        if (
            state is None
            or draft is None
            or state.application_state != "applying"
            or state.application_version != version
            or state.application_actor != actor
        ):
            return

        try:
            provider_api_key = self._open_secret(draft.sealed_secrets.get("provider_api_key"))
            await self.runtime.apply(version=version, settings=draft.settings, provider_api_key=provider_api_key)
        except RuntimeApplicationError as exc:
            state.application_state = "failed"
            state.application_message = self._safe_application_message(exc)
        except Exception:  # pragma: no cover - protects state transitions from third-party failures
            state.application_state = "failed"
            state.application_message = "runtime did not accept the saved configuration"
        else:
            state.active_version = version
            state.application_state = "active"
            state.application_message = "settings version is active"
        state.application_at = datetime.now(timezone.utc)
        await self.session.commit()

    async def restore_active_application(self) -> None:
        state = await self.session.get(SystemSettingsState, 1)
        if state is None or state.active_version is None:
            return
        draft = await self.session.get(SystemSettingsDraft, state.active_version)
        if draft is None:
            state.active_version = None
            state.application_state = "failed"
            state.application_message = "active settings version is unavailable"
            state.application_at = datetime.now(timezone.utc)
            await self.session.commit()
            return

        try:
            provider_api_key = self._open_secret(draft.sealed_secrets.get("provider_api_key"))
            await self.runtime.apply(
                version=state.active_version,
                settings=draft.settings,
                provider_api_key=provider_api_key,
            )
        except RuntimeApplicationError as exc:
            state.active_version = None
            state.application_state = "failed"
            state.application_message = self._safe_application_message(exc)
            state.application_at = datetime.now(timezone.utc)
            await self.session.commit()

    def _validate_saved_version(self, draft: SystemSettingsDraft) -> None:
        payload = {**draft.settings, "provider_api_key": None}
        self._validate(payload)

    @staticmethod
    def _parse_version(version: object) -> int:
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise AppError(status_code=400, code="SETTINGS_VERSION_INVALID", message="saved settings version is invalid")
        return version

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

    def _open_secret(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuntimeApplicationError("saved secret is unavailable")
        secret_box = self._secret_box()
        if secret_box is None:
            raise RuntimeApplicationError("secure secret storage is unavailable")
        try:
            return secret_box.decrypt(value.encode("ascii")).decode("utf-8")
        except Exception as exc:
            raise RuntimeApplicationError("saved secret is unavailable") from exc

    @staticmethod
    def _safe_application_message(exc: RuntimeApplicationError) -> str:
        message = str(exc)
        return message if message in {
            "runtime rejected the saved configuration",
            "runtime did not accept the saved configuration",
            "saved secret is unavailable",
            "secure secret storage is unavailable",
        } else "runtime did not accept the saved configuration"

    @staticmethod
    def _project(*, draft: SystemSettingsDraft | None, state: SystemSettingsState | None) -> dict:
        application = None
        if state and state.application_version is not None and state.application_actor and state.application_at:
            application = {
                "version": state.application_version,
                "actor": state.application_actor,
                "at": state.application_at.isoformat(),
                "message": state.application_message or "application outcome is unavailable",
            }
        active_version = state.active_version if state else None
        application_state = state.application_state if state else "draft_only"
        if draft is None:
            return {
                "draft": {**_DEFAULT_DRAFT, "provider_api_key": {"configured": False}},
                "saved_version": None,
                "active_version": active_version,
                "last_modified": None,
                "application_state": application_state,
                "application": application,
            }
        return {
            "draft": {
                **draft.settings,
                "provider_api_key": {"configured": bool(draft.sealed_secrets.get("provider_api_key"))},
            },
            "saved_version": draft.version,
            "active_version": active_version,
            "last_modified": {"actor": draft.saved_by, "at": draft.saved_at.isoformat()},
            "application_state": application_state,
            "application": application,
        }
