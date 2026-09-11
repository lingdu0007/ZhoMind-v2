from __future__ import annotations

import asyncio
import re
import uuid
from typing import Literal
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.config import get_settings
from app.common.exceptions import AppError
from app.delivery_acceptance.service import DeliveryAcceptanceService
from app.extensions.generation_factory import build_generation_provider
from app.extensions.provider_router import ApprovedGenerationRoute, ApprovedRouteProvider, ProviderRouter
from app.model.canonical import CanonicalEventModel, CanonicalRecordModel
from app.model.generation_route import GenerationRouteSecret, GenerationRouteState
from app.model.user import User
from app.rag.interfaces import GenerationCompletion
from app.settings.runtime import SystemSettingsRuntime

_ACTIVATION_CHECKS = frozenset({
    "check:generation-provider", "check:generation-failure",
    "check:generation-privacy", "check:generation-prompt-citation",
})


class ProviderInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    provider: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    provider_type: Literal["ark", "openai", "anthropic"]
    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
    service_url: str = Field(max_length=512)
    endpoint_class: Literal["public_https", "private_https"]
    data_scope: Literal["team_shared_pilot"]
    timeout_seconds: float = Field(gt=0, le=60, allow_inf_nan=False)
    provider_api_key: SecretStr = Field(min_length=1, max_length=1024)

    @field_validator("service_url")
    @classmethod
    def safe_endpoint(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid endpoint") from exc
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or port == 0 or value != value.strip()):
            raise ValueError("invalid endpoint")
        return value


class RouteInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    data_scope: Literal["team_shared_pilot"]
    providers: list[ProviderInput] = Field(min_length=1, max_length=4)
    max_attempts: int = Field(ge=1, le=4)
    total_timeout_seconds: float = Field(gt=0, le=60, allow_inf_nan=False)


class ActivationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    route_identity: str = Field(pattern=r"^provider_route:[0-9a-f]{64}$")
    expected_active_identity: str | None
    acceptance_record_identity: str = Field(pattern=r"^delivery_acceptance_record:[A-Za-z0-9._/-]+$")


def route_error(code: str, status: int = 409) -> AppError:
    return AppError(status_code=status, code=code, message="generation route request could not be accepted")


def secret_box() -> Fernet:
    try:
        return Fernet(get_settings().system_settings_encryption_key.encode("ascii"))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise route_error("GENERATION_SECRET_STORAGE_UNAVAILABLE") from exc


def route_contract(payload: dict) -> ApprovedGenerationRoute:
    return ApprovedGenerationRoute(
        identity=payload["route_identity"],
        data_scope=payload["data_scope"],
        providers=tuple(ApprovedRouteProvider(
            approval_identity=item["approval_identity"], provider=item["provider"], model=item["model"],
            data_scope=item["data_scope"], timeout_seconds=item["timeout_seconds"],
        ) for item in payload["providers"]),
        max_attempts=payload["max_attempts"],
        total_timeout_seconds=payload["total_timeout_seconds"],
    )


class GenerationRouteService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def capture(self) -> ProviderRouter:
        state = await self.session.get(GenerationRouteState, 1)
        if state is None or state.active_identity is None:
            return ProviderRouter(providers={})
        try:
            payload = await self._record(state.active_identity)
            evidence = await self._activation_evidence(state.active_identity)
            if evidence is None:
                return ProviderRouter(providers={})
            current = await self._verify_acceptance(evidence["record_identity"], payload)
            if current != evidence:
                return ProviderRouter(providers={})
            providers = await self.providers(payload)
            return ProviderRouter(providers=providers, approved_route=route_contract(payload))
        except (AppError, ValueError, TypeError, KeyError):
            return ProviderRouter(providers={})

    async def save(self, *, actor: str, payload: object) -> dict:
        try:
            request = RouteInput.model_validate(payload)
            normalized = request.model_dump(mode="json", exclude={"providers": {"__all__": {"provider_api_key"}}})
            box = secret_box()
            for provider, raw in zip(normalized["providers"], request.providers, strict=True):
                reference = uuid.uuid4().hex
                provider["credential_reference"] = reference
                provider["approval_identity"] = "configuration:" + canonical_json_sha256(provider)
                self.session.add(GenerationRouteSecret(
                    id=reference, ciphertext=box.encrypt(raw.provider_api_key.get_secret_value().encode("utf-8")).decode("ascii"),
                ))
            normalized["schema"] = "approved_generation_route/v1"
            normalized["route_identity"] = "provider_route:" + canonical_json_sha256(normalized)
            route_contract(normalized)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            await self.session.rollback()
            raise route_error("GENERATION_ROUTE_INVALID", 400) from exc
        identity = normalized["route_identity"]
        self.session.add(CanonicalRecordModel(
            stable_id=identity, identity_kind="provider_route", identity_value=identity.split(":", 1)[1],
            state="inactive", record_class="immutable", payload={**normalized, "configured_by": actor},
        ))
        state = await self._lock_state()
        if state is None:
            state = GenerationRouteState(id=1, version=0)
            self.session.add(state)
        state.version += 1
        state.draft_identity = identity
        await self.session.commit()
        return self._project(normalized, active=False)

    async def read(self) -> dict:
        state = await self.session.get(GenerationRouteState, 1)
        draft = await self._record(state.draft_identity) if state and state.draft_identity else None
        active = await self._record(state.active_identity) if state and state.active_identity else None
        evidence = await self._activation_evidence(state.active_identity) if state and state.active_identity else None
        if active:
            try:
                if evidence is None or await self._verify_acceptance(evidence["record_identity"], active) != evidence:
                    active = None
            except AppError:
                active = None
        draft_active = bool(active and state and state.active_identity == state.draft_identity)
        return {
            "current_route_identity": state.active_identity if state else None,
            "draft": self._project(draft, active=draft_active, evidence=evidence if draft_active else None) if draft else None,
            "active": self._project(active, active=True, evidence=evidence) if active else None,
        }

    async def verification_context(self) -> dict | None:
        """Freeze non-secret current activation evidence without changing route authority."""
        state = await self._lock_state()
        if state is None or state.active_identity is None:
            return None
        try:
            active = (await self.read())["active"]
        except AppError:
            return None
        if active is None:
            return None
        event = await self.session.scalar(
            select(CanonicalEventModel)
            .where(
                CanonicalEventModel.aggregate_id == state.active_identity,
                CanonicalEventModel.event_type == "generation_route_activated",
            )
            .order_by(CanonicalEventModel.occurred_at.desc(), CanonicalEventModel.id.desc())
            .limit(1)
        )
        if event is None:
            return None
        evidence = active["providers"][0]["validation_evidence"]
        return {
            "route_identity": state.active_identity,
            "activation_event_id": str(event.id),
            "acceptance_record_identity": evidence["record_identity"],
            "acceptance_evidence_sha256": evidence["evidence_sha256"],
        }

    async def verification_admission(self, route_identity: str, acceptance_identity: str, actor: User) -> dict:
        """Validate isolated invocation admission without constructing Providers or activating a route."""
        administrator = await self.session.scalar(
            select(User).where(User.id == actor.id).with_for_update().execution_options(populate_existing=True)
        )
        if administrator is None or not administrator.is_active or administrator.role != "admin":
            raise route_error("AUTH_FORBIDDEN", 403)
        await self.session.scalar(
            select(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == acceptance_identity).with_for_update()
        )
        route = await self._record(route_identity)
        evidence = await self._verify_acceptance(acceptance_identity, route)
        return {
            "route_identity": route_identity,
            "acceptance_record_identity": evidence["record_identity"],
            "acceptance_evidence_sha256": evidence["evidence_sha256"],
        }

    async def activate(self, *, actor: str, payload: object) -> dict:
        try:
            request = ActivationInput.model_validate(payload)
        except ValidationError as exc:
            raise route_error("GENERATION_ROUTE_INVALID", 400) from exc
        route = await self._record(request.route_identity)
        await self._verify_acceptance(request.acceptance_record_identity, route)
        await self.session.rollback()
        try:
            await self.validate_connection(route)
        except Exception as exc:
            await self.session.rollback()
            raise route_error("GENERATION_ROUTE_VALIDATION_FAILED") from exc
        await self.session.rollback()
        state = await self._lock_state()
        administrator = await self.session.scalar(
            select(User).where(User.id == uuid.UUID(actor.removeprefix("member:"))).with_for_update()
        )
        if administrator is None or not administrator.is_active or administrator.role != "admin":
            raise route_error("AUTH_FORBIDDEN", 403)
        if (state is None or state.draft_identity != request.route_identity
                or state.active_identity != request.expected_active_identity):
            raise route_error("GENERATION_ROUTE_STALE")
        await self.session.scalar(select(CanonicalRecordModel).where(
            CanonicalRecordModel.stable_id == request.acceptance_record_identity,
        ).with_for_update())
        evidence = await self._verify_acceptance(request.acceptance_record_identity, route)
        self.session.add(CanonicalEventModel(
            aggregate_id=request.route_identity, aggregate_kind="provider_route",
            event_type="generation_route_activated", from_state="inactive", to_state="active",
            recorded_by=actor,
            payload={"schema": "generation_route_activation/v1", **evidence, "previous_route_identity": state.active_identity},
        ))
        state.active_identity = request.route_identity
        state.version += 1
        await self.session.commit()
        return self._project(route, active=True, evidence=evidence)

    async def capture_verification(self, admission: dict, actor: User) -> ProviderRouter:
        current = await self.verification_admission(
            admission["route_identity"], admission["acceptance_record_identity"], actor,
        )
        if current != admission:
            raise route_error("GENERATION_ROUTE_STALE")
        route = await self._record(current["route_identity"])
        return ProviderRouter(providers=await self.providers(route), approved_route=route_contract(route))

    async def _lock_state(self) -> GenerationRouteState | None:
        if self.session.bind is not None and self.session.bind.dialect.name == "sqlite":
            await self.session.execute(update(GenerationRouteState).where(
                GenerationRouteState.id == 1,
            ).values(version=GenerationRouteState.version))
        return await self.session.scalar(select(GenerationRouteState).where(
            GenerationRouteState.id == 1,
        ).with_for_update().execution_options(populate_existing=True))

    async def _verify_acceptance(self, identity: str, route: dict) -> dict:
        try:
            projection = await DeliveryAcceptanceService(self.session).get_projection(identity)
            settings = get_settings()
            stages = (
                {"local_development"} if settings.generation_validation_mode == "local_development"
                else {"editorial_preview", "limited_team_pilot", "daily_use_release", "public_evidence_release"}
            )
            if (
                not re.fullmatch(r"deployment:[a-z0-9][a-z0-9._:-]{2,148}", settings.generation_deployment_identity)
                or not re.fullmatch(r"product_revision:[0-9a-f]{40}", settings.generation_product_revision)
                or projection["stage"] not in stages
                or projection["conditions"].get("generation_validation_mode") != settings.generation_validation_mode
                or projection["affected_scope"]["deployment_identity"] != settings.generation_deployment_identity
                or settings.generation_product_revision not in projection["product_identities"]
            ):
                raise ValueError("unbound activation environment")
            required = {route["route_identity"], *(p["approval_identity"] for p in route["providers"])}
            checks = {c["check_id"]: c for c in projection["checks"]}
            if (projection["current_status"] != "active" or projection["blockers"]
                    or not required.issubset(projection["product_identities"])
                    or not projection["approver_identities"]):
                raise ValueError("unverified acceptance")
            for check in _ACTIVATION_CHECKS:
                value = checks[check]
                if value["result"] not in {"passed", "carried_forward"} or not required.issubset(value["identity_dependencies"]):
                    raise ValueError("unbound activation evidence")
            return {"record_identity": identity, "evidence_sha256": canonical_json_sha256(jsonable_encoder(projection))}
        except (AppError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise route_error("GENERATION_ROUTE_ACCEPTANCE_REQUIRED") from exc

    async def _activation_evidence(self, identity: str) -> dict | None:
        event = await self.session.scalar(select(CanonicalEventModel).where(
            CanonicalEventModel.aggregate_id == identity,
            CanonicalEventModel.event_type == "generation_route_activated",
        ).order_by(CanonicalEventModel.occurred_at.desc()).limit(1))
        if event is not None and (
            event.aggregate_kind != "provider_route" or event.from_state != "inactive"
            or event.to_state != "active" or event.payload.get("schema") != "generation_route_activation/v1"
            or not (event.recorded_by or "").startswith("member:")
        ):
            raise route_error("GENERATION_ROUTE_INVALID")
        return (
            {"record_identity": event.payload["record_identity"], "evidence_sha256": event.payload["evidence_sha256"]}
            if event is not None else None
        )

    async def providers(self, payload: dict) -> dict:
        result = {}
        for item in payload["providers"]:
            secret = await self.session.get(GenerationRouteSecret, item["credential_reference"])
            if secret is None:
                raise route_error("GENERATION_SECRET_UNAVAILABLE")
            try:
                key = secret_box().decrypt(secret.ciphertext.encode("ascii")).decode("utf-8")
            except Exception as exc:
                raise route_error("GENERATION_SECRET_UNAVAILABLE") from exc
            settings = SystemSettingsRuntime._candidate_settings(settings=item, provider_api_key=key)
            provider = build_generation_provider(settings, timeout_seconds=item["timeout_seconds"])
            if provider is None:
                raise route_error("GENERATION_ROUTE_INVALID")
            result[item["provider"]] = provider
        return result

    async def validate_connection(self, payload: dict) -> None:
        providers = await self.providers(payload)
        async with asyncio.timeout(payload["total_timeout_seconds"]):
            for item in payload["providers"]:
                async with asyncio.timeout(item["timeout_seconds"]):
                    completion = await providers[item["provider"]].complete("Connection validation. Reply with OK.")
                    text = completion.text if isinstance(completion, GenerationCompletion) else completion
                    if text.strip() != "OK":
                        raise route_error("GENERATION_ROUTE_VALIDATION_FAILED")

    async def _record(self, identity: str) -> dict:
        record = await self.session.scalar(select(CanonicalRecordModel).where(CanonicalRecordModel.stable_id == identity))
        if record is None or record.identity_kind != "provider_route":
            raise route_error("GENERATION_ROUTE_NOT_FOUND", 404)
        payload = dict(record.payload)
        bound = {key: value for key, value in payload.items() if key not in {"configured_by", "route_identity"}}
        if (record.record_class != "immutable" or record.state != "inactive"
                or payload.get("schema") != "approved_generation_route/v1"
                or payload.get("route_identity") != identity
                or identity != "provider_route:" + canonical_json_sha256(bound)):
            raise route_error("GENERATION_ROUTE_INVALID")
        for provider in payload["providers"]:
            expected = canonical_json_sha256({key: value for key, value in provider.items() if key != "approval_identity"})
            if provider.get("approval_identity") != "configuration:" + expected:
                raise route_error("GENERATION_ROUTE_INVALID")
        route_contract(payload)
        return payload

    @staticmethod
    def _project(payload: dict, *, active: bool, evidence: dict | None = None) -> dict:
        return {
            "route_identity": payload["route_identity"],
            "schema": payload["schema"],
            "data_scope": payload["data_scope"],
            "max_attempts": payload["max_attempts"],
            "total_timeout_seconds": payload["total_timeout_seconds"],
            "activation_status": "active" if active else "inactive",
            "providers": [{
                **{key: item[key] for key in (
                    "provider", "provider_type", "model", "service_url", "endpoint_class",
                    "data_scope", "timeout_seconds", "approval_identity",
                )},
                "credential_configured": True,
                "validation_evidence": evidence,
            } for item in payload["providers"]],
        }
