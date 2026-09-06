from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.contracts.canonical import CanonicalRecordClass, StableIdentityKind
from app.model.canonical import CanonicalRecordModel
from app.reviewed_bundles.models import CandidateBuildJob

_SHA256_LENGTH = 64
_FROZEN_INPUT_HASH_FIELDS = (
    "schema",
    "bundle_id",
    "bundle_sha256",
    "bundle_item_id",
    "bundle_item_sha256",
    "entry_identity",
    "document_identity",
    "requested_generation",
    "editorial_source_revision",
    "input_sha256",
    "chunk_strategy",
    "embedding_configuration",
)


@dataclass(frozen=True)
class FrozenCandidateBuildInput:
    bundle_id: str
    bundle_sha256: str
    bundle_item_id: str
    bundle_item_sha256: str
    entry_identity: str
    document_identity: str
    requested_generation: int
    editorial_source_revision: str
    input_sha256: str
    frozen_input_sha256: str
    is_legacy_hash_backfill: bool
    chunk_strategy: dict[str, object]
    embedding_configuration: dict[str, object]
    artifact: dict[str, object]

    @property
    def embedding_fingerprint(self) -> str | None:
        if self.embedding_configuration.get("active") is not True:
            return None
        fingerprint = self.embedding_configuration.get("fingerprint")
        return fingerprint if isinstance(fingerprint, str) else None

    def matches_job(self, job: CandidateBuildJob) -> bool:
        return (
            job.bundle_id == self.bundle_id
            and job.bundle_item_id == self.bundle_item_id
            and job.entry_identity == self.entry_identity
            and job.document_identity == self.document_identity
            and job.requested_generation == self.requested_generation
            and job.editorial_source_revision == self.editorial_source_revision
            and job.input_sha256 == self.input_sha256
            and job.frozen_input_sha256 == self.frozen_input_sha256
            and job.chunk_strategy == self.chunk_strategy
            and job.embedding_configuration == self.embedding_configuration
        )


async def load_frozen_candidate_build_input(session: AsyncSession, job_id: str) -> FrozenCandidateBuildInput:
    input_record = await session.get(CanonicalRecordModel, f"build_generation:{job_id}")
    if input_record is None:
        _raise_input_integrity()
    _require_record(
        input_record,
        stable_id=f"build_generation:{job_id}",
        identity_kind=StableIdentityKind.BUILD_GENERATION.value,
        schema="candidate_build_input/v1",
    )
    input_payload = _require_payload(input_record)

    bundle_id = _require_string(input_payload, "bundle_id")
    bundle_sha256 = _require_sha256(input_payload, "bundle_sha256")
    bundle_item_id = _require_string(input_payload, "bundle_item_id")
    bundle_item_sha256 = _require_sha256(input_payload, "bundle_item_sha256")
    entry_identity = _require_string(input_payload, "entry_identity")
    document_identity = _require_string(input_payload, "document_identity")
    requested_generation = _require_positive_int(input_payload, "requested_generation")
    editorial_source_revision = _require_sha256(input_payload, "editorial_source_revision")
    input_sha256 = _require_sha256(input_payload, "input_sha256")
    chunk_strategy = _require_dict(input_payload, "chunk_strategy")
    embedding_configuration = _require_dict(input_payload, "embedding_configuration")
    frozen_input_sha256 = frozen_candidate_build_input_sha256(input_payload)
    stored_frozen_input_sha256 = input_payload.get("frozen_input_sha256")
    is_legacy_hash_backfill = "frozen_input_sha256" not in input_payload
    if not is_legacy_hash_backfill and (
        not isinstance(stored_frozen_input_sha256, str)
        or _require_sha256(input_payload, "frozen_input_sha256") != frozen_input_sha256
    ):
        _raise_input_integrity()

    item_record = await session.get(CanonicalRecordModel, bundle_item_id)
    if item_record is None:
        _raise_input_integrity()
    _require_record(
        item_record,
        stable_id=bundle_item_id,
        identity_kind=StableIdentityKind.BUNDLE_ITEM.value,
        schema="reviewed_release_bundle_item/v1",
    )
    item_payload = _require_payload(item_record)
    artifact = _require_dict(item_payload, "artifact")
    if (
        item_payload.get("bundle_id") != bundle_id
        or item_payload.get("entry_identity") != entry_identity
        or item_payload.get("artifact_sha256") != input_sha256
        or item_payload.get("bundle_item_sha256") != bundle_item_sha256
        or _structured_sha256(artifact) != input_sha256
        or _structured_sha256(
            {
                "bundle_item_id": item_record.identity_value,
                "operation": item_payload.get("operation"),
                "artifact_sha256": input_sha256,
                "artifact": artifact,
            }
        )
        != bundle_item_sha256
    ):
        _raise_input_integrity()

    bundle_record = await session.get(CanonicalRecordModel, bundle_id)
    if bundle_record is None:
        _raise_input_integrity()
    _require_record(
        bundle_record,
        stable_id=bundle_id,
        identity_kind=StableIdentityKind.BUNDLE.value,
        schema="reviewed_release_bundle/v1",
    )
    bundle_payload = _require_payload(bundle_record)
    manifest = _require_dict(bundle_payload, "manifest")
    if (
        bundle_payload.get("bundle_sha256") != bundle_sha256
        or manifest.get("bundle_sha256") != bundle_sha256
        or _structured_sha256({key: value for key, value in manifest.items() if key != "bundle_sha256"}) != bundle_sha256
    ):
        _raise_input_integrity()

    return FrozenCandidateBuildInput(
        bundle_id=bundle_id,
        bundle_sha256=bundle_sha256,
        bundle_item_id=bundle_item_id,
        bundle_item_sha256=bundle_item_sha256,
        entry_identity=entry_identity,
        document_identity=document_identity,
        requested_generation=requested_generation,
        editorial_source_revision=editorial_source_revision,
        input_sha256=input_sha256,
        frozen_input_sha256=frozen_input_sha256,
        is_legacy_hash_backfill=is_legacy_hash_backfill,
        chunk_strategy=chunk_strategy,
        embedding_configuration=embedding_configuration,
        artifact=artifact,
    )


def _require_record(
    record: CanonicalRecordModel,
    *,
    stable_id: str,
    identity_kind: str,
    schema: str,
) -> None:
    if (
        record.stable_id != stable_id
        or record.identity_kind != identity_kind
        or record.record_class != CanonicalRecordClass.IMMUTABLE.value
        or _require_payload(record).get("schema") != schema
    ):
        _raise_input_integrity()


def _require_payload(record: CanonicalRecordModel) -> dict[str, object]:
    if not isinstance(record.payload, dict):
        _raise_input_integrity()
    if any(not isinstance(key, str) for key in record.payload):
        _raise_input_integrity()
    return dict(record.payload)


def _require_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        _raise_input_integrity()
    return value


def _require_sha256(payload: dict[str, object], field: str) -> str:
    value = _require_string(payload, field)
    if len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        _raise_input_integrity()
    return value


def _require_positive_int(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _raise_input_integrity()
    return value


def _require_dict(payload: dict[str, object], field: str) -> dict[str, object]:
    value = payload.get(field)
    if not isinstance(value, dict):
        _raise_input_integrity()
    if any(not isinstance(key, str) for key in value):
        _raise_input_integrity()
    return dict(value)


def _structured_sha256(value: object) -> str:
    return canonical_json_sha256(value)


def frozen_candidate_build_input_sha256(payload: Mapping[str, object]) -> str:
    return canonical_json_sha256({field: payload[field] for field in _FROZEN_INPUT_HASH_FIELDS})


def _raise_input_integrity() -> NoReturn:
    raise AppError(
        status_code=409,
        code="CANDIDATE_INPUT_INTEGRITY_FAILED",
        message="candidate build immutable inputs no longer verify",
    )
