from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.rag.claim_evidence import ClaimEvidenceContract, ClaimResolution, ResolvedClaim
from app.rag.interfaces import EmbeddingProvider

_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SemanticClaimResolverArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class _ClaimProfile:
    claim_id: str
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class _EntryProfile:
    contract_sha256: str
    boundary_vectors: tuple[tuple[float, ...], ...]
    claims: tuple[_ClaimProfile, ...]


class SemanticClaimResolver:
    def __init__(
        self,
        *,
        resolver_id: str,
        calibration_id: str,
        calibration_version: str,
        profile_sha256: str,
        calibration_set_sha256: str,
        calibration_report_sha256: str,
        embedding_contract_fingerprint: str,
        embedding_provider: EmbeddingProvider,
        embedding_dimension: int,
        selection_threshold: float,
        boundary_margin: float,
        cross_entry_margin: float,
        resolved_confidence: float,
        profiles: Mapping[str, _EntryProfile],
    ) -> None:
        self.resolver_id = resolver_id
        self.calibration_id = calibration_id
        self.calibration_version = calibration_version
        self.profile_sha256 = profile_sha256
        self.calibration_set_sha256 = calibration_set_sha256
        self.calibration_report_sha256 = calibration_report_sha256
        self.embedding_contract_fingerprint = embedding_contract_fingerprint
        self._embedding_provider = embedding_provider
        self._embedding_dimension = embedding_dimension
        self._selection_threshold = selection_threshold
        self._boundary_margin = boundary_margin
        self._cross_entry_margin = cross_entry_margin
        self._resolved_confidence = resolved_confidence
        self._profiles = dict(profiles)

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    @staticmethod
    def _normalized_vector(value: object, *, dimension: int, field: str) -> tuple[float, ...]:
        if not isinstance(value, list) or len(value) != dimension:
            raise SemanticClaimResolverArtifactError(f"{field} must match the embedding dimension")
        vector: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise SemanticClaimResolverArtifactError(f"{field} must contain finite numbers")
            vector.append(float(item))
        norm = math.sqrt(sum(item * item for item in vector))
        if norm <= 0:
            raise SemanticClaimResolverArtifactError(f"{field} must have a non-zero norm")
        return tuple(item / norm for item in vector)

    async def resolve(
        self,
        question: str,
        contracts: Mapping[str, ClaimEvidenceContract],
    ) -> ClaimResolution:
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise SemanticClaimResolverArtifactError("question must be non-empty and bounded")
        if not contracts:
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_scope")

        active_profiles: dict[str, _EntryProfile] = {}
        expected_identity = (self.resolver_id, self.calibration_id, self.calibration_version)
        for entry_id, contract in contracts.items():
            profile = self._profiles.get(entry_id)
            if profile is None:
                raise SemanticClaimResolverArtifactError(f"profile is missing contract for {entry_id}")
            if profile.contract_sha256 != contract.sha256:
                raise SemanticClaimResolverArtifactError(f"contract hash mismatch for {entry_id}")
            contract_identity = (
                contract.resolver.resolver_id,
                contract.resolver.calibration_id,
                contract.resolver.calibration_version,
            )
            if contract_identity != expected_identity:
                raise SemanticClaimResolverArtifactError(f"resolver identity mismatch for {entry_id}")
            profile_claim_ids = {claim.claim_id for claim in profile.claims}
            contract_claim_ids = {claim.claim_id for claim in contract.claims}
            if profile_claim_ids != contract_claim_ids:
                raise SemanticClaimResolverArtifactError(f"claim profile mismatch for {entry_id}")
            if self._resolved_confidence < contract.resolver.minimum_confidence:
                raise SemanticClaimResolverArtifactError(f"calibrated confidence is too low for {entry_id}")
            active_profiles[entry_id] = profile

        query_vectors = await self._embedding_provider.embed([question.strip()])
        if not isinstance(query_vectors, list) or len(query_vectors) != 1:
            raise SemanticClaimResolverArtifactError("embedding provider returned an unexpected vector count")
        query_vector = self._normalized_vector(
            query_vectors[0],
            dimension=self._embedding_dimension,
            field="query vector",
        )

        entry_scores: dict[str, list[tuple[str, float]]] = {}
        for entry_id, profile in active_profiles.items():
            entry_scores[entry_id] = [
                (
                    claim.claim_id,
                    max(self._cosine(query_vector, vector) for vector in claim.vectors),
                )
                for claim in profile.claims
            ]
        ranked_entries = sorted(
            ((max(score for _claim_id, score in scores), entry_id) for entry_id, scores in entry_scores.items()),
            reverse=True,
        )
        top_score, top_entry_id = ranked_entries[0]
        if top_score < self._selection_threshold:
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_scope")
        if (
            len(ranked_entries) > 1
            and ranked_entries[1][0] >= self._selection_threshold
            and top_score - ranked_entries[1][0] < self._cross_entry_margin
        ):
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_ambiguity")
        boundary_score = max(
            self._cosine(query_vector, vector) for vector in active_profiles[top_entry_id].boundary_vectors
        )
        if top_score - boundary_score < self._boundary_margin:
            return ClaimResolution(required_claims=(), out_of_scope=True, reason="reject_claim_boundary")

        required_claims = tuple(
            ResolvedClaim(
                entry_id=top_entry_id,
                claim_id=claim_id,
                confidence=self._resolved_confidence,
            )
            for claim_id, score in entry_scores[top_entry_id]
            if score >= self._selection_threshold
        )
        return ClaimResolution(required_claims=required_claims, out_of_scope=False, reason="resolved_claims")


def _mapping(value: object, *, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SemanticClaimResolverArtifactError(f"{field} must contain exactly {sorted(keys)}")
    return value


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value.strip()) is None:
        raise SemanticClaimResolverArtifactError(f"{field} must be a stable lowercase identifier")
    return value.strip()


def _bounded_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 512:
        raise SemanticClaimResolverArtifactError(f"{field} must be a non-empty bounded string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SemanticClaimResolverArtifactError(f"{field} must be a lowercase sha256")
    return value


def _probability(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SemanticClaimResolverArtifactError(f"{field} must be a finite number")
    result = float(value)
    if not 0 < result <= 1:
        raise SemanticClaimResolverArtifactError(f"{field} must be greater than zero and at most one")
    return result


def load_semantic_claim_resolver(
    payload: bytes,
    *,
    expected_sha256: str,
    embedding_provider: EmbeddingProvider,
    active_embedding_model: str,
    active_embedding_dimension: int,
    active_embedding_contract_fingerprint: str,
) -> SemanticClaimResolver:
    expected_hash = _sha256(expected_sha256, field="expected profile sha256")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise SemanticClaimResolverArtifactError("profile sha256 does not match the protected expectation")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticClaimResolverArtifactError("profile must be valid UTF-8 JSON") from exc
    root = _mapping(
        raw,
        field="profile",
        keys={"schema_version", "resolver", "embedding", "calibration", "calibration_evidence", "contracts"},
    )
    if root.get("schema_version") != 1:
        raise SemanticClaimResolverArtifactError("profile.schema_version must be 1")

    resolver = _mapping(
        root.get("resolver"),
        field="profile.resolver",
        keys={"resolver_id", "calibration_id", "calibration_version"},
    )
    resolver_id = _safe_id(resolver.get("resolver_id"), field="profile.resolver.resolver_id")
    calibration_id = _safe_id(resolver.get("calibration_id"), field="profile.resolver.calibration_id")
    calibration_version = _bounded_text(
        resolver.get("calibration_version"),
        field="profile.resolver.calibration_version",
    )

    embedding = _mapping(
        root.get("embedding"),
        field="profile.embedding",
        keys={"model", "dimension", "contract_fingerprint"},
    )
    embedding_model = _bounded_text(embedding.get("model"), field="profile.embedding.model")
    dimension = embedding.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 8192:
        raise SemanticClaimResolverArtifactError("profile.embedding.dimension is invalid")
    if embedding_model != active_embedding_model:
        raise SemanticClaimResolverArtifactError("profile embedding model does not match the active embedding model")
    if dimension != active_embedding_dimension:
        raise SemanticClaimResolverArtifactError("profile embedding dimension does not match the active embedding dimension")
    embedding_contract_fingerprint = _sha256(
        embedding.get("contract_fingerprint"),
        field="profile.embedding.contract_fingerprint",
    )
    active_fingerprint = _sha256(
        active_embedding_contract_fingerprint,
        field="active embedding contract fingerprint",
    )
    if embedding_contract_fingerprint != active_fingerprint:
        raise SemanticClaimResolverArtifactError(
            "profile embedding contract fingerprint does not match the active embedding contract"
        )

    calibration = _mapping(
        root.get("calibration"),
        field="profile.calibration",
        keys={"selection_threshold", "boundary_margin", "cross_entry_margin", "resolved_confidence"},
    )
    selection_threshold = _probability(
        calibration.get("selection_threshold"),
        field="profile.calibration.selection_threshold",
    )
    boundary_margin = _probability(
        calibration.get("boundary_margin"),
        field="profile.calibration.boundary_margin",
    )
    cross_entry_margin = _probability(
        calibration.get("cross_entry_margin"),
        field="profile.calibration.cross_entry_margin",
    )
    resolved_confidence = _probability(
        calibration.get("resolved_confidence"),
        field="profile.calibration.resolved_confidence",
    )

    evidence = _mapping(
        root.get("calibration_evidence"),
        field="profile.calibration_evidence",
        keys={
            "calibration_set_sha256",
            "report_sha256",
            "sample_count",
            "precision",
            "recall",
            "boundary_rejection_rate",
        },
    )
    calibration_set_sha256 = _sha256(
        evidence.get("calibration_set_sha256"),
        field="profile.calibration_evidence.calibration_set_sha256",
    )
    calibration_report_sha256 = _sha256(
        evidence.get("report_sha256"),
        field="profile.calibration_evidence.report_sha256",
    )
    sample_count = evidence.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 100:
        raise SemanticClaimResolverArtifactError("profile.calibration_evidence.sample_count must be at least 100")
    calibrated_precision = _probability(
        evidence.get("precision"),
        field="profile.calibration_evidence.precision",
    )
    calibrated_recall = _probability(evidence.get("recall"), field="profile.calibration_evidence.recall")
    boundary_rejection_rate = _probability(
        evidence.get("boundary_rejection_rate"),
        field="profile.calibration_evidence.boundary_rejection_rate",
    )
    if calibrated_precision < 0.95:
        raise SemanticClaimResolverArtifactError("profile calibrated precision is below 0.95")
    if calibrated_recall < 0.90:
        raise SemanticClaimResolverArtifactError("profile calibrated recall is below 0.90")
    if boundary_rejection_rate != 1:
        raise SemanticClaimResolverArtifactError("profile calibrated boundary rejection must be 1.0")
    if resolved_confidence > calibrated_precision:
        raise SemanticClaimResolverArtifactError("resolved confidence exceeds calibrated precision")

    raw_contracts = root.get("contracts")
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise SemanticClaimResolverArtifactError("profile.contracts must be a non-empty list")
    profiles: dict[str, _EntryProfile] = {}
    for entry_index, raw_entry in enumerate(raw_contracts):
        entry = _mapping(
            raw_entry,
            field=f"profile.contracts[{entry_index}]",
            keys={"entry_id", "contract_sha256", "boundary_vectors", "claims"},
        )
        entry_id = _safe_id(entry.get("entry_id"), field=f"profile.contracts[{entry_index}].entry_id")
        if entry_id in profiles:
            raise SemanticClaimResolverArtifactError("profile.contracts contains duplicate entry_id")
        raw_boundary_vectors = entry.get("boundary_vectors")
        if not isinstance(raw_boundary_vectors, list) or not raw_boundary_vectors:
            raise SemanticClaimResolverArtifactError("profile boundary vectors must be non-empty")
        boundary_vectors = tuple(
            SemanticClaimResolver._normalized_vector(
                vector,
                dimension=dimension,
                field=f"profile.contracts[{entry_index}].boundary_vectors[{vector_index}]",
            )
            for vector_index, vector in enumerate(raw_boundary_vectors)
        )
        raw_claims = entry.get("claims")
        if not isinstance(raw_claims, list) or not raw_claims:
            raise SemanticClaimResolverArtifactError(f"profile.contracts[{entry_index}].claims must be non-empty")
        claims: list[_ClaimProfile] = []
        seen_claim_ids: set[str] = set()
        for claim_index, raw_claim in enumerate(raw_claims):
            claim = _mapping(
                raw_claim,
                field=f"profile.contracts[{entry_index}].claims[{claim_index}]",
                keys={"claim_id", "vectors"},
            )
            claim_id = _safe_id(
                claim.get("claim_id"),
                field=f"profile.contracts[{entry_index}].claims[{claim_index}].claim_id",
            )
            if claim_id in seen_claim_ids:
                raise SemanticClaimResolverArtifactError("profile claim_id must not repeat within an entry")
            seen_claim_ids.add(claim_id)
            raw_vectors = claim.get("vectors")
            if not isinstance(raw_vectors, list) or not raw_vectors:
                raise SemanticClaimResolverArtifactError("profile claim vectors must be non-empty")
            vectors = tuple(
                SemanticClaimResolver._normalized_vector(
                    vector,
                    dimension=dimension,
                    field=f"profile.contracts[{entry_index}].claims[{claim_index}].vectors[{vector_index}]",
                )
                for vector_index, vector in enumerate(raw_vectors)
            )
            claims.append(_ClaimProfile(claim_id=claim_id, vectors=vectors))
        profiles[entry_id] = _EntryProfile(
            contract_sha256=_sha256(
                entry.get("contract_sha256"),
                field=f"profile.contracts[{entry_index}].contract_sha256",
            ),
            boundary_vectors=boundary_vectors,
            claims=tuple(claims),
        )

    return SemanticClaimResolver(
        resolver_id=resolver_id,
        calibration_id=calibration_id,
        calibration_version=calibration_version,
        profile_sha256=actual_hash,
        calibration_set_sha256=calibration_set_sha256,
        calibration_report_sha256=calibration_report_sha256,
        embedding_contract_fingerprint=embedding_contract_fingerprint,
        embedding_provider=embedding_provider,
        embedding_dimension=dimension,
        selection_threshold=selection_threshold,
        boundary_margin=boundary_margin,
        cross_entry_margin=cross_entry_margin,
        resolved_confidence=resolved_confidence,
        profiles=profiles,
    )


def load_semantic_claim_resolver_file(
    path: Path,
    *,
    expected_sha256: str,
    embedding_provider: EmbeddingProvider,
    active_embedding_model: str,
    active_embedding_dimension: int,
    active_embedding_contract_fingerprint: str,
) -> SemanticClaimResolver:
    if not path.is_absolute():
        raise SemanticClaimResolverArtifactError("claim resolver profile path must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SemanticClaimResolverArtifactError("claim resolver profile is unavailable") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SemanticClaimResolverArtifactError("claim resolver profile must be a regular file")
        if file_stat.st_uid != os.geteuid():
            raise SemanticClaimResolverArtifactError("claim resolver profile must be owned by the service user")
        if file_stat.st_mode & 0o077:
            raise SemanticClaimResolverArtifactError(
                "claim resolver profile must not be accessible by group or other users"
            )
        if not 1 <= file_stat.st_size <= 32 * 1024 * 1024:
            raise SemanticClaimResolverArtifactError("claim resolver profile size is invalid")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(32 * 1024 * 1024 + 1)
    except OSError as exc:
        raise SemanticClaimResolverArtifactError("claim resolver profile cannot be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return load_semantic_claim_resolver(
        payload,
        expected_sha256=expected_sha256,
        embedding_provider=embedding_provider,
        active_embedding_model=active_embedding_model,
        active_embedding_dimension=active_embedding_dimension,
        active_embedding_contract_fingerprint=active_embedding_contract_fingerprint,
    )
