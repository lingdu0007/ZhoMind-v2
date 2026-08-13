import asyncio
import hashlib
import json
import math
from pathlib import Path

import pytest

from app.rag.claim_evidence import parse_claim_evidence_contract
from app.rag.semantic_claim_resolver import (
    SemanticClaimResolverArtifactError,
    load_semantic_claim_resolver,
    load_semantic_claim_resolver_file,
)


def _contract():
    return parse_claim_evidence_contract(
        {
            "schema_version": 1,
            "review_id": "editorial-review-20260813-resolver",
            "review_revision": "2026-08-13.1",
            "conflict_state": "none",
            "unknown_state": "none",
            "resolver": {
                "resolver_id": "calibrated-semantic-claim-resolver-v1",
                "calibration_id": "semantic-claim-calibration-20260813",
                "calibration_version": "2026-08-13",
                "minimum_confidence": 0.90,
            },
            "claims": [
                {
                    "claim_id": "workflow-controls-known-paths",
                    "scope": "Known execution paths use deterministic workflow control.",
                    "evidence": [{"section_id": "stable-principle", "source_id": "source-workflow"}],
                },
                {
                    "claim_id": "agent-handles-runtime-decisions",
                    "scope": "Runtime decisions that depend on tool observations use a bounded Agent.",
                    "evidence": [{"section_id": "recommendation", "source_id": "source-agent"}],
                },
            ],
        }
    )


def _artifact(contract_sha256: str) -> bytes:
    payload = {
        "schema_version": 1,
        "resolver": {
            "resolver_id": "calibrated-semantic-claim-resolver-v1",
            "calibration_id": "semantic-claim-calibration-20260813",
            "calibration_version": "2026-08-13",
        },
        "embedding": {"model": "fixture-embedding-v1", "dimension": 3, "contract_fingerprint": "f" * 64},
        "calibration": {
            "selection_threshold": 0.70,
            "boundary_margin": 0.25,
            "cross_entry_margin": 0.05,
            "resolved_confidence": 0.95,
        },
        "calibration_evidence": {
            "calibration_set_sha256": "a" * 64,
            "report_sha256": "b" * 64,
            "sample_count": 100,
            "precision": 0.96,
            "recall": 0.91,
            "boundary_rejection_rate": 1.0,
        },
        "contracts": [
            {
                "entry_id": "pae-workflow-001",
                "contract_sha256": contract_sha256,
                "boundary_vectors": [[0.0, 0.0, 1.0]],
                "claims": [
                    {
                        "claim_id": "workflow-controls-known-paths",
                        "vectors": [[1.0, 0.0, 0.0]],
                    },
                    {
                        "claim_id": "agent-handles-runtime-decisions",
                        "vectors": [[0.0, 1.0, 0.0]],
                    },
                ],
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


class _FixtureEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "direct": [1.0, 0.0, 0.0],
            "combined": [math.sqrt(0.5), math.sqrt(0.5), 0.0],
            "boundary": [0.0, 0.0, 1.0],
            "near-boundary": [0.8, 0.0, 0.6],
        }
        return [vectors[text] for text in texts]


def _resolver(payload: bytes, *, expected_sha256: str | None = None):
    return load_semantic_claim_resolver(
        payload,
        expected_sha256=expected_sha256 or hashlib.sha256(payload).hexdigest(),
        embedding_provider=_FixtureEmbeddingProvider(),
        active_embedding_model="fixture-embedding-v1",
        active_embedding_dimension=3,
        active_embedding_contract_fingerprint="f" * 64,
    )


def test_semantic_claim_resolver_resolves_direct_combined_and_boundary_queries() -> None:
    contract = _contract()
    resolver = _resolver(_artifact(contract.sha256))

    direct = asyncio.run(resolver.resolve("direct", {"pae-workflow-001": contract}))
    combined = asyncio.run(resolver.resolve("combined", {"pae-workflow-001": contract}))
    boundary = asyncio.run(resolver.resolve("boundary", {"pae-workflow-001": contract}))
    near_boundary = asyncio.run(resolver.resolve("near-boundary", {"pae-workflow-001": contract}))

    assert [(item.entry_id, item.claim_id, item.confidence) for item in direct.required_claims] == [
        ("pae-workflow-001", "workflow-controls-known-paths", 0.95)
    ]
    assert {item.claim_id for item in combined.required_claims} == {
        "workflow-controls-known-paths",
        "agent-handles-runtime-decisions",
    }
    assert (boundary.required_claims, boundary.out_of_scope, boundary.reason) == (
        (),
        True,
        "reject_claim_scope",
    )
    assert (near_boundary.required_claims, near_boundary.out_of_scope, near_boundary.reason) == (
        (),
        True,
        "reject_claim_boundary",
    )
    assert resolver.profile_sha256 == hashlib.sha256(_artifact(contract.sha256)).hexdigest()
    assert resolver.calibration_set_sha256 == "a" * 64
    assert resolver.calibration_report_sha256 == "b" * 64


def test_semantic_claim_resolver_rejects_tampered_profile_and_contract_or_embedding_identity_mismatch() -> None:
    contract = _contract()
    payload = _artifact(contract.sha256)

    with pytest.raises(SemanticClaimResolverArtifactError, match="sha256"):
        _resolver(payload + b" ", expected_sha256=hashlib.sha256(payload).hexdigest())

    with pytest.raises(SemanticClaimResolverArtifactError, match="embedding model"):
        load_semantic_claim_resolver(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            embedding_provider=_FixtureEmbeddingProvider(),
            active_embedding_model="different-model",
            active_embedding_dimension=3,
            active_embedding_contract_fingerprint="f" * 64,
        )

    with pytest.raises(SemanticClaimResolverArtifactError, match="contract fingerprint"):
        load_semantic_claim_resolver(
            payload,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            embedding_provider=_FixtureEmbeddingProvider(),
            active_embedding_model="fixture-embedding-v1",
            active_embedding_dimension=3,
            active_embedding_contract_fingerprint="e" * 64,
        )

    resolver = _resolver(payload)
    other_contract = _contract().to_record()
    other_contract["review_revision"] = "2026-08-13.2"
    parsed_other = parse_claim_evidence_contract(other_contract)
    with pytest.raises(SemanticClaimResolverArtifactError, match="contract hash"):
        asyncio.run(resolver.resolve("direct", {"pae-workflow-001": parsed_other}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision", 0.94, "precision"),
        ("recall", 0.89, "recall"),
        ("boundary_rejection_rate", 0.99, "boundary rejection"),
    ],
)
def test_semantic_claim_resolver_rejects_profile_below_calibration_gate(
    field: str,
    value: float,
    message: str,
) -> None:
    raw = json.loads(_artifact(_contract().sha256))
    raw["calibration_evidence"][field] = value
    payload = json.dumps(raw, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(SemanticClaimResolverArtifactError, match=message):
        _resolver(payload)


def test_semantic_claim_resolver_file_requires_absolute_non_writable_expected_artifact(tmp_path: Path) -> None:
    payload = _artifact(_contract().sha256)
    profile = tmp_path / "profile.json"
    profile.write_bytes(payload)
    profile.chmod(0o600)

    resolver = load_semantic_claim_resolver_file(
        profile,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        embedding_provider=_FixtureEmbeddingProvider(),
        active_embedding_model="fixture-embedding-v1",
        active_embedding_dimension=3,
        active_embedding_contract_fingerprint="f" * 64,
    )
    assert resolver.profile_sha256 == hashlib.sha256(payload).hexdigest()

    profile.chmod(0o620)
    with pytest.raises(SemanticClaimResolverArtifactError, match="must not be accessible"):
        load_semantic_claim_resolver_file(
            profile,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            embedding_provider=_FixtureEmbeddingProvider(),
            active_embedding_model="fixture-embedding-v1",
            active_embedding_dimension=3,
            active_embedding_contract_fingerprint="f" * 64,
        )

    profile.chmod(0o600)
    symlink = tmp_path / "profile-symlink.json"
    symlink.symlink_to(profile)
    with pytest.raises(SemanticClaimResolverArtifactError, match="unavailable"):
        load_semantic_claim_resolver_file(
            symlink,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            embedding_provider=_FixtureEmbeddingProvider(),
            active_embedding_model="fixture-embedding-v1",
            active_embedding_dimension=3,
            active_embedding_contract_fingerprint="f" * 64,
        )
