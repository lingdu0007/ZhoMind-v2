import hashlib
import json
from pathlib import Path

import pytest

from app.common.config import get_settings
from app.extensions.registry import ExtensionRegistry, get_extension_registry
from app.rag.semantic_claim_resolver import SemanticClaimResolverArtifactError


def test_capability_registry_roundtrip() -> None:
    registry = ExtensionRegistry()
    registry.register_capability("llm", "chat-default-llm", {"supports_stream": True, "cost_tier": "medium"})
    cap = registry.get_capability("llm", "chat-default-llm")
    assert cap is not None
    assert cap["supports_stream"] is True


def test_choose_provider_by_capability() -> None:
    registry = ExtensionRegistry()
    registry.register_capability("llm", "provider-a", {"supports_stream": False})
    registry.register_capability("llm", "provider-b", {"supports_stream": True})
    assert registry.choose_provider("llm", ["provider-a", "provider-b"], {"supports_stream": True}) == "provider-b"


def test_registry_never_discovers_generation_providers_from_available_credentials(monkeypatch) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    monkeypatch.setenv("BASE_URL", "https://ark.example.com/v3")
    monkeypatch.setenv("MODEL", "ark-model")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    registry = get_extension_registry()

    assert registry.get_llm("ark") is None
    assert registry.get_llm("openai") is None
    assert registry.get_llm("anthropic") is None
    assert registry.llm_providers == {}

    get_settings.cache_clear()
    get_extension_registry.cache_clear()


def _resolver_profile() -> bytes:
    payload = {
        "schema_version": 1,
        "resolver": {
            "resolver_id": "calibrated-semantic-claim-resolver-v1",
            "calibration_id": "semantic-claim-calibration-20260813",
            "calibration_version": "2026-08-13",
        },
        "embedding": {
            "model": "fixture-embedding-v1",
            "dimension": 3,
            "contract_fingerprint": hashlib.sha256(
                json.dumps(
                    {
                        "embedding_base_url": "https://embedding.example.com/v1",
                        "embedding_model": "fixture-embedding-v1",
                        "dense_embedding_dim": 3,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "calibration": {
            "selection_threshold": 0.7,
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
                "contract_sha256": "c" * 64,
                "boundary_vectors": [[0.0, 0.0, 1.0]],
                "claims": [{"claim_id": "workflow-controls-known-paths", "vectors": [[1.0, 0.0, 0.0]]}],
            }
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def test_registry_registers_only_hash_pinned_semantic_claim_resolver(monkeypatch, tmp_path: Path) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    payload = _resolver_profile()
    profile = tmp_path / "claim-resolver-profile.json"
    profile.write_bytes(payload)
    profile.chmod(0o600)
    monkeypatch.setenv("EMBEDDING_API_KEY", "fixture-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example.com/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fixture-embedding-v1")
    monkeypatch.setenv("DENSE_EMBEDDING_DIM", "3")
    monkeypatch.setenv("CLAIM_RESOLVER_PROFILE_PATH", str(profile))
    monkeypatch.setenv("CLAIM_RESOLVER_PROFILE_SHA256", hashlib.sha256(payload).hexdigest())

    resolver = get_extension_registry().get_claim_resolver("chat-default-claim-resolver")

    assert resolver is not None
    assert resolver.resolver_id == "calibrated-semantic-claim-resolver-v1"
    get_settings.cache_clear()
    get_extension_registry.cache_clear()


def test_registry_fails_closed_when_resolver_configuration_is_partial_or_tampered(monkeypatch, tmp_path: Path) -> None:
    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    profile = tmp_path / "claim-resolver-profile.json"
    profile.write_bytes(_resolver_profile())
    profile.chmod(0o600)
    monkeypatch.setenv("CLAIM_RESOLVER_PROFILE_PATH", str(profile))

    with pytest.raises(SemanticClaimResolverArtifactError, match="PROFILE_PATH and CLAIM_RESOLVER_PROFILE_SHA256"):
        get_extension_registry()

    get_settings.cache_clear()
    get_extension_registry.cache_clear()
    monkeypatch.setenv("CLAIM_RESOLVER_PROFILE_SHA256", "0" * 64)
    monkeypatch.setenv("EMBEDDING_API_KEY", "fixture-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example.com/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fixture-embedding-v1")
    monkeypatch.setenv("DENSE_EMBEDDING_DIM", "3")
    with pytest.raises(SemanticClaimResolverArtifactError, match="sha256"):
        get_extension_registry()

    get_settings.cache_clear()
    get_extension_registry.cache_clear()
