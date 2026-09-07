"""Public Evaluation Retriever comparison contract tests.

The tests exercise the evaluation-facing runner that powers
``retrieval-evidence evaluate``.  Dense vectors are deterministic fixtures so
the contract can be verified without a live provider; live-provider execution
remains a remote Release Gate concern.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import retrieval_evidence
from app.evaluation.evaluate import (
    DENSE_CANDIDATE_DEPTH,
    METRIC_NAMES,
    RRF_K,
    fuse_rrf_candidates,
    run_retrieval_evaluation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = REPOSITORY_ROOT / "evaluation"
VALIDATOR = REPOSITORY_ROOT / "scripts" / "validate-evidence-bundle.py"
REVISION = "f2b6c4a8e0d1f3a5b7c9d1e3f5a7b9c1d3e5f7a9"


class FixtureEmbeddingProvider:
    """A deterministic public adapter fixture, never a provider substitute."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float((sum(ord(character) for character in text) % 31) + 1), 1.0] for text in texts]


def _run_comparison(tmp_path: Path, *, bundle_dir: Path | None = None) -> dict:
    return asyncio.run(
        run_retrieval_evaluation(
            evaluation_dir=EVALUATION_DIR,
            output_dir=tmp_path / "runs",
            source_revision=REVISION,
            modes=("sparse_bm25", "dense", "hybrid_rrf"),
            embedding_provider=FixtureEmbeddingProvider(),
            embedding_identity="fixture-embedding-identity",
            run_id="comparison-test-run-001",
            bundle_dir=bundle_dir,
            now=lambda: datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC),
        )
    )


def test_evaluation_selects_all_three_modes_with_compatible_metrics_and_traces(tmp_path: Path) -> None:
    manifest = _run_comparison(tmp_path)

    assert manifest["modes"] == ["sparse_bm25", "dense", "hybrid_rrf"]
    assert manifest["algorithm"]["dense_candidate_depth"] == DENSE_CANDIDATE_DEPTH == 20
    assert manifest["algorithm"]["rrf_k"] == RRF_K == 60
    assert manifest["embedding_identity"] == "fixture-embedding-identity"

    mode_results = manifest["mode_results"]
    assert set(mode_results) == {"sparse_bm25", "dense", "hybrid_rrf"}
    for mode, result in mode_results.items():
        assert sorted(result["metrics"]) == sorted(METRIC_NAMES), mode
        assert len(result["query_results"]) == 16
        assert result["answerable_query_count"] == 12
        assert result["boundary_query_count"] == 4
        for query in result["query_results"]:
            assert len(query["candidates"]) <= 10
            if mode == "hybrid_rrf":
                chunk_ids = [candidate["chunk_id"] for candidate in query["candidates"]]
                assert len(chunk_ids) == len(set(chunk_ids))


def test_hybrid_rrf_uses_rank_only_deduplication_and_the_accepted_constant() -> None:
    sparse = [
        {"chunk_id": "shared", "score": 99.0},
        {"chunk_id": "sparse-only", "score": 1.0},
    ]
    dense = [
        {"chunk_id": "dense-only", "score": 0.1},
        {"chunk_id": "shared", "score": 0.01},
    ]

    fused = fuse_rrf_candidates(sparse, dense)

    assert [candidate["chunk_id"] for candidate in fused] == ["shared", "dense-only", "sparse-only"]
    assert fused[0]["score"] == round((1 / (RRF_K + 1)) + (1 / (RRF_K + 2)), 8)
    assert all("original_score" not in candidate for candidate in fused)


def test_comparison_bundle_records_shared_provenance_and_validator_rejects_drift(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    manifest = _run_comparison(tmp_path, bundle_dir=bundle_dir)

    section_path = bundle_dir / "sections" / "retrieval.json"
    section = json.loads(section_path.read_text(encoding="utf-8"))
    bundle_manifest_path = bundle_dir / "manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    assert bundle_manifest["schema_version"] == "1.1.0"
    assert section["modes"] == ["sparse_bm25", "dense", "hybrid_rrf"]
    mode_provenance = section["conditions"]["mode_provenance"]
    assert {item["mode"] for item in mode_provenance} == set(manifest["modes"])
    assert {item["source_revision"] for item in mode_provenance} == {REVISION}
    assert {item["embedding_identity"] for item in mode_provenance} == {"fixture-embedding-identity"}

    accepted = subprocess.run(
        [sys.executable, str(VALIDATOR), str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr

    del section["conditions"]["mode_provenance"]
    _rewrite_section_artifact_hash(section_path, section, bundle_manifest_path, bundle_manifest)
    missing_provenance = subprocess.run(
        [sys.executable, str(VALIDATOR), str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_provenance.returncode == 1
    assert "cross-mode provenance" in missing_provenance.stdout

    section["conditions"]["mode_provenance"] = mode_provenance
    mode_provenance[1]["corpus_sha256"] = "0" * 64
    _rewrite_section_artifact_hash(section_path, section, bundle_manifest_path, bundle_manifest)

    rejected = subprocess.run(
        [sys.executable, str(VALIDATOR), str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert "cross-mode provenance" in rejected.stdout


def _rewrite_section_artifact_hash(
    section_path: Path,
    section: dict,
    bundle_manifest_path: Path,
    bundle_manifest: dict,
) -> None:
    section_content = json.dumps(section, ensure_ascii=False, indent=2) + "\n"
    section_path.write_text(section_content, encoding="utf-8")
    for artifact in bundle_manifest["artifacts"]:
        if artifact["path"] == "sections/retrieval.json":
            artifact["sha256"] = sha256(section_content.encode("utf-8")).hexdigest()
    bundle_manifest_path.write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mode", "expected_modes"),
    [
        ("dense", ("dense",)),
        ("hybrid_rrf", ("hybrid_rrf",)),
        ("all", ("sparse_bm25", "dense", "hybrid_rrf")),
    ],
)
def test_evaluate_cli_selects_requested_comparison_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    expected_modes: tuple[str, ...],
) -> None:
    settings = SimpleNamespace(
        embedding_api_key="fixture-api-key",
        embedding_base_url_normalized="https://fixture.invalid/v1",
        embedding_model_normalized="fixture-model",
    )
    captured: dict[str, object] = {}
    embedding_provider_kwargs: dict[str, object] = {}

    async def fake_run_retrieval_evaluation(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "outcome": "passed",
            "run_id": kwargs["run_id"],
            "modes": list(kwargs["modes"]),
        }

    monkeypatch.setattr(retrieval_evidence, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retrieval_evidence.DenseEmbeddingContract,
        "from_settings",
        lambda _settings: SimpleNamespace(active=True, dimension=1536),
    )
    monkeypatch.setattr(
        retrieval_evidence,
        "OpenAIEmbeddingProvider",
        lambda **kwargs: embedding_provider_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(retrieval_evidence, "build_embedding_contract_fingerprint", lambda _settings: "fixture")
    monkeypatch.setattr(retrieval_evidence, "run_retrieval_evaluation", fake_run_retrieval_evaluation)

    exit_code = retrieval_evidence.main(
        [
            "evaluate",
            "--output-dir",
            str(tmp_path / "runs"),
            "--source-revision",
            REVISION,
            "--run-id",
            "comparison-cli-test-run",
            "--evaluation-dir",
            str(EVALUATION_DIR),
            "--mode",
            mode,
        ]
    )

    assert exit_code == 0
    assert captured["modes"] == expected_modes
    assert captured["embedding_identity"] == "fixture-model:fixture"
    assert embedding_provider_kwargs == {
        "api_key": "fixture-api-key",
        "base_url": "https://fixture.invalid/v1",
        "dimensions": 1536,
        "model": "fixture-model",
    }
    assert json.loads(capsys.readouterr().out) == {
        "modes": list(expected_modes),
        "outcome": "passed",
        "run_id": "comparison-cli-test-run",
    }
