"""Sparse BM25 evaluation (retrieval-evidence evaluate) tests.

The seam is the public ``run_sparse_bm25_evaluation`` entry point that the
``retrieval-evidence evaluate`` command invokes over the frozen evaluation
inputs. Tests assert the metric contract (evidence recall@3/5/10, context
precision@3/5/10, first gold evidence rank, retrieval duration), Boundary
Query exclusion, deterministic candidates, and the exported Evidence Package
accepted by the Public Evidence Bundle validator.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.evaluation.evaluate import (
    METRIC_NAMES,
    OUTPUT_DEPTHS,
    aggregate_metrics,
    chunk_covers_passage,
    run_sparse_bm25_evaluation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = REPOSITORY_ROOT / "evaluation"
VALIDATOR = REPOSITORY_ROOT / "scripts" / "validate-evidence-bundle.py"

# Placeholder, clearly non-sensitive test revision (40 hex chars).
REVISION = "f2b6c4a8e0d1f3a5b7c9d1e3f5a7b9c1d3e5f7a9"
FROZEN_CORPUS_SHA = "894d10004f25a2a6462870c1a869586ca83dfc28f0407f75f8cc33d9757ac39d"
FROZEN_QUERY_SET_SHA = "a468a1af9fc0c7b979ec1c5dfa8874a5ad27e25bcc79de1e0142564db992df20"
BOUNDARY_QUERY_IDS = {
    "boundary-lexical-is-bm25",
    "boundary-chunking-300",
    "boundary-hybrid-winner",
    "boundary-production-ports",
}


def _run_evaluate(
    tmp_path: Path,
    *,
    run_id: str = "evaluate-test-run-001",
    bundle_dir: Path | None = None,
    evaluation_dir: Path = EVALUATION_DIR,
) -> dict:
    return run_sparse_bm25_evaluation(
        evaluation_dir=evaluation_dir,
        output_dir=tmp_path / "runs",
        source_revision=REVISION,
        run_id=run_id,
        bundle_dir=bundle_dir,
        now=lambda: datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC),
    )


def test_gold_coverage_requires_half_of_the_passage() -> None:
    assert chunk_covers_passage(chunk_start=0, chunk_end=100, passage_start=10, passage_end=90)
    assert chunk_covers_passage(chunk_start=0, chunk_end=100, passage_start=0, passage_end=150)  # 2/3 covered
    assert chunk_covers_passage(chunk_start=0, chunk_end=100, passage_start=50, passage_end=150)  # exactly half
    assert not chunk_covers_passage(chunk_start=0, chunk_end=100, passage_start=60, passage_end=260)
    assert not chunk_covers_passage(chunk_start=0, chunk_end=100, passage_start=100, passage_end=200)


def test_evaluate_anchors_the_frozen_evaluation_inputs(tmp_path: Path) -> None:
    manifest = _run_evaluate(tmp_path)
    assert manifest["outcome"] == "passed"
    inputs = manifest["evaluation_inputs"]
    assert inputs["corpus_id"] == "project-derived-corpus"
    assert inputs["corpus_sha256"] == FROZEN_CORPUS_SHA
    assert inputs["query_set_id"] == "evaluation-query-set"
    assert inputs["query_set_sha256"] == FROZEN_QUERY_SET_SHA
    assert inputs["qa_chunking"] == {"policy": "qa", "chunk_chars": 500, "overlap_chars": 50}


def test_evaluate_writes_the_internal_run_manifest(tmp_path: Path) -> None:
    manifest = _run_evaluate(tmp_path)
    persisted = json.loads((tmp_path / "runs" / "evaluate-test-run-001" / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest
    assert manifest["command"] == "retrieval-evidence evaluate"
    assert manifest["mode"] == "sparse_bm25"
    assert "test-revision-secret" not in json.dumps(manifest)


def test_evaluate_reports_the_full_metric_family_with_valid_ranges(tmp_path: Path) -> None:
    manifest = _run_evaluate(tmp_path)
    metrics = manifest["metrics"]
    assert sorted(metrics) == sorted(METRIC_NAMES)
    for name, value in metrics.items():
        if name.startswith(("evidence_recall", "context_precision")):
            assert 0 <= value <= 1, f"{name} must be a ratio in [0, 1], got {value}"
        elif name == "first_gold_rank":
            assert value >= 1, f"first_gold_rank must be >= 1, got {value}"
        else:
            assert value >= 0, f"{name} must be non-negative, got {value}"


def test_evaluate_aggregates_only_answerable_queries(tmp_path: Path) -> None:
    manifest = _run_evaluate(tmp_path)
    assert manifest["answerable_query_count"] == 12
    assert manifest["boundary_query_count"] == 4
    query_results = manifest["query_results"]
    assert len(query_results) == 16
    answerable = [item for item in query_results if item["category"] != "boundary"]
    assert len(answerable) == 12
    for depth in OUTPUT_DEPTHS:
        expected_recall = sum(item["evidence_recall"][str(depth)] for item in answerable) / len(answerable)
        assert manifest["metrics"][f"evidence_recall@{depth}"] == pytest.approx(expected_recall)
        expected_precision = sum(item["context_precision"][str(depth)] for item in answerable) / len(answerable)
        assert manifest["metrics"][f"context_precision@{depth}"] == pytest.approx(expected_precision)
    expected_rank = sum(item["first_gold_rank"] for item in answerable) / len(answerable)
    assert manifest["metrics"]["first_gold_rank"] == pytest.approx(expected_rank)


def test_aggregate_metrics_is_a_macro_mean_over_the_given_queries() -> None:
    per_query = [
        {
            "evidence_recall@3": 1.0,
            "evidence_recall@5": 1.0,
            "evidence_recall@10": 1.0,
            "context_precision@3": 0.5,
            "context_precision@5": 0.4,
            "context_precision@10": 0.3,
            "first_gold_rank": 1.0,
            "retrieval_duration_ms": 2.0,
        },
        {
            "evidence_recall@3": 0.5,
            "evidence_recall@5": 1.0,
            "evidence_recall@10": 1.0,
            "context_precision@3": 0.33,
            "context_precision@5": 0.4,
            "context_precision@10": 0.3,
            "first_gold_rank": 4.0,
            "retrieval_duration_ms": 6.0,
        },
    ]
    aggregated = aggregate_metrics(per_query)
    assert aggregated["evidence_recall@3"] == pytest.approx(0.75)
    assert aggregated["evidence_recall@10"] == pytest.approx(1.0)
    assert aggregated["context_precision@5"] == pytest.approx(0.4)
    assert aggregated["first_gold_rank"] == pytest.approx(2.5)
    assert aggregated["retrieval_duration_ms"] == pytest.approx(4.0)


def test_boundary_queries_are_excluded_and_keep_their_diagnostics(tmp_path: Path) -> None:
    manifest = _run_evaluate(tmp_path)
    diagnostics = manifest["boundary_query_diagnostics"]
    assert len(diagnostics) == 4
    assert {item["query_id"] for item in diagnostics} == BOUNDARY_QUERY_IDS
    for diagnostic in diagnostics:
        assert diagnostic["reason_kind"] in {"insufficient", "conflicting", "stale"}
        assert diagnostic["explanation"].strip()
    boundary_results = [item for item in manifest["query_results"] if item["category"] == "boundary"]
    for item in boundary_results:
        assert item["gold_count"] == 0
        # Boundary queries still record diagnostic candidates but never enter aggregates.
        assert len(item["candidates"]) <= 10


def test_evaluate_candidates_are_deterministic_except_duration(tmp_path: Path) -> None:
    first = _run_evaluate(tmp_path, run_id="run-det-1")
    second = _run_evaluate(tmp_path, run_id="run-det-2")
    for name in METRIC_NAMES:
        if name == "retrieval_duration_ms":
            continue
        assert first["metrics"][name] == second["metrics"][name], f"{name} must be deterministic"
    for a, b in zip(first["query_results"], second["query_results"], strict=True):
        assert a["query_id"] == b["query_id"]
        assert [candidate["chunk_id"] for candidate in a["candidates"]] == [
            candidate["chunk_id"] for candidate in b["candidates"]
        ]


def test_evaluate_rejects_a_drifted_corpus(tmp_path: Path) -> None:
    drifted = tmp_path / "drifted-evaluation"
    shutil.copytree(EVALUATION_DIR, drifted)
    (drifted / "corpus" / "tampered.md").write_text("tampered statement\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus hash mismatch"):
        _run_evaluate(tmp_path, evaluation_dir=drifted)


def test_evaluate_exports_a_validated_public_evidence_bundle(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    manifest = _run_evaluate(tmp_path, bundle_dir=bundle_dir)
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "sections" / "retrieval.json").exists()
    assert (bundle_dir / "annotations.json").exists()
    assert (bundle_dir / "candidates.json").exists()
    assert (bundle_dir / "run-conditions.json").exists()
    assert (bundle_dir / "REPORT.md").exists()
    assert (bundle_dir / "REPORT.zh-CN.md").exists()

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(bundle_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"validator rejected the exported bundle:\n{result.stdout}{result.stderr}"

    bundle_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert bundle_manifest["bundle_id"] == "sparse-bm25-01"
    assert bundle_manifest["source_revision"] == REVISION
    run = bundle_manifest["provenance"]["runs"][0]
    assert run["run_id"] == manifest["run_id"]
    assert run["kind"] == "retrieval-evaluation"
    assert run["outcome"] == "passed"
    assert bundle_manifest["provenance"]["corpora"][0]["sha256"] == FROZEN_CORPUS_SHA
    assert bundle_manifest["provenance"]["query_sets"][0]["sha256"] == FROZEN_QUERY_SET_SHA

    section = json.loads((bundle_dir / "sections" / "retrieval.json").read_text(encoding="utf-8"))
    assert section["modes"] == ["sparse_bm25"]
    assert section["run_ids"] == [manifest["run_id"]]
    assert section["conditions"]["chunking"] == {"policy": "qa", "chunk_chars": 500, "overlap_chars": 50}
    assert section["conditions"]["output_depths"] == [3, 5, 10]
    assert section["conditions"]["corpus_version"] == "1.0.0"
    assert section["conditions"]["query_set_version"] == "1.0.0"
    metric_names = {(metric["name"], metric["mode"]) for metric in section["metrics"]}
    assert len(metric_names) == len(METRIC_NAMES)
    assert all((name, "sparse_bm25") in metric_names for name in METRIC_NAMES)


def test_evaluate_bundle_reports_are_bilingual_with_matching_literals(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    _run_evaluate(tmp_path, bundle_dir=bundle_dir)
    report_en = (bundle_dir / "REPORT.md").read_text(encoding="utf-8")
    report_zh = (bundle_dir / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    for literal in ("evidence_recall@3", "context_precision@5", "first_gold_rank", "retrieval_duration_ms", "sparse_bm25"):
        assert literal in report_en
        assert literal in report_zh
    # Fenced code block counts must match for the bilingual parity check.
    assert report_en.count("```") == report_zh.count("```")


def test_retrieval_evidence_evaluate_command_writes_artifacts_and_exits_zero(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli-runs"
    bundle_dir = tmp_path / "cli-bundle"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.retrieval_evidence",
            "evaluate",
            "--output-dir",
            str(run_dir),
            "--source-revision",
            REVISION,
            "--evaluation-dir",
            str(EVALUATION_DIR),
            "--bundle-dir",
            str(bundle_dir),
        ],
        cwd=REPOSITORY_ROOT / "backend",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"retrieval-evidence evaluate failed:\n{result.stdout}{result.stderr}"
    output = json.loads(result.stdout)
    assert output["outcome"] == "passed"
    assert (run_dir / output["run_id"] / "manifest.json").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "sections" / "retrieval.json").exists()


def test_retrieval_evidence_script_exposes_the_evaluate_profile() -> None:
    script = (REPOSITORY_ROOT / "retrieval-evidence").read_text(encoding="utf-8")
    assert "retrieval-evidence {smoke|fallback|evaluate|generation-smoke}" in script
    assert "evaluate" in script
