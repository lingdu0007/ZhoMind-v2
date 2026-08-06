"""Public Evidence Bundle validator CLI tests.

Behavior is asserted through the public command: exit code and artifact
diagnostics of ``python3 scripts/validate-evidence-bundle.py <bundle-dir>``.
The seam is the validator command itself, exactly as the deterministic PR Gate
invokes it. Complete example bundles are built in a temporary directory from
the same structures the checked-in ``public-evidence/example`` bundle uses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "scripts" / "validate-evidence-bundle.py"
EXAMPLE = REPO_ROOT / "public-evidence" / "example"

# Placeholder, clearly non-sensitive example revision (40 hex chars).
REVISION = "f2b6c4a8e0d1f3a5b7c9d1e3f5a7b9c1d3e5f7a9"
CORPUS_SHA = "a" * 64
QUERY_SET_SHA = "b" * 64

SECTIONS: dict[str, dict[str, object]] = {
    "retrieval": {
        "section": "retrieval",
        "schema_version": "1.0.0",
        "run_ids": ["example-run-20260805"],
        "corpus_id": "project-derived-corpus",
        "query_set_id": "evaluation-query-set",
        "modes": ["sparse_bm25", "dense", "hybrid_rrf", "migration"],
        "metrics": [
            {"name": "evidence_recall@3", "mode": "sparse_bm25", "value": 0.66, "unit": "ratio"},
            {"name": "evidence_recall@5", "mode": "sparse_bm25", "value": 1.0, "unit": "ratio"},
            {"name": "evidence_recall@10", "mode": "sparse_bm25", "value": 1.0, "unit": "ratio"},
            {"name": "context_precision@3", "mode": "sparse_bm25", "value": 0.5, "unit": "ratio"},
            {"name": "context_precision@5", "mode": "sparse_bm25", "value": 0.5, "unit": "ratio"},
            {"name": "context_precision@10", "mode": "sparse_bm25", "value": 0.5, "unit": "ratio"},
            {"name": "first_gold_rank", "mode": "sparse_bm25", "value": 1, "unit": "rank"},
            {"name": "retrieval_duration_ms", "mode": "sparse_bm25", "value": 12.0, "unit": "ms"},
            {"name": "evidence_recall@3", "mode": "dense", "value": 0.5, "unit": "ratio"},
            {"name": "evidence_recall@5", "mode": "dense", "value": 0.75, "unit": "ratio"},
            {"name": "evidence_recall@10", "mode": "dense", "value": 1.0, "unit": "ratio"},
            {"name": "context_precision@3", "mode": "dense", "value": 0.5, "unit": "ratio"},
            {"name": "context_precision@5", "mode": "dense", "value": 0.5, "unit": "ratio"},
            {"name": "context_precision@10", "mode": "dense", "value": 0.5, "unit": "ratio"},
            {"name": "first_gold_rank", "mode": "dense", "value": 1, "unit": "rank"},
            {"name": "retrieval_duration_ms", "mode": "dense", "value": 18.0, "unit": "ms"},
            {"name": "evidence_recall@3", "mode": "hybrid_rrf", "value": 0.66, "unit": "ratio"},
            {"name": "evidence_recall@5", "mode": "hybrid_rrf", "value": 1.0, "unit": "ratio"},
            {"name": "evidence_recall@10", "mode": "hybrid_rrf", "value": 1.0, "unit": "ratio"},
            {"name": "context_precision@3", "mode": "hybrid_rrf", "value": 0.5, "unit": "ratio"},
            {"name": "context_precision@5", "mode": "hybrid_rrf", "value": 0.5, "unit": "ratio"},
            {"name": "context_precision@10", "mode": "hybrid_rrf", "value": 0.5, "unit": "ratio"},
            {"name": "first_gold_rank", "mode": "hybrid_rrf", "value": 1, "unit": "rank"},
            {"name": "retrieval_duration_ms", "mode": "hybrid_rrf", "value": 24.0, "unit": "ms"},
        ],
        "boundary_query_diagnostics": [
            {
                "query_id": "boundary-conflicting-01",
                "reason_kind": "conflicting",
                "explanation": "The corpus contains two published versions with contradictory instructions for this boundary case.",
            }
        ],
        "conditions": {
            "chunking": {"policy": "qa_chunking", "chunk_chars": 500, "overlap_chars": 50},
            "output_depths": [3, 5, 10],
            "corpus_version": "1.0.0",
            "query_set_version": "1.0.0",
        },
    },
    "answer": {
        "section": "answer",
        "schema_version": "1.0.0",
        "run_ids": ["example-run-20260805"],
        "cases": [
            {"case_id": "answer-evidence-gated-01", "outcome": "evidence_gated_answer", "citations_count": 2, "contract": "passed"},
            {"case_id": "answer-insufficient-01", "outcome": "insufficient_evidence_reply", "citations_count": 0, "contract": "passed"},
        ],
    },
    "prompt-injection": {
        "section": "prompt-injection",
        "schema_version": "1.0.0",
        "run_ids": ["example-run-20260805"],
        "cases": [
            {
                "case_id": "injection-instruction-override-01",
                "kind": "instruction_override",
                "outcome": "insufficient_evidence_reply",
                "pass_fail": "pass",
                "citation_counts": {"source_count": 0, "evidence_count": 0},
                "failure_classification": "none",
            }
        ],
    },
    "performance": {
        "section": "performance",
        "schema_version": "1.0.0",
        "run_ids": ["example-run-20260805"],
        "load": {"concurrency": 1, "requests": 20},
        "metrics": {
            "ttft_ms": {"p50": 1200.0, "p95": 3100.0, "p99": 5200.0},
            "total_ms": {"p50": 3400.0, "p95": 8600.0, "p99": 12900.0},
            "error_rate": 0.0,
            "retrieval_ms": 180.0,
            "provider_ms": 2400.0,
            "persistence_ms": 60.0,
        },
        "target": {
            "p95_seconds_target": 12,
            "met": True,
            "note": "Example-only placeholder values; not a production claim.",
        },
    },
    "production-acceptance": {
        "section": "production-acceptance",
        "schema_version": "1.0.0",
        "run_ids": ["example-run-20260805"],
        "path": "/api/v1/chat",
        "results": [
            {
                "case_id": "accept-normal-01",
                "check": "normal_answer",
                "outcome": "evidence_gated_answer",
                "passed": True,
                "limitations": [],
            }
        ],
    },
}

REPORT_EN = """# Example Public Evidence Bundle

## Provenance

The example bundle records source_revision, one run identity, the project-derived-corpus, and the evaluation-query-set.

## Retrieval Metrics

Answerable-query aggregates report evidence_recall@3, context_precision@5, first_gold_rank, and retrieval_duration_ms
for sparse_bm25, dense, hybrid_rrf, and migration modes.

## Answer Outcomes

Deterministic answer cases record the evidence_gated_answer, insufficient_evidence_reply,
non_knowledge_base_reply, and generation_unavailable outcomes.

## Prompt Injection

Adversarial cases classify instruction_override, secret_extraction, forged_source, and unsupported_answer_pressure.

## Performance

TTFT and total P50, P95, and P99 percentiles are reported with error_rate and stage durations.

## Limits

The evidence does not generalize beyond the recorded run conditions; the twelve-second P95 remains a target.
"""

REPORT_ZH = """# 示例 Public Evidence Bundle

## Provenance

示例 bundle 记录 source_revision、一个 run identity、project-derived-corpus 和 evaluation-query-set。

## Retrieval Metrics

可回答查询聚合报告 sparse_bm25、dense、hybrid_rrf 和 migration 模式的 evidence_recall@3、context_precision@5、
first_gold_rank 与 retrieval_duration_ms。

## Answer Outcomes

确定性回答 cases 记录 evidence_gated_answer、insufficient_evidence_reply、non_knowledge_base_reply 和 generation_unavailable 四种结果。

## Prompt Injection

对抗 cases 对 instruction_override、secret_extraction、forged_source 和 unsupported_answer_pressure 进行分类。

## Performance

报告 TTFT 和 total P50、P95、P99 百分位，以及 error_rate 和各阶段时长。

## Limits

证据不推广到记录的 run conditions 之外；十二秒 P95 仍然是 target。
"""


def sha256_hex(content: bytes) -> str:
    return sha256(content).hexdigest()


def make_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "bundle_id": "example-bundle-001",
        "kind": "public-evidence-bundle",
        "canonical_language": "en",
        "mirror_language": "zh-CN",
        "source_revision": REVISION,
        "release_candidate": {
            "identity": "portfolio-release-candidate-01",
            "revision": REVISION,
            "status": "candidate",
            "created_at": "2026-08-05T00:00:00Z",
        },
        "provenance": {
            "runs": [
                {
                    "run_id": "example-run-20260805",
                    "kind": "retrieval-evaluation",
                    "source_revision": REVISION,
                    "started_at": "2026-08-05T00:00:00Z",
                    "finished_at": "2026-08-05T00:30:00Z",
                    "outcome": "passed",
                }
            ],
            "corpora": [
                {
                    "corpus_id": "project-derived-corpus",
                    "version": "1.0.0",
                    "sha256": CORPUS_SHA,
                    "description": "Example corpus placeholder.",
                }
            ],
            "query_sets": [
                {
                    "query_set_id": "evaluation-query-set",
                    "version": "1.0.0",
                    "sha256": QUERY_SET_SHA,
                    "query_count": 16,
                }
            ],
            "revisions": [REVISION],
        },
        "artifacts": [],
        "limits": [
            {
                "name": "no_generalization",
                "kind": "qualification",
                "statement": "Metrics describe only the recorded run conditions.",
            },
            {
                "name": "twelve_second_p95_target",
                "kind": "target",
                "statement": "The twelve-second P95 remains a target until a recorded run satisfies it.",
            },
        ],
    }
    manifest.update(overrides)
    return manifest


def write_bundle(
    root: Path,
    manifest: dict[str, object],
    *,
    sections: dict[str, dict[str, object]] | None = None,
    reports: tuple[str, str] | None = None,
    fix_artifacts: bool = True,
) -> Path:
    """Write a complete bundle under ``root`` and return the bundle directory."""
    bundle_dir = root / "example-bundle-001"
    sections_dir = bundle_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    for kind, content in (sections or SECTIONS).items():
        (sections_dir / f"{kind}.json").write_text(
            json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    en_text, zh_text = reports or (REPORT_EN, REPORT_ZH)
    (bundle_dir / "REPORT.md").write_text(en_text, encoding="utf-8")
    (bundle_dir / "REPORT.zh-CN.md").write_text(zh_text, encoding="utf-8")
    if fix_artifacts:
        manifest = fix_artifact_hashes(manifest, bundle_dir)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle_dir


def fix_artifact_hashes(manifest: dict[str, object], bundle_dir: Path) -> dict[str, object]:
    """Rewrite artifact sha256 entries to match the files actually written."""
    fixed = dict(manifest)
    artifacts = [dict(item) for item in manifest["artifacts"]]  # type: ignore[arg-type]
    seen_manifest = False
    for item in artifacts:
        path = str(item["path"])
        if path == "manifest.json":
            seen_manifest = True
            continue
        target = (bundle_dir / path).resolve()
        if target.exists() and target.is_relative_to(bundle_dir):
            item["sha256"] = sha256_hex(target.read_bytes())
    if not seen_manifest:
        artifacts.insert(0, {"path": "manifest.json", "kind": "manifest", "sha256": "0" * 64, "role": "manifest"})
    fixed["artifacts"] = artifacts
    return fixed


def complete_manifest() -> dict[str, object]:
    manifest = make_manifest()
    manifest["artifacts"] = [
        {"path": "manifest.json", "kind": "manifest", "sha256": "0" * 64, "role": "manifest"},
        {"path": "sections/retrieval.json", "kind": "retrieval", "sha256": "", "role": "section"},
        {"path": "sections/answer.json", "kind": "answer", "sha256": "", "role": "section"},
        {"path": "sections/prompt-injection.json", "kind": "prompt-injection", "sha256": "", "role": "section"},
        {"path": "sections/performance.json", "kind": "performance", "sha256": "", "role": "section"},
        {"path": "sections/production-acceptance.json", "kind": "production-acceptance", "sha256": "", "role": "section"},
        {"path": "REPORT.md", "kind": "report", "sha256": "", "role": "report-en"},
        {"path": "REPORT.zh-CN.md", "kind": "report", "sha256": "", "role": "report-zh"},
    ]
    return manifest


def run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def assert_failure(result: subprocess.CompletedProcess[str], check: str, detail: str = "") -> None:
    assert result.returncode == 1, f"expected exit 1, got {result.returncode}: {result.stdout}\n{result.stderr}"
    assert "ERROR " in result.stdout, f"expected ERROR diagnostic: {result.stdout}"
    assert check in result.stdout, f"expected check '{check}' in: {result.stdout}"
    if detail:
        assert detail in result.stdout, f"expected detail '{detail}' in: {result.stdout}"


class TestValidBundles:
    def test_checked_in_example_bundle_passes(self) -> None:
        result = run_validator(str(EXAMPLE))
        assert result.returncode == 0, f"example bundle must validate: {result.stdout}\n{result.stderr}"
        assert "OK" in result.stdout

    def test_built_complete_bundle_passes(self, tmp_path: Path) -> None:
        bundle_dir = write_bundle(tmp_path, complete_manifest())
        result = run_validator(str(bundle_dir))
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    def test_all_flag_discovers_and_validates_example(self) -> None:
        result = run_validator("--all")
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "public-evidence/example" in result.stdout or str(EXAMPLE) in result.stdout

    def test_list_flag_lists_example(self) -> None:
        result = run_validator("--list")
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "example" in result.stdout


class TestUsage:
    def test_no_arguments_is_usage_error(self) -> None:
        result = run_validator()
        assert result.returncode == 2, f"expected usage error, got {result.returncode}: {result.stdout}\n{result.stderr}"

    def test_nonexistent_bundle_directory_fails(self, tmp_path: Path) -> None:
        result = run_validator(str(tmp_path / "does-not-exist"))
        assert result.returncode == 1
        assert "missing-manifest" in result.stdout


class TestMissingAndMalformed:
    def test_empty_bundle_missing_manifest_fails(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "empty"
        bundle_dir.mkdir()
        result = run_validator(str(bundle_dir))
        assert_failure(result, "missing-manifest")

    def test_invalid_json_manifest_fails(self, tmp_path: Path) -> None:
        bundle_dir = tmp_path / "broken"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text("{not json", encoding="utf-8")
        result = run_validator(str(bundle_dir))
        assert_failure(result, "invalid-json")

    def test_missing_provenance_key_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        del manifest["provenance"]
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "provenance")

    def test_manifest_not_in_inventory_fails(self, tmp_path: Path) -> None:
        bundle_dir = write_bundle(tmp_path, complete_manifest())
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"] = [item for item in manifest["artifacts"] if item["path"] != "manifest.json"]
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = run_validator(str(bundle_dir))
        assert_failure(result, "manifest-inventory")

    def test_unknown_sensitive_field_rejected(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["credentials"] = {"api_key": "placeholder-value"}
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "credentials")

    def test_forbidden_section_field_rejected(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        sections["answer"]["cases"] = [
            {
                "case_id": "x",
                "outcome": "evidence_gated_answer",
                "citations_count": 1,
                "contract": "passed",
                "answer": "full model answer text",
            }
        ]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "answer")

    def test_wrong_schema_version_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["schema_version"] = "0.9.0"
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "schema_version")

    def test_missing_limits_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["limits"] = []
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "missing-limits")

    def test_malformed_metric_value_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        metrics = [dict(item) for item in sections["retrieval"]["metrics"]]  # type: ignore[arg-type]
        metrics[0]["value"] = "not-a-number"
        sections["retrieval"]["metrics"] = metrics  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "value")

    def test_unknown_metric_name_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        metrics = [dict(item) for item in sections["retrieval"]["metrics"]]  # type: ignore[arg-type]
        metrics[0]["name"] = "made_up_metric"
        sections["retrieval"]["metrics"] = metrics  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "made_up_metric")


class TestArtifactIntegrity:
    def test_artifact_sha256_mismatch_fails(self, tmp_path: Path) -> None:
        bundle_dir = write_bundle(tmp_path, complete_manifest())
        section = bundle_dir / "sections" / "answer.json"
        content = json.loads(section.read_text(encoding="utf-8"))
        content["cases"][0]["citations_count"] = 3
        section.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = run_validator(str(bundle_dir))
        assert_failure(result, "artifact-hash", "sections/answer.json")

    def test_missing_artifact_file_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["artifacts"].append(
            {"path": "sections/ghost.json", "kind": "answer", "sha256": "c" * 64, "role": "section"}  # type: ignore[arg-type]
        )
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "artifact-hash", "sections/ghost.json")

    def test_path_escaping_bundle_dir_rejected(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["artifacts"].append(
            {"path": "../outside.json", "kind": "report", "sha256": "c" * 64, "role": "report-en"}  # type: ignore[arg-type]
        )
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "artifact-hash", "..")

    def test_section_kind_mismatch_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        for item in manifest["artifacts"]:  # type: ignore[arg-type]
            if item["path"] == "sections/answer.json":
                item["kind"] = "performance"
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "schema-violation", "performance")

    def test_missing_report_pair_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["artifacts"] = [  # type: ignore[assignment]
            item for item in manifest["artifacts"] if item["path"] != "REPORT.zh-CN.md"  # type: ignore[arg-type]
        ]
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "missing-report")


class TestProvenanceReferences:
    def test_unknown_run_reference_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        sections["answer"]["run_ids"] = ["run-that-does-not-exist"]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "run-that-does-not-exist")

    def test_unknown_corpus_reference_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        sections["retrieval"]["corpus_id"] = "corpus-not-in-provenance"
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "corpus-not-in-provenance")

    def test_unknown_query_set_reference_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        sections["retrieval"]["query_set_id"] = "query-set-not-in-provenance"
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "query-set-not-in-provenance")

    def test_conditions_corpus_version_mismatch_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        conditions = dict(sections["retrieval"]["conditions"])  # type: ignore[arg-type]
        conditions["corpus_version"] = "9.9.9"
        sections["retrieval"]["conditions"] = conditions  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "conditions.corpus_version")

    def test_run_revision_not_in_provenance_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        runs = [dict(item) for item in manifest["provenance"]["runs"]]  # type: ignore[arg-type]
        runs[0]["source_revision"] = "0" * 40
        manifest["provenance"]["runs"] = runs  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference")

    def test_release_candidate_revision_mismatch_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        candidate = dict(manifest["release_candidate"])  # type: ignore[arg-type]
        candidate["revision"] = "1" * 40
        manifest["release_candidate"] = candidate  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "release_candidate.revision")

    def test_duplicate_corpus_id_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        corpora = [dict(item) for item in manifest["provenance"]["corpora"]]  # type: ignore[arg-type]
        corpora.append(dict(corpora[0]))
        manifest["provenance"]["corpora"] = corpora  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "duplicate corpus_id")

    def test_duplicate_run_id_fails(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        runs = [dict(item) for item in manifest["provenance"]["runs"]]  # type: ignore[arg-type]
        runs.append(dict(runs[0]))
        manifest["provenance"]["runs"] = runs  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "provenance-reference", "duplicate")


class TestMetricCompleteness:
    def test_incomplete_metric_family_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        metrics = [dict(item) for item in sections["retrieval"]["metrics"]]  # type: ignore[arg-type]
        metrics = [
            item
            for item in metrics
            if not (item["mode"] == "dense" and item["name"] == "retrieval_duration_ms")
        ]
        sections["retrieval"]["metrics"] = metrics  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "metric-completeness", "retrieval_duration_ms")

    def test_ratio_metric_out_of_bounds_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        metrics = [dict(item) for item in sections["retrieval"]["metrics"]]  # type: ignore[arg-type]
        metrics[0]["value"] = 1.5
        sections["retrieval"]["metrics"] = metrics  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "metric-completeness", "evidence_recall@3")

    def test_rank_metric_below_one_fails(self, tmp_path: Path) -> None:
        sections = {kind: dict(content) for kind, content in SECTIONS.items()}
        metrics = [dict(item) for item in sections["retrieval"]["metrics"]]  # type: ignore[arg-type]
        metrics[6]["value"] = 0
        sections["retrieval"]["metrics"] = metrics  # type: ignore[assignment]
        bundle_dir = write_bundle(tmp_path, complete_manifest(), sections=sections)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "metric-completeness", "first_gold_rank")


class TestSensitiveShapes:
    def test_provider_api_key_shape_rejected(self, tmp_path: Path) -> None:
        en_text = REPORT_EN + "\nA stray token sk-0123456789abcdef0123456789abcdef appeared in a scratch note.\n"
        bundle_dir = write_bundle(tmp_path, complete_manifest(), reports=(en_text, REPORT_ZH))
        result = run_validator(str(bundle_dir))
        assert_failure(result, "sensitive-shape")

    def test_sensitive_shape_in_non_section_artifact_rejected(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["artifacts"].append(  # type: ignore[arg-type]
            {"path": "sections/trace-data.json", "kind": "trace", "sha256": "c" * 64, "role": "data"}
        )
        bundle_dir = tmp_path / "example-bundle-001"
        (bundle_dir / "sections").mkdir(parents=True, exist_ok=True)
        (bundle_dir / "sections" / "trace-data.json").write_text(
            json.dumps({"trace": "contains 192.0.2.5:7000 host detail"}),
            encoding="utf-8",
        )
        write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "sensitive-shape", "trace-data.json")

    def test_private_key_shape_rejected(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["limits"].append(
            {
                "name": "scratch_note",
                "kind": "qualification",
                "statement": "-----BEGIN PRIVATE KEY----- placeholder",
            }  # type: ignore[arg-type]
        )
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "sensitive-shape")

    def test_environment_assignment_rejected(self, tmp_path: Path) -> None:
        en_text = REPORT_EN + "\nJWT_SECRET=change-me\n"
        bundle_dir = write_bundle(tmp_path, complete_manifest(), reports=(en_text, REPORT_ZH))
        result = run_validator(str(bundle_dir))
        assert_failure(result, "sensitive-shape")

    def test_non_secret_assignment_like_metric_note_allowed(self, tmp_path: Path) -> None:
        en_text = REPORT_EN + "\nA measured total P95=8600ms was recorded under load.\n"
        bundle_dir = write_bundle(tmp_path, complete_manifest(), reports=(en_text, REPORT_ZH))
        result = run_validator(str(bundle_dir))
        assert result.returncode == 0, f"non-secret assignment must not be rejected: {result.stdout}\n{result.stderr}"

    def test_ipv4_address_rejected(self, tmp_path: Path) -> None:
        manifest = complete_manifest()
        manifest["limits"].append(
            {"name": "host_note", "kind": "qualification", "statement": "Reachable at 192.0.2.10:8000."}  # type: ignore[arg-type]
        )
        bundle_dir = write_bundle(tmp_path, manifest)
        result = run_validator(str(bundle_dir))
        assert_failure(result, "sensitive-shape")


class TestBilingualParity:
    def test_heading_drift_fails(self, tmp_path: Path) -> None:
        zh_text = REPORT_ZH.replace("## Answer Outcomes\n", "")
        bundle_dir = write_bundle(tmp_path, complete_manifest(), reports=(REPORT_EN, zh_text))
        result = run_validator(str(bundle_dir))
        assert_failure(result, "bilingual-parity", "heading")

    def test_translated_literal_fails(self, tmp_path: Path) -> None:
        zh_text = REPORT_ZH.replace("source_revision", "来源修订")
        bundle_dir = write_bundle(tmp_path, complete_manifest(), reports=(REPORT_EN, zh_text))
        result = run_validator(str(bundle_dir))
        assert_failure(result, "bilingual-parity", "literal")

    def test_literal_substring_not_enough_fails(self, tmp_path: Path) -> None:
        zh_text = REPORT_ZH.replace("source_revision", "source_revisionX")
        bundle_dir = write_bundle(tmp_path, complete_manifest(), reports=(REPORT_EN, zh_text))
        result = run_validator(str(bundle_dir))
        assert_failure(result, "bilingual-parity", "source_revision")
