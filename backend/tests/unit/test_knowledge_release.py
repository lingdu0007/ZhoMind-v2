import json
from datetime import UTC, datetime

import pytest

from app.knowledge_release import (
    KnowledgeBaseReleaseGate,
    KnowledgeEditionManifestBuilder,
    PublicReleaseScanner,
    ReleaseGateError,
)


def _observations() -> list[dict]:
    items = []
    for entry_index in range(6):
        entry_id = f"pae-entry-{entry_index + 1:03d}"
        for kind in ("direct", "paraphrase", "combined"):
            items.append(
                {
                    "query_id": f"{entry_id}-{kind}",
                    "entry_id": entry_id,
                    "kind": kind,
                    "expected_outcome": "evidence_gated_answer",
                    "actual_outcome": "evidence_gated_answer",
                    "top_three_evidence_covered": True,
                    "citation_openable": True,
                    "citation_snapshot_exact": True,
                }
            )
        items.append(
            {
                "query_id": f"{entry_id}-boundary",
                "entry_id": entry_id,
                "kind": "boundary",
                "expected_outcome": "insufficient_evidence_reply",
                "actual_outcome": "insufficient_evidence_reply",
                "top_three_evidence_covered": True,
                "citation_openable": True,
                "citation_snapshot_exact": True,
            }
        )
    return items


def test_release_gate_enforces_project_thresholds_and_first_edition_label() -> None:
    declared = [f"pae-entry-{index + 1:03d}" for index in range(6)]
    result = KnowledgeBaseReleaseGate(declared).evaluate(_observations(), requested_label="First Edition")

    assert result["passed"] is True
    assert result["edition_label"] == "First Edition"
    assert result["published_entry_ids"] == declared
    assert result["failed_entry_ids"] == []
    assert result["metrics"] == {
        "direct_top_three_coverage": 1.0,
        "paraphrase_combined_top_three_coverage": 1.0,
        "unsupported_boundary_answers": 0,
        "exact_openable_citation_snapshots": 1.0,
    }


def test_release_gate_fails_closed_and_excludes_failed_entries_from_claims() -> None:
    observations = _observations()
    observations[0]["top_three_evidence_covered"] = False
    observations[5]["citation_snapshot_exact"] = False
    declared = [f"pae-entry-{index + 1:03d}" for index in range(6)]

    result = KnowledgeBaseReleaseGate(declared).evaluate(observations, requested_label="Pilot Edition")

    assert result["passed"] is False
    assert result["edition_label"] == "Unreleased Candidate"
    assert result["failed_entry_ids"] == ["pae-entry-001", "pae-entry-002"]
    assert result["published_entry_ids"] == declared[2:]
    assert result["metrics"]["direct_top_three_coverage"] < 1.0
    assert result["metrics"]["exact_openable_citation_snapshots"] < 1.0

    with pytest.raises(ReleaseGateError, match="First Edition"):
        KnowledgeBaseReleaseGate(declared).evaluate(observations, requested_label="First Edition", strict_label=True)


def test_generation_unavailable_fails_boundary_without_counting_an_unsupported_answer() -> None:
    observations = _observations()
    observations[3]["actual_outcome"] = "generation_unavailable"
    declared = [f"pae-entry-{index + 1:03d}" for index in range(6)]

    result = KnowledgeBaseReleaseGate(declared).evaluate(observations, requested_label="First Edition")

    assert result["passed"] is False
    assert result["failed_entry_ids"] == ["pae-entry-001"]
    assert result["metrics"]["unsupported_boundary_answers"] == 0


def test_manifest_is_non_sensitive_reproducible_and_supports_rollback(tmp_path) -> None:
    entries = [
        {"entry_id": "pae-a-001", "sha256": "a" * 64},
        {"entry_id": "pae-b-001", "sha256": "b" * 64},
    ]
    builder = KnowledgeEditionManifestBuilder()
    kwargs = {
        "edition_id": "pilot-2026-08-12",
        "edition_label": "Pilot Edition",
        "entries": entries,
        "corpus_sha256": "c" * 64,
        "acceptance_set_sha256": "d" * 64,
        "source_revision": "9c663fe",
        "model_identities": {"generation": "approved:model-a", "embedding": "approved:model-b"},
        "published_at": datetime(2026, 8, 12, 6, 0, tzinfo=UTC),
        "release_gate": {"passed": True, "failed_entry_ids": []},
    }
    first = builder.build(**kwargs)
    second = builder.build(**kwargs)
    assert first == second
    assert set(first) == {
        "schema_version",
        "edition_id",
        "edition_label",
        "entries",
        "corpus_sha256",
        "acceptance_set_sha256",
        "source_revision",
        "model_identities",
        "published_at",
        "release_gate",
        "manifest_sha256",
    }
    assert "path" not in json.dumps(first)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(first), encoding="utf-8")
    plan = builder.rollback_plan(manifest_path)
    assert plan == {
        "action": "republish_prior_manifest",
        "edition_id": "pilot-2026-08-12",
        "manifest_sha256": first["manifest_sha256"],
        "entry_ids": ["pae-a-001", "pae-b-001"],
    }


@pytest.mark.parametrize(
    ("name", "content", "finding"),
    [
        ("raw.json", '{"query":"complete private question"}', "forbidden-field:query"),
        ("excerpt.json", '{"source_excerpt":"raw source material"}', "forbidden-field:source_excerpt"),
        ("secret.txt", "sk-abcdefghijklmnopqrstuv", "credential-shape"),
        ("url.txt", "http://localhost:8000/private", "private-url"),
        ("entry.md", "# Decision Question\nprivate corpus body", "private-corpus-shape"),
    ],
)
def test_public_release_scanner_rejects_private_material(tmp_path, name, content, finding) -> None:
    (tmp_path / name).write_text(content, encoding="utf-8")
    assert finding in PublicReleaseScanner().scan(tmp_path)


def test_public_release_scanner_accepts_normalized_security_and_manifest_records(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "edition_id": "pilot-01",
                "entry_ids": ["pae-tools-001"],
                "source_revision": "9c663fe",
                "run_id": "agent-security-01",
                "case_id": "agent-unsafe-code-01",
                "pass_fail": "pass",
                "outcome": "evidence_gated_answer",
                "citation_count": 1,
                "failure_classification": "none",
            }
        ),
        encoding="utf-8",
    )
    assert PublicReleaseScanner().scan(tmp_path) == []
