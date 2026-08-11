from __future__ import annotations

import json
from pathlib import Path

from app.documents.chunker import CHUNK_STRATEGY_PRESETS
from app.evaluation_inputs import corpus_sha256, load_evaluation_inputs, query_set_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_DIR = REPOSITORY_ROOT / "evaluation"
CORPUS_DIR = EVALUATION_DIR / "corpus"
QUERY_SET_PATH = EVALUATION_DIR / "queries" / "query-set.json"
MANIFEST_PATH = EVALUATION_DIR / "corpus-manifest.json"

BOUNDARY_REASON_KINDS = {"insufficient", "conflicting", "stale"}
QUERY_CATEGORIES = {"exact_constraint", "semantic_paraphrase", "combined_condition", "boundary"}


def _corpus_text() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in CORPUS_DIR.glob("*.md")}


def test_evaluation_manifest_is_committed_and_matches_frozen_inputs() -> None:
    assert MANIFEST_PATH.is_file(), "evaluation/corpus-manifest.json must be committed with the frozen inputs"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["corpus"]["corpus_id"] == "project-derived-corpus"
    assert manifest["corpus"]["version"] == "1.0.0"
    assert manifest["corpus"]["sha256"] == corpus_sha256(CORPUS_DIR)
    assert manifest["query_set"]["query_set_id"] == "evaluation-query-set"
    assert manifest["query_set"]["version"] == "1.0.0"
    assert manifest["query_set"]["sha256"] == query_set_sha256(QUERY_SET_PATH)
    assert manifest["query_set"]["query_count"] == 16


def test_corpus_hash_is_deterministic_and_sensitive_to_names_and_contents() -> None:
    first = corpus_sha256(CORPUS_DIR)
    second = corpus_sha256(CORPUS_DIR)
    assert first == second
    assert len(first) == 64

    # A hash that ignores file names must differ from the name-aware corpus hash,
    # proving the corpus hash is anchored to both names and contents.
    contents_only = _contents_only_sha256(CORPUS_DIR)
    assert first != contents_only


def _contents_only_sha256(corpus_dir: Path) -> str:
    import hashlib

    joined = "\n".join(sorted(path.read_bytes().hex() for path in corpus_dir.glob("*.md"))).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def test_load_evaluation_inputs_returns_anchored_identity() -> None:
    inputs = load_evaluation_inputs(EVALUATION_DIR)
    assert inputs["corpus_id"] == "project-derived-corpus"
    assert inputs["corpus_version"] == "1.0.0"
    assert len(inputs["corpus_sha256"]) == 64
    assert inputs["query_set_id"] == "evaluation-query-set"
    assert inputs["query_set_version"] == "1.0.0"
    assert len(inputs["query_set_sha256"]) == 64
    assert inputs["query_count"] == 16
    assert inputs["qa_chunking"] == {"policy": "qa", "chunk_chars": 500, "overlap_chars": 50}


def test_load_evaluation_inputs_rejects_drifted_corpus(tmp_path: Path) -> None:
    import shutil

    drifted = tmp_path / "evaluation"
    shutil.copytree(EVALUATION_DIR, drifted)
    (drifted / "corpus" / "tampered.md").write_text("tampered statement\n", encoding="utf-8")
    try:
        load_evaluation_inputs(drifted)
    except ValueError as exc:
        assert "corpus hash mismatch" in str(exc)
    else:
        raise AssertionError("drifted corpus must be rejected")


def test_query_set_has_exactly_four_of_each_category() -> None:
    query_set = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    assert query_set["schema_version"] == 1
    queries = query_set["queries"]
    assert len(queries) == 16
    categories = [item["category"] for item in queries]
    for category in QUERY_CATEGORIES:
        assert categories.count(category) == 4, f"{category} must have exactly four queries"
    query_ids = [item["query_id"] for item in queries]
    assert len(query_ids) == len(set(query_ids)), "query ids must be unique"


def test_answerable_queries_record_claims_and_gold_evidence_from_corpus() -> None:
    query_set = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    corpus = _corpus_text()
    for item in query_set["queries"]:
        if item["category"] == "boundary":
            continue
        assert item.get("required_claims"), f"{item['query_id']} must record required claims"
        gold_evidence = item.get("gold_evidence")
        assert gold_evidence, f"{item['query_id']} must record gold evidence"
        for evidence in gold_evidence:
            document = evidence["document"]
            passage = evidence["passage"]
            assert document in corpus, f"{item['query_id']} references unknown corpus document {document}"
            assert passage in corpus[document], (
                f"{item['query_id']} gold passage must be verbatim text of {document}"
            )


def test_boundary_queries_record_diagnostics_without_empty_result_requirement() -> None:
    query_set = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    for item in query_set["queries"]:
        if item["category"] != "boundary":
            continue
        diagnostic = item.get("boundary_diagnostic")
        assert diagnostic, f"{item['query_id']} must record a boundary diagnostic"
        assert diagnostic["reason_kind"] in BOUNDARY_REASON_KINDS
        assert diagnostic["explanation"].strip()


def test_qa_chunking_matches_the_accepted_chunker_preset() -> None:
    inputs = load_evaluation_inputs(EVALUATION_DIR)
    preset = CHUNK_STRATEGY_PRESETS["qa"]
    assert inputs["qa_chunking"] == {
        "policy": "qa",
        "chunk_chars": preset["chunk_size"],
        "overlap_chars": preset["chunk_overlap"],
    }
    assert preset == {"chunk_size": 500, "chunk_overlap": 50}
