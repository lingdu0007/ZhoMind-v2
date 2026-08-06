"""Sparse BM25 evaluation for the ``retrieval-evidence evaluate`` command.

Runs genuine Sparse BM25 over the frozen Project-Derived Corpus and the
16-query Evaluation Query Set with the fixed QA Chunking policy (500 characters,
50 overlap). It computes the accepted answerable-query aggregates — Evidence
Recall@3/5/10, Context Precision@3/5/10, first Gold Evidence rank, and retrieval
duration — keeps Boundary Query Diagnostics outside those aggregates, writes the
internal run manifest, and exports a sanitized Public Evidence Bundle that the
deterministic validator (scripts/validate-evidence-bundle.py) accepts.

This module is evaluation-only. It never changes production Migration Retrieval
and never relabels the Lexical Heuristic as Sparse BM25 (ADR-0001).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from app.documents.chunker import CHUNK_STRATEGY_PRESETS, _iter_chunks
from app.evaluation.sparse_bm25 import Bm25Chunk, SparseBm25Index
from app.evaluation_inputs import load_evaluation_inputs
from app.rag.interfaces import EmbeddingProvider

METRIC_NAMES: tuple[str, ...] = (
    "evidence_recall@3",
    "evidence_recall@5",
    "evidence_recall@10",
    "context_precision@3",
    "context_precision@5",
    "context_precision@10",
    "first_gold_rank",
    "retrieval_duration_ms",
)
OUTPUT_DEPTHS: tuple[int, ...] = (3, 5, 10)
MAX_OUTPUT_DEPTH = 10
DENSE_CANDIDATE_DEPTH = 20
RRF_K = 60
GOLD_COVERAGE_RATIO = 0.5
BM25_K1 = 1.5
BM25_B = 0.75
BUNDLE_ID = "sparse-bm25-01"
BUNDLE_SCHEMA_VERSION = "1.0.0"
RELEASE_CANDIDATE_IDENTITY = "portfolio-release-candidate-01"
COMPARISON_BUNDLE_ID = "dense-hybrid-rrf-01"
COMPARISON_MODES: tuple[str, ...] = ("sparse_bm25", "dense", "hybrid_rrf")

_METRIC_UNITS: dict[str, str] = {
    "evidence_recall@3": "ratio",
    "evidence_recall@5": "ratio",
    "evidence_recall@10": "ratio",
    "context_precision@3": "ratio",
    "context_precision@5": "ratio",
    "context_precision@10": "ratio",
    "first_gold_rank": "rank",
    "retrieval_duration_ms": "ms",
}


def _qa_chunks_with_spans(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Apply the QA Chunking sliding window; return (piece, start, end) triples.

    Chunk text comes from the production chunker's sliding window
    (``backend/app/documents/chunker.py``) so this evaluation reuses the exact
    frozen chunking policy; spans are computed with the same step.
    """
    step = max(1, chunk_size - chunk_overlap)
    pieces: list[tuple[str, int, int]] = []
    start = 0
    for piece in _iter_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        pieces.append((piece, start, start + len(piece)))
        start += step
    return pieces


def chunk_covers_passage(
    *,
    chunk_start: int,
    chunk_end: int,
    passage_start: int,
    passage_end: int,
) -> bool:
    """True when the chunk's character span covers at least half of the passage span.

    Gold Evidence passages are verbatim corpus text; most fit inside one chunk,
    but longer passages can straddle QA Chunking boundaries. The 50% coverage
    rule treats a straddling passage as covered by the chunk that contains most
    of it, without requiring a full containment that chunking can mechanically
    violate.
    """
    overlap = max(0, min(chunk_end, passage_end) - max(chunk_start, passage_start))
    passage_length = passage_end - passage_start
    if passage_length <= 0:
        return False
    return overlap >= GOLD_COVERAGE_RATIO * passage_length


def aggregate_metrics(per_query_metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Macro mean of every metric name over the given (answerable) queries."""
    if not per_query_metrics:
        raise ValueError("cannot aggregate metrics over an empty query set")
    names = list(per_query_metrics[0])
    return {
        name: sum(float(item[name]) for item in per_query_metrics) / len(per_query_metrics)
        for name in names
    }


class DenseVectorIndex:
    """Evaluation-only full-corpus dense search over precomputed chunk vectors."""

    def __init__(self, chunks: Sequence[Bm25Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("dense evaluation vector count does not match the frozen corpus chunks")
        self._chunks = list(chunks)
        self._vectors = [self._normalized(vector) for vector in vectors]
        self._dimension = len(self._vectors[0]) if self._vectors else 0
        if any(len(vector) != self._dimension for vector in self._vectors):
            raise ValueError("dense evaluation vectors do not share one dimension")

    @staticmethod
    def _normalized(vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("dense evaluation received an invalid embedding vector")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ValueError("dense evaluation received a zero-norm embedding vector")
        return [value / norm for value in values]

    def search(self, vector: Sequence[float], *, top_k: int) -> list[dict[str, Any]]:
        query = self._normalized(vector)
        if len(query) != self._dimension:
            raise ValueError("dense evaluation query vector dimension does not match the corpus vectors")
        scored = [
            (sum(left * right for left, right in zip(query, stored, strict=True)), index)
            for index, stored in enumerate(self._vectors)
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                self._chunks[item[1]].chunk_index,
                self._chunks[item[1]].chunk_id,
            )
        )
        return [
            {
                "chunk_id": self._chunks[index].chunk_id,
                "document": self._chunks[index].document,
                "chunk_index": self._chunks[index].chunk_index,
                "score": round(score, 8),
                "content_preview": self._chunks[index].content[:160],
            }
            for score, index in scored[:top_k]
        ]


def fuse_rrf_candidates(
    sparse_candidates: Sequence[Mapping[str, Any]],
    dense_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse fixed-depth Sparse and Dense ranks with RRF, never raw scores.

    Candidates are deduplicated by ``chunk_id`` before the final order.  The
    only numeric contribution is ``1 / (RRF_K + rank)`` for each source list;
    this intentionally contains no score averaging or reranking behavior.
    """
    fused: dict[str, dict[str, Any]] = {}
    for source_candidates in (sparse_candidates, dense_candidates):
        for rank, candidate in enumerate(source_candidates, start=1):
            chunk_id = str(candidate["chunk_id"])
            current = fused.get(chunk_id)
            if current is None:
                current = dict(candidate)
                current["score"] = 0.0
                fused[chunk_id] = current
            current["score"] = float(current["score"]) + 1.0 / (RRF_K + rank)
    return sorted(
        (
            {**candidate, "score": round(float(candidate["score"]), 8)}
            for candidate in fused.values()
        ),
        key=lambda candidate: (-float(candidate["score"]), str(candidate["chunk_id"])),
    )


def _chunk_corpus(
    corpus_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Bm25Chunk], dict[str, str], dict[str, list[tuple[int, int]]]]:
    """Chunk every corpus document; return chunks, full document text, and per-document spans."""
    chunks: list[Bm25Chunk] = []
    full_texts: dict[str, str] = {}
    spans_by_document: dict[str, list[tuple[int, int]]] = {}
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        full_texts[path.name] = text
        spans: list[tuple[int, int]] = []
        for index, (piece, start, end) in enumerate(
            _qa_chunks_with_spans(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                Bm25Chunk(
                    chunk_id=f"{path.name}#{index}",
                    document=path.name,
                    chunk_index=index,
                    content=piece,
                )
            )
            spans.append((start, end))
        spans_by_document[path.name] = spans
    return chunks, full_texts, spans_by_document


def _evaluate_answerable_query(
    *,
    query_item: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    full_texts: Mapping[str, str],
    spans_by_document: Mapping[str, list[tuple[int, int]]],
    duration_ms: float,
) -> dict[str, Any]:
    gold_evidence = query_item["gold_evidence"]
    gold_count = len(gold_evidence)
    first_covering_rank: list[int] = []
    covered_ranks: set[int] = set()
    for evidence in gold_evidence:
        document = evidence["document"]
        passage = evidence["passage"]
        text = full_texts.get(document, "")
        passage_start = text.find(passage)
        first_rank = None
        if passage_start >= 0:
            passage_end = passage_start + len(passage)
            spans = spans_by_document.get(document, [])
            for rank, candidate in enumerate(candidates, start=1):
                if candidate["document"] != document:
                    continue
                chunk_index = int(candidate["chunk_index"])
                if chunk_index >= len(spans):
                    continue
                chunk_start, chunk_end = spans[chunk_index]
                if chunk_covers_passage(
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    passage_start=passage_start,
                    passage_end=passage_end,
                ):
                    first_rank = rank
                    covered_ranks.add(rank)
                    break
        first_covering_rank.append(first_rank if first_rank is not None else MAX_OUTPUT_DEPTH + 1)

    matched_at = {
        depth: sum(1 for rank in first_covering_rank if rank <= depth) for depth in OUTPUT_DEPTHS
    }
    evidence_recall = {
        str(depth): (matched_at[depth] / gold_count if gold_count else 0.0) for depth in OUTPUT_DEPTHS
    }
    context_precision = {
        str(depth): sum(1 for rank in covered_ranks if rank <= depth) / depth for depth in OUTPUT_DEPTHS
    }
    first_gold_rank = min(first_covering_rank)
    return {
        "gold_count": gold_count,
        "gold_matched_at": {str(depth): matched_at[depth] for depth in OUTPUT_DEPTHS},
        "evidence_recall": evidence_recall,
        "context_precision": context_precision,
        "first_gold_rank": first_gold_rank,
        "duration_ms": duration_ms,
        "metrics": {
            **{f"evidence_recall@{depth}": evidence_recall[str(depth)] for depth in OUTPUT_DEPTHS},
            **{f"context_precision@{depth}": context_precision[str(depth)] for depth in OUTPUT_DEPTHS},
            "first_gold_rank": float(first_gold_rank),
            "retrieval_duration_ms": duration_ms,
        },
    }


def _candidate_records(candidates: list[dict[str, Any]], *, with_preview: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        record: dict[str, Any] = {
            "rank": rank,
            "chunk_id": candidate["chunk_id"],
            "document": candidate["document"],
            "chunk_index": candidate["chunk_index"],
            "score": candidate["score"],
        }
        if with_preview:
            record["content_preview"] = candidate["content_preview"]
        records.append(record)
    return records


def run_sparse_bm25_evaluation(
    *,
    evaluation_dir: Path,
    output_dir: Path,
    source_revision: str,
    run_id: str | None = None,
    bundle_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the Sparse BM25 evaluation and return the internal run manifest.

    Raises ValueError when the frozen evaluation inputs drift from the committed
    manifest (via ``load_evaluation_inputs``).
    """
    current_time = now or (lambda: datetime.now(UTC))
    started_at = current_time()
    inputs = load_evaluation_inputs(evaluation_dir)
    chunk_size = int(inputs["qa_chunking"]["chunk_chars"])
    chunk_overlap = int(inputs["qa_chunking"]["overlap_chars"])
    preset = CHUNK_STRATEGY_PRESETS["qa"]
    if chunk_size != preset["chunk_size"] or chunk_overlap != preset["chunk_overlap"]:
        raise ValueError("QA Chunking drift: frozen evaluation inputs no longer match the accepted chunker preset")

    chunks, full_texts, spans_by_document = _chunk_corpus(
        evaluation_dir / "corpus",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    index = SparseBm25Index(chunks, k1=BM25_K1, b=BM25_B)
    # Warm up jieba so the first query's search time is not dominated by
    # dictionary loading; duration records per-query search only.
    index.search("预热", top_k=1)

    query_set = json.loads((evaluation_dir / "queries" / "query-set.json").read_text(encoding="utf-8"))
    query_results: list[dict[str, Any]] = []
    boundary_query_diagnostics: list[dict[str, Any]] = []
    answerable_metrics: list[dict[str, float]] = []

    for query_item in query_set["queries"]:
        started = perf_counter()
        candidates = index.search(query_item["query"], top_k=MAX_OUTPUT_DEPTH)
        duration_ms = (perf_counter() - started) * 1000.0
        if query_item["category"] == "boundary":
            query_results.append(
                {
                    "query_id": query_item["query_id"],
                    "category": query_item["category"],
                    "candidates": _candidate_records(candidates, with_preview=True),
                    "gold_count": 0,
                }
            )
            boundary_query_diagnostics.append(
                {
                    "query_id": query_item["query_id"],
                    "reason_kind": query_item["boundary_diagnostic"]["reason_kind"],
                    "explanation": query_item["boundary_diagnostic"]["explanation"],
                }
            )
            continue
        evaluated = _evaluate_answerable_query(
            query_item=query_item,
            candidates=candidates,
            full_texts=full_texts,
            spans_by_document=spans_by_document,
            duration_ms=duration_ms,
        )
        answerable_metrics.append(evaluated["metrics"])
        query_results.append(
            {
                "query_id": query_item["query_id"],
                "category": query_item["category"],
                "candidates": _candidate_records(candidates, with_preview=True),
                "gold_count": evaluated["gold_count"],
                "gold_matched_at": evaluated["gold_matched_at"],
                "evidence_recall": evaluated["evidence_recall"],
                "context_precision": evaluated["context_precision"],
                "first_gold_rank": evaluated["first_gold_rank"],
                "duration_ms": evaluated["duration_ms"],
            }
        )

    metrics = aggregate_metrics(answerable_metrics)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id or f"evaluate-{started_at.strftime('%Y%m%dT%H%M%SZ')}",
        "command": "retrieval-evidence evaluate",
        "mode": "sparse_bm25",
        "source_revision": source_revision,
        "started_at": started_at.isoformat(),
        "finished_at": current_time().isoformat(),
        "outcome": "passed",
        "evaluation_inputs": dict(inputs),
        "algorithm": {
            "retrieval_mode": "sparse_bm25",
            "tokenizer": "literal_preserving_jieba",
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "gold_coverage_ratio": GOLD_COVERAGE_RATIO,
            "output_depths": list(OUTPUT_DEPTHS),
            "first_gold_rank_unmatched_value": MAX_OUTPUT_DEPTH + 1,
        },
        "answerable_query_count": len(answerable_metrics),
        "boundary_query_count": len(boundary_query_diagnostics),
        "metrics": metrics,
        "query_results": query_results,
        "boundary_query_diagnostics": boundary_query_diagnostics,
    }

    run_directory = output_dir / manifest["run_id"]
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if bundle_dir is not None:
        _export_bundle(
            bundle_dir=bundle_dir,
            manifest=manifest,
            inputs=inputs,
            metrics=metrics,
            query_results=query_results,
            boundary_query_diagnostics=boundary_query_diagnostics,
            now=current_time,
        )
    return manifest


def _export_bundle(
    *,
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    inputs: Mapping[str, Any],
    metrics: Mapping[str, float],
    query_results: Sequence[Mapping[str, Any]],
    boundary_query_diagnostics: Sequence[Mapping[str, Any]],
    now: Callable[[], datetime],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = bundle_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    run: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "kind": "retrieval-evaluation",
        "source_revision": manifest["source_revision"],
        "started_at": manifest["started_at"],
        "finished_at": manifest["finished_at"],
        "outcome": manifest["outcome"],
    }
    bundle_manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": BUNDLE_ID,
        "kind": "public-evidence-bundle",
        "canonical_language": "en",
        "mirror_language": "zh-CN",
        "source_revision": manifest["source_revision"],
        "release_candidate": {
            "identity": RELEASE_CANDIDATE_IDENTITY,
            "revision": manifest["source_revision"],
            "status": "candidate",
            "created_at": now().isoformat(),
        },
        "provenance": {
            "runs": [run],
            "corpora": [
                {
                    "corpus_id": inputs["corpus_id"],
                    "version": inputs["corpus_version"],
                    "sha256": inputs["corpus_sha256"],
                }
            ],
            "query_sets": [
                {
                    "query_set_id": inputs["query_set_id"],
                    "version": inputs["query_set_version"],
                    "sha256": inputs["query_set_sha256"],
                    "query_count": inputs["query_count"],
                }
            ],
            "revisions": [manifest["source_revision"]],
        },
        "artifacts": [],
        "limits": [
            {
                "name": "no_generalization",
                "kind": "qualification",
                "statement": (
                    "Metrics describe only the recorded Sparse BM25 run over the frozen "
                    "Project-Derived Corpus and Evaluation Query Set; they do not generalize "
                    "to other corpora, models, chunking policies, or languages."
                ),
            },
            {
                "name": "sparse_bm25_only",
                "kind": "exclusion",
                "statement": (
                    "This bundle contains Sparse BM25 evidence only; Dense, Hybrid RRF, "
                    "answer-quality, and universal-improvement claims are not made."
                ),
            },
            {
                "name": "unmatched_rank_convention",
                "kind": "qualification",
                "statement": (
                    "first_gold_rank counts an answerable query with no gold-covered candidate "
                    "inside the top 10 as rank 11 (max output depth + 1)."
                ),
            },
            {
                "name": "gold_coverage_ratio",
                "kind": "qualification",
                "statement": (
                    "A candidate covers a Gold Evidence passage when its character span "
                    "overlaps at least 50% of the passage span, accommodating passages that "
                    "straddle QA Chunking boundaries."
                ),
            },
            {
                "name": "duration_scope",
                "kind": "qualification",
                "statement": (
                    "retrieval_duration_ms is the mean per-query in-process Sparse BM25 "
                    "search time over the 12 answerable queries; it excludes corpus chunking "
                    "and index construction."
                ),
            },
        ],
    }

    retrieval_section: dict[str, Any] = {
        "section": "retrieval",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_ids": [manifest["run_id"]],
        "corpus_id": inputs["corpus_id"],
        "query_set_id": inputs["query_set_id"],
        "modes": ["sparse_bm25"],
        "metrics": [
            {
                "name": name,
                "mode": "sparse_bm25",
                "value": round(float(metrics[name]), 6),
                "unit": _METRIC_UNITS[name],
            }
            for name in METRIC_NAMES
        ],
        "boundary_query_diagnostics": [dict(item) for item in boundary_query_diagnostics],
        "conditions": {
            "chunking": {
                "policy": inputs["qa_chunking"]["policy"],
                "chunk_chars": inputs["qa_chunking"]["chunk_chars"],
                "overlap_chars": inputs["qa_chunking"]["overlap_chars"],
            },
            "output_depths": list(OUTPUT_DEPTHS),
            "corpus_version": inputs["corpus_version"],
            "query_set_version": inputs["query_set_version"],
        },
    }

    annotations: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "query_set_id": inputs["query_set_id"],
        "query_set_version": inputs["query_set_version"],
        "queries": [],
    }
    candidates_export: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "mode": "sparse_bm25",
        "run_id": manifest["run_id"],
        "output_depths": list(OUTPUT_DEPTHS),
        "queries": [],
    }
    for item in query_results:
        annotations["queries"].append(
            {
                "query_id": item["query_id"],
                "category": item["category"],
                "gold_count": item["gold_count"],
                "gold_matched_at": item.get("gold_matched_at"),
                "first_gold_rank": item.get("first_gold_rank"),
            }
        )
        candidates_export["queries"].append(
            {
                "query_id": item["query_id"],
                "category": item["category"],
                "candidates": [
                    {
                        "rank": candidate["rank"],
                        "chunk_id": candidate["chunk_id"],
                        "document": candidate["document"],
                        "chunk_index": candidate["chunk_index"],
                        "score": candidate["score"],
                    }
                    for candidate in item["candidates"]
                ],
            }
        )

    run_conditions: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "mode": "sparse_bm25",
        "tokenizer": {
            "name": "literal_preserving_jieba",
            "segments_chinese_with": "jieba",
            "preserves_literals": [
                "configuration_keys",
                "paths",
                "status_codes",
                "versions",
                "model_names",
                "underscore_identifiers",
            ],
        },
        "bm25": {"k1": BM25_K1, "b": BM25_B, "idf": "lucene"},
        "chunking": dict(inputs["qa_chunking"]),
        "output_depths": list(OUTPUT_DEPTHS),
        "gold_evidence_matching": {
            "rule": "character_span_overlap",
            "coverage_ratio": GOLD_COVERAGE_RATIO,
        },
        "first_gold_rank_unmatched_value": MAX_OUTPUT_DEPTH + 1,
        "evaluation_inputs": dict(inputs),
        "metric_definitions": {
            name: {
                "unit": _METRIC_UNITS[name],
                "aggregate": "macro mean over the 12 answerable queries",
            }
            for name in METRIC_NAMES
        },
    }

    report_en = _report_en(manifest=manifest, inputs=inputs, metrics=metrics)
    report_zh = _report_zh(manifest=manifest, inputs=inputs, metrics=metrics)

    bundle_files: dict[str, tuple[str, str, str]] = {
        "manifest.json": ("manifest", "manifest", json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n"),
        "sections/retrieval.json": ("retrieval", "section", json.dumps(retrieval_section, ensure_ascii=False, indent=2) + "\n"),
        "annotations.json": ("annotation", "data", json.dumps(annotations, ensure_ascii=False, indent=2) + "\n"),
        "candidates.json": ("trace", "data", json.dumps(candidates_export, ensure_ascii=False, indent=2) + "\n"),
        "run-conditions.json": ("retrieval", "data", json.dumps(run_conditions, ensure_ascii=False, indent=2) + "\n"),
        "REPORT.md": ("report", "report-en", report_en),
        "REPORT.zh-CN.md": ("report", "report-zh", report_zh),
    }
    artifacts: list[dict[str, Any]] = []
    for relative_path, (kind, role, content) in bundle_files.items():
        target = bundle_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        artifacts.append(
            {
                "path": relative_path,
                "kind": kind,
                "role": role,
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    bundle_manifest["artifacts"] = artifacts
    (bundle_dir / "manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric_rows(metrics: Mapping[str, float]) -> str:
    return "\n".join(
        f"| {name} | {float(metrics[name]):.4f} |" for name in METRIC_NAMES
    )


def _report_en(*, manifest: Mapping[str, Any], inputs: Mapping[str, Any], metrics: Mapping[str, float]) -> str:
    return f"""# Sparse BM25 Evidence Package

## Summary

This bundle records the first quality-oriented retrieval evaluation of the ZhoMind-v2 Portfolio
Release: genuine Sparse BM25 (Literal-Preserving Tokenization with jieba) run through the
Evaluation Retriever's `retrieval-evidence evaluate` command over the frozen Project-Derived
Corpus and 16-query Evaluation Query Set. It is a tracer bullet for the Retrieval Evidence
Baseline; it makes no Dense, Hybrid RRF, answer-quality, or universal-improvement claim.

## Method

### Corpus

- corpus_id `{inputs["corpus_id"]}` version {inputs["corpus_version"]}, sha256 `{inputs["corpus_sha256"]}`
- Six Chinese-First Markdown documents; English identifiers, paths, status codes, versions, and model names kept as literals

### Query set

- query_set_id `{inputs["query_set_id"]}` version {inputs["query_set_version"]}, sha256 `{inputs["query_set_sha256"]}`
- 16 queries: four `exact_constraint`, four `semantic_paraphrase`, four `combined_condition`,
  and four `boundary` queries; the 12 answerable queries record `required_claims` and `gold_evidence`

### Chunking and tokenization

- QA Chunking fixed at 500 characters with 50 characters of overlap (the `qa` preset of `CHUNK_STRATEGY_PRESETS`)
- Sparse BM25 uses Literal-Preserving Tokenization: Chinese prose is segmented with jieba;
  configuration keys, paths, status codes, versions, model names, and underscore-bearing
  identifiers remain whole tokens. BM25 uses k1 = 1.5 and b = 0.75.

### Metrics

Answerable-query aggregates are macro means over the 12 answerable queries:
`evidence_recall@3`, `evidence_recall@5`, `evidence_recall@10`, `context_precision@3`,
`context_precision@5`, `context_precision@10`, `first_gold_rank`, and
`retrieval_duration_ms`. A candidate covers a Gold Evidence passage when its character span
overlaps at least 50% of the passage span. An answerable query with no gold-covered candidate
inside the top 10 counts as rank 11 for `first_gold_rank`. Boundary Query Diagnostics stay
outside these aggregates.

## Results

### Evidence Recall and Context Precision

| metric | value |
| --- | --- |
{_metric_rows(metrics)}

### First Gold Evidence Rank

- first_gold_rank: {float(metrics['first_gold_rank']):.4f} (1-based; 11 = no gold-covered candidate inside the top 10)

### Retrieval Duration

- retrieval_duration_ms: {float(metrics['retrieval_duration_ms']):.2f} (mean per-query
  in-process Sparse BM25 search time)

## Boundary Query Diagnostics

The four Boundary Queries (`boundary-lexical-is-bm25`, `boundary-chunking-300`,
`boundary-hybrid-winner`, `boundary-production-ports`) are excluded from answerable-query
aggregates and retain their corpus-derived insufficiency, conflict, or staleness explanations
in `sections/retrieval.json`.

## Run Conditions

- source revision: `{manifest["source_revision"]}`
- run id: `{manifest["run_id"]}`
- mode: sparse_bm25
- output depths: 3, 5, 10
- tokenizer: literal_preserving_jieba

## Limits

- The metrics describe only the recorded run conditions and do not generalize to other corpora, models, chunking policies, or languages.
- This bundle contains Sparse BM25 evidence only; Dense, Hybrid RRF, answer-quality, and universal-improvement claims are not made.
- `retrieval_duration_ms` excludes corpus chunking and index construction.
"""


def _report_zh(*, manifest: Mapping[str, Any], inputs: Mapping[str, Any], metrics: Mapping[str, float]) -> str:
    return f"""# Sparse BM25 Evidence Package（稀疏 BM25 证据包）

## Summary（摘要）

本证据包记录 ZhoMind-v2 Portfolio Release 的首次质量导向检索评估：真正的 Sparse BM25（使用
jieba 的 Literal-Preserving Tokenization）通过 Evaluation Retriever 的
`retrieval-evidence evaluate` 命令，在冻结的 Project-Derived Corpus 与 16 条 Evaluation
Query Set 上运行。它是 Retrieval Evidence Baseline 的 tracer bullet；本包不做任何 Dense、
Hybrid RRF、回答质量或普遍提升的声明。

## Method（方法）

### Corpus（语料）

- corpus_id `{inputs["corpus_id"]}`，版本 {inputs["corpus_version"]}，sha256 `{inputs["corpus_sha256"]}`
- 六份 Chinese-First Markdown 文档；英文标识符、路径、状态码、版本号与模型名保留为字面量

### Query set（查询集）

- query_set_id `{inputs["query_set_id"]}`，版本 {inputs["query_set_version"]}，sha256 `{inputs["query_set_sha256"]}`
- 16 条查询：四条 `exact_constraint`、四条 `semantic_paraphrase`、四条 `combined_condition`
  与四条 `boundary` 查询；12 条 answerable 查询记录 `required_claims` 与 `gold_evidence`

### Chunking and tokenization（分块与分词）

- QA Chunking 固定为 500 字符、50 字符重叠（`CHUNK_STRATEGY_PRESETS` 的 `qa` 预置）
- Sparse BM25 使用 Literal-Preserving Tokenization：中文散文用 jieba 分词；配置键、路径、
  状态码、版本号、模型名与带下划线的标识符保留为完整 token。BM25 使用 k1 = 1.5 与 b = 0.75。

### Metrics（指标）

Answerable-query 聚合是 12 条 answerable 查询的宏平均：`evidence_recall@3`、
`evidence_recall@5`、`evidence_recall@10`、`context_precision@3`、`context_precision@5`、
`context_precision@10`、`first_gold_rank` 与 `retrieval_duration_ms`。当候选 chunk 的字符区间
与 Gold Evidence passage 区间的重叠达到该 passage 长度的至少 50% 时，判定该 passage 被覆盖。
若某 answerable 查询在 top 10 内没有覆盖任何 Gold Evidence 的候选，`first_gold_rank` 记为 11。
Boundary Query Diagnostics 不计入这些聚合。

## Results（结果）

### Evidence Recall and Context Precision（证据召回与上下文精确率）

| metric | value |
| --- | --- |
{_metric_rows(metrics)}

### First Gold Evidence Rank（首个 Gold 证据排名）

- first_gold_rank: {float(metrics['first_gold_rank']):.4f}（1 基；11 表示 top 10 内没有
  覆盖 Gold Evidence 的候选）

### Retrieval Duration（检索耗时）

- retrieval_duration_ms: {float(metrics['retrieval_duration_ms']):.2f}（12 条 answerable
  查询的每条查询进程内 Sparse BM25 检索平均耗时）

## Boundary Query Diagnostics（边界查询诊断）

四条 Boundary Query（`boundary-lexical-is-bm25`、`boundary-chunking-300`、
`boundary-hybrid-winner`、`boundary-production-ports`）不计入 answerable-query 聚合，并在
`sections/retrieval.json` 中保留语料推导出的 insufficient、conflicting 或 stale 解释。

## Run Conditions（运行条件）

- source revision: `{manifest["source_revision"]}`
- run id: `{manifest["run_id"]}`
- mode: sparse_bm25
- output depths: 3、5、10
- tokenizer: literal_preserving_jieba

## Limits（适用限制）

- 指标仅描述已记录的运行条件，不推广到其他语料、模型、分块策略或语言。
- 本证据包只包含 Sparse BM25 证据；不做 Dense、Hybrid RRF、回答质量或普遍提升的声明。
- `retrieval_duration_ms` 不包括语料分块与索引构建耗时。
"""


def _selected_modes(modes: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(modes)
    if not selected:
        raise ValueError("retrieval evaluation requires at least one mode")
    if len(set(selected)) != len(selected):
        raise ValueError("retrieval evaluation modes must not repeat")
    unsupported = set(selected).difference(COMPARISON_MODES)
    if unsupported:
        raise ValueError(f"unsupported evaluation modes: {', '.join(sorted(unsupported))}")
    return selected


async def _evaluate_mode(
    *,
    mode: str,
    sparse_index: SparseBm25Index,
    dense_index: DenseVectorIndex | None,
    embedding_provider: EmbeddingProvider | None,
    query_set: Mapping[str, Any],
    full_texts: Mapping[str, str],
    spans_by_document: Mapping[str, list[tuple[int, int]]],
) -> dict[str, Any]:
    query_results: list[dict[str, Any]] = []
    boundary_query_diagnostics: list[dict[str, Any]] = []
    answerable_metrics: list[dict[str, float]] = []

    for query_item in query_set["queries"]:
        query = str(query_item["query"])
        started = perf_counter()
        if mode == "sparse_bm25":
            candidates = sparse_index.search(query, top_k=DENSE_CANDIDATE_DEPTH)
        else:
            if dense_index is None or embedding_provider is None:
                raise ValueError(f"{mode} evaluation requires an active embedding provider")
            vectors = await embedding_provider.embed([query])
            if len(vectors) != 1:
                raise ValueError("embedding provider returned an unexpected query vector count")
            dense_candidates = dense_index.search(vectors[0], top_k=DENSE_CANDIDATE_DEPTH)
            if mode == "dense":
                candidates = dense_candidates
            else:
                sparse_candidates = sparse_index.search(query, top_k=DENSE_CANDIDATE_DEPTH)
                candidates = fuse_rrf_candidates(sparse_candidates, dense_candidates)
        duration_ms = (perf_counter() - started) * 1000.0
        output_candidates = candidates[:MAX_OUTPUT_DEPTH]

        if query_item["category"] == "boundary":
            query_results.append(
                {
                    "query_id": query_item["query_id"],
                    "category": query_item["category"],
                    "candidates": _candidate_records(output_candidates, with_preview=True),
                    "gold_count": 0,
                }
            )
            boundary_query_diagnostics.append(
                {
                    "query_id": query_item["query_id"],
                    "reason_kind": query_item["boundary_diagnostic"]["reason_kind"],
                    "explanation": query_item["boundary_diagnostic"]["explanation"],
                }
            )
            continue

        evaluated = _evaluate_answerable_query(
            query_item=query_item,
            candidates=output_candidates,
            full_texts=full_texts,
            spans_by_document=spans_by_document,
            duration_ms=duration_ms,
        )
        answerable_metrics.append(evaluated["metrics"])
        query_results.append(
            {
                "query_id": query_item["query_id"],
                "category": query_item["category"],
                "candidates": _candidate_records(output_candidates, with_preview=True),
                "gold_count": evaluated["gold_count"],
                "gold_matched_at": evaluated["gold_matched_at"],
                "evidence_recall": evaluated["evidence_recall"],
                "context_precision": evaluated["context_precision"],
                "first_gold_rank": evaluated["first_gold_rank"],
                "duration_ms": evaluated["duration_ms"],
            }
        )

    return {
        "mode": mode,
        "metrics": aggregate_metrics(answerable_metrics),
        "query_results": query_results,
        "boundary_query_diagnostics": boundary_query_diagnostics,
        "answerable_query_count": len(answerable_metrics),
        "boundary_query_count": len(boundary_query_diagnostics),
    }


async def run_retrieval_evaluation(
    *,
    evaluation_dir: Path,
    output_dir: Path,
    source_revision: str,
    modes: Sequence[str],
    embedding_provider: EmbeddingProvider | None,
    embedding_identity: str | None,
    run_id: str | None = None,
    bundle_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Evaluate selected quality modes over one frozen corpus and query set.

    This is the Evaluation Retriever seam used by ``retrieval-evidence
    evaluate``. It deliberately has no dependency on production Migration
    Retrieval and rejects Dense or Hybrid evaluation without an explicit active
    embedding provider and non-sensitive model identity.
    """
    selected_modes = _selected_modes(modes)
    current_time = now or (lambda: datetime.now(UTC))
    started_at = current_time()
    inputs = load_evaluation_inputs(evaluation_dir)
    chunk_size = int(inputs["qa_chunking"]["chunk_chars"])
    chunk_overlap = int(inputs["qa_chunking"]["overlap_chars"])
    preset = CHUNK_STRATEGY_PRESETS["qa"]
    if chunk_size != preset["chunk_size"] or chunk_overlap != preset["chunk_overlap"]:
        raise ValueError("QA Chunking drift: frozen evaluation inputs no longer match the accepted chunker preset")

    requires_dense = any(mode in {"dense", "hybrid_rrf"} for mode in selected_modes)
    if requires_dense and (embedding_provider is None or not embedding_identity):
        raise ValueError("Dense and Hybrid RRF evaluation require an active embedding provider identity")

    chunks, full_texts, spans_by_document = _chunk_corpus(
        evaluation_dir / "corpus",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    sparse_index = SparseBm25Index(chunks, k1=BM25_K1, b=BM25_B)
    sparse_index.search("预热", top_k=1)
    dense_index: DenseVectorIndex | None = None
    if requires_dense:
        assert embedding_provider is not None
        corpus_vectors = await embedding_provider.embed([chunk.content for chunk in chunks])
        dense_index = DenseVectorIndex(chunks, corpus_vectors)

    query_set = json.loads((evaluation_dir / "queries" / "query-set.json").read_text(encoding="utf-8"))
    run_identity = run_id or f"evaluate-{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    mode_results: dict[str, dict[str, Any]] = {}
    for mode in selected_modes:
        mode_result = await _evaluate_mode(
            mode=mode,
            sparse_index=sparse_index,
            dense_index=dense_index,
            embedding_provider=embedding_provider,
            query_set=query_set,
            full_texts=full_texts,
            spans_by_document=spans_by_document,
        )
        mode_result["run_id"] = f"{run_identity}-{mode}"
        mode_results[mode] = mode_result

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_identity,
        "command": "retrieval-evidence evaluate",
        "mode": selected_modes[0] if len(selected_modes) == 1 else "comparison",
        "modes": list(selected_modes),
        "source_revision": source_revision,
        "started_at": started_at.isoformat(),
        "finished_at": current_time().isoformat(),
        "outcome": "passed",
        "evaluation_inputs": dict(inputs),
        "embedding_identity": embedding_identity or "not-applicable",
        "algorithm": {
            "sparse_bm25": {"k1": BM25_K1, "b": BM25_B, "idf": "lucene"},
            "dense_candidate_depth": DENSE_CANDIDATE_DEPTH,
            "rrf_k": RRF_K,
            "rrf_deduplicate_by": "chunk_id",
            "reranker": "not-used",
            "score_averaging": "not-used",
            "gold_coverage_ratio": GOLD_COVERAGE_RATIO,
            "output_depths": list(OUTPUT_DEPTHS),
            "first_gold_rank_unmatched_value": MAX_OUTPUT_DEPTH + 1,
        },
        "mode_results": mode_results,
    }
    run_directory = output_dir / run_identity
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if bundle_dir is not None:
        _export_comparison_bundle(
            bundle_dir=bundle_dir,
            manifest=manifest,
            inputs=inputs,
            now=current_time,
        )
    return manifest


def _comparison_metric_rows(mode_results: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["| metric | sparse_bm25 | dense | hybrid_rrf |", "| --- | ---: | ---: | ---: |"]
    for name in METRIC_NAMES:
        lines.append(
            "| {name} | {sparse:.4f} | {dense:.4f} | {hybrid:.4f} |".format(
                name=name,
                sparse=float(mode_results["sparse_bm25"]["metrics"][name]),
                dense=float(mode_results["dense"]["metrics"][name]),
                hybrid=float(mode_results["hybrid_rrf"]["metrics"][name]),
            )
        )
    return "\n".join(lines)


def _failure_case_ids(mode_result: Mapping[str, Any]) -> list[str]:
    return [
        str(item["query_id"])
        for item in mode_result["query_results"]
        if item["category"] != "boundary" and int(item["first_gold_rank"]) > MAX_OUTPUT_DEPTH
    ]


def _comparison_report_en(*, manifest: Mapping[str, Any], inputs: Mapping[str, Any]) -> str:
    mode_results = manifest["mode_results"]
    failures = {mode: _failure_case_ids(result) for mode, result in mode_results.items()}
    return f"""# Dense and Hybrid RRF Comparison Evidence

## Summary

This bundle records one controlled Evaluation Retriever comparison of `sparse_bm25`, `dense`,
and `hybrid_rrf` over the identical frozen Project-Derived Corpus and Evaluation Query Set. It
does not change or relabel production Migration Retrieval, which remains availability and fallback
behavior rather than Hybrid Retrieval.

## Conditions

- corpus_id `{inputs["corpus_id"]}` version {inputs["corpus_version"]}, sha256 `{inputs["corpus_sha256"]}`
- query_set_id `{inputs["query_set_id"]}` version {inputs["query_set_version"]}, sha256 `{inputs["query_set_sha256"]}`
- QA Chunking: 500 characters with 50 characters overlap; output depths: 3, 5, 10
- active embedding identity: `{manifest["embedding_identity"]}`
- `hybrid_rrf`: top {DENSE_CANDIDATE_DEPTH} Sparse BM25 plus top {DENSE_CANDIDATE_DEPTH}
  Dense candidates, `chunk_id` deduplication, reciprocal-rank fusion `k={RRF_K}`, no reranker,
  no score averaging

## Results

{_comparison_metric_rows(mode_results)}

## Failure Cases

- sparse_bm25 unmatched top-10 Gold Evidence queries: {", ".join(failures["sparse_bm25"]) or "none"}
- dense unmatched top-10 Gold Evidence queries: {", ".join(failures["dense"]) or "none"}
- hybrid_rrf unmatched top-10 Gold Evidence queries: {", ".join(failures["hybrid_rrf"]) or "none"}

## Boundary Query Diagnostics

The four Boundary Queries remain outside answerable-query aggregates. Their insufficient,
conflicting, or stale explanations are retained without turning non-empty candidates into ordinary
answerable-query failures.

## Improvement Claim

No overall improvement claim is established. The reported values are a controlled, per-metric
observation for this one corpus, query set, chunking policy, output depth, and embedding identity;
they do not declare a universal winner or generalize to production Migration Retrieval.

## Limits

- `retrieval_duration_ms` is a mean per-query evaluation retrieval duration; it excludes corpus chunking and corpus vector construction.
- Candidate traces are sanitized and omit content previews, prompts, answers, runtime values, and server details.
- Dense and Hybrid RRF depend on the recorded active embedding identity; changing it requires a new controlled comparison.
"""


def _comparison_report_zh(*, manifest: Mapping[str, Any], inputs: Mapping[str, Any]) -> str:
    mode_results = manifest["mode_results"]
    failures = {mode: _failure_case_ids(result) for mode, result in mode_results.items()}
    return f"""# Dense and Hybrid RRF Comparison Evidence（Dense 与 Hybrid RRF 对比证据）

## Summary（摘要）

本证据包记录 `sparse_bm25`、`dense` 与 `hybrid_rrf` 在同一冻结 Project-Derived Corpus 和
Evaluation Query Set 上的一次受控 Evaluation Retriever 对比。它不改变或重命名生产 Migration
Retrieval；后者仍是 availability/fallback 行为，不是 Hybrid Retrieval。

## Conditions（条件）

- corpus_id `{inputs["corpus_id"]}`，版本 {inputs["corpus_version"]}，sha256 `{inputs["corpus_sha256"]}`
- query_set_id `{inputs["query_set_id"]}`，版本 {inputs["query_set_version"]}，sha256 `{inputs["query_set_sha256"]}`
- QA Chunking：500 字符、50 字符重叠；输出深度：3、5、10
- active embedding identity：`{manifest["embedding_identity"]}`
- `hybrid_rrf`：Sparse BM25 前 {DENSE_CANDIDATE_DEPTH} 条与 Dense 前
  {DENSE_CANDIDATE_DEPTH} 条候选，按 `chunk_id` 去重，使用 reciprocal-rank fusion
  `k={RRF_K}`，无 reranker、无 score averaging

## Results（结果）

{_comparison_metric_rows(mode_results)}

## Failure Cases（失败 Case）

- sparse_bm25 在 top-10 内未覆盖 Gold Evidence 的查询：{", ".join(failures["sparse_bm25"]) or "none"}
- dense 在 top-10 内未覆盖 Gold Evidence 的查询：{", ".join(failures["dense"]) or "none"}
- hybrid_rrf 在 top-10 内未覆盖 Gold Evidence 的查询：{", ".join(failures["hybrid_rrf"]) or "none"}

## Boundary Query Diagnostics（边界查询诊断）

四条 Boundary Query 不计入 answerable-query 聚合。它们的 insufficient、conflicting 或 stale
解释被保留，不会因为候选非空而被记作普通 answerable-query failure。

## Improvement Claim（提升声明）

未建立整体 improvement claim。报告值只是这一份 corpus、query set、chunking policy、output depth
和 embedding identity 条件下的受控逐指标观察；它们不预设普遍赢家，也不推广到生产 Migration Retrieval。

## Limits（适用限制）

- `retrieval_duration_ms` 是每条查询评测检索耗时的平均值；不包含语料分块和语料向量构建。
- Candidate traces 已脱敏，不包含 content previews、prompts、answers、运行时值或服务器信息。
- Dense 与 Hybrid RRF 依赖记录的 active embedding identity；变更后必须重新运行受控对比。
"""


def _export_comparison_bundle(
    *,
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    inputs: Mapping[str, Any],
    now: Callable[[], datetime],
) -> None:
    modes = list(manifest["modes"])
    if modes != list(COMPARISON_MODES):
        raise ValueError("a Public Evidence comparison bundle requires sparse_bm25, dense, and hybrid_rrf")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "sections").mkdir(exist_ok=True)
    mode_results = manifest["mode_results"]
    mode_provenance = [
        {
            "mode": mode,
            "run_id": mode_results[mode]["run_id"],
            "source_revision": manifest["source_revision"],
            "corpus_sha256": inputs["corpus_sha256"],
            "query_set_sha256": inputs["query_set_sha256"],
            "embedding_identity": manifest["embedding_identity"],
        }
        for mode in modes
    ]
    runs = [
        {
            "run_id": item["run_id"],
            "kind": "retrieval-evaluation",
            "source_revision": manifest["source_revision"],
            "started_at": manifest["started_at"],
            "finished_at": manifest["finished_at"],
            "outcome": manifest["outcome"],
        }
        for item in (mode_results[mode] for mode in modes)
    ]
    bundle_manifest: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_id": COMPARISON_BUNDLE_ID,
        "kind": "public-evidence-bundle",
        "canonical_language": "en",
        "mirror_language": "zh-CN",
        "source_revision": manifest["source_revision"],
        "release_candidate": {
            "identity": RELEASE_CANDIDATE_IDENTITY,
            "revision": manifest["source_revision"],
            "status": "candidate",
            "created_at": now().isoformat(),
        },
        "provenance": {
            "runs": runs,
            "corpora": [
                {
                    "corpus_id": inputs["corpus_id"],
                    "version": inputs["corpus_version"],
                    "sha256": inputs["corpus_sha256"],
                }
            ],
            "query_sets": [
                {
                    "query_set_id": inputs["query_set_id"],
                    "version": inputs["query_set_version"],
                    "sha256": inputs["query_set_sha256"],
                    "query_count": inputs["query_count"],
                }
            ],
            "revisions": [manifest["source_revision"]],
        },
        "artifacts": [],
        "limits": [
            {
                "name": "no_generalization",
                "kind": "qualification",
                "statement": (
                    "Metrics describe only this controlled three-mode run over the frozen Project-Derived "
                    "Corpus and Evaluation Query Set; they do not generalize to other corpora, models, "
                    "chunking policies, or languages."
                ),
            },
            {
                "name": "no_universal_winner",
                "kind": "exclusion",
                "statement": (
                    "The comparison records per-metric results without declaring Dense or Hybrid RRF a "
                    "universal winner or replacing production Migration Retrieval."
                ),
            },
            {
                "name": "rrf_contract",
                "kind": "qualification",
                "statement": (
                    "Hybrid RRF fuses fixed top-20 Sparse BM25 and Dense candidates by chunk_id with "
                    "k=60, without a reranker or score averaging."
                ),
            },
            {
                "name": "duration_scope",
                "kind": "qualification",
                "statement": (
                    "retrieval_duration_ms is a mean per-query evaluation retrieval duration and excludes "
                    "corpus chunking and corpus vector construction."
                ),
            },
        ],
    }
    retrieval_section = {
        "section": "retrieval",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_ids": [item["run_id"] for item in mode_provenance],
        "corpus_id": inputs["corpus_id"],
        "query_set_id": inputs["query_set_id"],
        "modes": modes,
        "metrics": [
            {
                "name": name,
                "mode": mode,
                "value": round(float(mode_results[mode]["metrics"][name]), 6),
                "unit": _METRIC_UNITS[name],
            }
            for mode in modes
            for name in METRIC_NAMES
        ],
        "boundary_query_diagnostics": [
            dict(item) for item in mode_results["sparse_bm25"]["boundary_query_diagnostics"]
        ],
        "conditions": {
            "chunking": {
                "policy": inputs["qa_chunking"]["policy"],
                "chunk_chars": inputs["qa_chunking"]["chunk_chars"],
                "overlap_chars": inputs["qa_chunking"]["overlap_chars"],
            },
            "output_depths": list(OUTPUT_DEPTHS),
            "corpus_version": inputs["corpus_version"],
            "query_set_version": inputs["query_set_version"],
            "candidate_depth": DENSE_CANDIDATE_DEPTH,
            "model_identity": manifest["embedding_identity"],
            "mode_provenance": mode_provenance,
        },
    }
    annotations = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "query_set_id": inputs["query_set_id"],
        "query_set_version": inputs["query_set_version"],
        "queries": [
            {
                "query_id": item["query_id"],
                "category": item["category"],
                "gold_count": item["gold_count"],
                "gold_matched_at": item.get("gold_matched_at"),
                "first_gold_rank": item.get("first_gold_rank"),
            }
            for item in mode_results["sparse_bm25"]["query_results"]
        ],
    }
    candidates = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "modes": modes,
        "run_ids": [item["run_id"] for item in mode_provenance],
        "output_depths": list(OUTPUT_DEPTHS),
        "mode_runs": [
            {
                "mode": mode,
                "run_id": mode_results[mode]["run_id"],
                "queries": [
                    {
                        "query_id": query["query_id"],
                        "category": query["category"],
                        "candidates": [
                            {
                                key: candidate[key]
                                for key in ("rank", "chunk_id", "document", "chunk_index", "score")
                            }
                            for candidate in query["candidates"]
                        ],
                    }
                    for query in mode_results[mode]["query_results"]
                ],
            }
            for mode in modes
        ],
    }
    run_conditions = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "modes": modes,
        "embedding_identity": manifest["embedding_identity"],
        "chunking": dict(inputs["qa_chunking"]),
        "output_depths": list(OUTPUT_DEPTHS),
        "candidate_depth": DENSE_CANDIDATE_DEPTH,
        "hybrid_rrf": {
            "k": RRF_K,
            "deduplicate_by": "chunk_id",
            "reranker": False,
            "score_averaging": False,
        },
        "evaluation_inputs": dict(inputs),
        "metric_definitions": {
            name: {
                "unit": _METRIC_UNITS[name],
                "aggregate": "macro mean over the 12 answerable queries",
            }
            for name in METRIC_NAMES
        },
    }
    bundle_files: dict[str, tuple[str, str, str]] = {
        "manifest.json": ("manifest", "manifest", json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n"),
        "sections/retrieval.json": ("retrieval", "section", json.dumps(retrieval_section, ensure_ascii=False, indent=2) + "\n"),
        "annotations.json": ("annotation", "data", json.dumps(annotations, ensure_ascii=False, indent=2) + "\n"),
        "candidates.json": ("trace", "data", json.dumps(candidates, ensure_ascii=False, indent=2) + "\n"),
        "run-conditions.json": ("retrieval", "data", json.dumps(run_conditions, ensure_ascii=False, indent=2) + "\n"),
        "REPORT.md": ("report", "report-en", _comparison_report_en(manifest=manifest, inputs=inputs)),
        "REPORT.zh-CN.md": ("report", "report-zh", _comparison_report_zh(manifest=manifest, inputs=inputs)),
    }
    artifacts: list[dict[str, Any]] = []
    for relative_path, (kind, role, content) in bundle_files.items():
        target = bundle_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        artifacts.append({"path": relative_path, "kind": kind, "role": role, "sha256": sha256(content.encode("utf-8")).hexdigest()})
    bundle_manifest["artifacts"] = artifacts
    (bundle_dir / "manifest.json").write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
