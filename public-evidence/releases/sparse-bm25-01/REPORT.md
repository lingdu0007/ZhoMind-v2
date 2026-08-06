# Sparse BM25 Evidence Package

## Summary

This bundle records the first quality-oriented retrieval evaluation of the ZhoMind-v2 Portfolio
Release: genuine Sparse BM25 (Literal-Preserving Tokenization with jieba) run through the
Evaluation Retriever's `retrieval-evidence evaluate` command over the frozen Project-Derived
Corpus and 16-query Evaluation Query Set. It is a tracer bullet for the Retrieval Evidence
Baseline; it makes no Dense, Hybrid RRF, answer-quality, or universal-improvement claim.

## Method

### Corpus

- corpus_id `project-derived-corpus` version 1.0.0, sha256 `894d10004f25a2a6462870c1a869586ca83dfc28f0407f75f8cc33d9757ac39d`
- Six Chinese-First Markdown documents; English identifiers, paths, status codes, versions, and model names kept as literals

### Query set

- query_set_id `evaluation-query-set` version 1.0.0, sha256 `a468a1af9fc0c7b979ec1c5dfa8874a5ad27e25bcc79de1e0142564db992df20`
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
| evidence_recall@3 | 1.0000 |
| evidence_recall@5 | 1.0000 |
| evidence_recall@10 | 1.0000 |
| context_precision@3 | 0.3333 |
| context_precision@5 | 0.2000 |
| context_precision@10 | 0.1000 |
| first_gold_rank | 1.1667 |
| retrieval_duration_ms | 0.3554 |

### First Gold Evidence Rank

- first_gold_rank: 1.1667 (1-based; 11 = no gold-covered candidate inside the top 10)

### Retrieval Duration

- retrieval_duration_ms: 0.36 (mean per-query
  in-process Sparse BM25 search time)

## Boundary Query Diagnostics

The four Boundary Queries (`boundary-lexical-is-bm25`, `boundary-chunking-300`,
`boundary-hybrid-winner`, `boundary-production-ports`) are excluded from answerable-query
aggregates and retain their corpus-derived insufficiency, conflict, or staleness explanations
in `sections/retrieval.json`.

## Run Conditions

- source revision: `95dad2987a182f5a250f244ac96e58bb11f50bf8`
- run id: `20260806T110748Z-105402-20694`
- mode: sparse_bm25
- output depths: 3, 5, 10
- tokenizer: literal_preserving_jieba

## Limits

- The metrics describe only the recorded run conditions and do not generalize to other corpora, models, chunking policies, or languages.
- This bundle contains Sparse BM25 evidence only; Dense, Hybrid RRF, answer-quality, and universal-improvement claims are not made.
- `retrieval_duration_ms` excludes corpus chunking and index construction.
