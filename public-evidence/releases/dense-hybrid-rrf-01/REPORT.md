# Dense and Hybrid RRF Comparison Evidence

## Summary

This bundle records one controlled Evaluation Retriever comparison of `sparse_bm25`, `dense`,
and `hybrid_rrf` over the identical frozen Project-Derived Corpus and Evaluation Query Set. It
does not change or relabel production Migration Retrieval, which remains availability and fallback
behavior rather than Hybrid Retrieval.

## Conditions

- corpus_id `project-derived-corpus` version 1.0.0, sha256 `894d10004f25a2a6462870c1a869586ca83dfc28f0407f75f8cc33d9757ac39d`
- query_set_id `evaluation-query-set` version 1.0.0, sha256 `a468a1af9fc0c7b979ec1c5dfa8874a5ad27e25bcc79de1e0142564db992df20`
- QA Chunking: 500 characters with 50 characters overlap; output depths: 3, 5, 10
- active embedding identity: `Qwen/Qwen3-Embedding-8B:8cfc4ef25f497deccfd2eb0d23b4a8bdd7360e588db0ea9b80ffae616f5d82b9`
- `hybrid_rrf`: top 20 Sparse BM25 plus top 20
  Dense candidates, `chunk_id` deduplication, reciprocal-rank fusion `k=60`, no reranker,
  no score averaging

## Results

| metric | sparse_bm25 | dense | hybrid_rrf |
| --- | ---: | ---: | ---: |
| evidence_recall@3 | 1.0000 | 1.0000 | 1.0000 |
| evidence_recall@5 | 1.0000 | 1.0000 | 1.0000 |
| evidence_recall@10 | 1.0000 | 1.0000 | 1.0000 |
| context_precision@3 | 0.3333 | 0.3333 | 0.3333 |
| context_precision@5 | 0.2000 | 0.2000 | 0.2000 |
| context_precision@10 | 0.1000 | 0.1000 | 0.1000 |
| first_gold_rank | 1.1667 | 1.2500 | 1.1667 |
| retrieval_duration_ms | 0.4503 | 1904.8889 | 2061.8107 |

## Failure Cases

- sparse_bm25 unmatched top-10 Gold Evidence queries: none
- dense unmatched top-10 Gold Evidence queries: none
- hybrid_rrf unmatched top-10 Gold Evidence queries: none

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
