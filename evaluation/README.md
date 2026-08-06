# Evaluation Inputs: Project-Derived Corpus and Evaluation Query Set

This directory freezes the inputs of the Retrieval Evidence Baseline: a Project-Derived Corpus
and the 16-query Evaluation Query Set. It is versioned with the source so every run can anchor its
manifests to deterministic hashes.

## Project-Derived Corpus

`corpus/` contains six Markdown documents written in Chinese (Chinese-First Corpus). English
identifiers, paths, status codes, versions, and model names are kept as literals. Every statement
is traceable to ZhoMind-v2's README, interface contracts, ADRs, or verified tests; each document
ends with a `## 来源` (sources) section that maps its facts to concrete repository locations.

| Document | Topic |
| --- | --- |
| `01-product-overview.md` | Evidence-Gated Answer, Answer Execution Outcome, Answer Evidence Set, Insufficient Evidence Reply, Generation Unavailable, Non-Knowledge-Base Reply |
| `02-retrieval-and-chunking.md` | Chunking presets, QA Chunking, Migration Retrieval, Lexical Heuristic, fallback trace, Evaluation Retriever contract, RRF Fusion |
| `03-embedding-and-dense-index.md` | Qwen Embedding, DenseEmbeddingContract, contract fingerprint, Milvus collection and row identity |
| `04-publication-lifecycle.md` | Upload/build/publish flow, job states, Withdrawn Source |
| `05-auth-and-membership.md` | Bootstrap Administrator, Team Invitation, Administrator Promotion, Deactivated Member |
| `06-configuration-and-deployment.md` | Locked dependencies, PR Gate, isolated Retrieval Smoke |

## Evaluation Query Set

`queries/query-set.json` holds the fixed 16 queries:

- four `exact_constraint` queries (exact identifiers, paths, versions, constants),
- four `semantic_paraphrase` queries (natural rewordings of the same facts),
- four `combined_condition` queries (multiple conditions in one question),
- four `boundary` queries (insufficient, conflicting, or stale evidence).

Every answerable query records `required_claims` and `gold_evidence`. Each Gold Evidence item
references one corpus document and one verbatim `passage`; the deterministic tests assert that
every passage exists in the referenced document. Every Boundary Query records a
`boundary_diagnostic` with a `reason_kind` of `insufficient`, `conflicting`, or `stale` and an
explanation of why its evidence must not be treated as normally answerable. Boundary Query
Diagnostics stay outside answerable-query aggregates.

## QA Chunking

QA Chunking is fixed at `chunk_size = 500`, `chunk_overlap = 50` (the `qa` preset of
`CHUNK_STRATEGY_PRESETS` in `backend/app/documents/chunker.py`). It is held constant across all
retrieval-mode comparisons; any chunking ablation is a later, separately labeled experiment.
Literal preservation (keeping identifiers, paths, versions, and model names as whole tokens) is a
Sparse BM25 tokenization rule recorded here for the frozen baseline; it is not implemented by this
ticket.

## Deterministic hashes

`corpus-manifest.json` records the frozen `corpus_id`, `query_set_id`, versions, and sha256 hashes
plus the QA Chunking policy. The hashing rules (also implemented in
`scripts/hash-evaluation-assets.py`, pure stdlib) are:

- **corpus sha256**: sort every `*.md` file under `corpus/` by relative path; for each file take
  the sha256 hex digest of its bytes; build the string
  `"<relative-path>\0<per-file-sha256-hex>"` joined by newlines in sorted order; return the sha256
  hex digest of that joined string.
- **query-set sha256**: the sha256 hex digest of the exact bytes of `queries/query-set.json`.

Regenerate or verify with:

```bash
python3 scripts/hash-evaluation-assets.py        # rewrite corpus-manifest.json
python3 scripts/hash-evaluation-assets.py --check  # verify the committed manifest (PR Gate)
```

Run manifests anchor to these values through the `evaluation_inputs` block, so a run's corpus and
query-set identity are reproducible from the committed files without storing credentials.

## Scope

This directory freezes baseline inputs only. It does not implement Sparse BM25, Dense or
Hybrid RRF retrieval, evaluation metrics, or quality comparison; those belong to later
Retrieval Evidence work.

Run manifests project the Migration Retrieval fallback trace as a normalized, non-sensitive
subset: `provider_error` is recorded as `code` and `type` only (never the provider message, which
may carry operational detail), alongside `fallback_used`, `lexical_scope`, and bounded candidate
counts. The corpus document `02-retrieval-and-chunking.md` describes the full application-level
`RetrieveResult` trace, which also carries the message field.
