# Public Evidence Bundle Contract

Version: **1.1.0** (schema `evidence-bundle.schema.json`, section schemas in `sections/`)

## Purpose

A **Public Evidence Bundle** is the reviewed, non-sensitive evidence subset published with the ZhoMind-v2 Portfolio Release. It links every public claim to a source revision, run identities, and corpus/query hashes, without exposing credentials, raw environment values, private questions, complete model answers, source excerpts, host addresses, or operational credentials.

This contract is the single public evidence format. Future evidence producers (retrieval, deterministic answer, Prompt Injection, performance, production acceptance) write their data into the sections defined here; they must not invent alternate public evidence formats. The deterministic validator `scripts/validate-evidence-bundle.py` enforces this contract in the PR Gate.

## Bundle layout

A bundle is one directory containing a manifest, typed section files, and the bilingual report pair:

```
<bundle-dir>/
  manifest.json                     # required: allowlisted manifest (schema 1.1.0)
  sections/
    retrieval.json                  # optional typed section
    answer.json                     # optional typed section
    prompt-injection.json           # optional typed section
    performance.json                # optional typed section
    production-acceptance.json      # optional typed section
  REPORT.md                         # required: English canonical report
  REPORT.zh-CN.md                   # required: complete Chinese mirror
```

A bundle must contain at least one typed section. Section files and reports are declared in `manifest.artifacts` with their sha256; the validator recomputes every hash.

## Manifest fields (explicit allowlist)

`manifest.json` is an object with exactly these top-level fields (`additionalProperties: false`):

| Field | Type | Rule |
| --- | --- | --- |
| `schema_version` | string | Legacy bundles use `1.0.0`; controlled three-mode retrieval comparisons must use `1.1.0`. |
| `bundle_id` | string | `[a-z0-9][a-z0-9-]{0,127}`, unique across bundles checked in one invocation (`--all` or explicit directories). |
| `kind` | string | Must equal `public-evidence-bundle`. |
| `canonical_language` | string | Must equal `en`. |
| `mirror_language` | string | Must equal `zh-CN`. |
| `source_revision` | string | Full 40-hex source commit id the bundle describes. |
| `release_candidate` | object | `identity`, `revision` (must equal `source_revision`), `status` (`candidate`/`accepted`/`released`), `created_at` (ISO 8601 UTC). |
| `provenance` | object | `runs`, `corpora`, `query_sets`, `revisions` (see below). |
| `artifacts` | array | Inventory of every bundle file: `path`, `kind`, `sha256`, `role`. |
| `limits` | array | Declared limits, each `name`/`kind` (`boundary`/`qualification`/`exclusion`/`target`)/`statement`. At least one is required. |

`provenance.runs` records non-sensitive run identities: `run_id`, `kind` (`retrieval-smoke`, `generation-smoke`, `retrieval-evaluation`, `answer-acceptance`, `prompt-injection-run`, `performance-run`, `production-acceptance`), `source_revision`, `started_at`, `finished_at`, `outcome` (`passed`/`failed`/`completed-with-exceptions`).

`provenance.corpora` and `provenance.query_sets` record `sha256`-anchored identities (version + deterministic hash). `provenance.revisions` lists every revision referenced anywhere in the bundle.

## Typed sections

Each section is one JSON file validated against its own allowlisted schema. Section files are optional now; producers may add them over time, but once present they must conform.

| Section file | Schema | Required content |
| --- | --- | --- |
| `sections/retrieval.json` | `retrieval.schema.json` | `run_ids`, `corpus_id`, `query_set_id`, `modes`, `metrics` (Evidence Recall@3/@5/@10, Context Precision@3/@5/@10, first Gold Evidence rank, retrieval duration), `boundary_query_diagnostics`, `conditions` (chunking, output depths, corpus/query-set versions). |
| `sections/answer.json` | `answer.schema.json` | Deterministic answer cases: `case_id`, `outcome` (one of the four closed Answer Execution Outcomes), `citations_count`, `contract` (`passed`/`failed`). |
| `sections/prompt-injection.json` | `prompt-injection.schema.json` | Adversarial cases: `case_id`, `kind`, `outcome`, `pass_fail`, `citation_counts`, `failure_classification`. |
| `sections/performance.json` | `performance.schema.json` | `load` (concurrency, requests), `metrics` (TTFT and total P50/P95/P99, error rate, retrieval/provider/persistence durations), `target` (12-second P95 target, `met`, `note`). |
| `sections/production-acceptance.json` | `production-acceptance.schema.json` | `path` (must equal `/api/v1/chat`), `results` (bounded per-check outcomes and limitations). |

Every declared evaluation mode (all `modes` except `migration`, which is the separately reported availability regime) must carry the complete eight-metric family. Values are unit-bounded: `ratio` in [0, 1], `rank` >= 1, `ms` >= 0. Missing metric families and out-of-bounds values fail with `metric-completeness` diagnostics. A retrieval section that reports only `migration` availability may carry an empty `metrics` array.

An exact three-mode comparison (`sparse_bm25`, `dense`, and `hybrid_rrf`, with no `migration` mode) is a `1.1.0` retrieval section. It must record `conditions.candidate_depth = 20`, one non-sensitive `model_identity`, and one `mode_provenance` item per mode. Each item carries the mode's run identity, source revision, corpus hash, query-set hash, and embedding identity; the validator rejects missing fields or cross-mode drift.

## Sensitive-field policy

Two layers enforce exclusion of sensitive material.

1. **Field allowlist.** The schemas use `additionalProperties: false` everywhere. Fields that would carry sensitive content (`credentials`, `api_key`, `password`, `token`, `secret`, `answer`, `excerpt`, `prompt`, `question`, `environment`, `host`, `address`, `private_content`) are not part of the contract, so any occurrence is a schema violation with a specific diagnostic.

2. **Value-shape rejection.** The validator scans the text content of the manifest, every artifact file (up to a 4 MiB per-file text limit; binary or non-UTF-8 files are not text-scannable), and both reports for sensitive shapes and rejects the bundle when any is found:
   - Provider/API key shapes (e.g. long `key-prefix-...` tokens, AWS-style access keys, GitHub tokens).
   - Private key blocks and SSH key references (`id_ed25519`, `id_rsa`, `.ssh/` paths).
   - JSON Web Tokens and bearer tokens.
   - URLs embedding credentials (`scheme://user:password@host`).
   - IPv4 addresses and `host:port` forms.
   - Environment assignments (`KEY=value` lines) and `${VAR}` references.
   - `user@ip-address` SSH-style login forms.

Sensitive values are never printed back; diagnostics name the shape class and the offending file only.

## Cross-artifact reference rules

The validator rejects any report, section, or artifact that references something absent from the bundle's own provenance:

1. `release_candidate.revision` must equal `source_revision`.
2. `source_revision` must appear in `provenance.revisions`.
3. Every `run.source_revision` must appear in `provenance.revisions` (or equal `source_revision`).
4. `run_id`, `corpus_id`, and `query_set_id` are unique within their arrays.
5. Every section `run_ids` entry must exist in `provenance.runs`; `corpus_id` and `query_set_id` must exist in `provenance.corpora` and `provenance.query_sets`.
6. `manifest.json` itself must be listed in `artifacts` (kind `manifest`, role `manifest`). Its `sha256` is a self-reference placeholder and is not recomputed by the validator.
7. Every other artifact path must resolve inside the bundle directory, exist, and match its declared sha256.
8. Section artifacts must exist and their JSON `section` discriminator must equal the artifact `kind`.
9. `REPORT.md` (role `report-en`) and `REPORT.zh-CN.md` (role `report-zh`) must both be present and declared.

## Bilingual parity

`REPORT.md` is canonical; `REPORT.zh-CN.md` is the complete mirror updated in the same change. The validator enforces two machine-checkable rules:

1. **Structure parity.** Identical H1/H2/H3 heading outline counts and identical fenced code block counts.
2. **Literal parity.** Identifiers, metric names, status values, model names, and versions are preserved as literals. The validator extracts a literal set from the English report (snake_case identifiers, `Metric@N` names, `x.y.z` versions, and a fixed acronym list such as TTFT, P50, P95, P99, RRF, BM25, QA, SSE, API, JSON, CLI, PR) and requires every extracted literal to appear in the Chinese report. Metric names and status values are never translated.

## Validator

```bash
# Validate the example bundle and every tracked release bundle
python3 scripts/validate-evidence-bundle.py --all

# Validate specific bundle directories
python3 scripts/validate-evidence-bundle.py public-evidence/example

# List bundles discovered by --all without validating
python3 scripts/validate-evidence-bundle.py --list
```

Exit code `0` means every checked bundle conforms; `1` means at least one check failed (with per-bundle diagnostics on stdout); `2` is a usage error. The validator is pure Python 3 stdlib and needs no installed dependencies, matching `check-docs-parity.py` and `scan-secrets.py`.

## Versioning

The contract is versioned by `schema_version` inside each schema. Version `1.1.0` adds the controlled three-mode retrieval-comparison rules and the validator support in this same change. Existing `1.0.0` bundles remain on an explicit legacy validation path; only a three-mode comparison must use `1.1.0`, so legacy evidence is not silently reinterpreted as comparison evidence.

## Terminology

Domain vocabulary follows `CONTEXT.md`: Evidence Package, Evidence Manifest, Public Evidence Bundle, Portfolio Release, Stable Experimental Baseline, Evaluation Query Set, Boundary Query, Answer Evidence Set, Answer Execution Outcome, and Production Answer Acceptance keep their canonical definitions.
