# ZhoMind-v2 Portfolio Release v1.0.0

[Chinese mirror](PORTFOLIO-RELEASE-v1.0.0.zh-CN.md) | [Project landing page](../../README.md) | [Interview Dossier](../portfolio/INTERVIEW-DOSSIER.md) | [Public Evidence Bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.md)

## Release Scope

This Portfolio Release publishes the reviewed project narrative and the accepted, non-sensitive evidence subset for source candidate `91753f1c1ff6fc07bc262dfa50fb719a63210e0b`. The release record and tag identify the exact merged revision; the source candidate remains the provenance identity for accepted runtime observations because a generated bundle cannot self-reference the commit that adds it.

## Meaningful Development History

- Locked deterministic PR Gate: committed backend and frontend locks, backend Ruff/Pyright/full deterministic tests, frontend unit/build/disposable-API browser acceptance, bilingual parity, bundle validation, and secret scanning.
- Publication and access boundaries: invitation-gated Knowledge User admission, explicit Candidate Build to Published Knowledge Version lifecycle, single-provider settings lifecycle, privacy/retention, operational-event, and first-release capacity boundaries.
- Evidence-Gated Answer Execution: one immutable Answer Evidence Set and Evidence Excerpt Snapshot move consistently across normal response, SSE, persistence, and history; insufficient evidence and provider unavailability are closed outcomes.
- Retrieval evidence: frozen Chinese-first evaluation inputs, Migration Retrieval fallback kept separate, Sparse BM25 baseline, and controlled Dense/Hybrid RRF comparison with `k=60`.
- Security evidence: structured untrusted-source regions and a bounded four-case Prompt Injection observation through the authenticated product path.
- Candidate acceptance: repaired incomplete answer-evidence anchoring, measured c1/c5 performance, reran deterministic and isolated gates, and recorded persistent-stack product acceptance in `portfolio-release-candidate-01`.

## Accepted Evidence

The [Public Evidence Bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.md) contains only reviewed, non-sensitive data. Its [manifest](../../public-evidence/releases/portfolio-release-candidate-01/manifest.json) records artifact hashes, source revision, corpus/query hashes, and normalized run identities. It covers retrieval, closed answer outcomes, Prompt Injection classifications, c1/c5 performance decomposition, clean-start smoke outcomes, and bounded production acceptance.

The deterministic PR Gate recorded `355 passed, 1 skipped` for the backend, frontend unit/build success, and browser acceptance `49/49`. The explicit live-Milvus opt-in skip is not live evidence. Persistent-stack acceptance includes normal answer, SSE, history, insufficient evidence, generation unavailable with zero fallback hops, withdrawal, deactivation, health, and role boundaries.

## Known Limits

The P95 target of 12 seconds remains unmet: c1 total P95 was 15.930 seconds and c5 total P95 was 74.465 seconds. External provider and embedding-provider time are reported separately from project-controlled timing. Retrieval comparison findings apply only to the recorded corpus, query set, model identity, load, host envelope, and observation window; they do not establish a universal retrieval winner.

The release does not require a continuously hosted public demo. Public evidence excludes credentials, runtime configuration, private questions, complete answers, source excerpts, host details, and operational credentials.

## Verification Links

- [Bilingual project narrative](../../README.md)
- [Bilingual Interview Dossier](../portfolio/INTERVIEW-DOSSIER.md)
- [Public Evidence Bundle report](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.md)
- [Bundle manifest](../../public-evidence/releases/portfolio-release-candidate-01/manifest.json)
- [Bundle validator contract](../../public-evidence/contract/README.md)
