# ZhoMind-v2 Interview Dossier

[Chinese mirror](INTERVIEW-DOSSIER.zh-CN.md) | [Project landing page](../../README.md) | [Public Evidence Bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.md) | [Release Notes](../releases/PORTFOLIO-RELEASE-v1.0.0.md)

## Positioning

ZhoMind-v2 is a production-oriented, evidence-gated RAG application for a Team-Shared Knowledge Base. Its reviewable unit is not a broad retrieval result: it is one Answer Evidence Set that is carried unchanged through generation, citations, SSE, and history. The accepted source candidate is `91753f1c1ff6fc07bc262dfa50fb719a63210e0b`; `portfolio-release-candidate-01` is the reviewed public evidence subset for that candidate.

## Architecture Claim

The architecture separates Migration Retrieval, Evaluation Retriever, Direct Retrieval Diagnostic, Evidence-Gated Answer Execution, and Production Answer Acceptance. The source boundary is implemented in [`backend/app/rag/answer_execution.py`](../../backend/app/rag/answer_execution.py), while the public acceptance record is [production acceptance](../../public-evidence/releases/portfolio-release-candidate-01/sections/production-acceptance.json). This separation prevents a broad diagnostic candidate set from being represented as an answer or citation decision.

## Failure And Repair Claim

The release candidate repaired a false positive in which dense nearest-neighbor candidates, question scaffolding, or generic tokens could appear answerable without a complete lexical anchor. The repair keeps dense candidates diagnostic until complete answer evidence exists; it adds no score threshold, relevance judge, reranker, or answer fallback. The accepted source and regression coverage are recorded in [`backend/tests/unit/test_document_retrieval_anchor.py`](../../backend/tests/unit/test_document_retrieval_anchor.py) and the [candidate bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.md).

## Evaluation Claim

The Evaluation Retriever compares Sparse BM25, Dense, and Hybrid RRF over the same frozen inputs: Chinese-first Project-Derived Corpus, 16-query Evaluation Query Set, QA Chunking 500/50, candidate depth 20, and RRF `k=60`. The bundle's [retrieval section](../../public-evidence/releases/portfolio-release-candidate-01/sections/retrieval.json) records Evidence Recall@3/@5/@10, Context Precision@3/@5/@10, first Gold Evidence rank, duration, and Boundary Query diagnostics. The recorded comparison did not establish a universal winner; all three modes had matching recall/precision, with larger recorded duration for Dense and Hybrid RRF.

## Security Claim

Published source text is untrusted input. [`backend/app/rag/prompt_regions.py`](../../backend/app/rag/prompt_regions.py) separates policy, user question, and evidence snapshots, while [`backend/app/rag/answer_execution.py`](../../backend/app/rag/answer_execution.py) preserves the closed outcomes. The bounded Prompt Injection evidence records four fixed case kinds through the authenticated product path, with normalized case identity, pass/fail, outcome, citation count, and failure classification only. See the [prompt-injection section](../../public-evidence/releases/portfolio-release-candidate-01/sections/prompt-injection.json). This is not a general security guarantee.

## Performance Claim

The accepted c1 and c5 observations used 20 requests each and recorded zero errors. The [performance sections](../../public-evidence/releases/portfolio-release-candidate-01/sections/performance-c1.json) report TTFT P50/P95/P99, total P50/P95/P99, retrieval, provider, embedding-provider, persistence, and application-controlled timing. c1 total P95 was 15.930 seconds; c5 total P95 was 74.465 seconds. The P95 target of 12 seconds remains unmet. The observed project-controlled envelope is reported separately and does not turn external provider latency into an application claim.

## Production Claim

The final accepted persistent-stack record covers authenticated normal answer, SSE, history, insufficient evidence, generation unavailable with zero fallback hops, withdrawal, deactivation, health, and role boundaries. Two independent clean-start Retrieval Smokes and the deterministic PR Gate are included in the same provenance chain. The accepted run identities and non-sensitive outcomes are in the [bundle manifest](../../public-evidence/releases/portfolio-release-candidate-01/manifest.json) and [candidate report](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.md).

## Scope And Limits

The dossier makes no claim that retrieval evaluation changes Migration Retrieval, that Hybrid RRF is always superior, or that the 12-second P95 target was achieved. Results are bounded to the recorded corpus, query set, model identity, load, host envelope, and observation window. Public artifacts deliberately exclude credentials, raw environment values, private questions, complete answers, source excerpts, and operational details.
