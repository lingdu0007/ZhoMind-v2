# ZhoMind-v2 Interview Dossier

[English](INTERVIEW-DOSSIER.md) | [Project landing page](../../README.zh-CN.md) | [Public Evidence Bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md) | [Release Notes](../releases/PORTFOLIO-RELEASE-v1.0.0.zh-CN.md)

## Positioning

ZhoMind-v2 是面向 Team-Shared Knowledge Base 的 production-oriented、evidence-gated RAG application。可审阅的 unit 不是广泛 retrieval result，而是一个在 generation、citations、SSE 和 history 中保持不变的 Answer Evidence Set。已接受 source candidate 为 `91753f1c1ff6fc07bc262dfa50fb719a63210e0b`；`portfolio-release-candidate-01` 是该 candidate 经审阅的 public evidence subset。

## Architecture Claim

架构分离 Migration Retrieval、Evaluation Retriever、Direct Retrieval Diagnostic、Evidence-Gated Answer Execution 与 Production Answer Acceptance。source boundary 实现在 [`backend/app/rag/answer_execution.py`](../../backend/app/rag/answer_execution.py)，公开 acceptance record 在 [production acceptance](../../public-evidence/releases/portfolio-release-candidate-01/sections/production-acceptance.json)。这一分离避免把 broad diagnostic candidate set 表述为 answer 或 citation decision。

## Failure And Repair Claim

release candidate 修复了一种 false positive：dense nearest-neighbor candidates、question scaffolding 或 generic tokens 会在没有 complete lexical anchor 时显得可回答。该 repair 使 dense candidates 保持 diagnostic，直到存在 complete answer evidence；没有增加 score threshold、relevance judge、reranker 或 answer fallback。已接受 source 与 regression coverage 记录在 [`backend/tests/unit/test_document_retrieval_anchor.py`](../../backend/tests/unit/test_document_retrieval_anchor.py) 和 [candidate bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md)。

## Evaluation Claim

Evaluation Retriever 在相同冻结 inputs 上比较 Sparse BM25、Dense 与 Hybrid RRF：Chinese-first Project-Derived Corpus、16-query Evaluation Query Set、QA Chunking 500/50、candidate depth 20 和 RRF `k=60`。bundle 的 [retrieval section](../../public-evidence/releases/portfolio-release-candidate-01/sections/retrieval.json) 记录 Evidence Recall@3/@5/@10、Context Precision@3/@5/@10、first Gold Evidence rank、duration 与 Boundary Query diagnostics。记录的 comparison 没有建立 universal winner；三种 modes 的 recall/precision 一致，Dense 与 Hybrid RRF 的 recorded duration 更长。

## Security Claim

Published source text 是 untrusted input。[`backend/app/rag/prompt_regions.py`](../../backend/app/rag/prompt_regions.py) 分离 policy、user question 与 evidence snapshots，而 [`backend/app/rag/answer_execution.py`](../../backend/app/rag/answer_execution.py) 保持 closed outcomes。有限的 Prompt Injection evidence 通过 authenticated product path 记录四类固定 cases，仅包含 normalized case identity、pass/fail、outcome、citation count 与 failure classification。见 [prompt-injection section](../../public-evidence/releases/portfolio-release-candidate-01/sections/prompt-injection.json)。这不是 general security guarantee。

## Performance Claim

已接受的 c1 与 c5 observations 各使用 20 requests，并记录 zero errors。[performance sections](../../public-evidence/releases/portfolio-release-candidate-01/sections/performance-c1.json) 报告 TTFT P50/P95/P99、total P50/P95/P99、retrieval、provider、embedding-provider、persistence 与 application-controlled timing。c1 total P95 为 15.930 seconds；c5 total P95 为 74.465 seconds。P95 target of 12 seconds remains unmet。observed project-controlled envelope 单独报告，不把 external provider latency 转成 application claim。

## Production Claim

最终 accepted persistent-stack record 覆盖 authenticated normal answer、SSE、history、insufficient evidence、generation unavailable with zero fallback hops、withdrawal、deactivation、health 与 role boundaries。两次独立 clean-start Retrieval Smokes 与 deterministic PR Gate 位于同一 provenance chain。已接受 run identities 与 non-sensitive outcomes 在 [bundle manifest](../../public-evidence/releases/portfolio-release-candidate-01/manifest.json) 与 [candidate report](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md) 中。

## Scope And Limits

该 dossier 不主张 retrieval evaluation 改变 Migration Retrieval，不主张 Hybrid RRF 总是更优，也不主张 12-second P95 target 已达成。结果受 recorded corpus、query set、model identity、load、host envelope 与 observation window 限制。public artifacts 刻意排除 credentials、raw environment values、private questions、complete answers、source excerpts 与 operational details。
