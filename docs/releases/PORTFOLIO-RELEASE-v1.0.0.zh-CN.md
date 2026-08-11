# ZhoMind-v2 Portfolio Release v1.0.0

[English](PORTFOLIO-RELEASE-v1.0.0.md) | [Project landing page](../../README.zh-CN.md) | [Interview Dossier](../portfolio/INTERVIEW-DOSSIER.zh-CN.md) | [Public Evidence Bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md)

## Release Scope

本次 Portfolio Release 发布经审阅的 project narrative，以及 runtime source candidate `91753f1c1ff6fc07bc262dfa50fb719a63210e0b` 的 accepted、non-sensitive evidence subset。release artifact revision 是包含 bundle 与这些双语 materials 的精确 merged commit；release record 与 tag 标识该 revision。source candidate 保持 accepted runtime observations 的 provenance identity，因为 generated bundle 不能 self-reference 添加它的 commit。

## Meaningful Development History

- Locked deterministic PR Gate：committed backend 与 frontend locks、backend Ruff/Pyright/full deterministic tests、frontend unit/build/disposable-API browser acceptance、bilingual parity、bundle validation 和 secret scanning。
- Publication and access boundaries：invitation-gated Knowledge User admission、显式 Candidate Build 到 Published Knowledge Version lifecycle、single-provider settings lifecycle、privacy/retention、operational-event 与 first-release capacity boundaries。
- Evidence-Gated Answer Execution：一个 immutable Answer Evidence Set 与 Evidence Excerpt Snapshot 一致地经过 normal response、SSE、persistence 和 history；insufficient evidence 与 provider unavailability 是 closed outcomes。
- Retrieval evidence：冻结 Chinese-first evaluation inputs、保持分离的 Migration Retrieval fallback、Sparse BM25 baseline，以及使用 `k=60` 的 controlled Dense/Hybrid RRF comparison。
- Security evidence：structured untrusted-source regions，以及通过 authenticated product path 的 bounded four-case Prompt Injection observation。
- Candidate acceptance：修复 incomplete answer-evidence anchoring、测量 c1/c5 performance、重跑 deterministic 与 isolated gates，并把 persistent-stack product acceptance 记录进 `portfolio-release-candidate-01`。

## Accepted Evidence

[Public Evidence Bundle](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md) 仅包含 reviewed、non-sensitive data。其 [manifest](../../public-evidence/releases/portfolio-release-candidate-01/manifest.json) 记录 artifact hashes、source revision、corpus/query hashes 与 normalized run identities。它覆盖 retrieval、closed answer outcomes、Prompt Injection classifications、c1/c5 performance decomposition、clean-start smoke outcomes 与 bounded production acceptance。

deterministic PR Gate 记录 backend `355 passed, 1 skipped`、frontend unit/build success 与 browser acceptance `49/49`。explicit live-Milvus opt-in skip 不是 live evidence。persistent-stack acceptance 包含 normal answer、SSE、history、insufficient evidence、generation unavailable with zero fallback hops、withdrawal、deactivation、health 与 role boundaries。

## Known Limits

P95 target of 12 seconds remains unmet：c1 total P95 为 15.930 seconds，c5 total P95 为 74.465 seconds。external provider 与 embedding-provider time 与 project-controlled timing 分开报告。retrieval comparison findings 只适用于 recorded corpus、query set、model identity、load、host envelope 与 observation window；不建立 universal retrieval winner。

release 不要求 continuously hosted public demo。public evidence 排除 credentials、runtime configuration、private questions、complete answers、source excerpts、host details 与 operational credentials。

## Verification Links

- [Bilingual project narrative](../../README.zh-CN.md)
- [Bilingual Interview Dossier](../portfolio/INTERVIEW-DOSSIER.zh-CN.md)
- [Public Evidence Bundle report](../../public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md)
- [Bundle manifest](../../public-evidence/releases/portfolio-release-candidate-01/manifest.json)
- [Bundle validator contract](../../public-evidence/contract/README.zh-CN.md)
