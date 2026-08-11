# ZhoMind-v2

[English](README.md) | [Interview Dossier](docs/portfolio/INTERVIEW-DOSSIER.zh-CN.md) | [Release Notes](docs/releases/PORTFOLIO-RELEASE-v1.0.0.zh-CN.md) | [Public Evidence Bundle](public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md)

ZhoMind-v2 是一个首发版本的 Team-Shared Knowledge Base 应用，处理需要把回答追溯到已批准来源材料的场景。管理员控制成员准入、显式构建和发布来源版本，并管理唯一的 Approved Generation Provider。Knowledge User 要么获得带可检查来源摘要的 Evidence-Gated Answer，要么获得明确的封闭结果。项目不要求持续托管的公网 Demo 才能审阅。

![脱敏后的中文 application entry](docs/assets/portfolio-release-chinese-entry.png)

该截图来自现有中文 application entry 的有界 static render：不包含 private content、credentials、host information、raw diagnostics 或 production data。

## Portfolio Release

已接受的代码 candidate 为 `91753f1c1ff6fc07bc262dfa50fb719a63210e0b`。其经过审阅、非敏感的 [Public Evidence Bundle](public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md) 名为 `portfolio-release-candidate-01`；[manifest](public-evidence/releases/portfolio-release-candidate-01/manifest.json) 将 run identities、冻结的 corpus/query hashes、artifact hashes 与 declared limits 绑定到该 revision。

[Interview Dossier](docs/portfolio/INTERVIEW-DOSSIER.zh-CN.md) 将每一条 portfolio claim 映射到 source file、已接受测试或 bundle artifact。[Release Notes](docs/releases/PORTFOLIO-RELEASE-v1.0.0.zh-CN.md) 汇总开发序列和发布边界，审阅者无需重建完整 commit history。

## User Problem

当来源资格与回答支撑必须可审阅时，仅有“已上传来源的知识库”并不足够。一次成功 build 不会自动变得可搜索，广泛的 retrieval diagnostic 不是生成回答已经被支撑的证明，provider error 也不能变成无引用回答。

ZhoMind-v2 将这些状态转换显式化。管理员只有在 inspection 后才能发布来源 generation。Knowledge User 会看到受支撑回答的 source identity 与有界 excerpt。没有有效 Answer Evidence Set 时，产品返回 Insufficient Evidence Reply，且不调用 generation provider。

## Architecture

该 release 将五个边界分开。名称是有意区分的：它们回答不同问题，不能相互替代。

| Boundary | Purpose | It does not prove |
| --- | --- | --- |
| Migration Retrieval | dense coverage 或 dense query availability 不完整时的 availability behavior；可回退至现有 Lexical Heuristic。 | quality-oriented Sparse BM25 或 Hybrid Retrieval quality。 |
| Evaluation Retriever | 在一个冻结的 Project-Derived Corpus 和 Evaluation Query Set 上进行的 experiment-only comparison；评估 `sparse_bm25`、`dense` 与 `hybrid_rrf`。 | production routing 或回答可被支撑。 |
| Direct Retrieval Diagnostic | 可暴露更广泛 ordered candidate set 的 administrator retrieval-health surface。 | citation selection、Answer Evidence Set membership 或 production answer path。 |
| Evidence-Gated Answer Execution | 选择单个封闭 Answer Execution Outcome 的 transport-independent execution boundary。 | diagnostic candidate 可以被引用，或 provider 可在无 evidence 时回答。 |
| Production Answer Acceptance | 通过 normal、SSE 与 history projections 对 authenticated `/chat` 执行 acceptance。 | 单独的 direct retrieval result。 |

公开架构因此是一条 product flow，而不是 feature list：

```text
Candidate Build -> explicit Published Knowledge Version -> retrieval
                                                     -> Answer Evidence Set
                                                     -> Evidence-Gated Answer Execution
                                                     -> normal, SSE, and history projections
```

文档化的 production checks 断言最终 user-facing path；Evaluation Retriever 与 Direct Retrieval Diagnostic 保持独立 evidence surfaces。

## Evidence-Gated Answer

Evidence-Gated Answer Execution 从 final retrieval order 形成一个 immutable Answer Evidence Set。按首发策略，它最多将前三个有效 passage 提供给 Approved Generation Provider，且每个 item 都携带一个 immutable Evidence Excerpt Snapshot。prompt、persisted citation、normal response、SSE response 和 history projection 都引用同一个 set 与 snapshot。

封闭 outcomes 保持 fail-closed behavior：

- `evidence_gated_answer` 要求来自 Published Knowledge Version 的非空 Answer Evidence Set，并展示 title、version 或 publication time，以及可打开的相关 excerpt。
- `insufficient_evidence_reply` 没有 citation，且不调用 generation provider。
- `generation_unavailable` 在 Approved Generation Provider 不可用时没有 generation fallback route。
- `non_knowledge_base_reply` 是 allowlisted social prompts 的狭窄 pre-retrieval exception，不是无引用 knowledge answer。

这是该 release 的核心 invariant：Direct Retrieval Diagnostic 可以包含更广泛 candidates，但绝不选择或修复 Answer Evidence Set。

## Publication Lifecycle

首发版本采用 Manual Publication Workflow。管理员上传或替换材料、开始 build、检查 Candidate Build，并显式 publish 已批准 generation。只有 Published Knowledge Version 能参与后续 retrieval 和 citations。withdrawal 会立即把该来源从未来 retrieval 中移除；历史 projections 保留 identity 和 withdrawal notice，而不是继续暴露此前 excerpt。

团队模型刻意保持狭窄：一个 Team-Shared Knowledge Base、invitation-gated Knowledge User admission、Bootstrap Administrator setup、administrator promotion，以及 deactivation 而非立即 account deletion。它不是 private-document vault，也不是 document-level access-control system。

## Evaluation Method

[evaluation inputs](evaluation/README.md) 冻结 Chinese-first Project-Derived Corpus 与 16 条 Evaluation Query Set：四条 exact-constraint、四条 semantic-paraphrase、四条 combined-condition、四条 Boundary Query。可回答 queries 测量 `Evidence Recall@3/@5/@10`、`Context Precision@3/@5/@10`、first Gold Evidence rank 与 retrieval duration。Boundary Query diagnostics 不计入 answerable-query aggregates。

已接受的 comparison 在 Sparse BM25、Dense 与 Hybrid RRF 间保持 corpus hash、query-set hash、QA Chunking 500/50、output depth 20 与 model identity 一致。Hybrid RRF 按 `chunk_id` 去重并使用 rank-only `k=60`；它既不加入 score averaging，也不加入 reranking。记录的 run 中，三种 modes 有相同的 recall 和 context-precision values，而 Dense 与 Hybrid RRF 有更长 retrieval duration。这一条记录的结果不建立 universal winner，也不重命名 Migration Retrieval。

## Security Boundary

Published source content 被视为 untrusted data。Evidence-Gated Answer Execution 将 system policy、normalized user question 和 immutable Evidence Excerpt Snapshots 放入分离的 structured regions；provider-visible data 排除 credentials、runtime configuration、private history 与 administrator diagnostics。

已接受的有界 Prompt Injection observation 覆盖 instruction override、configuration or secret extraction、forged-source instructions 和 pressure to answer without evidence。公开记录只包含 normalized case identity、pass/fail、closed outcome、citation count 和 failure classification。这是对已记录 cases 与 conditions 的证据，不是 general security guarantee。

## Production Acceptance

该 accepted candidate 的 persistent-stack acceptance 记录：authenticated normal answer、SSE、history、insufficient evidence、generation unavailable with zero fallback hops、withdrawal、deactivation、service health、两次 clean-start Retrieval Smokes，以及 Administrator/Knowledge User browser role boundaries 均通过。支撑该结论的公开 section 是 [production acceptance](public-evidence/releases/portfolio-release-candidate-01/sections/production-acceptance.json)；它有意只包含 bounded outcomes，不包含 conversations、prompts、source excerpts 或 operations data。

deterministic PR Gate 可在没有 provider credentials 的情况下复现。已接受的 remote run 记录 backend Ruff、Pyright 与 `355 passed, 1 skipped`；frontend unit tests 加 production build；以及 disposable-API browser acceptance `49/49`。live-Milvus test 是 explicit opt-in，skip 时绝不表述为 live evidence。

## Performance And Limits

性能是 observation，不是 service-level claim。已接受的 20-request runs 在 c1 和 c5 均报告 zero errors。c1 total P95 为 15.930 seconds，c5 total P95 为 74.465 seconds；P95 target of 12 seconds remains unmet。generation-provider 与 embedding-provider durations 与 project-controlled timing 分开报告；后者的 observed 40-sample regression envelope threshold 为 6128.299 ms。

结果只适用于已记录的 corpus、query set、model identity、load、host envelope 和 observation window。release 还具有明确的 first-release capacity boundaries：25 active members、five concurrent Q&A requests、500 published knowledge versions/documents、25 MiB uploads 与 one document-build worker。达到边界时需显式 scale-up 或 archive action。

## Local Reproduction

公开 deterministic gate 使用 committed locks，不需要 provider credentials、persistent service 或 private host access：

```bash
cd backend
uv sync --frozen
uv run ruff check .
uv run pyright
uv run pytest -q -ra

cd ../frontend
npm ci
npm run test:unit
npm run build
npx playwright install chromium
npm run test:browser

cd ..
python3 scripts/check-docs-parity.py
python3 scripts/validate-evidence-bundle.py --all
python3 scripts/scan-secrets.py
python3 scripts/verify-portfolio-release.py --expected-source-revision 91753f1c1ff6fc07bc262dfa50fb719a63210e0b
```

真实 embedding、Milvus、provider、Compose 与 persistent-stack acceptance 都是刻意分离的 remote-only checks。请通过 [Public Evidence Bundle](public-evidence/releases/portfolio-release-candidate-01/REPORT.zh-CN.md) 审阅其 recorded conditions 与 limits，不要把本地 deterministic run 当作 live-provider 或 production evidence。
