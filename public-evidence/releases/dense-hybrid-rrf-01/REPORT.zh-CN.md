# Dense and Hybrid RRF Comparison Evidence（Dense 与 Hybrid RRF 对比证据）

## Summary（摘要）

本证据包记录 `sparse_bm25`、`dense` 与 `hybrid_rrf` 在同一冻结 Project-Derived Corpus 和
Evaluation Query Set 上的一次受控 Evaluation Retriever 对比。它不改变或重命名生产 Migration
Retrieval；后者仍是 availability/fallback 行为，不是 Hybrid Retrieval。

## Conditions（条件）

- corpus_id `project-derived-corpus`，版本 1.0.0，sha256 `894d10004f25a2a6462870c1a869586ca83dfc28f0407f75f8cc33d9757ac39d`
- query_set_id `evaluation-query-set`，版本 1.0.0，sha256 `a468a1af9fc0c7b979ec1c5dfa8874a5ad27e25bcc79de1e0142564db992df20`
- QA Chunking：500 字符、50 字符重叠；输出深度：3、5、10
- active embedding identity：`Qwen/Qwen3-Embedding-8B:8cfc4ef25f497deccfd2eb0d23b4a8bdd7360e588db0ea9b80ffae616f5d82b9`
- `hybrid_rrf`：Sparse BM25 前 20 条与 Dense 前
  20 条候选，按 `chunk_id` 去重，使用 reciprocal-rank fusion
  `k=60`，无 reranker、无 score averaging

## Results（结果）

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

## Failure Cases（失败 Case）

- sparse_bm25 在 top-10 内未覆盖 Gold Evidence 的查询：none
- dense 在 top-10 内未覆盖 Gold Evidence 的查询：none
- hybrid_rrf 在 top-10 内未覆盖 Gold Evidence 的查询：none

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
