# Sparse BM25 Evidence Package（稀疏 BM25 证据包）

## Summary（摘要）

本证据包记录 ZhoMind-v2 Portfolio Release 的首次质量导向检索评估：真正的 Sparse BM25（使用
jieba 的 Literal-Preserving Tokenization）通过 Evaluation Retriever 的
`retrieval-evidence evaluate` 命令，在冻结的 Project-Derived Corpus 与 16 条 Evaluation
Query Set 上运行。它是 Retrieval Evidence Baseline 的 tracer bullet；本包不做任何 Dense、
Hybrid RRF、回答质量或普遍提升的声明。

## Method（方法）

### Corpus（语料）

- corpus_id `project-derived-corpus`，版本 1.0.0，sha256 `894d10004f25a2a6462870c1a869586ca83dfc28f0407f75f8cc33d9757ac39d`
- 六份 Chinese-First Markdown 文档；英文标识符、路径、状态码、版本号与模型名保留为字面量

### Query set（查询集）

- query_set_id `evaluation-query-set`，版本 1.0.0，sha256 `a468a1af9fc0c7b979ec1c5dfa8874a5ad27e25bcc79de1e0142564db992df20`
- 16 条查询：四条 `exact_constraint`、四条 `semantic_paraphrase`、四条 `combined_condition`
  与四条 `boundary` 查询；12 条 answerable 查询记录 `required_claims` 与 `gold_evidence`

### Chunking and tokenization（分块与分词）

- QA Chunking 固定为 500 字符、50 字符重叠（`CHUNK_STRATEGY_PRESETS` 的 `qa` 预置）
- Sparse BM25 使用 Literal-Preserving Tokenization：中文散文用 jieba 分词；配置键、路径、
  状态码、版本号、模型名与带下划线的标识符保留为完整 token。BM25 使用 k1 = 1.5 与 b = 0.75。

### Metrics（指标）

Answerable-query 聚合是 12 条 answerable 查询的宏平均：`evidence_recall@3`、
`evidence_recall@5`、`evidence_recall@10`、`context_precision@3`、`context_precision@5`、
`context_precision@10`、`first_gold_rank` 与 `retrieval_duration_ms`。当候选 chunk 的字符区间
与 Gold Evidence passage 区间的重叠达到该 passage 长度的至少 50% 时，判定该 passage 被覆盖。
若某 answerable 查询在 top 10 内没有覆盖任何 Gold Evidence 的候选，`first_gold_rank` 记为 11。
Boundary Query Diagnostics 不计入这些聚合。

## Results（结果）

### Evidence Recall and Context Precision（证据召回与上下文精确率）

| metric | value |
| --- | --- |
| evidence_recall@3 | 1.0000 |
| evidence_recall@5 | 1.0000 |
| evidence_recall@10 | 1.0000 |
| context_precision@3 | 0.3333 |
| context_precision@5 | 0.2000 |
| context_precision@10 | 0.1000 |
| first_gold_rank | 1.1667 |
| retrieval_duration_ms | 0.3458 |

### First Gold Evidence Rank（首个 Gold 证据排名）

- first_gold_rank: 1.1667（1 基；11 表示 top 10 内没有
  覆盖 Gold Evidence 的候选）

### Retrieval Duration（检索耗时）

- retrieval_duration_ms: 0.35（12 条 answerable
  查询的每条查询进程内 Sparse BM25 检索平均耗时）

## Boundary Query Diagnostics（边界查询诊断）

四条 Boundary Query（`boundary-lexical-is-bm25`、`boundary-chunking-300`、
`boundary-hybrid-winner`、`boundary-production-ports`）不计入 answerable-query 聚合，并在
`sections/retrieval.json` 中保留语料推导出的 insufficient、conflicting 或 stale 解释。

## Run Conditions（运行条件）

- source revision: `ece8f00f8d3aba65332aeb780a665820063a8710`
- run id: `0434b82787cf4173ba567a2c884906ba`
- mode: sparse_bm25
- output depths: 3、5、10
- tokenizer: literal_preserving_jieba

## Limits（适用限制）

- 指标仅描述已记录的运行条件，不推广到其他语料、模型、分块策略或语言。
- 本证据包只包含 Sparse BM25 证据；不做 Dense、Hybrid RRF、回答质量或普遍提升的声明。
- `retrieval_duration_ms` 不包括语料分块与索引构建耗时。
