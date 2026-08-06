# 检索与分块策略

ZhoMind-v2 的文档构建与检索依赖一套固定的分块策略和两种检索制度：Migration Retrieval 与 Evaluation Retriever。

## 分块策略

文档按字符滑窗切分，步长为 `chunk_size - chunk_overlap`。预置策略定义在 `CHUNK_STRATEGY_PRESETS`：

- `general`：chunk_size 1000，chunk_overlap 100，文档上传默认策略；
- `paper`：chunk_size 1500，chunk_overlap 150；
- `qa`：chunk_size 500，chunk_overlap 50，即 QA Chunking 的固定策略。

每个 chunk 记录 `source_file`、`strategy`、`chunk_size`、`chunk_overlap`、`input_length` 元数据。DocumentChunk 以 `document_id`、`generation` 与 `chunk_index` 唯一标识，`chunk_id` 为随机 UUID。

QA Chunking 是 Retrieval Evidence Baseline 的第一轮固定策略：500 字符、50 字符重叠，在所有检索模式对比中保持不变。任何分块消融实验都必须是后续独立标注的实验。

## Migration Retrieval

Migration Retrieval 是面向可用性的检索制度，用于稠密索引覆盖或稠密查询可用性不完整的时期，它不是质量导向的 Hybrid Retrieval。当 `DenseEmbeddingContract` 处于 active 状态时，检索策略为 `dense_plus_lexical_migration`；否则为 `sparse_only`。

词法检索使用现有的 Lexical Heuristic 打分器：全文紧凑形式包含加 5.0 分，每个 token 重叠加 1.0 分，bigram 重叠最多 6 个、每个加 0.8 分。该打分器是自定义的 word、substring、bigram 重叠评分器，保留用于稳定性和迁移对比，绝不标记为 BM25。

当稠密查询失败时，Migration Retrieval 回退到全量已发布实时语料：`lexical_scope` 为 `full_published_live`，不施加 `not_dense_ready_published` 分支中的 200 候选上限与 dense-ready 过滤。检索结果携带归一化的失败与回退痕迹：`fallback_used`、`lexical_scope`、`provider_error`（含 `code`、`message`、`type`），以及 `dense_candidate_count`、`dense_hydrated_count`、`lexical_candidate_count`、`merged_count` 等有界计数。RAG 执行图的 retrieve 步骤会把这些字段写入 provider trace。

## Evaluation Retriever 合约

Evaluation Retriever 是仅用于实验的检索边界，不改变 Migration Retrieval 行为。第一轮支持三种质量模式：`sparse_bm25`、`dense`、`hybrid_rrf`。所有质量模式在同一 Project-Derived Corpus、同一 Evaluation Query Set、同一 QA Chunking 与同一输出深度下对比，测试不得预设某一模式必须获胜。

Sparse BM25 使用 Literal-Preserving Tokenization：中文散文用 jieba 分词，配置键、路径、状态码、版本号、模型名与带下划线的标识符保留为完整 token。它区别于现有的 Lexical Heuristic。

Hybrid Retrieval 使用固定的 RRF Fusion 合约：Sparse BM25 与 Dense 各取前 20 个候选，按 reciprocal rank 融合，`k = 60`，按 `chunk_id` 去重，第一轮不加入 reranker。

## 来源

- 分块预置与滑窗：`backend/app/documents/chunker.py` 的 `CHUNK_STRATEGY_PRESETS`
- chunk 唯一性与元数据：`backend/app/model/document.py` 的 `DocumentChunk`
- Migration Retrieval 与回退痕迹：`backend/app/service/document_retrieval_service.py` 的 `MixedModeDocumentRetrieverService.retrieve`
- 归一化 provider 错误：`backend/app/rag/interfaces.py` 的 `ProviderExecError`
- retrieve 步骤 trace：`backend/app/rag/runtime/default_nodes.py`
- Evaluation Retriever 与 RRF 合约、Sparse BM25 定义：CONTEXT 领域词表、`.scratch/zhomind-retrieval-evidence/PRD.md`
