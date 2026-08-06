# 嵌入与稠密向量索引

ZhoMind-v2 通过 Qwen Embedding 集成把文档块编码为稠密向量，并写入 Milvus 向量库用于稠密检索。嵌入相关的配置项包括 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`DENSE_EMBEDDING_DIM` 与 `MILVUS_URI`。

## 嵌入合约

`DenseEmbeddingContract` 由配置派生：当 `EMBEDDING_API_KEY` 已配置、`EMBEDDING_BASE_URL` 与 `EMBEDDING_MODEL` 非空、`DENSE_EMBEDDING_DIM` 大于 0 且 `MILVUS_URI` 非空时，合约处于 active 状态。

嵌入合约指纹 `build_embedding_contract_fingerprint` 计算 `embedding_base_url`、`embedding_model`、`dense_embedding_dim` 三个维度的 JSON 规范化串的 sha256 摘要，用于标识一个确定的嵌入契约。`DENSE_EMBEDDING_DIM` 的默认值为 1024。

Milvus collection 名称由指纹派生：前缀 `document_chunks_` 加上指纹的十六进制字符。Milvus 中每一行的标识为 `document_id:generation:chunk_index:content_sha256`，行字段包括 `document_id`、`generation`、`chunk_index`、`content_sha256`、`embedding_contract_fingerprint` 与向量本身。

## 稠密检索

检索时先对查询文本调用嵌入 provider 得到向量，再对 `collection_name` 执行批量向量搜索，`batch_size` 取 `max(top_k, top_k * 4)`。候选行按 `document_id`、`generation`、`chunk_index` 水合为文档块，直到达到 `top_k`。

嵌入 provider 通过扩展注册表解析，默认名称为 `embedding-default`；未注册时使用 `OpenAIEmbeddingProvider`，即通过 OpenAI 兼容接口调用 Qwen Embedding 服务。本地 Milvus 的默认地址为 `http://localhost:19530`。

构建文档块时，稠密索引服务对每个 chunk 调用嵌入 provider 得到向量，并写入与嵌入合约指纹对应的 Milvus collection。`/ops/dense-backfill` 与 `/ops/dense-reconcile` 提供稠密维护操作，用于在嵌入配置变化后重建或对齐稠密索引。

## 来源

- 嵌入配置项与默认值：`backend/app/common/config.py` 的 `Settings`
- 嵌入合约与指纹：`backend/app/rag/dense_contract.py`
- 稠密索引写入：`backend/app/service/dense_index_service.py` 的 `DenseIndexService`
- 稠密检索与水合：`backend/app/service/document_retrieval_service.py` 的 `_dense_search`
- Milvus 行结构与 collection 命名：`backend/app/infra/milvus_document_index.py`
- 嵌入 provider 解析：`backend/app/rag/langchain_embedding_providers.py` 与扩展注册表
