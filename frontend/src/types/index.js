export const defaultConfig = {
  llm_model: 'Qwen/Qwen3-32B',
  embedding_model: 'BAAI/bge-m3',
  rerank_model: 'Qwen/Qwen3-Reranker-8B',
  top_k: 5,
  score_threshold: 0.3,
  hybrid_dense_weight: 0.7,
  hybrid_sparse_weight: 0.3,
  milvus_uri: 'http://localhost:19530',
  redis_url: 'redis://localhost:6379',
  storage_path: './data/bm25',
  silicon_api_key: '',
  callback_url: '',
  rate_limit_per_minute: 60
};
