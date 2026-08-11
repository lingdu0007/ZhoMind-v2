"""Sparse BM25 Literal-Preserving Tokenization and scoring tests.

Sparse BM25 is the true lexical retrieval baseline exposed only through the
Evaluation Retriever's ``retrieval-evidence evaluate`` command. It is distinct
from the production Lexical Heuristic and is never used to change Migration
Retrieval. Literal-Preserving Tokenization segments Chinese prose with jieba
while keeping configuration keys, paths, status codes, versions, model names,
and underscore-bearing identifiers as whole tokens.
"""

from __future__ import annotations

from app.evaluation.sparse_bm25 import Bm25Chunk, LiteralPreservingTokenizer, SparseBm25Index


def test_tokenizer_segments_chinese_prose_into_multiple_tokens() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("QA Chunking 的 chunk_size 和 chunk_overlap 分别是多少？")
    assert "chunk_size" in tokens
    assert "chunk_overlap" in tokens
    assert "QA" in tokens
    assert "Chunking" in tokens
    # Chinese prose is actually segmented: the sentence yields several tokens.
    assert len(tokens) >= 6


def test_tokenizer_keeps_configuration_keys_whole() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("DENSE_EMBEDDING_DIM 的默认值为 1024。")
    assert "DENSE_EMBEDDING_DIM" in tokens
    assert "1024" in tokens
    assert not any("DENSE" in token and token != "DENSE_EMBEDDING_DIM" for token in tokens)


def test_tokenizer_keeps_paths_whole() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("通过 POST /api/v1/documents/{document_id}/publish 发布。")
    assert "/api/v1/documents/{document_id}/publish" in tokens
    assert "POST" in tokens


def test_tokenizer_keeps_versions_and_status_codes_whole() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("corpus version 1.0.0 与 HTTP 404 状态码。")
    assert "1.0.0" in tokens
    assert "404" in tokens


def test_tokenizer_keeps_model_names_whole() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("嵌入模型为 Qwen/Qwen3-Embedding-8B。")
    assert "Qwen/Qwen3-Embedding-8B" in tokens


def test_tokenizer_keeps_snake_case_identifiers_whole() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("前缀 document_chunks_ 加指纹。")
    assert "document_chunks_" in tokens
    tokens = tokenizer.tokenize("lexical_scope 为 full_published_live。")
    assert "full_published_live" in tokens


def test_tokenizer_keeps_camel_case_identifiers_whole() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("DenseEmbeddingContract 处于 active 状态。")
    assert "DenseEmbeddingContract" in tokens


def test_tokenizer_filters_whitespace_and_punctuation() -> None:
    tokenizer = LiteralPreservingTokenizer()
    tokens = tokenizer.tokenize("  你好，世界！  ")
    assert tokens == ["你好", "世界"]
    assert all(token.strip() for token in tokens)


def test_tokenizer_is_reusable_across_documents() -> None:
    tokenizer = LiteralPreservingTokenizer()
    first = tokenizer.tokenize("chunk_size 500，chunk_overlap 50。")
    second = tokenizer.tokenize("chunk_size 500，chunk_overlap 50。")
    assert first == second


def test_bm25_ranks_the_chunk_containing_query_terms_first() -> None:
    chunks = [
        Bm25Chunk(chunk_id="c1", document="doc.md", chunk_index=0, content="ZhoMind-v2 支持 Evidence-Gated Answer。"),
        Bm25Chunk(chunk_id="c2", document="doc.md", chunk_index=1, content="分块预置策略定义在 CHUNK_STRATEGY_PRESETS。"),
    ]
    index = SparseBm25Index(chunks)
    results = index.search("CHUNK_STRATEGY_PRESETS 预置策略", top_k=2)
    # Only the chunk sharing query terms scores above zero; the unrelated chunk
    # is filtered out rather than returned with a zero score.
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["score"] > 0.0


def test_bm25_returns_no_candidates_when_no_term_matches() -> None:
    index = SparseBm25Index([Bm25Chunk(chunk_id="c1", document="doc.md", chunk_index=0, content="与查询完全无关的中文内容。")])
    assert index.search("milvus vector search", top_k=5) == []


def test_bm25_is_case_insensitive_across_literal_forms() -> None:
    chunks = [
        Bm25Chunk(chunk_id="c1", document="doc.md", chunk_index=0, content="MILVUS_URI 非空时合约 active。"),
        Bm25Chunk(chunk_id="c2", document="doc.md", chunk_index=1, content="发布生命周期与 Candidate Build。"),
    ]
    index = SparseBm25Index(chunks)
    results = index.search("milvus_uri 非空", top_k=1)
    assert results[0]["chunk_id"] == "c1"


def test_bm25_ties_are_broken_deterministically_by_chunk_order() -> None:
    chunks = [
        Bm25Chunk(chunk_id="c3", document="doc.md", chunk_index=2, content="相同的 token 文本。"),
        Bm25Chunk(chunk_id="c1", document="doc.md", chunk_index=0, content="相同的 token 文本。"),
    ]
    index = SparseBm25Index(chunks)
    results = index.search("相同的 token 文本", top_k=2)
    assert [item["chunk_id"] for item in results] == ["c1", "c3"]
