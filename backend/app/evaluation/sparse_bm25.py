"""Sparse BM25 with Literal-Preserving Tokenization.

Sparse BM25 is the genuine lexical retrieval baseline of the Evaluation
Retriever. It is distinct from the production Lexical Heuristic (word,
substring, and bigram overlap) and is never used to change Migration Retrieval.

Literal-Preserving Tokenization segments Chinese prose with jieba while keeping
configuration keys, paths, status codes, versions, model names, and
underscore-bearing identifiers as whole tokens (CONTEXT: Literal-Preserving
Tokenization).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jieba

_PATH_RE = re.compile(r"(?:/[A-Za-z0-9_.{}(),;=\-+]+)+")
_MODEL_NAME_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\b")
_VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b")
_CONFIG_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_SNAKE_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+_?\b")
_CAMEL_IDENTIFIER_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[A-Z][a-z0-9]+){1,}\b")
_HYPHENATED_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")

_LITERAL_PATTERNS = (
    _MODEL_NAME_RE,
    _PATH_RE,
    _VERSION_RE,
    _CONFIG_KEY_RE,
    _SNAKE_IDENTIFIER_RE,
    _CAMEL_IDENTIFIER_RE,
    _HYPHENATED_RE,
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """Merged spans of the literal classes that must stay whole tokens."""
    spans: list[tuple[int, int]] = []
    for pattern in _LITERAL_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(text))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _segment(text: str) -> list[str]:
    """Segment a non-literal fragment: jieba for Chinese-bearing text, word regex otherwise."""
    if _CJK_RE.search(text):
        return [token for token in jieba.cut(text) if token.strip() and any(ch.isalnum() for ch in token)]
    return _WORD_RE.findall(text)


class LiteralPreservingTokenizer:
    """Tokenizer that segments Chinese prose while preserving literal identifiers."""

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        cursor = 0
        for start, end in _protected_spans(text):
            if start > cursor:
                tokens.extend(_segment(text[cursor:start]))
            tokens.append(text[start:end])
            cursor = end
        if cursor < len(text):
            tokens.extend(_segment(text[cursor:]))
        return tokens


@dataclass(frozen=True)
class Bm25Chunk:
    chunk_id: str
    document: str
    chunk_index: int
    content: str


class SparseBm25Index:
    """Deterministic BM25 (Lucene variant, k1 = 1.5, b = 0.75) over fixed chunks.

    Term matching is case-folded so literal variants such as ``MILVUS_URI`` and
    ``milvus_uri`` match each other; the original token shape is preserved for
    reporting.
    """

    def __init__(
        self,
        chunks: Sequence[Bm25Chunk],
        *,
        tokenizer: LiteralPreservingTokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = list(chunks)
        self._tokenizer = tokenizer or LiteralPreservingTokenizer()
        self._k1 = k1
        self._b = b
        self._doc_terms: list[dict[str, int]] = []
        self._lengths: list[int] = []
        for chunk in self._chunks:
            tokens = [token.casefold() for token in self._tokenizer.tokenize(chunk.content)]
            self._lengths.append(len(tokens))
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self._doc_terms.append(term_freq)
        self._doc_count = len(self._chunks)
        self._avgdl = sum(self._lengths) / self._doc_count if self._doc_count else 0.0
        self._df: dict[str, int] = {}
        for terms in self._doc_terms:
            for term in terms:
                self._df[term] = self._df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        length = self._lengths[doc_index]
        if length == 0:
            return 0.0
        terms = self._doc_terms[doc_index]
        score = 0.0
        for term in query_terms:
            freq = terms.get(term, 0)
            if freq == 0:
                continue
            denominator = freq + self._k1 * (1.0 - self._b + self._b * length / self._avgdl)
            score += self._idf(term) * (freq * (self._k1 + 1.0)) / denominator
        return score

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = [token.casefold() for token in self._tokenizer.tokenize(query)]
        if not query_terms:
            return []
        scored = [(self._score(query_terms, index), index) for index in range(self._doc_count)]
        scored.sort(
            key=lambda pair: (
                -pair[0],
                self._chunks[pair[1]].chunk_index,
                self._chunks[pair[1]].chunk_id,
            )
        )
        results: list[dict[str, Any]] = []
        for score, index in scored:
            if score <= 0.0:
                continue
            chunk = self._chunks[index]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document": chunk.document,
                    "chunk_index": chunk.chunk_index,
                    "score": round(score, 4),
                    "content_preview": chunk.content[:160],
                }
            )
            if len(results) >= top_k:
                break
        return results
