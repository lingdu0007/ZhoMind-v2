"""Deterministic Sparse BM25 with Literal-Preserving Tokenization."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jieba

_ASCII_LITERAL_START = r"(?<![A-Za-z0-9_])"
_ASCII_LITERAL_END = r"(?![A-Za-z0-9_])"
_PATH_RE = re.compile(_ASCII_LITERAL_START + r"(?:/[A-Za-z0-9_.{}(),;=\-+]+)+" + _ASCII_LITERAL_END)
_MODEL_NAME_RE = re.compile(
    _ASCII_LITERAL_START
    + r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
    + _ASCII_LITERAL_END
)
_VERSION_RE = re.compile(_ASCII_LITERAL_START + r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?" + _ASCII_LITERAL_END)
_CONFIG_KEY_RE = re.compile(_ASCII_LITERAL_START + r"[A-Z][A-Z0-9_]{2,}" + _ASCII_LITERAL_END)
_SNAKE_IDENTIFIER_RE = re.compile(
    _ASCII_LITERAL_START + r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+_?" + _ASCII_LITERAL_END
)
_CAMEL_IDENTIFIER_RE = re.compile(
    _ASCII_LITERAL_START + r"[A-Za-z][A-Za-z0-9]*(?:[A-Z][a-z0-9]+){1,}" + _ASCII_LITERAL_END
)
_HYPHENATED_RE = re.compile(
    _ASCII_LITERAL_START + r"[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)+" + _ASCII_LITERAL_END
)
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
    if _CJK_RE.search(text):
        return [token for token in jieba.cut(text) if token.strip() and any(char.isalnum() for char in token)]
    return _WORD_RE.findall(text)


class LiteralPreservingTokenizer:
    """Segment Chinese prose while keeping technical literals as whole tokens."""

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
    tie_breaker: tuple[str, int, str] | None = None

    @property
    def ranking_tie_breaker(self) -> tuple[str, int, str]:
        return self.tie_breaker or (self.document, self.chunk_index, self.chunk_id)


class SparseBm25Index:
    """Lucene-style BM25 over a fixed corpus with deterministic ordering."""

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
        document_frequency = self._df.get(term, 0)
        return math.log(1.0 + (self._doc_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def _score(self, query_terms: list[str], document_index: int) -> float:
        length = self._lengths[document_index]
        if length == 0:
            return 0.0
        terms = self._doc_terms[document_index]
        score = 0.0
        for term in query_terms:
            frequency = terms.get(term, 0)
            if frequency == 0:
                continue
            denominator = frequency + self._k1 * (1.0 - self._b + self._b * length / self._avgdl)
            score += self._idf(term) * (frequency * (self._k1 + 1.0)) / denominator
        return score

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_terms = [token.casefold() for token in self._tokenizer.tokenize(query)]
        if not query_terms:
            return []
        scored = [(self._score(query_terms, index), index) for index in range(self._doc_count)]
        scored.sort(key=lambda pair: (-pair[0], self._chunks[pair[1]].ranking_tie_breaker))

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
