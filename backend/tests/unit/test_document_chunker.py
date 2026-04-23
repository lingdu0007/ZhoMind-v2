from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from app.documents.chunker import CHUNK_STRATEGY_PRESETS, chunk_document
from app.documents.types import ChunkRecord, ParsedDocument


def test_chunk_record_is_dataclass() -> None:
    assert is_dataclass(ChunkRecord)


def test_chunk_document_uses_supported_presets_and_metadata_keys() -> None:
    parsed = ParsedDocument(
        source_file="guide.txt",
        file_type="txt",
        text=("0123456789 " * 400).strip(),
    )

    for strategy in ("general", "paper", "qa"):
        chunks = chunk_document(parsed, strategy=strategy)

        assert chunks
        preset = CHUNK_STRATEGY_PRESETS[strategy]
        for index, chunk in enumerate(chunks):
            assert chunk.chunk_index == index
            assert chunk.metadata["source_file"] == "guide.txt"
            assert chunk.metadata["strategy"] == strategy
            assert chunk.metadata["chunk_size"] == preset["chunk_size"]
            assert chunk.metadata["chunk_overlap"] == preset["chunk_overlap"]
            assert chunk.metadata["input_length"] == len(parsed.text)


def test_chunk_index_is_stable_for_same_input() -> None:
    parsed = ParsedDocument(
        source_file="stable.md",
        file_type="md",
        text=("lorem ipsum dolor sit amet " * 300).strip(),
    )

    first = chunk_document(parsed, strategy="general")
    second = chunk_document(parsed, strategy="general")

    assert [chunk.chunk_index for chunk in first] == [chunk.chunk_index for chunk in second]
    assert [chunk.content for chunk in first] == [chunk.content for chunk in second]


def test_chunk_document_preserves_whitespace_at_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = "whitespace_fidelity"
    monkeypatch.setitem(CHUNK_STRATEGY_PRESETS, strategy, {"chunk_size": 5, "chunk_overlap": 2})

    parsed = ParsedDocument(
        source_file="spaces.txt",
        file_type="txt",
        text="abc  def",
    )

    chunks = chunk_document(parsed, strategy=strategy)

    assert [chunk.content for chunk in chunks] == ["abc  ", "  def", "ef"]
