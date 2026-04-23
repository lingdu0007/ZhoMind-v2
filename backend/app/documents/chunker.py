from __future__ import annotations

from app.documents.types import ChunkRecord, ParsedDocument

CHUNK_STRATEGY_PRESETS: dict[str, dict[str, int]] = {
    "general": {"chunk_size": 1000, "chunk_overlap": 100},
    "paper": {"chunk_size": 1500, "chunk_overlap": 150},
    "qa": {"chunk_size": 500, "chunk_overlap": 50},
}


def _iter_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_document(parsed_document: ParsedDocument, *, strategy: str = "general") -> list[ChunkRecord]:
    preset = CHUNK_STRATEGY_PRESETS.get(strategy, CHUNK_STRATEGY_PRESETS["general"])
    chunk_size = preset["chunk_size"]
    chunk_overlap = preset["chunk_overlap"]

    pieces = _iter_chunks(parsed_document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    input_length = len(parsed_document.text)

    return [
        ChunkRecord(
            chunk_index=index,
            content=piece,
            metadata={
                "source_file": parsed_document.source_file,
                "strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "input_length": input_length,
            },
        )
        for index, piece in enumerate(pieces)
    ]
