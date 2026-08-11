from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.documents.types import ChunkRecord, ParsedDocument

CHUNK_STRATEGY_PRESETS: dict[str, dict[str, int]] = {
    "general": {"chunk_size": 1000, "chunk_overlap": 100},
    "paper": {"chunk_size": 1500, "chunk_overlap": 150},
    "qa": {"chunk_size": 500, "chunk_overlap": 50},
}
_AGENT_CHUNK_SIZE = 1200


@dataclass(frozen=True)
class _MarkdownSection:
    heading: str
    section_id: str
    body: str


def _section_id(heading: str) -> str:
    label = heading.lstrip("#").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", label).strip("-")
    return normalized or "section"


def _markdown_sections(text: str) -> list[_MarkdownSection]:
    sections: list[_MarkdownSection] = []
    heading = "# Entry"
    body_lines: list[str] = []
    in_fence = False

    def append_section() -> None:
        body = "\n".join(body_lines).strip()
        if body or heading != "# Entry":
            sections.append(_MarkdownSection(heading=heading, section_id=_section_id(heading), body=body))

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            body_lines.append(line)
            continue
        if not in_fence and re.match(r"^#{1,6}\s+\S", line):
            append_section()
            heading = line.strip()
            body_lines = []
            continue
        body_lines.append(line)
    append_section()
    return sections


def _markdown_blocks(body: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in body.splitlines():
        stripped = line.lstrip()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        if not in_fence and not line.strip() and current:
            blocks.append("\n".join(current).strip())
            current = []
            continue
        current.append(line)
        if is_fence:
            in_fence = not in_fence
            if not in_fence:
                blocks.append("\n".join(current).strip())
                current = []
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _split_sentences(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars or text.startswith(("```", "~~~")):
        return [text]
    sentences = [item.strip() for item in re.split(r"(?<=[。！？.!?])\s*", text) if item.strip()]
    if len(sentences) <= 1:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current)
            current = ""
        if len(sentence) > max_chars:
            pieces.extend(sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars))
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def _split_agent_section(section: _MarkdownSection) -> list[str]:
    max_body_chars = max(200, _AGENT_CHUNK_SIZE - len(section.heading) - 2)
    blocks: list[str] = []
    for block in _markdown_blocks(section.body):
        blocks.extend(_split_sentences(block, max_chars=max_body_chars))

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0
    for block in blocks:
        separator_length = 2 if current else 0
        if current and current_length + separator_length + len(block) > max_body_chars:
            pieces.append("\n\n".join(current))
            current = []
            current_length = 0
        current.append(block)
        current_length += separator_length + len(block)
    if current or not pieces:
        pieces.append("\n\n".join(current))
    return [f"{section.heading}\n\n{piece}".rstrip() for piece in pieces]


def _agent_chunk_metadata(parsed_document: ParsedDocument, *, section: _MarkdownSection) -> dict[str, Any]:
    metadata = parsed_document.metadata
    sources = metadata.get("sources") if isinstance(metadata.get("sources"), list) else []
    primary_source = sources[0] if sources and isinstance(sources[0], dict) else {}
    return {
        "source_file": parsed_document.source_file,
        "strategy": "agent",
        "entry_id": metadata.get("entry_id"),
        "entry_title": metadata.get("title"),
        "domain": metadata.get("domain"),
        "section_id": section.section_id,
        "section_title": section.heading.lstrip("#").strip(),
        "review_status": metadata.get("review_status"),
        "review_date": metadata.get("review_date"),
        "applicable_versions": metadata.get("applicable_versions", []),
        "sources": sources,
        "source_title": primary_source.get("title"),
        "source_authority": primary_source.get("authority"),
        "source_url": primary_source.get("url"),
        "source_version": primary_source.get("version"),
        "source_availability": primary_source.get("availability"),
        "evidence_conflict": metadata.get("evidence_conflict"),
        "approved_summary": metadata.get("approved_summary"),
        "suggested_query": metadata.get("suggested_query"),
    }


def _chunk_agent_document(parsed_document: ParsedDocument) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for section in _markdown_sections(parsed_document.text):
        metadata = _agent_chunk_metadata(parsed_document, section=section)
        for content in _split_agent_section(section):
            chunks.append(ChunkRecord(chunk_index=len(chunks), content=content, metadata=dict(metadata)))
    return chunks


def _iter_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_document(parsed_document: ParsedDocument, *, strategy: str = "general") -> list[ChunkRecord]:
    if strategy == "agent":
        return _chunk_agent_document(parsed_document)

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
