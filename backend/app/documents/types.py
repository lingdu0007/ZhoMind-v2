from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedDocument:
    source_file: str
    file_type: str
    text: str


@dataclass
class ChunkRecord:
    chunk_index: int
    content: str
    metadata: dict[str, Any]
