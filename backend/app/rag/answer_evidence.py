from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlsplit

_CITATION_METADATA_KEYS = (
    "title",
    "publication_version",
    "entry_id",
    "entry_title",
    "domain",
    "section_id",
    "source_title",
    "source_authority",
    "source_url",
    "source_version",
    "review_date",
    "review_status",
    "source_availability",
    "filename",
    "source_file",
    "source",
    "document_name",
    "path",
)
_AGENT_CITATION_KEYS = (
    "entry_id",
    "entry_title",
    "domain",
    "section_id",
    "source_title",
    "source_authority",
    "source_url",
    "source_version",
    "review_date",
)
_UNSAFE_URL_QUERY_PARTS = ("credential", "password", "redirect", "secret", "signature", "token")


def _safe_public_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or hostname == "localhost"
        or hostname.endswith((".local", ".internal"))
        or "." not in hostname
    ):
        return False
    return not any(
        any(part in key.lower() for part in _UNSAFE_URL_QUERY_PARTS)
        for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
    )


def _agent_metadata_is_eligible(metadata: Mapping[str, object]) -> bool:
    if not isinstance(metadata.get("entry_id"), str):
        return True
    if metadata.get("section_id") == "evidence-conflicts-and-unknowns":
        return False
    if metadata.get("review_status") != "approved" or metadata.get("source_availability") != "verified":
        return False
    if metadata.get("evidence_conflict") == "unresolved":
        return False
    if any(not isinstance(metadata.get(key), str) or not str(metadata[key]).strip() for key in _AGENT_CITATION_KEYS):
        return False
    if not _safe_public_url(str(metadata["source_url"])):
        return False
    if metadata.get("section_id") == "version-mapping":
        try:
            reviewed = date.fromisoformat(str(metadata["review_date"]))
        except ValueError:
            return False
        if datetime.now(UTC).date() - reviewed > timedelta(days=90):
            return False
    return True


def _normalize_excerpt(value: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text[:max_chars]


@dataclass(frozen=True)
class AnswerEvidence:
    source_id: str
    document_id: str
    generation: int
    chunk_index: int
    title: str
    publication_version: str
    excerpt: str
    retrieval_source: str | None
    score: float | None
    metadata_items: tuple[tuple[str, str], ...]

    @classmethod
    def from_candidate(cls, candidate: object, *, max_excerpt_chars: int) -> AnswerEvidence | None:
        if not isinstance(candidate, Mapping):
            return None

        raw_source_id = candidate.get("chunk_id") or candidate.get("source_id")
        raw_document_id = candidate.get("document_id")
        generation = candidate.get("generation")
        metadata = candidate.get("metadata")
        if (
            not isinstance(raw_source_id, str)
            or not raw_source_id.strip()
            or not isinstance(raw_document_id, str)
            or not raw_document_id.strip()
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            return None
        if not isinstance(metadata, Mapping):
            return None
        if not _agent_metadata_is_eligible(metadata):
            return None

        raw_title = metadata.get("title")
        raw_publication_version = metadata.get("publication_version")
        if not isinstance(raw_title, str) or not raw_title.strip():
            return None
        if raw_publication_version is not None and not isinstance(raw_publication_version, str):
            return None
        raw_excerpt = candidate.get("content_preview") or candidate.get("content")
        if not isinstance(raw_excerpt, str):
            return None

        source_id = raw_source_id.strip()
        document_id = raw_document_id.strip()
        title = raw_title.strip()
        publication_version = (raw_publication_version or f"v{generation}").strip()
        excerpt = _normalize_excerpt(
            raw_excerpt,
            max_chars=max_excerpt_chars,
        )
        if not title or not publication_version or not excerpt:
            return None

        raw_chunk_index = candidate.get("chunk_index", 0)
        chunk_index = raw_chunk_index if isinstance(raw_chunk_index, int) and not isinstance(raw_chunk_index, bool) else 0
        raw_score = candidate.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool) else None
        retrieval_source = candidate.get("retrieval_source")
        retrieval_source = str(retrieval_source).strip() if isinstance(retrieval_source, str) and retrieval_source.strip() else None

        citation_metadata = {
            key: str(metadata[key]).strip()
            for key in _CITATION_METADATA_KEYS
            if isinstance(metadata.get(key), str) and str(metadata[key]).strip()
        }
        if isinstance(metadata.get("entry_id"), str):
            for legacy_location_key in ("filename", "source_file", "source", "document_name", "path"):
                citation_metadata.pop(legacy_location_key, None)
        citation_metadata["title"] = title
        citation_metadata["publication_version"] = publication_version
        return cls(
            source_id=source_id,
            document_id=document_id,
            generation=generation,
            chunk_index=chunk_index,
            title=title,
            publication_version=publication_version,
            excerpt=excerpt,
            retrieval_source=retrieval_source,
            score=score,
            metadata_items=tuple(citation_metadata.items()),
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "chunk_id": self.source_id,
            "document_id": self.document_id,
            "generation": self.generation,
            "chunk_index": self.chunk_index,
            "content_preview": self.excerpt,
            "metadata": dict(self.metadata_items),
        }
        if self.retrieval_source is not None:
            record["retrieval_source"] = self.retrieval_source
        if self.score is not None:
            record["score"] = self.score
        return record

    def to_source(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "metadata": dict(self.metadata_items),
            "excerpt": self.excerpt,
        }

    def is_agent_entry(self) -> bool:
        return isinstance(dict(self.metadata_items).get("entry_id"), str)

    def to_public_citation(self, citation_id: str) -> dict[str, str]:
        metadata = dict(self.metadata_items)
        return {
            "citation_id": citation_id,
            **{key: metadata[key] for key in _AGENT_CITATION_KEYS if key in metadata},
            "publication_version": self.publication_version,
            "excerpt": self.excerpt,
        }


def select_answer_evidence(
    candidates: list[dict],
    *,
    max_items: int,
    max_excerpt_chars: int,
) -> tuple[AnswerEvidence, ...]:
    selected: list[AnswerEvidence] = []
    for candidate in candidates:
        evidence = AnswerEvidence.from_candidate(candidate, max_excerpt_chars=max_excerpt_chars)
        if evidence is None:
            continue
        selected.append(evidence)
        if len(selected) >= max_items:
            break
    return tuple(selected)


def evidence_summary_from_trace(rag_trace: object) -> dict[str, Any]:
    trace = rag_trace if isinstance(rag_trace, Mapping) else {}
    evidence = trace.get("evidence")
    evidence_items = evidence if isinstance(evidence, (list, tuple)) else []
    sources: list[dict[str, Any]] = []
    for item in evidence_items:
        if not isinstance(item, Mapping):
            continue
        source_id = str(item.get("chunk_id") or item.get("source_id") or "").strip()
        if not source_id:
            continue
        metadata = item.get("metadata")
        source_metadata = (
            {
                key: str(metadata[key]).strip()
                for key in _CITATION_METADATA_KEYS
                if isinstance(metadata, Mapping)
                and isinstance(metadata.get(key), str)
                and str(metadata[key]).strip()
            }
            if isinstance(metadata, Mapping)
            else {}
        )
        if isinstance(source_metadata.get("entry_id"), str):
            citation: dict[str, Any] = {
                "citation_id": f"S{len(sources) + 1}",
                **{key: source_metadata[key] for key in _AGENT_CITATION_KEYS if key in source_metadata},
                "publication_version": source_metadata.get("publication_version") or f"v{item.get('generation', 1)}",
            }
            if item.get("withdrawn") is True:
                citation["withdrawal_notice"] = "This source has been withdrawn."
            else:
                excerpt = str(item.get("content_preview") or item.get("content") or "")
                if not excerpt:
                    continue
                citation["excerpt"] = excerpt
            sources.append(citation)
            continue
        if item.get("withdrawn") is True:
            sources.append(
                {
                    "source_id": source_id,
                    "metadata": source_metadata,
                    "withdrawal_notice": "This source has been withdrawn.",
                }
            )
            continue
        excerpt = str(item.get("content_preview") or item.get("content") or "")
        if not excerpt:
            continue
        sources.append({"source_id": source_id, "metadata": source_metadata, "excerpt": excerpt})

    outcome = trace.get("outcome")
    _gate_value = trace.get("gate")
    gate = _gate_value if isinstance(_gate_value, Mapping) else {}
    if outcome == "insufficient_evidence_reply" or gate.get("passed") is False:
        coverage = "insufficient"
    elif sources:
        coverage = "sufficient"
    else:
        coverage = "unavailable"
    return {"coverage": coverage, "source_count": len(sources), "sources": sources}
