from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.exceptions import AppError
from app.documents.parsers import validate_canonical_source_url
from app.model.document import Document, DocumentChunk

THEMES = (
    ("workflow-vs-agent", "Workflow 与 Agent"),
    ("tools-and-mcp", "Tools 与 MCP"),
    ("context-memory-rag", "Context、Memory 与 RAG"),
    ("orchestration", "Agent Orchestration"),
    ("reliability-safety-evaluation", "Reliability、Safety 与 Evaluation"),
    ("operating-constraints", "Latency、Cost 与 Concurrency"),
)
_THEME_ORDER = {domain: index for index, (domain, _label) in enumerate(THEMES)}
_MAX_REVIEW_AGE = timedelta(days=90)


def _plain_section(content: str) -> str:
    body = re.sub(r"^#{1,6}\s+[^\n]+\n+", "", content.strip())
    return re.sub(r"\s+", " ", body).strip()


def _fresh_review_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        reviewed = date.fromisoformat(value)
    except ValueError:
        return None
    if datetime.now(UTC).date() - reviewed > _MAX_REVIEW_AGE:
        return None
    return value


def _public_sources(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    projected: list[dict[str, str]] = []
    for source in value:
        if not isinstance(source, dict) or source.get("availability") != "verified":
            return None
        fields = {key: source.get(key) for key in ("title", "authority", "url", "version")}
        if any(not isinstance(item, str) or not item.strip() for item in fields.values()):
            return None
        try:
            validate_canonical_source_url(str(fields["url"]), source_probe=lambda _url: None)
        except AppError:
            return None
        projected.append({key: str(item).strip() for key, item in fields.items()})
    return projected


class KnowledgeMapService:
    """Project current published Agent entries through one read-only interface."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self) -> dict[str, Any]:
        result = await self._session.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.deleted_at.is_(None),
                Document.published_generation > 0,
                Document.chunk_strategy == "agent",
                DocumentChunk.generation == Document.published_generation,
            )
            .order_by(Document.id.asc(), DocumentChunk.chunk_index.asc())
        )
        grouped: dict[str, list[tuple[DocumentChunk, Document]]] = {}
        for chunk, document in result.all():
            metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
            entry_id = metadata.get("entry_id")
            if isinstance(entry_id, str) and entry_id:
                grouped.setdefault(entry_id, []).append((chunk, document))

        entries = [entry for items in grouped.values() if (entry := self._project_entry(items)) is not None]
        theme_groups: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            theme_groups.setdefault(entry["domain"], []).append(entry)
        themes = [
            {
                "domain": domain,
                "label": label,
                "entries": [
                    {key: value for key, value in entry.items() if key != "domain"}
                    for entry in sorted(theme_groups[domain], key=lambda item: (item["title"], item["entry_id"]))
                ],
            }
            for domain, label in THEMES
            if domain in theme_groups
        ]
        unknown_domains = sorted(set(theme_groups) - _THEME_ORDER.keys())
        themes.extend(
            {
                "domain": domain,
                "label": domain,
                "entries": [{key: value for key, value in entry.items() if key != "domain"} for entry in theme_groups[domain]],
            }
            for domain in unknown_domains
        )
        return {"total_entries": len(entries), "themes": themes}

    async def get(self, entry_id: str) -> dict[str, Any]:
        payload = await self.list()
        for theme in payload["themes"]:
            for entry in theme["entries"]:
                if entry["entry_id"] == entry_id:
                    return entry
        raise AppError(status_code=404, code="RESOURCE_NOT_FOUND", message="published knowledge entry not found")

    def _project_entry(self, items: list[tuple[DocumentChunk, Document]]) -> dict[str, Any] | None:
        first_chunk, document = items[0]
        metadata = first_chunk.chunk_metadata if isinstance(first_chunk.chunk_metadata, dict) else {}
        required = ("entry_id", "entry_title", "domain")
        if any(not isinstance(metadata.get(key), str) or not str(metadata[key]).strip() for key in required):
            return None
        if metadata.get("review_status") != "approved" or metadata.get("evidence_conflict") == "unresolved":
            return None
        review_date = _fresh_review_date(metadata.get("review_date"))
        sources = _public_sources(metadata.get("sources"))
        versions = metadata.get("applicable_versions")
        if review_date is None or sources is None or not isinstance(versions, list):
            return None
        applicable_versions = [str(version).strip() for version in versions if isinstance(version, str) and version.strip()]
        if not applicable_versions:
            return None

        by_section: dict[str, list[str]] = {}
        for chunk, _document in items:
            chunk_metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
            section_id = chunk_metadata.get("section_id")
            if isinstance(section_id, str):
                by_section.setdefault(section_id, []).append(_plain_section(chunk.content))
        metadata_summary = metadata.get("approved_summary")
        metadata_query = metadata.get("suggested_query")
        summary = (
            metadata_summary.strip()[:320]
            if isinstance(metadata_summary, str) and metadata_summary.strip()
            else " ".join(by_section.get("recommendation", [])).strip()[:320]
        )
        suggested_query = (
            metadata_query.strip()[:240]
            if isinstance(metadata_query, str) and metadata_query.strip()
            else " ".join(by_section.get("decision-question", [])).strip()[:240]
        )
        if not summary or not suggested_query:
            return None
        return {
            "entry_id": str(metadata["entry_id"]).strip(),
            "title": str(metadata["entry_title"]).strip(),
            "approved_summary": summary,
            "review_date": review_date,
            "applicable_versions": applicable_versions,
            "publication_version": f"v{document.published_generation}",
            "public_source_count": len(sources),
            "suggested_query": suggested_query,
            "sources": sources,
            "domain": str(metadata["domain"]).strip(),
        }
