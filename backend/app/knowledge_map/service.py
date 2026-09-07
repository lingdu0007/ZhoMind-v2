from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.canonical_json import canonical_json_sha256
from app.common.exceptions import AppError
from app.documents.parsers import validate_canonical_source_url
from app.editorial_authority.service import EditorialAuthorityService
from app.model.document import Document, DocumentChunk
from app.rag.answer_evidence import has_safe_source_locator

THEMES = (
    ("workflow-vs-agent", "Workflow 与 Agent"),
    ("tools-and-mcp", "Tools 与 MCP"),
    ("context-memory-rag", "Context、Memory 与 RAG"),
    ("orchestration", "Agent Orchestration"),
    ("reliability-safety-evaluation", "Reliability、Safety 与 Evaluation"),
    ("operating-constraints", "Latency、Cost 与 Concurrency"),
)
_THEME_ORDER = {domain: index for index, (domain, _label) in enumerate(THEMES)}
_COVERAGE_POSITION_THEMES = {
    "rag_source_admission_and_chunking": "context-memory-rag",
    "sparse_dense_hybrid_and_reranking_choices": "context-memory-rag",
    "evidence_sufficiency_refusal_and_acceptance": "reliability-safety-evaluation",
    "tools_and_mcp_permissions_and_failure_behavior": "tools-and-mcp",
    "agent_context_state_and_memory": "context-memory-rag",
    "orchestration_retry_human_intervention_and_side_effects": "orchestration",
    "provider_failure_and_observability": "reliability-safety-evaluation",
    "prompt_injection_isolation_and_security": "reliability-safety-evaluation",
}
_ASSURANCE_LEVELS = frozenset({"source_grounded", "claim_linked", "release_assured"})


class CurrentKnowledgeMapAuthority(Protocol):
    async def get_retrieval_authority(self, entry_id: str, *, now: datetime | None = None) -> dict[str, Any]: ...


def _plain_section(content: str) -> str:
    body = re.sub(r"^#{1,6}\s+[^\n]+\n+", "", content.strip())
    return re.sub(r"\s+", " ", body).strip()


def _review_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        return None
    return value


def _access_appropriate_sources(value: object) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    projected: list[dict[str, str]] = []
    for source in value:
        if not isinstance(source, dict):
            return None
        access_scope = source.get("access_scope")
        if access_scope not in {"public", "controlled_internal"}:
            return None
        locator = source.get("controlled_locator") if access_scope == "controlled_internal" else source.get("public_url")
        fields = {key: source.get(key) for key in ("title", "authority", "version")}
        if any(not isinstance(item, str) or not item.strip() for item in fields.values()):
            return None
        if access_scope == "public":
            if not isinstance(locator, str) or not locator.strip():
                return None
            try:
                validate_canonical_source_url(locator.strip(), source_probe=lambda _url: None)
            except AppError:
                return None
        elif not has_safe_source_locator(locator, access_scope=access_scope):
            return None
        projected.append(
            {
                **{key: str(item).strip() for key, item in fields.items()},
                "url": str(locator).strip(),
                "access_scope": str(access_scope),
            }
        )
    return projected


def _displayable_assurance_level(value: object) -> str | None:
    if value in _ASSURANCE_LEVELS:
        return str(value)
    return None


def _theme_from_coverage_position(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _COVERAGE_POSITION_THEMES.get(value.strip())


class KnowledgeMapService:
    """Project current published Agent entries through one read-only interface."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        editorial_authority: CurrentKnowledgeMapAuthority | None = None,
        now: datetime | None = None,
    ) -> None:
        self._session = session
        self._authority = editorial_authority or EditorialAuthorityService(session)
        self._now = now

    async def list(self) -> dict[str, Any]:
        result = await self._session.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                Document.deleted_at.is_(None),
                Document.published_generation > 0,
                DocumentChunk.generation == Document.published_generation,
            )
            .order_by(Document.id.asc(), DocumentChunk.chunk_index.asc())
        )
        publications: dict[tuple[str, int, str], list[tuple[DocumentChunk, Document]]] = {}
        for chunk, document in result.all():
            metadata = chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {}
            entry_id = metadata.get("entry_id")
            if isinstance(entry_id, str) and entry_id:
                publications.setdefault((document.id, document.published_generation, entry_id), []).append((chunk, document))

        current_entries: dict[str, tuple[tuple[int, str], dict[str, Any]]] = {}
        for items in publications.values():
            entry = await self._project_entry(items)
            if entry is None:
                continue
            document = items[0][1]
            publication_rank = (document.published_generation, document.id)
            existing = current_entries.get(entry["entry_id"])
            if existing is None or publication_rank > existing[0]:
                current_entries[entry["entry_id"]] = (publication_rank, entry)
        entries = [entry for _rank, entry in current_entries.values()]
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

    async def _project_entry(self, items: list[tuple[DocumentChunk, Document]]) -> dict[str, Any] | None:
        first_chunk, document = items[0]
        metadata = first_chunk.chunk_metadata if isinstance(first_chunk.chunk_metadata, dict) else {}
        entry_id = metadata.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            return None
        authority = await self._current_authority(entry_id)
        if authority is None or authority.get("answer_eligible") is not True:
            return None
        domain = _theme_from_coverage_position(authority.get("coverage_position"))
        if domain is None:
            return None
        if any(
            not self._metadata_matches_authority(
                chunk.chunk_metadata if isinstance(chunk.chunk_metadata, dict) else {},
                authority,
            )
            for chunk, _document in items
        ):
            return None
        review_date = _review_date(authority.get("review_date"))
        sources = _access_appropriate_sources(authority.get("source_definitions"))
        versions = authority.get("applicable_versions")
        title = authority.get("entry_title")
        decision_query = authority.get("decision_query")
        if (
            review_date is None
            or sources is None
            or not isinstance(versions, list)
            or not isinstance(title, str)
            or not title.strip()
            or not isinstance(decision_query, str)
            or not decision_query.strip()
        ):
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
        summary = (
            metadata_summary.strip()[:320]
            if isinstance(metadata_summary, str) and metadata_summary.strip()
            else " ".join(
                by_section.get("recommendation_or_reviewed_branches", [])
                or by_section.get("recommendation", [])
            ).strip()[:320]
        )
        if not summary:
            return None
        entry = {
            "entry_id": entry_id.strip(),
            "title": title.strip(),
            "approved_summary": summary,
            "review_date": review_date,
            "applicable_versions": applicable_versions,
            "publication_version": f"v{document.published_generation}",
            "source_count": len(sources),
            "public_source_count": sum(source["access_scope"] == "public" for source in sources),
            "controlled_source_count": sum(source["access_scope"] == "controlled_internal" for source in sources),
            "suggested_query": decision_query.strip()[:240],
            "sources": sources,
            "domain": domain.strip(),
        }
        assurance_level = _displayable_assurance_level(authority.get("assurance_level"))
        if assurance_level is not None:
            entry["assurance_level"] = assurance_level
        return entry

    async def _current_authority(self, entry_id: str) -> dict[str, Any] | None:
        try:
            authority = await self._authority.get_retrieval_authority(entry_id, now=self._now)
        except (AppError, RuntimeError, ValueError):
            return None
        return authority if isinstance(authority, dict) else None

    @staticmethod
    def _metadata_matches_authority(metadata: dict[str, Any], authority: dict[str, Any]) -> bool:
        if metadata.get("candidate_build") is True:
            return False
        entry_id = authority.get("entry_id")
        entry_identity = authority.get("entry_identity")
        revision_identity = authority.get("editorial_revision_identity")
        section_id = metadata.get("section_id")
        relationships = authority.get("section_source_relationships")
        assurance_level = authority.get("assurance_level")
        applicability_conditions = authority.get("applicability_conditions")
        freshness_triggers = authority.get("freshness_triggers")
        if (
            not isinstance(entry_id, str)
            or not isinstance(entry_identity, str)
            or not isinstance(revision_identity, str)
            or not isinstance(section_id, str)
            or not section_id
            or not isinstance(relationships, dict)
            or not isinstance(relationships.get(section_id), list)
            or assurance_level not in _ASSURANCE_LEVELS
            or not isinstance(applicability_conditions, list)
            or not applicability_conditions
            or not isinstance(freshness_triggers, list)
            or not freshness_triggers
        ):
            return False
        if (
            metadata.get("entry_id") != entry_id
            or metadata.get("entry_identity") != entry_identity
            or metadata.get("editorial_revision_identity") != revision_identity
            or metadata.get("assurance_level") != assurance_level
        ):
            return False
        try:
            return (
                canonical_json_sha256(metadata.get("source_relationships"))
                == canonical_json_sha256(relationships[section_id])
                and canonical_json_sha256(metadata.get("applicability_conditions"))
                == canonical_json_sha256(applicability_conditions)
                and canonical_json_sha256(metadata.get("freshness_triggers"))
                == canonical_json_sha256(freshness_triggers)
            )
        except (TypeError, ValueError):
            return False
