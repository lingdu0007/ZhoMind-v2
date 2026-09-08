from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from app.common.canonical_json import canonical_json_sha256
from app.contracts.canonical import StableIdentity, StableIdentityKind
from app.rag.generation_observation import observed_generation_envelope

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
_GATE_METADATA_KEYS = (
    "entry_id",
    "entry_identity",
    "editorial_revision_identity",
    "section_id",
    "review_status",
    "evidence_conflict",
    "source_id",
    "source_identity",
    "source_availability",
    "source_review_date",
    "source_freshness_days",
    "source_access_scope",
    "source_tier",
    "assurance_level",
    "claim_evidence_contract",
    "claim_evidence_contract_sha256",
    "candidate_evidence_source_identity",
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
_DISPLAY_METADATA_KEYS = (
    "applicability_conditions",
    "non_applicability_conditions",
)
_UNSAFE_URL_QUERY_PARTS = ("credential", "password", "redirect", "secret", "signature", "token")
_CONTROLLED_LOCATOR = re.compile(r"^controlled://[a-z0-9][a-z0-9._/-]{2,159}$")


def _freeze_display_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_display_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_display_metadata(item) for item in value)
    return value


def _thaw_display_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_display_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_display_metadata(item) for item in value]
    return value


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


def _safe_source_locator(value: str, *, access_scope: object) -> bool:
    return has_safe_source_locator(value, access_scope=access_scope)


def has_safe_source_locator(value: object, *, access_scope: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    locator = value.strip()
    if access_scope == "controlled_internal":
        return (
            _CONTROLLED_LOCATOR.fullmatch(locator) is not None
            and not any(part in locator.lower() for part in _UNSAFE_URL_QUERY_PARTS)
        )
    return _safe_public_url(locator)


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
    if not _safe_source_locator(
        str(metadata["source_url"]),
        access_scope=metadata.get("source_access_scope"),
    ):
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


def evidence_snapshot_id(
    *,
    title: str,
    publication_version: str,
    excerpt: str,
    citation_metadata: Mapping[str, object] | None = None,
) -> str:
    """Return the stable identity for one provider-visible evidence snapshot.

    The identity binds the exact normalized excerpt to its published source
    identity and citation metadata. It is intentionally independent of the
    citation marker (``S1``), which is a presentation detail that can change
    when retrieval order changes. Only provider-visible citation metadata is
    accepted, so internal retrieval and runtime fields cannot affect it.
    """
    payload: dict[str, object] = {
        "title": title.strip(),
        "publication_version": publication_version.strip(),
        "excerpt": excerpt,
    }
    if citation_metadata:
        payload["citation_metadata"] = {
            key: str(citation_metadata[key]).strip()
            for key in _AGENT_CITATION_KEYS
            if isinstance(citation_metadata.get(key), str) and str(citation_metadata[key]).strip()
        }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
    metadata_items: tuple[tuple[str, Any], ...]

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

        citation_metadata: dict[str, Any] = {
            key: str(metadata[key]).strip()
            for key in _CITATION_METADATA_KEYS
            if isinstance(metadata.get(key), str) and str(metadata[key]).strip()
        }
        gate_metadata: dict[str, Any] = {
            key: str(metadata[key]).strip()
            for key in _GATE_METADATA_KEYS
            if isinstance(metadata.get(key), (str, int, float)) and str(metadata[key]).strip()
        }
        display_metadata: dict[str, Any] = {
            key: list(metadata[key])
            for key in _DISPLAY_METADATA_KEYS
            if isinstance(metadata.get(key), list)
        }
        if isinstance(metadata.get("entry_id"), str):
            for legacy_location_key in ("filename", "source_file", "source", "document_name", "path"):
                citation_metadata.pop(legacy_location_key, None)
        citation_metadata["title"] = title
        citation_metadata["publication_version"] = publication_version
        citation_metadata.update(gate_metadata)
        citation_metadata.update(display_metadata)
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
            metadata_items=tuple(
                (key, _freeze_display_metadata(value) if key in _DISPLAY_METADATA_KEYS else value)
                for key, value in citation_metadata.items()
            ),
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "chunk_id": self.source_id,
            "document_id": self.document_id,
            "generation": self.generation,
            "chunk_index": self.chunk_index,
            "content_preview": self.excerpt,
            "metadata": {
                key: _thaw_display_metadata(value) if key in _DISPLAY_METADATA_KEYS else value
                for key, value in self.metadata_items
            },
            "snapshot_id": self.snapshot_id,
        }
        if self.retrieval_source is not None:
            record["retrieval_source"] = self.retrieval_source
        if self.score is not None:
            record["score"] = self.score
        return record

    def to_source(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "metadata": {
                key: _thaw_display_metadata(value) if key in _DISPLAY_METADATA_KEYS else value
                for key, value in self.metadata_items
            },
            "excerpt": self.excerpt,
            "snapshot_id": self.snapshot_id,
        }

    @property
    def snapshot_id(self) -> str:
        metadata = dict(self.metadata_items)
        return evidence_snapshot_id(
            title=self.title,
            publication_version=self.publication_version,
            excerpt=self.excerpt,
            citation_metadata=metadata if self.is_agent_entry() else None,
        )

    def is_agent_entry(self) -> bool:
        return isinstance(dict(self.metadata_items).get("entry_id"), str)

    @property
    def section_id(self) -> str:
        section_id = dict(self.metadata_items).get("section_id")
        return section_id if isinstance(section_id, str) else ""

    def to_public_citation(self, citation_id: str) -> dict[str, str]:
        metadata = dict(self.metadata_items)
        citation = {
            "citation_id": citation_id,
            **{key: metadata[key] for key in _AGENT_CITATION_KEYS if key in metadata},
            "publication_version": self.publication_version,
            "excerpt": self.excerpt,
        }
        if self.is_agent_entry():
            citation["snapshot_id"] = self.snapshot_id
        return citation


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


def _identity_has_kind(value: object, kind: StableIdentityKind) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return StableIdentity.from_stable_id(value).kind is kind
    except ValueError:
        return False


def _frozen_identity_binding_is_valid(
    binding: object,
    *,
    item_identity: str,
    snapshot_id: str,
    evidence: Mapping[str, object],
    metadata: Mapping[str, object],
    excerpt: str | None,
) -> bool:
    if not isinstance(binding, Mapping):
        return False

    required_fields = frozenset(
        {
            "entry_identity",
            "editorial_revision_identity",
            "publication_identity",
            "section_identity",
            "chunk_identity",
            "snapshot_id",
            "source_content_length",
        }
    )
    source_binding_field = "candidate_evidence_source_identity"
    binding_fields = frozenset(binding)
    if binding_fields not in (
        required_fields,
        required_fields | {source_binding_field},
    ):
        return False
    if canonical_json_sha256(binding) != item_identity:
        return False
    entry_identity = binding.get("entry_identity")
    editorial_revision_identity = binding.get("editorial_revision_identity")
    publication_identity = binding.get("publication_identity")
    section_identity = binding.get("section_identity")
    chunk_identity = binding.get("chunk_identity")
    source_content_length = binding.get("source_content_length")
    source_evidence_identity = binding.get(source_binding_field)
    if source_binding_field in binding:
        if not isinstance(source_evidence_identity, str):
            return False
        try:
            source_identity = StableIdentity.from_stable_id(source_evidence_identity)
        except ValueError:
            return False
        if (
            source_identity.kind is not StableIdentityKind.SOURCE
            or source_evidence_identity != metadata.get("source_identity")
            or source_identity.value != metadata.get("source_id")
        ):
            return False
    if (
        not _identity_has_kind(entry_identity, StableIdentityKind.ENTRY)
        or not _identity_has_kind(editorial_revision_identity, StableIdentityKind.EDITORIAL_REVISION)
        or not _identity_has_kind(publication_identity, StableIdentityKind.PUBLISHED_KNOWLEDGE_VERSION)
        or not isinstance(section_identity, str)
        or not isinstance(chunk_identity, Mapping)
        or set(chunk_identity) != {"document_id", "generation", "chunk_index", "content_sha256"}
        or not isinstance(source_content_length, int)
        or isinstance(source_content_length, bool)
        or source_content_length < 1
        or binding.get("snapshot_id") != snapshot_id
    ):
        return False
    assert isinstance(entry_identity, str)
    try:
        entry = StableIdentity.from_stable_id(entry_identity)
    except ValueError:
        return False
    if (
        metadata.get("entry_id") != entry.value
        or metadata.get("entry_identity") != entry_identity
        or metadata.get("editorial_revision_identity") != editorial_revision_identity
        or metadata.get("section_id") is None
        or section_identity != f"{entry_identity}#{metadata['section_id']}"
        or chunk_identity.get("document_id") != evidence.get("document_id")
        or chunk_identity.get("generation") != evidence.get("generation")
        or chunk_identity.get("chunk_index") != evidence.get("chunk_index")
        or not isinstance(chunk_identity.get("content_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(chunk_identity["content_sha256"])) is None
        or (excerpt is not None and source_content_length < len(excerpt))
    ):
        return False
    return True


def _sources_from_frozen_answer_evidence_set(value: Mapping[str, object]) -> list[dict[str, Any]]:
    evidence_set_identity = value.get("identity")
    query_condition_set_identity = value.get("query_condition_set_identity")
    items = value.get("items")
    governing_citation = value.get("governing_citation")
    if (
        not isinstance(evidence_set_identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", evidence_set_identity)
        or not isinstance(query_condition_set_identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", query_condition_set_identity)
        or not isinstance(items, list)
        or not items
        or not isinstance(governing_citation, Mapping)
    ):
        return []

    sources: list[dict[str, Any]] = []
    item_identities: list[str] = []
    citations_by_marker: dict[str, Mapping[str, object]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            return []
        item_identity = item.get("item_identity")
        identity_binding = item.get("identity_binding")
        snapshot_id = item.get("snapshot_id")
        evidence = item.get("evidence")
        citation = item.get("citation")
        if (
            not isinstance(item_identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", item_identity)
            or not isinstance(snapshot_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", snapshot_id)
            or not isinstance(evidence, Mapping)
            or not isinstance(citation, Mapping)
        ):
            return []
        metadata = evidence.get("metadata")
        if not isinstance(metadata, Mapping):
            return []
        title = metadata.get("title")
        publication_version = metadata.get("publication_version")
        excerpt = evidence.get("content_preview")
        withdrawn = evidence.get("withdrawn") is True or item.get("withdrawn") is True
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(publication_version, str)
            or not publication_version.strip()
            or not isinstance(metadata.get("entry_id"), str)
            or not isinstance(metadata.get("section_id"), str)
        ):
            return []
        if not withdrawn and (not isinstance(excerpt, str) or not excerpt.strip()):
            return []
        expected_snapshot_id = (
            str(evidence.get("snapshot_id") or "")
            if withdrawn
            else evidence_snapshot_id(
                title=title,
                publication_version=publication_version,
                excerpt=str(excerpt),
                citation_metadata=metadata,
            )
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_snapshot_id)
            or evidence.get("snapshot_id") != expected_snapshot_id
            or snapshot_id != expected_snapshot_id
            or not _frozen_identity_binding_is_valid(
                identity_binding,
                item_identity=item_identity,
                snapshot_id=expected_snapshot_id,
                evidence=evidence,
                metadata=metadata,
                excerpt=None if withdrawn else str(excerpt),
            )
        ):
            return []
        citation_id = citation.get("citation_id")
        citation_identity = citation.get("citation_identity")
        if (
            not isinstance(citation_id, str)
            or re.fullmatch(r"S[1-9][0-9]*", citation_id) is None
            or citation_id in citations_by_marker
            or not isinstance(citation_identity, str)
            or not re.fullmatch(r"[0-9a-f]{64}", citation_identity)
            or citation.get("item_identity") != item_identity
            or citation.get("snapshot_id") != expected_snapshot_id
            or citation.get("entry_id") != metadata.get("entry_id")
            or citation.get("section_id") != metadata.get("section_id")
            or citation_identity
            != canonical_json_sha256(
                {
                    "evidence_set_identity": evidence_set_identity,
                    "item_identity": item_identity,
                }
            )
        ):
            return []
        citations_by_marker[citation_id] = citation
        item_identities.append(item_identity)
        source: dict[str, Any] = {
            "citation_id": citation_id,
            "citation_identity": citation_identity,
            **{key: str(metadata[key]).strip() for key in _AGENT_CITATION_KEYS if isinstance(metadata.get(key), str)},
            "publication_version": publication_version,
            "snapshot_id": expected_snapshot_id,
        }
        if isinstance(metadata.get("assurance_level"), str):
            source["assurance_level"] = metadata["assurance_level"].strip()
        if isinstance(metadata.get("review_status"), str):
            source["review_status"] = metadata["review_status"].strip()
        for key in _DISPLAY_METADATA_KEYS:
            if isinstance(metadata.get(key), list):
                source[key] = _thaw_display_metadata(metadata[key])
        if metadata.get("source_access_scope") == "controlled_internal":
            source["source_access_scope"] = "controlled_internal"
        if withdrawn:
            source["withdrawal_notice"] = "This source has been withdrawn."
        else:
            source["excerpt"] = excerpt
        sources.append(source)

    governing_marker = governing_citation.get("citation_id")
    governing_item_identity = governing_citation.get("item_identity")
    expected_set_identity = canonical_json_sha256(
        {
            "query_condition_set_identity": query_condition_set_identity,
            "item_identities": item_identities,
            "governing_item_identity": governing_item_identity,
        }
    )
    if (
        expected_set_identity != evidence_set_identity
        or not isinstance(governing_marker, str)
        or governing_marker not in citations_by_marker
        or citations_by_marker[governing_marker].get("citation_identity") != governing_citation.get("citation_identity")
        or governing_citation.get("section_id") != "recommendation_or_reviewed_branches"
    ):
        return []
    return sources


def evidence_summary_from_trace(rag_trace: object) -> dict[str, Any]:
    trace = rag_trace if isinstance(rag_trace, Mapping) else {}
    answer_evidence_set = trace.get("answer_evidence_set")
    if isinstance(answer_evidence_set, Mapping):
        sources = _sources_from_frozen_answer_evidence_set(answer_evidence_set)
    else:
        evidence = trace.get("evidence")
        evidence_items = evidence if isinstance(evidence, (list, tuple)) else []
        sources = []
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
                    if isinstance(item.get("snapshot_id"), str) and item["snapshot_id"].strip():
                        citation["snapshot_id"] = item["snapshot_id"].strip()
                else:
                    excerpt = str(item.get("content_preview") or item.get("content") or "")
                    if not excerpt:
                        continue
                    citation["excerpt"] = excerpt
                    citation["snapshot_id"] = evidence_snapshot_id(
                        title=str(source_metadata.get("title") or ""),
                        publication_version=str(
                            source_metadata.get("publication_version") or f"v{item.get('generation', 1)}"
                        ),
                        excerpt=excerpt,
                        citation_metadata=source_metadata,
                    )
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
    summary: dict[str, Any] = {"coverage": coverage, "source_count": len(sources), "sources": sources}
    raw_runtime = trace.get("runtime")
    runtime = raw_runtime if isinstance(raw_runtime, Mapping) else {}
    observed_envelope = observed_generation_envelope(runtime.get("provider_generation_envelope"))
    agent_sources = [source for source in sources if isinstance(source.get("entry_id"), str)]
    citation_snapshot_ids = [source.get("snapshot_id") for source in agent_sources]
    if (
        agent_sources
        and len(agent_sources) == len(sources)
        and observed_envelope is not None
        and observed_envelope["snapshot_ids"] == citation_snapshot_ids
    ):
        summary["provider_prompt_snapshot_ids"] = observed_envelope["snapshot_ids"]
        summary["provider_generation_envelope"] = observed_envelope
    return summary


def evidence_summary_from_execution(execution_result: object) -> dict[str, Any]:
    """Project a completed closed execution without deriving its outcome."""

    result = execution_result if isinstance(execution_result, Mapping) else {}
    outcome = result.get("outcome")
    if outcome == "insufficient_evidence_reply":
        return {"coverage": "insufficient", "source_count": 0, "sources": []}
    if outcome == "non_knowledge_base_reply":
        return {"coverage": "unavailable", "source_count": 0, "sources": []}
    if outcome not in {"evidence_gated_answer", "generation_unavailable"}:
        raise ValueError("completed answer execution has no accepted closed outcome")

    answer_evidence_set = result.get("evidence_set")
    if not isinstance(answer_evidence_set, Mapping):
        raise ValueError("evidence-bound answer execution has no frozen Answer Evidence Set")
    sources = _sources_from_frozen_answer_evidence_set(answer_evidence_set)
    if not sources:
        raise ValueError("frozen Answer Evidence Set cannot be projected")
    if outcome == "generation_unavailable":
        return {"coverage": "unavailable", "source_count": 0, "sources": []}
    return {"coverage": "sufficient", "source_count": len(sources), "sources": sources}
