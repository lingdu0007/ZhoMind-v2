from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.contracts.canonical import SourceAccessScope
from app.reviewed_bundles.assurance import candidate_assurance_metadata

_SOURCE_ACCESS_SCOPES = frozenset(scope.value for scope in SourceAccessScope)
_PUBLIC_SOURCE_ACCESS_SCOPE = SourceAccessScope.PUBLIC.value


def candidate_artifact_entry(
    *,
    artifact: Mapping[str, object],
    entry_identity: str,
) -> dict[str, object]:
    entry = artifact.get("entry")
    if (
        not isinstance(entry, dict)
        or entry.get("entry_id") != artifact.get("entry_id")
        or artifact.get("entry_identity") != entry_identity
        or not isinstance(artifact.get("editorial_revision_identity"), str)
    ):
        raise ValueError("Candidate artifact entry is invalid")
    return entry


def candidate_source_definitions(
    *,
    artifact: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    raw_sources = artifact.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Candidate artifact sources are invalid")
    definitions: dict[str, dict[str, object]] = {}
    for snapshot in raw_sources:
        source_identity = snapshot.get("source_identity") if isinstance(snapshot, dict) else None
        source = snapshot.get("source") if isinstance(snapshot, dict) else None
        if (
            not isinstance(source_identity, str)
            or not isinstance(source, dict)
            or snapshot.get("availability") != "verified_usable"
            or not isinstance(source.get("source_id"), str)
            or not source.get("source_id")
            or source_identity != f"source:{source['source_id']}"
            or source.get("access_scope") not in _SOURCE_ACCESS_SCOPES
            or source_identity in definitions
        ):
            raise ValueError("Candidate artifact source snapshot is invalid")
        definitions[source_identity] = dict(source)
    return definitions


def candidate_section_source_relationships(
    *,
    artifact: Mapping[str, object],
    entry_identity: str,
    section_id: str,
) -> list[dict[str, str]]:
    entry = candidate_artifact_entry(artifact=artifact, entry_identity=entry_identity)
    definitions = candidate_source_definitions(artifact=artifact)
    raw_relationships = entry.get("section_source_relationships")
    if not isinstance(raw_relationships, list):
        raise ValueError("Candidate artifact source relationships are invalid")
    matched = [
        relationship
        for relationship in raw_relationships
        if isinstance(relationship, dict) and relationship.get("section_id") == section_id
    ]
    if len(matched) != 1:
        raise ValueError("Candidate artifact section has no exact source relationship")
    source_ids = matched[0].get("source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise ValueError("Candidate artifact section source relationship is invalid")
    relationships: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_id in source_ids:
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Candidate artifact section source relationship is invalid")
        source_identity = f"source:{source_id}"
        source = definitions.get(source_identity)
        if source is None or source_identity in seen:
            raise ValueError("Candidate artifact section source relationship is invalid")
        seen.add(source_identity)
        relationships.append(
            {
                "source_identity": source_identity,
                "availability": "verified_usable",
                "access_scope": str(source["access_scope"]),
            }
        )
    return sorted(relationships, key=lambda item: item["source_identity"])


def candidate_source_evidence_projection(
    *,
    artifact: Mapping[str, object],
    entry_identity: str,
    section_id: str,
    source_identity: str,
) -> dict[str, str]:
    entry = candidate_artifact_entry(artifact=artifact, entry_identity=entry_identity)
    relationships = candidate_section_source_relationships(
        artifact=artifact,
        entry_identity=entry_identity,
        section_id=section_id,
    )
    definitions = candidate_source_definitions(artifact=artifact)
    if source_identity not in {relationship["source_identity"] for relationship in relationships}:
        raise ValueError("Candidate artifact source relationship is invalid")
    source = definitions.get(source_identity)
    if source is None:
        raise ValueError("Candidate artifact source relationship is invalid")
    review_date = entry.get("review_date")
    source_title = source.get("title")
    source_authority = source.get("authority")
    source_version = source.get("version_or_date")
    access_scope = source.get("access_scope")
    source_locator = (
        source.get("public_url")
        if access_scope == _PUBLIC_SOURCE_ACCESS_SCOPE
        else source.get("controlled_locator")
    )
    if (
        not isinstance(review_date, str)
        or not review_date.strip()
        or not isinstance(source_title, str)
        or not source_title.strip()
        or not isinstance(source_authority, str)
        or not source_authority.strip()
        or not isinstance(source_version, str)
        or not source_version.strip()
        or not isinstance(source_locator, str)
        or not source_locator.strip()
        or not isinstance(source.get("source_tier"), str)
        or not source["source_tier"].strip()
        or not isinstance(source.get("source_id"), str)
        or not source["source_id"].strip()
    ):
        raise ValueError("Candidate artifact citation metadata is invalid")
    return {
        "source_identity": source_identity,
        "source_id": str(source["source_id"]),
        "source_tier": str(source["source_tier"]),
        "source_access_scope": str(access_scope),
        "source_title": source_title,
        "source_authority": source_authority,
        "source_url": source_locator,
        "source_version": source_version,
        "source_review_date": review_date,
    }


def candidate_source_evidence_projections(
    *,
    artifact: Mapping[str, object],
    entry_identity: str,
    section_id: str,
) -> list[dict[str, str]]:
    relationships = candidate_section_source_relationships(
        artifact=artifact,
        entry_identity=entry_identity,
        section_id=section_id,
    )
    return [
        candidate_source_evidence_projection(
            artifact=artifact,
            entry_identity=entry_identity,
            section_id=section_id,
            source_identity=relationship["source_identity"],
        )
        for relationship in relationships
    ]


def candidate_frozen_chunk_metadata(
    *,
    artifact: dict[str, object],
    entry_identity: str,
    chunk_strategy: dict[str, object],
    section_id: str,
) -> dict[str, object]:
    entry = candidate_artifact_entry(
        artifact=artifact,
        entry_identity=entry_identity,
    )
    relationships = candidate_section_source_relationships(
        artifact=artifact,
        entry_identity=entry_identity,
        section_id=section_id,
    )
    return candidate_build_chunk_metadata(
        artifact=artifact,
        entry=entry,
        entry_identity=entry_identity,
        editorial_revision_identity=artifact.get("editorial_revision_identity"),
        chunk_strategy=chunk_strategy,
        section_id=section_id,
        source_relationships=relationships,
    )


def candidate_frozen_published_chunk_metadata(
    *,
    artifact: dict[str, object],
    entry_identity: str,
    chunk_strategy: dict[str, object],
    section_id: str,
    publication_identity: str,
    source_identity: str | None = None,
    include_source_evidence_projections: bool = True,
) -> dict[str, object]:
    entry = candidate_artifact_entry(
        artifact=artifact,
        entry_identity=entry_identity,
    )
    expected = candidate_frozen_chunk_metadata(
        artifact=artifact,
        entry_identity=entry_identity,
        chunk_strategy=chunk_strategy,
        section_id=section_id,
    )
    source_relationships = expected.get("source_relationships")
    if not isinstance(source_relationships, list) or not source_relationships:
        raise ValueError("Candidate artifact source relationship is invalid")
    selected_source_identity = source_identity or source_relationships[0].get("source_identity")
    if not isinstance(selected_source_identity, str):
        raise ValueError("Candidate artifact source relationship is invalid")
    source_projection = candidate_source_evidence_projection(
        artifact=artifact,
        entry_identity=entry_identity,
        section_id=section_id,
        source_identity=selected_source_identity,
    )
    is_claim_linked = expected.get("assurance_level") == "claim_linked"
    source_evidence_projections = (
        candidate_source_evidence_projections(
            artifact=artifact,
            entry_identity=entry_identity,
            section_id=section_id,
        )
        if is_claim_linked and include_source_evidence_projections
        else None
    )
    if (
        is_claim_linked
        and include_source_evidence_projections
        and len(source_evidence_projections or []) != len(source_relationships)
    ):
        raise ValueError("Candidate artifact source relationship is invalid")
    body = entry.get("body")
    decision_query = body.get("decision_query") if isinstance(body, dict) else None
    if not isinstance(decision_query, str) or not decision_query.strip():
        raise ValueError("Candidate artifact decision query is invalid")
    metadata: dict[str, object] = {
        **expected,
        "candidate_build": False,
        "lifecycle_state": "published",
        "published_knowledge_version_identity": publication_identity,
        "publication_version": publication_identity,
        "review_status": "approved",
        "source_availability": "verified",
        **source_projection,
        "review_date": source_projection["source_review_date"],
        "title": entry["title"],
        "entry_title": entry["title"],
        "decision_query": decision_query.strip(),
    }
    if is_claim_linked:
        metadata["candidate_evidence_source_identity"] = selected_source_identity
    if source_evidence_projections is not None:
        metadata["source_evidence_projections"] = source_evidence_projections
    return metadata


def candidate_build_chunk_metadata(
    *,
    artifact: dict[str, object],
    entry: dict[str, object],
    entry_identity: str,
    editorial_revision_identity: object,
    chunk_strategy: dict[str, object],
    section_id: str,
    source_relationships: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the immutable Candidate metadata shared by build and publication checks."""

    body = entry.get("body")
    applicability_conditions = entry.get("applicability_conditions") or []
    non_applicability_conditions = entry.get("non_applicability_conditions") or []
    freshness_triggers = entry.get("freshness_triggers") or []
    if (
        not isinstance(body, dict)
        or section_id not in body
        or not isinstance(body.get(section_id), str)
        or not body[section_id].strip()
        or not isinstance(entry.get("title"), str)
        or not entry["title"].strip()
        or not isinstance(entry.get("entry_id"), str)
        or not isinstance(entry.get("coverage_position"), str)
        or not isinstance(entry.get("assurance_level"), str)
        or not isinstance(applicability_conditions, list)
        or not isinstance(non_applicability_conditions, list)
        or not isinstance(freshness_triggers, list)
        or not isinstance(editorial_revision_identity, str)
        or not editorial_revision_identity
        or not isinstance(chunk_strategy.get("strategy_id"), str)
        or not chunk_strategy["strategy_id"].strip()
        or not source_relationships
    ):
        raise ValueError("Candidate artifact chunk metadata is invalid")

    source_identities: list[str] = []
    normalized_relationships: list[dict[str, object]] = []
    for relationship in source_relationships:
        source_identity = relationship.get("source_identity")
        availability = relationship.get("availability")
        access_scope = relationship.get("access_scope")
        if (
            not isinstance(source_identity, str)
            or not source_identity
            or availability != "verified_usable"
            or access_scope not in _SOURCE_ACCESS_SCOPES
        ):
            raise ValueError("Candidate artifact chunk metadata is invalid")
        source_identities.append(source_identity)
        normalized_relationships.append(
            {
                "source_identity": source_identity,
                "availability": availability,
                "access_scope": access_scope,
            }
        )
    if len(source_identities) != len(set(source_identities)):
        raise ValueError("Candidate artifact chunk metadata is invalid")

    return {
        "entry_id": entry["entry_id"],
        "domain": entry["coverage_position"],
        "entry_identity": entry_identity,
        "editorial_revision_identity": editorial_revision_identity,
        "section_id": section_id,
        "section_title": section_id.replace("_", " ").strip().title(),
        "chunk_strategy_id": chunk_strategy["strategy_id"],
        "source_identities": source_identities,
        "source_relationships": normalized_relationships,
        "assurance_level": entry["assurance_level"],
        "applicability_conditions": applicability_conditions,
        "non_applicability_conditions": non_applicability_conditions,
        "freshness_triggers": freshness_triggers,
        "lifecycle_state": "candidate_build",
        "candidate_build": True,
        **candidate_assurance_metadata(
            artifact=artifact,
            entry=entry,
            entry_identity=entry_identity,
        ),
    }
