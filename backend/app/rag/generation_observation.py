from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("provider-visible JSON object contains a duplicate key")
        value[key] = item
    return value


def _provider_visible_json_object(user_prompt: str) -> Mapping[str, Any] | None:
    try:
        envelope = json.loads(user_prompt, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError):
        return None
    return envelope if isinstance(envelope, Mapping) else None


def generation_envelope_observation(
    *,
    user_prompt: str,
    system_prompt: str | None,
    snapshot_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Return a content-free identity for the exact generation envelope.

    The hash is computed at the provider-call seam from both prompt regions and
    the ordered Evidence Excerpt Snapshot identities. ``None`` is retained as
    distinct from an empty system prompt because it is part of the provider
    call contract. Only the hash and the already public snapshot identities are
    retained for later comparison.
    """
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "snapshot_ids": list(snapshot_ids),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "identity": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "snapshot_ids": list(snapshot_ids),
        "source_count": len(snapshot_ids),
    }


def wire_generation_envelope_observation(
    *,
    wire_payload: bytes,
    snapshot_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Record the identity of the provider's final serialized input.

    Providers call this after constructing their wire/message envelope and
    before handing it to the transport or model. The payload itself is never
    returned or persisted.
    """
    return {
        "identity": hashlib.sha256(wire_payload).hexdigest(),
        "snapshot_ids": list(snapshot_ids),
        "source_count": len(snapshot_ids),
    }


def provider_visible_snapshot_ids(user_prompt: str) -> tuple[str, ...]:
    """Derive snapshot identities from the exact JSON sent to the provider."""
    envelope = _provider_visible_json_object(user_prompt)
    if envelope is None:
        return ()
    raw_sources = envelope.get("evidence_sources")
    if not isinstance(raw_sources, list):
        return ()
    snapshot_ids: list[str] = []
    citation_keys = (
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
    for source in raw_sources:
        if not isinstance(source, Mapping):
            return ()
        raw_title = source.get("title") or source.get("entry_title")
        raw_version = source.get("publication_version")
        raw_excerpt = source.get("excerpt")
        if not isinstance(raw_title, str) or not raw_title.strip():
            return ()
        if not isinstance(raw_version, str) or not raw_version.strip():
            return ()
        if not isinstance(raw_excerpt, str) or not raw_excerpt.strip():
            return ()
        payload: dict[str, object] = {
            "title": raw_title.strip(),
            "publication_version": raw_version.strip(),
            "excerpt": raw_excerpt,
        }
        metadata = {
            key: str(source[key]).strip()
            for key in citation_keys
            if isinstance(source.get(key), str) and str(source[key]).strip()
        }
        if metadata:
            payload["citation_metadata"] = metadata
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        snapshot_ids.append(hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    return tuple(snapshot_ids)


def provider_visible_generation_input(user_prompt: str) -> dict[str, Any] | None:
    """Validate the complete frozen input visible at the provider boundary."""

    envelope = _provider_visible_json_object(user_prompt)
    if envelope is None:
        return None

    required_regions = {
        "user_question",
        "evidence_sources",
        "query_condition_set",
        "answer_evidence_set_identity",
        "knowledge_version_identities",
    }
    allowed_regions = required_regions | {"response_contract"}
    if set(envelope) not in {frozenset(required_regions), frozenset(allowed_regions)}:
        return None

    question = envelope.get("user_question")
    evidence_set_identity = envelope.get("answer_evidence_set_identity")
    query_conditions = envelope.get("query_condition_set")
    knowledge_version_identities = envelope.get("knowledge_version_identities")
    sources = envelope.get("evidence_sources")
    if (
        not isinstance(question, str)
        or not question.strip()
        or question != question.strip()
        or not isinstance(evidence_set_identity, str)
        or not evidence_set_identity
        or not isinstance(query_conditions, Mapping)
        or set(query_conditions) != {"identity", "conditions"}
        or not isinstance(knowledge_version_identities, list)
        or not isinstance(sources, list)
        or not sources
    ):
        return None

    condition_set_identity = query_conditions.get("identity")
    conditions = query_conditions.get("conditions")
    if (
        not isinstance(condition_set_identity, str)
        or not condition_set_identity.strip()
        or condition_set_identity != condition_set_identity.strip()
        or not isinstance(conditions, list)
    ):
        return None
    frozen_conditions: list[dict[str, str]] = []
    condition_keys: set[tuple[str, str]] = set()
    condition_ids: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, Mapping) or set(condition) != {
            "condition_id",
            "field",
            "operator",
            "value",
        }:
            return None
        frozen_condition: dict[str, str] = {}
        for key in ("condition_id", "field", "operator", "value"):
            value = condition.get(key)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                return None
            frozen_condition[key] = value
        key = (frozen_condition["field"], frozen_condition["operator"])
        if key in condition_keys or frozen_condition["condition_id"] in condition_ids:
            return None
        condition_keys.add(key)
        condition_ids.add(frozen_condition["condition_id"])
        frozen_conditions.append(frozen_condition)

    item_identities: list[str] = []
    citation_identities: list[str] = []
    frozen_sources: list[dict[str, object]] = []
    for source in sources:
        if (
            not isinstance(source, Mapping)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in source.items())
        ):
            return None
        title = source.get("title") or source.get("entry_title")
        publication_version = source.get("publication_version")
        excerpt = source.get("excerpt")
        item_identity = source.get("item_identity")
        citation_identity = source.get("citation_identity")
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(publication_version, str)
            or not publication_version.strip()
            or not isinstance(excerpt, str)
            or not excerpt.strip()
            or not isinstance(item_identity, str)
            or not item_identity.strip()
            or item_identity != item_identity.strip()
            or not isinstance(citation_identity, str)
            or not citation_identity.strip()
            or citation_identity != citation_identity.strip()
        ):
            return None
        citation_id = source.get("citation_id")
        if citation_id is not None and (
            not isinstance(citation_id, str) or not citation_id.strip() or citation_id != citation_id.strip()
        ):
            return None
        item_identities.append(item_identity)
        citation_identities.append(citation_identity)
        frozen_sources.append(dict(source))
    if len(set(item_identities)) != len(item_identities):
        return None
    if len(set(citation_identities)) != len(citation_identities):
        return None

    if (
        any(
            not isinstance(identity, str) or not identity.strip() or identity != identity.strip()
            for identity in knowledge_version_identities
        )
        or len(set(knowledge_version_identities)) != len(knowledge_version_identities)
    ):
        return None
    snapshot_ids = provider_visible_snapshot_ids(user_prompt)
    if len(snapshot_ids) != len(sources):
        return None
    if any(source.get("snapshot_id") != snapshot_id for source, snapshot_id in zip(sources, snapshot_ids, strict=True)):
        return None

    observed: dict[str, Any] = {
        "user_question": question,
        "evidence_sources": frozen_sources,
        "query_condition_set": {
            "identity": condition_set_identity,
            "conditions": frozen_conditions,
        },
        "answer_evidence_set_identity": evidence_set_identity,
        "knowledge_version_identities": list(knowledge_version_identities),
    }
    if "response_contract" in envelope:
        response_contract = envelope.get("response_contract")
        if not isinstance(response_contract, Mapping):
            return None
        observed["response_contract"] = dict(response_contract)
    return observed


def observed_generation_envelope(value: object) -> dict[str, Any] | None:
    """Validate a persisted observer projection without accepting prompt text."""
    if not isinstance(value, Mapping):
        return None
    identity = value.get("identity")
    snapshot_ids = value.get("snapshot_ids")
    source_count = value.get("source_count")
    if (
        not isinstance(identity, str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity)
        or not isinstance(snapshot_ids, list)
        or not all(isinstance(item, str) and len(item) == 64 for item in snapshot_ids)
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count != len(snapshot_ids)
    ):
        return None
    return {
        "identity": identity,
        "snapshot_ids": list(snapshot_ids),
        "source_count": source_count,
    }
