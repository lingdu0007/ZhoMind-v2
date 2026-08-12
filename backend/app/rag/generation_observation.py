from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


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
    try:
        envelope = json.loads(user_prompt)
    except (TypeError, ValueError):
        return ()
    if not isinstance(envelope, Mapping):
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
