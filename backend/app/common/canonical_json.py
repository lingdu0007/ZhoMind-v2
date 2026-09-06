from __future__ import annotations

import hashlib
import json


def canonical_json_sha256(value: object) -> str:
    """Hash canonical JSON without changing established authority export bytes."""

    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
