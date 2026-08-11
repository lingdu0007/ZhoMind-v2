from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_EVALUATION_MANIFEST = "corpus-manifest.json"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def corpus_sha256(corpus_dir: Path) -> str:
    """Deterministic Project-Derived Corpus hash.

    Rule (shared with scripts/hash-evaluation-assets.py): for every `*.md` file under the corpus
    directory sorted by relative path, take the sha256 hex digest of its bytes and build the string
    ``"<relative-path>\\0<per-file-sha256-hex>"`` joined by newlines in sorted order; return the
    sha256 hex digest of that joined string.
    """
    digests: list[str] = []
    for path in sorted(corpus_dir.glob("*.md")):
        relative_path = path.relative_to(corpus_dir).as_posix()
        digests.append(f"{relative_path}\0{_sha256_bytes(path.read_bytes())}")
    return _sha256_bytes("\n".join(digests).encode("utf-8"))


def query_set_sha256(query_set_path: Path) -> str:
    """Deterministic Evaluation Query Set hash: sha256 of the exact file bytes."""
    return _sha256_bytes(query_set_path.read_bytes())


def load_evaluation_inputs(evaluation_dir: Path) -> dict[str, Any]:
    """Read the frozen corpus/query-set manifest and verify its hashes against the actual files.

    Raises ValueError when the evaluation directory is incomplete or the committed hashes drift
    from the frozen corpus and query-set inputs.
    """
    manifest_path = evaluation_dir / _EVALUATION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus = manifest["corpus"]
    query_set = manifest["query_set"]

    computed_corpus = corpus_sha256(evaluation_dir / "corpus")
    if computed_corpus != corpus["sha256"]:
        raise ValueError("evaluation corpus hash mismatch: committed manifest does not match frozen corpus")

    computed_query_set = query_set_sha256(evaluation_dir / "queries" / "query-set.json")
    if computed_query_set != query_set["sha256"]:
        raise ValueError("evaluation query-set hash mismatch: committed manifest does not match frozen query set")

    return {
        "corpus_id": corpus["corpus_id"],
        "corpus_version": corpus["version"],
        "corpus_sha256": corpus["sha256"],
        "query_set_id": query_set["query_set_id"],
        "query_set_version": query_set["version"],
        "query_set_sha256": query_set["sha256"],
        "query_count": int(query_set["query_count"]),
        "qa_chunking": dict(manifest.get("qa_chunking") or {}),
    }
