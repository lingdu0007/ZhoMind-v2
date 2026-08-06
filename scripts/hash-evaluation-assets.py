#!/usr/bin/env python3
"""Compute and verify the deterministic hashes of the Project-Derived Corpus and Evaluation Query Set.

The hashing rules are frozen for the Retrieval Evidence Baseline and are documented in
`evaluation/README.md`:

- corpus sha256: for every `*.md` file under `evaluation/corpus/`, sort by relative path,
  compute the per-file sha256 hex digest, build the string
  `"<relative-path>\\0<per-file-sha256-hex>"` joined by newlines in that sorted order, and
  return the sha256 hex digest of that joined string. This makes the hash sensitive to both
  file contents and file names.
- query-set sha256: the sha256 hex digest of the exact bytes of `evaluation/queries/query-set.json`.

This script is pure-stdlib so it runs on any Python 3 without installing dependencies. With no
arguments it writes `evaluation/corpus-manifest.json` (deterministic; unchanged content keeps the
same hashes). With `--check` it only verifies the committed manifest and exits nonzero on drift.

Exit code 0 means the manifest matches the frozen inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = REPO_ROOT / "evaluation"
CORPUS_DIR = EVALUATION_DIR / "corpus"
QUERY_SET_PATH = EVALUATION_DIR / "queries" / "query-set.json"
MANIFEST_PATH = EVALUATION_DIR / "corpus-manifest.json"

CORPUS_ID = "project-derived-corpus"
CORPUS_VERSION = "1.0.0"
QUERY_SET_ID = "evaluation-query-set"
QUERY_SET_VERSION = "1.0.0"
QA_CHUNKING = {"policy": "qa", "chunk_chars": 500, "overlap_chars": 50}
SCHEMA_VERSION = 1


def corpus_documents() -> list[tuple[str, Path]]:
    documents: list[tuple[str, Path]] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        documents.append((path.relative_to(CORPUS_DIR).as_posix(), path))
    return documents


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def corpus_sha256() -> str:
    digests: list[str] = []
    for relative_path, path in corpus_documents():
        digests.append(f"{relative_path}\0{sha256_bytes(path.read_bytes())}")
    return sha256_bytes("\n".join(digests).encode("utf-8"))


def query_set_sha256() -> str:
    return sha256_bytes(QUERY_SET_PATH.read_bytes())


def build_manifest() -> dict:
    query_set = json.loads(QUERY_SET_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus": {
            "corpus_id": CORPUS_ID,
            "version": CORPUS_VERSION,
            "sha256": corpus_sha256(),
            "documents": [
                {"path": relative_path, "sha256": sha256_bytes(path.read_bytes())}
                for relative_path, path in corpus_documents()
            ],
        },
        "query_set": {
            "query_set_id": QUERY_SET_ID,
            "version": QUERY_SET_VERSION,
            "sha256": query_set_sha256(),
            "query_count": len(query_set["queries"]),
        },
        "qa_chunking": QA_CHUNKING,
    }


def manifest_matches(committed: dict) -> list[str]:
    expected = build_manifest()
    violations: list[str] = []
    if committed != expected:
        violations.append(
            "committed evaluation/corpus-manifest.json does not match the frozen corpus/query-set inputs"
        )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed manifest instead of rewriting it",
    )
    args = parser.parse_args()

    if not CORPUS_DIR.is_dir() or not QUERY_SET_PATH.is_file():
        print(f"missing evaluation inputs: corpus={CORPUS_DIR} query-set={QUERY_SET_PATH}", file=sys.stderr)
        return 2

    if args.check:
        try:
            committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"cannot read committed manifest: {exc}", file=sys.stderr)
            return 2
        violations = manifest_matches(committed)
        for violation in violations:
            print(f"FAIL: {violation}", file=sys.stderr)
        if violations:
            return 1
        print("evaluation/corpus-manifest.json is consistent with the frozen inputs")
        return 0

    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} "
        f"(corpus {CORPUS_ID}@{CORPUS_VERSION} sha256={manifest['corpus']['sha256']}, "
        f"query-set {QUERY_SET_ID}@{QUERY_SET_VERSION} sha256={manifest['query_set']['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
