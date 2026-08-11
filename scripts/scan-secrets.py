#!/usr/bin/env python3
"""Scan tracked repository files for accidental secret material.

Runs on the same inputs the PR Gate uses: git-tracked files (source, tests,
workflows, docs, generated public evidence). It never needs credentials and it
never sends anything anywhere.

Detection is deliberately conservative: it flags high-signal secret shapes and
non-empty assignments to known secret-bearing settings keys. Test fixtures that
use clearly non-secret placeholder values (for example `browser-acceptance-...`
or `test-milvus-e2e-...`) do not match these shapes.

Allowlist: one `path:pattern-name` entry per line in
`scripts/secret-scan-allowlist.txt`. Use it only for a deliberate, reviewed
example value that must stay in the repository (e.g. a documented sample key).

Exit code 0 means no finding; nonzero lists every finding.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = REPO_ROOT / "scripts" / "secret-scan-allowlist.txt"
MAX_BYTES = 2 * 1024 * 1024
_BINARY_SUFFIXES = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".woff", ".woff2",
        ".ttf", ".eot", ".pdf", ".pyc", ".so", ".dylib", ".zip", ".gz", ".lock",
    }
)

# (name, compiled pattern). Names are referenced by the allowlist.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws-secret-key", re.compile(r"\b(?:aws)?_?secret\s*[:=]\s*[A-Za-z0-9/+=]{40,}")),
    ("openai-api-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("anthropic-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z0-9 ]* )?PRIVATE KEY-----")),
    # Assignment-style patterns: value must be a bare literal (no quotes, code
    # references, or trailing commas) so Python keyword args and runtime calls
    # like `password = token_urlsafe(32)` are not reported as leaks.
    ("jwt-secret-value", re.compile(r"(?im)^\s*(?:JWT_SECRET|SESSION_SECRET|SECRET_KEY)\s*=\s*[^'\"#,\s()\[\]{}]+$")),
    ("basic-auth-password", re.compile(r"(?im)^\s*(?:PASSWORD|DB_PASSWORD|POSTGRES_PASSWORD)\s*=\s*[^'\"#,\s()\[\]{}]+$")),
    ("database-url-with-password", re.compile(r"\b(?:postgres|postgresql|mysql)://[^/\s:@]+:[^@\s/]+@")),
]


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / item for item in result.stdout.split("\0") if item]


def load_allowlist() -> set[str]:
    if not ALLOWLIST_PATH.exists():
        return set()
    allowed: set[str] = set()
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and not entry.startswith("#"):
            allowed.add(entry)
    return allowed


def scan_file(path: Path, allowlist: set[str]) -> list[str]:
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return []
    try:
        size = path.stat().st_size
        if size > MAX_BYTES:
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    relative = path.relative_to(REPO_ROOT).as_posix()
    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in PATTERNS:
            if pattern.search(line) and f"{relative}:{name}" not in allowlist:
                findings.append(f"{relative}:{line_number}:{name}")
    return findings


def main() -> int:
    allowlist = load_allowlist()
    findings: list[str] = []
    for path in tracked_files():
        findings.extend(scan_file(path, allowlist))

    if findings:
        print("Secret scan findings (review before public publication):")
        for item in sorted(findings):
            print(f"  - {item}")
        print(f"Allowlist consulted: {ALLOWLIST_PATH.relative_to(REPO_ROOT)}")
        return 1
    print("Secret scan OK: no high-signal secret shapes in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
