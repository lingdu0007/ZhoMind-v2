#!/usr/bin/env python3
"""Check English/Chinese canonical-document parity in the repository.

Rule (see AGENTS.md "Bilingual materials"): an English file is the canonical
skill-facing version; a `.zh-CN.md` mirror must stay structurally in sync with
its English counterpart in the same change.

This gate enforces:
  1. Every tracked `X.zh-CN.md` has a tracked `X.md` counterpart.
  2. For every tracked pair, the heading-level outline (count of H1/H2/H3 per
     level) is identical, and the count of fenced code blocks is identical.
  3. A file with no mirror is allowed (English is canonical), but a mirror
     without its English counterpart is an error.

It is pure-stdlib so it runs on any Python 3 without installing packages.
Exit code 0 means parity holds; nonzero lists every violation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_FENCE_RE = re.compile(r"^```")


def tracked_files() -> list[Path]:
    """Return all git-tracked files under the repository root."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / item for item in result.stdout.split("\0") if item]


def outline_counts(path: Path) -> tuple[list[int], int]:
    """Return (per-level heading counts for H1..H3, fenced block count)."""
    heading_counts = [0, 0, 0]
    fence_count = 0
    in_fence = False
    with path.open(encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if _FENCE_RE.match(stripped):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            match = _HEADING_RE.match(line)
            if match:
                level = len(match.group(1))
                if 1 <= level <= 3:
                    heading_counts[level - 1] += 1
    return heading_counts, fence_count


def main() -> int:
    files = tracked_files()
    by_path = {item.relative_to(REPO_ROOT).as_posix(): item for item in files}
    mirrored = {name.removesuffix(".zh-CN.md") + ".md": item for name, item in by_path.items() if name.endswith(".zh-CN.md")}

    violations: list[str] = []
    for english_name, mirror in sorted(mirrored.items()):
        english = by_path.get(english_name)
        if english is None:
            violations.append(f"mirror without canonical counterpart: {english_name} (mirror {english_name}.zh-CN.md)")
            continue
        try:
            en_headings, en_fences = outline_counts(english)
            zh_headings, zh_fences = outline_counts(mirror)
        except UnicodeDecodeError as exc:
            violations.append(f"non-UTF-8 document pair: {english_name} ({exc})")
            continue
        if en_headings != zh_headings:
            violations.append(
                f"heading outline drift: {english_name} has H1/H2/H3 = {en_headings}, "
                f"but {english_name}.zh-CN.md has {zh_headings}"
            )
        if en_fences != zh_fences:
            violations.append(
                f"fenced code block count drift: {english_name} has {en_fences}, "
                f"but {english_name}.zh-CN.md has {zh_fences}"
            )

    if violations:
        print("Documentation parity violations:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("Documentation parity OK: every tracked .zh-CN.md mirrors its canonical English counterpart.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
