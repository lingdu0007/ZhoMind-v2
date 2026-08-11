#!/usr/bin/env python3
"""Verify the public Portfolio Release documentation and evidence navigation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "docs/portfolio/INTERVIEW-DOSSIER.md",
    ROOT / "docs/portfolio/INTERVIEW-DOSSIER.zh-CN.md",
    ROOT / "docs/releases/PORTFOLIO-RELEASE-v1.0.0.md",
    ROOT / "docs/releases/PORTFOLIO-RELEASE-v1.0.0.zh-CN.md",
)
REQUIRED_README_LITERALS = (
    "Migration Retrieval",
    "Evaluation Retriever",
    "Direct Retrieval Diagnostic",
    "Evidence-Gated Answer Execution",
    "Production Answer Acceptance",
    "Answer Evidence Set",
    "Insufficient Evidence Reply",
    "Published Knowledge Version",
    "Public Evidence Bundle",
    "91753f1c1ff6fc07bc262dfa50fb719a63210e0b",
    "74.465",
)
REQUIRED_DOSSIER_LITERALS = (
    "91753f1c1ff6fc07bc262dfa50fb719a63210e0b",
    "portfolio-release-candidate-01",
    "Evidence Recall@3",
    "Context Precision@3",
    "RRF",
    "k=60",
    "Prompt Injection",
    "P95",
)
FORBIDDEN_CLAIMS = (
    re.compile(r"\bMCP\b", re.IGNORECASE),
    re.compile(r"\bmulti[ -]?agent\b", re.IGNORECASE),
    re.compile(r"\bagent platform\b", re.IGNORECASE),
    re.compile(r"(?:universal|general-purpose).{0,32}Prompt Injection", re.IGNORECASE),
    re.compile(r"Hybrid RRF.{0,48}(?:always|universally|proved).{0,48}(?:better|winner)", re.IGNORECASE),
    re.compile(r"12(?:\.0+)?\s*(?:second|seconds|秒).{0,24}P95.{0,32}(?:met|achieved|达成)", re.IGNORECASE),
)
LINK_PATTERN = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_document(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        fail(errors, f"missing required release document: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8")


def verify_links(path: Path, content: str, errors: list[str]) -> None:
    for target in LINK_PATTERN.findall(content):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            fail(errors, f"link escapes repository in {path.relative_to(ROOT)}: {target}")
            continue
        if not resolved.exists():
            fail(errors, f"broken release link in {path.relative_to(ROOT)}: {target}")


def verify_pair(english: Path, chinese: Path, errors: list[str]) -> tuple[str, str]:
    english_text = read_document(english, errors)
    chinese_text = read_document(chinese, errors)
    english_headings = re.findall(r"^#{1,3} ", english_text, flags=re.MULTILINE)
    chinese_headings = re.findall(r"^#{1,3} ", chinese_text, flags=re.MULTILINE)
    if english_headings != chinese_headings:
        fail(errors, f"heading-level drift between {english.name} and {chinese.name}")
    for literal in REQUIRED_README_LITERALS if english.name == "README.md" else REQUIRED_DOSSIER_LITERALS:
        if literal not in english_text or literal not in chinese_text:
            fail(errors, f"missing bilingual literal {literal!r} in {english.name}")
    return english_text, chinese_text


def git_revision(revision: str, errors: list[str], label: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        fail(errors, f"{label} is not a resolvable commit")
        return ""
    return result.stdout.strip()


def verify_release_artifact(revision: str, errors: list[str]) -> str:
    resolved_revision = git_revision(revision, errors, "release artifact revision")
    head_revision = git_revision("HEAD", errors, "checkout HEAD")
    if resolved_revision and head_revision and resolved_revision != head_revision:
        fail(errors, "release artifact revision must equal the checked-out release candidate")

    bundle_root = ROOT / "public-evidence/releases/portfolio-release-candidate-01"
    for artifact in sorted(path for path in bundle_root.rglob("*") if path.is_file()):
        relative_path = artifact.relative_to(ROOT).as_posix()
        result = subprocess.run(
            ["git", "show", f"{resolved_revision}:{relative_path}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode or result.stdout != artifact.read_bytes():
            fail(errors, f"release artifact revision does not contain the checked bundle artifact: {relative_path}")
    return resolved_revision


def verify_bundle(errors: list[str], release_artifact_revision: str) -> str:
    manifest_path = ROOT / "public-evidence/releases/portfolio-release-candidate-01/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(errors, f"cannot read accepted Public Evidence Bundle manifest: {error}")
        return ""
    source_revision = manifest.get("source_revision", "")
    candidate = manifest.get("release_candidate", {})
    if candidate.get("status") != "accepted":
        fail(errors, "Public Evidence Bundle is not accepted")
    if candidate.get("revision") != source_revision:
        fail(errors, "Public Evidence Bundle candidate revision differs from source revision")
    if source_revision not in manifest.get("provenance", {}).get("revisions", []):
        fail(errors, "Public Evidence Bundle source revision is missing from provenance")
    resolved_source_revision = git_revision(source_revision, errors, "Public Evidence Bundle source revision")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_source_revision, release_artifact_revision],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        fail(errors, "Public Evidence Bundle source revision is not an ancestor of the release artifact revision")
    return source_revision


def verify_claims(documents: tuple[Path, ...], errors: list[str]) -> None:
    for path in documents:
        content = read_document(path, errors)
        verify_links(path, content, errors)
        for pattern in FORBIDDEN_CLAIMS:
            if pattern.search(content):
                fail(errors, f"unsupported public claim in {path.relative_to(ROOT)}: {pattern.pattern}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-release-revision", default="HEAD")
    args = parser.parse_args()
    errors: list[str] = []

    readme_en, readme_zh = verify_pair(DOCUMENTS[0], DOCUMENTS[1], errors)
    dossier_en, dossier_zh = verify_pair(DOCUMENTS[2], DOCUMENTS[3], errors)
    release_en = read_document(DOCUMENTS[4], errors)
    release_zh = read_document(DOCUMENTS[5], errors)
    if re.findall(r"^#{1,3} ", release_en, flags=re.MULTILINE) != re.findall(
        r"^#{1,3} ", release_zh, flags=re.MULTILINE
    ):
        fail(errors, "heading-level drift between release-note mirrors")
    release_artifact_revision = verify_release_artifact(args.expected_release_revision, errors)
    source_revision = verify_bundle(errors, release_artifact_revision)
    if args.expected_source_revision and source_revision != args.expected_source_revision:
        fail(errors, "accepted Public Evidence Bundle does not match --expected-source-revision")
    if not (ROOT / "docs/assets/portfolio-release-chinese-entry.png").is_file():
        fail(errors, "missing sanitized Chinese product screenshot")
    verify_claims(DOCUMENTS, errors)
    if "Public Evidence Bundle" not in release_en or "Public Evidence Bundle" not in release_zh:
        fail(errors, "release-note mirrors must link the Public Evidence Bundle")
    if "[中文" not in readme_en or "[English" not in readme_zh:
        fail(errors, "README mirrors must link prominently to each other")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "PASS: Portfolio Release documents, accepted source revision "
        f"{source_revision}, and release artifact revision {release_artifact_revision} are coherent"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
