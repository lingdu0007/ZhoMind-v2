from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ReleaseGateError(ValueError):
    pass


class KnowledgeBaseReleaseGate:
    """Evaluate only normalized Acceptance Set observations."""

    _KINDS = frozenset({"direct", "paraphrase", "combined", "boundary"})

    def __init__(self, declared_entry_ids: list[str]) -> None:
        if not declared_entry_ids or len(declared_entry_ids) != len(set(declared_entry_ids)):
            raise ReleaseGateError("declared entry identities must be non-empty and unique")
        self._declared = list(declared_entry_ids)

    def evaluate(
        self,
        observations: list[dict[str, Any]],
        *,
        requested_label: str,
        strict_label: bool = False,
    ) -> dict[str, Any]:
        grouped = {entry_id: [] for entry_id in self._declared}
        for item in observations:
            entry_id = item.get("entry_id")
            kind = item.get("kind")
            if entry_id not in grouped or kind not in self._KINDS:
                raise ReleaseGateError("observation references an undeclared entry or query kind")
            grouped[entry_id].append(item)
        if any({item.get("kind") for item in items} != self._KINDS for items in grouped.values()):
            raise ReleaseGateError("each declared entry requires direct, paraphrase, combined, and boundary observations")

        direct = [item for item in observations if item["kind"] == "direct"]
        semantic = [item for item in observations if item["kind"] in {"paraphrase", "combined"}]
        boundaries = [item for item in observations if item["kind"] == "boundary"]
        answerable = [item for item in observations if item["kind"] != "boundary"]
        metrics = {
            "direct_top_three_coverage": self._ratio(direct, "top_three_evidence_covered"),
            "paraphrase_combined_top_three_coverage": self._ratio(semantic, "top_three_evidence_covered"),
            "unsupported_boundary_answers": sum(
                item.get("actual_outcome") != "insufficient_evidence_reply" for item in boundaries
            ),
            "exact_openable_citation_snapshots": self._citation_ratio(answerable),
        }
        failed_entries = [entry_id for entry_id, items in grouped.items() if not self._entry_passes(items)]
        passed = (
            metrics["direct_top_three_coverage"] == 1.0
            and metrics["paraphrase_combined_top_three_coverage"] >= 0.9
            and metrics["unsupported_boundary_answers"] == 0
            and metrics["exact_openable_citation_snapshots"] == 1.0
            and not failed_entries
        )
        if requested_label == "First Edition" and not passed and strict_label:
            raise ReleaseGateError("First Edition requires every declared entry to pass the Knowledge Base Release Gate")
        label = requested_label if passed else "Unreleased Candidate"
        return {
            "passed": passed,
            "edition_label": label,
            "metrics": metrics,
            "published_entry_ids": [entry_id for entry_id in self._declared if entry_id not in failed_entries],
            "failed_entry_ids": failed_entries,
        }

    @staticmethod
    def _ratio(items: list[dict[str, Any]], field: str) -> float:
        return sum(item.get(field) is True for item in items) / len(items) if items else 0.0

    @staticmethod
    def _citation_ratio(items: list[dict[str, Any]]) -> float:
        return (
            sum(item.get("citation_openable") is True and item.get("citation_snapshot_exact") is True for item in items)
            / len(items)
            if items
            else 0.0
        )

    def _entry_passes(self, items: list[dict[str, Any]]) -> bool:
        return all(
            (
                item.get("actual_outcome") == "insufficient_evidence_reply"
                if item["kind"] == "boundary"
                else item.get("actual_outcome") == "evidence_gated_answer"
                and item.get("top_three_evidence_covered") is True
                and item.get("citation_openable") is True
                and item.get("citation_snapshot_exact") is True
            )
            for item in items
        )


class KnowledgeEditionManifestBuilder:
    _SHA256 = re.compile(r"^[0-9a-f]{64}$")

    def build(
        self,
        *,
        edition_id: str,
        edition_label: str,
        entries: list[dict[str, str]],
        corpus_sha256: str,
        acceptance_set_sha256: str,
        source_revision: str,
        model_identities: dict[str, str],
        published_at: datetime,
        release_gate: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._SHA256.fullmatch(corpus_sha256) or not self._SHA256.fullmatch(acceptance_set_sha256):
            raise ReleaseGateError("manifest hashes must be lowercase SHA-256 values")
        projected_entries = sorted(
            ({"entry_id": item["entry_id"], "sha256": item["sha256"]} for item in entries),
            key=lambda item: item["entry_id"],
        )
        if any(not self._SHA256.fullmatch(item["sha256"]) for item in projected_entries):
            raise ReleaseGateError("entry hashes must be lowercase SHA-256 values")
        manifest = {
            "schema_version": 1,
            "edition_id": edition_id,
            "edition_label": edition_label,
            "entries": projected_entries,
            "corpus_sha256": corpus_sha256,
            "acceptance_set_sha256": acceptance_set_sha256,
            "source_revision": source_revision,
            "model_identities": dict(sorted(model_identities.items())),
            "published_at": published_at.isoformat(),
            "release_gate": release_gate,
        }
        manifest["manifest_sha256"] = self._hash(manifest)
        return manifest

    def rollback_plan(self, manifest_path: Path) -> dict[str, Any]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.pop("manifest_sha256", None)
        if not isinstance(expected, str) or expected != self._hash(manifest):
            raise ReleaseGateError("prior manifest hash verification failed")
        return {
            "action": "republish_prior_manifest",
            "edition_id": manifest["edition_id"],
            "manifest_sha256": expected,
            "entry_ids": [item["entry_id"] for item in manifest["entries"]],
        }

    @staticmethod
    def _hash(value: dict[str, Any]) -> str:
        canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


class PublicReleaseScanner:
    _FORBIDDEN_FIELDS = frozenset({"query", "prompt", "answer", "excerpt", "source_excerpt", "document_source"})
    _CREDENTIAL = re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
        r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*[^\s,}]+)",
        re.IGNORECASE,
    )
    _PRIVATE_CORPUS = re.compile(r"(?im)^#{1,2}\s+(?:Decision Question|Stable Principle|Claim-Evidence Links)\s*$")

    def scan(self, directory: Path) -> list[str]:
        findings: set[str] = set()
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if self._CREDENTIAL.search(text):
                findings.add("credential-shape")
            if self._PRIVATE_CORPUS.search(text):
                findings.add("private-corpus-shape")
            for match in re.finditer(r"https?://[^\s\"'<>]+", text):
                parsed = urlsplit(match.group(0).rstrip(".,)"))
                host = (parsed.hostname or "").lower()
                if parsed.scheme != "https" or host in {"localhost", "127.0.0.1", "0.0.0.0"} or host.endswith((".local", ".internal")):
                    findings.add("private-url")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            self._scan_fields(payload, findings)
        return sorted(findings)

    def _scan_fields(self, value: object, findings: set[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in self._FORBIDDEN_FIELDS:
                    findings.add(f"forbidden-field:{key}")
                self._scan_fields(item, findings)
        elif isinstance(value, list):
            for item in value:
                self._scan_fields(item, findings)
