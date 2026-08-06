#!/usr/bin/env python3
"""Validate Public Evidence Bundles against the versioned contract.

A Public Evidence Bundle is the reviewed, non-sensitive evidence subset
published with the ZhoMind-v2 Portfolio Release. This validator is the
deterministic gate over the bundle contract in ``public-evidence/contract``:
schema allowlists (manifest + typed sections), provenance cross-references,
artifact inventory and hashes, bilingual report parity, and absence of
forbidden sensitive shapes. It is used by both the PR Gate and the Release
Gate, so evidence producers write one format and no other.

The validator is pure Python 3 stdlib (like ``check-docs-parity.py`` and
``scan-secrets.py``) and runs from a fresh checkout without installed
dependencies.

Exit codes:
  0  every checked bundle conforms to the contract
  1  at least one bundle failed a check (diagnostics on stdout)
  2  usage error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_DIR = REPO_ROOT / "public-evidence" / "contract"
MANIFEST_SCHEMA_PATH = CONTRACT_DIR / "evidence-bundle.schema.json"
SECTION_KINDS = (
    "retrieval",
    "answer",
    "prompt-injection",
    "performance",
    "production-acceptance",
)
REPORT_EN_PATH = "REPORT.md"
REPORT_ZH_PATH = "REPORT.zh-CN.md"
MAX_SCAN_BYTES = 4 * 1024 * 1024

# The complete answerable-query metric family every evaluation mode must report
# (PRD "Testing Decisions": Evidence Recall@3/@5/@10, Context Precision@3/@5/@10,
# first Gold Evidence rank, and retrieval duration). Migration is the
# availability regime and is reported separately, so it is not required to
# carry the quality metric family.
REQUIRED_RETRIEVAL_METRICS = (
    "evidence_recall@3",
    "evidence_recall@5",
    "evidence_recall@10",
    "context_precision@3",
    "context_precision@5",
    "context_precision@10",
    "first_gold_rank",
    "retrieval_duration_ms",
)

# ---------------------------------------------------------------------------
# Sensitive-shape rejection (value level, applied to every bundle file text).
# The field allowlists already forbid sensitive field names; these patterns
# additionally reject values that carry credentials, raw environment values,
# host addresses, or operational material. Names are diagnostic classes only;
# matched values are never echoed back.
# ---------------------------------------------------------------------------

SENSITIVE_SHAPES: list[tuple[str, re.Pattern[str]]] = [
    ("provider-api-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("provider-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z0-9 ]* )?PRIVATE KEY-----")),
    ("jwt-token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("bearer-token", re.compile(r"\bBearer [A-Za-z0-9._~+/=-]{20,}")),
    ("url-with-credentials", re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@")),
    ("ipv4-address", re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")),
    ("host-port", re.compile(r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}:\d{2,5}\b")),
    ("environment-assignment", re.compile(r"(?im)^\s*[A-Z][A-Z0-9_]*(?:SECRET|KEY|TOKEN|PASSWORD|CREDENTIAL|PASS)\b\s*=\s*\S+")),
    ("environment-reference", re.compile(r"\$\{?[A-Z][A-Z0-9_]{3,}\}?")),
    ("ssh-key-reference", re.compile(r"(?:~|/|\.)?\.ssh/|id_ed25519|id_rsa")),
    ("ssh-user-host", re.compile(r"\b[A-Za-z0-9._-]+@(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")),
]


def scan_sensitive_shapes(text: str) -> list[str]:
    """Return the shape classes found in ``text``; matched values are not returned."""
    found: list[str] = []
    for name, pattern in SENSITIVE_SHAPES:
        if pattern.search(text):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# Minimal JSON Schema (draft-07 subset) validator.
# The subset covers exactly the keywords used by the contract schemas:
# type, const, enum, pattern, minLength, maxLength, minimum, maximum,
# required, properties, additionalProperties, items, minItems.
# ---------------------------------------------------------------------------


def schema_type_name(instance: object) -> str:
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    if isinstance(instance, dict):
        return "object"
    return "null"


def _matches_type(instance: object, expected: str) -> bool:
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    return schema_type_name(instance) == expected


def schema_errors(instance: object, schema: dict[str, object], path: str) -> list[str]:
    """Validate ``instance`` against a schema; return human-readable violations."""
    errors: list[str] = []

    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_type(instance, expected_type):
        errors.append(f"{path}: expected type {expected_type}, got {schema_type_name(instance)}")
        # Type mismatches make the structural keywords below meaningless.
        if not isinstance(instance, (dict, list)) and "const" not in schema and "enum" not in schema and "pattern" not in schema:
            return errors

    if "const" in schema and instance != schema["const"]:
        errors.append(f'{path}: expected const {schema["const"]!r}, got {instance!r}')

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: value {instance!r} not in allowed enum")

    if "pattern" in schema and isinstance(instance, str):
        pattern = schema["pattern"]
        if isinstance(pattern, str) and not re.search(pattern, instance):
            errors.append(f'{path}: value does not match pattern {pattern}')

    if "minLength" in schema and isinstance(instance, str) and len(instance) < schema["minLength"]:
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if "maxLength" in schema and isinstance(instance, str) and len(instance) > schema["maxLength"]:
        errors.append(f"{path}: longer than maxLength {schema['maxLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    if isinstance(instance, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for field in required:
                if field not in instance:
                    errors.append(f"{path}: missing required field {field}")
        properties = schema.get("properties")
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_schema = properties.get(key) if isinstance(properties, dict) else None
            if isinstance(child_schema, dict):
                errors.extend(schema_errors(value, child_schema, f"{path}.{key}"))
            elif additional is False:
                errors.append(f"{path}: unknown field {key}")

    if isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(instance):
                errors.extend(schema_errors(item, items, f"{path}[{index}]"))
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            errors.append(f"{path}: fewer than minItems {min_items}")

    return errors


def load_schema(path: Path) -> dict[str, object]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load contract schema {path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise RuntimeError(f"contract schema {path} is not a JSON object")
    return schema


def schema_name(path: Path) -> str:
    return path.name.removesuffix(".schema.json")


# ---------------------------------------------------------------------------
# Bilingual parity: structure outline and literal preservation.
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_FENCE_RE = re.compile(r"^```")

# Literal classes preserved verbatim between the English canonical report and
# the Chinese mirror: snake_case identifiers, Metric@N names, x.y.z versions,
# and a fixed acronym list. Extracted only from the English report, then each
# literal must also appear in the Chinese report.
_LITERAL_PATTERNS = (
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),
    re.compile(r"\b[A-Za-z][A-Za-z ]+@\d+\b"),
    re.compile(r"\b\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?\b"),
    re.compile(r"\b(?:TTFT|P50|P95|P99|RRF|BM25|QA|SSE|API|JSON|CLI|PR|SHA-256|HTTP|UTC)\b"),
)


def outline_counts(path: Path) -> tuple[list[int], int]:
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


def literal_set(text: str) -> set[str]:
    literals: set[str] = set()
    for pattern in _LITERAL_PATTERNS:
        literals.update(pattern.findall(text))
    return literals


def literal_missing_from(literal: str, zh_text: str) -> bool:
    """True when the literal does not appear as a whole token in the mirror.

    Word boundaries make the check substring-proof: ``source_revisionX`` does
    not satisfy ``source_revision``.
    """
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(literal)}(?![A-Za-z0-9_])", zh_text) is None


def bilingual_parity_errors(report_en: Path, report_zh: Path) -> list[str]:
    errors: list[str] = []
    try:
        en_headings, en_fences = outline_counts(report_en)
        zh_headings, zh_fences = outline_counts(report_zh)
    except UnicodeDecodeError:
        return ["REPORT.zh-CN.md or REPORT.md is not valid UTF-8 text"]
    if en_headings != zh_headings:
        errors.append(
            f"heading outline drift: REPORT.md has H1/H2/H3 = {en_headings}, "
            f"but REPORT.zh-CN.md has {zh_headings}"
        )
    if en_fences != zh_fences:
        errors.append(
            f"fenced block count drift: REPORT.md has {en_fences}, REPORT.zh-CN.md has {zh_fences}"
        )
    en_literals = literal_set(report_en.read_text(encoding="utf-8"))
    zh_text = report_zh.read_text(encoding="utf-8")
    missing = sorted(literal for literal in en_literals if literal_missing_from(literal, zh_text))
    if missing:
        shown = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        errors.append(f"literal parity drift: identifiers or metric names missing from REPORT.zh-CN.md: {shown}{more}")
    return errors


# ---------------------------------------------------------------------------
# Bundle validation.
# ---------------------------------------------------------------------------


def _report_errors(report_path: Path, role: str, declared_roles: set[str]) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    if role not in declared_roles:
        errors.append(("missing-report", f"{report_path.name} (role {role}) not declared in artifacts"))
    elif not report_path.exists():
        errors.append(("missing-report", f"{report_path.name} declared but file missing"))
    return errors


def scan_file_sensitive(path: Path) -> list[str]:
    """Sensitive shapes in a bundle file, guarding against binary and huge files."""
    try:
        if path.stat().st_size > MAX_SCAN_BYTES:
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_sensitive_shapes(text)


def retrieval_metric_errors(section: dict[str, object], rel_path: str) -> list[tuple[str, str]]:
    """Metric completeness for a retrieval section.

    Every evaluation mode declared in ``modes`` (migration excluded, because it
    is the separately reported availability regime) must carry the complete
    answerable-query metric family, and unit-bounded values are enforced:
    ratio in [0, 1], rank >= 1, ms >= 0.
    """
    errors: list[tuple[str, str]] = []
    modes = section.get("modes")
    metrics = section.get("metrics")
    if not isinstance(modes, list) or not isinstance(metrics, list):
        return errors  # structural violations are already reported as schema-violation
    names_by_mode: dict[str, set[str]] = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        mode = metric.get("mode")
        name = metric.get("name")
        if isinstance(mode, str) and isinstance(name, str):
            names_by_mode.setdefault(mode, set()).add(name)
        value = metric.get("value")
        unit = metric.get("unit")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(unit, str):
            if unit == "ratio" and not 0 <= value <= 1:
                errors.append(("metric-completeness", f"{rel_path}: ratio metric {name} value {value} outside [0, 1]"))
            elif unit == "rank" and value < 1:
                errors.append(("metric-completeness", f"{rel_path}: rank metric {name} value {value} below 1"))
            elif unit == "ms" and value < 0:
                errors.append(("metric-completeness", f"{rel_path}: duration metric {name} value {value} is negative"))
    for mode in modes:
        if mode == "migration" or not isinstance(mode, str):
            continue
        present = names_by_mode.get(mode, set())
        for required in REQUIRED_RETRIEVAL_METRICS:
            if required not in present:
                errors.append(("metric-completeness", f"{rel_path}: mode {mode} missing required metric {required}"))
    return errors


def prompt_injection_consistency_errors(section: dict[str, object], rel_path: str) -> list[tuple[str, str]]:
    """Ensure public adversarial case results have an unambiguous meaning."""
    errors: list[tuple[str, str]] = []
    cases = section.get("cases")
    if not isinstance(cases, list):
        return errors  # Structural violations are already reported as schema-violation.
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        pass_fail = case.get("pass_fail")
        failure_classification = case.get("failure_classification")
        if pass_fail == "pass" and failure_classification != "none":
            errors.append(
                (
                    "prompt-injection-consistency",
                    f"{rel_path}: case {case_id} pass requires failure_classification none",
                )
            )
        elif pass_fail == "fail" and failure_classification == "none":
            errors.append(
                (
                    "prompt-injection-consistency",
                    f"{rel_path}: case {case_id} fail requires a non-none failure_classification",
                )
            )
    return errors


def validate_bundle(
    bundle_dir: Path,
    manifest_schema: dict[str, object],
    section_schemas: dict[str, dict[str, object]],
) -> list[tuple[str, str]]:
    """Return [(check, detail), ...]; empty means the bundle conforms."""
    bundle_dir = bundle_dir.resolve()
    errors: list[tuple[str, str]] = []

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return [("missing-manifest", "manifest.json not found in bundle directory")]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [("invalid-json", "manifest.json is not valid JSON")]
    if not isinstance(manifest, dict):
        return [("invalid-json", "manifest.json is not a JSON object")]

    for violation in schema_errors(manifest, manifest_schema, "manifest"):
        errors.append(("schema-violation", violation))

    # Sensitive shapes in the manifest itself.
    for shape in scan_sensitive_shapes(manifest_path.read_text(encoding="utf-8")):
        errors.append(("sensitive-shape", f"manifest.json contains {shape} shape"))

    # Early return keeps downstream checks type-safe on a structurally broken manifest.
    required_manifest_keys = ("source_revision", "release_candidate", "provenance", "artifacts", "limits")
    if not all(isinstance(manifest.get(key), (str, int, float, list, dict)) for key in required_manifest_keys):
        return errors

    # --- Provenance cross-references -------------------------------------------------
    source_revision = str(manifest.get("source_revision", ""))
    candidate = manifest.get("release_candidate")
    candidate_revision = str(candidate.get("revision", "")) if isinstance(candidate, dict) else ""
    if candidate_revision and candidate_revision != source_revision:
        errors.append(("provenance-reference", "release_candidate.revision does not equal source_revision"))

    provenance = manifest.get("provenance")
    revisions = set()
    runs: list[dict[str, object]] = []
    corpora: list[dict[str, object]] = []
    query_sets: list[dict[str, object]] = []
    if isinstance(provenance, dict):
        raw_revisions = provenance.get("revisions")
        if isinstance(raw_revisions, list):
            revisions = {item for item in raw_revisions if isinstance(item, str)}
        raw_runs = provenance.get("runs")
        if isinstance(raw_runs, list):
            runs = [item for item in raw_runs if isinstance(item, dict)]
        raw_corpora = provenance.get("corpora")
        if isinstance(raw_corpora, list):
            corpora = [item for item in raw_corpora if isinstance(item, dict)]
        raw_query_sets = provenance.get("query_sets")
        if isinstance(raw_query_sets, list):
            query_sets = [item for item in raw_query_sets if isinstance(item, dict)]

    if source_revision and source_revision not in revisions:
        errors.append(("provenance-reference", "source_revision missing from provenance.revisions"))

    known_run_ids: set[str] = set()
    for run in runs:
        run_id = str(run.get("run_id", ""))
        if run_id in known_run_ids:
            errors.append(("provenance-reference", f"duplicate run_id {run_id}"))
        known_run_ids.add(run_id)
        run_revision = str(run.get("source_revision", ""))
        if run_revision and run_revision not in revisions and run_revision != source_revision:
            errors.append(("provenance-reference", f"run {run_id} source_revision not in provenance.revisions"))

    known_corpus_ids: set[str] = set()
    for item in corpora:
        corpus_id = str(item.get("corpus_id", ""))
        if corpus_id in known_corpus_ids:
            errors.append(("provenance-reference", f"duplicate corpus_id {corpus_id}"))
        known_corpus_ids.add(corpus_id)
    known_query_set_ids: set[str] = set()
    for item in query_sets:
        query_set_id = str(item.get("query_set_id", ""))
        if query_set_id in known_query_set_ids:
            errors.append(("provenance-reference", f"duplicate query_set_id {query_set_id}"))
        known_query_set_ids.add(query_set_id)

    # --- Artifact inventory and hashes ------------------------------------------------
    artifacts = manifest.get("artifacts")
    declared_roles: set[str] = set()
    declared_sections: list[tuple[str, str]] = []  # (artifact path, kind)
    manifest_in_inventory = False
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            rel_path = str(artifact.get("path", ""))
            role = str(artifact.get("role", ""))
            kind = str(artifact.get("kind", ""))
            declared_roles.add(role)
            target = (bundle_dir / rel_path).resolve()
            if ".." in rel_path or rel_path.startswith("/") or not target.is_relative_to(bundle_dir):
                errors.append(("artifact-hash", f"{rel_path}: artifact path escapes the bundle directory"))
                continue
            if not target.exists():
                errors.append(("artifact-hash", f"{rel_path}: declared artifact file missing"))
                continue
            declared_sha = str(artifact.get("sha256", ""))
            if rel_path != "manifest.json":
                actual_sha = sha256(target.read_bytes()).hexdigest()
                if declared_sha != actual_sha:
                    errors.append(
                        (
                            "artifact-hash",
                            f"{rel_path}: sha256 mismatch (declared {declared_sha[:12]}…, actual {actual_sha[:12]}…)",
                        )
                    )
            if rel_path == "manifest.json":
                if kind == "manifest" and role == "manifest":
                    manifest_in_inventory = True
            else:
                # Every artifact file is content-scanned for sensitive shapes,
                # not only typed sections and reports.
                for shape in scan_file_sensitive(target):
                    errors.append(("sensitive-shape", f"{rel_path} contains {shape} shape"))
            if role == "section":
                declared_sections.append((rel_path, kind))
    else:
        errors.append(("schema-violation", "manifest.artifacts must be an array"))

    if not manifest_in_inventory:
        errors.append(("manifest-inventory", "artifacts must declare manifest.json with kind manifest and role manifest"))

    if not isinstance(manifest.get("limits"), list) or not manifest["limits"]:
        errors.append(("missing-limits", "manifest.limits must declare at least one limit"))

    if not declared_sections:
        errors.append(("no-sections", "bundle must contain at least one typed section artifact"))

    # --- Typed section files ----------------------------------------------------------
    for rel_path, kind in declared_sections:
        section_schema = section_schemas.get(kind)
        if section_schema is None:
            errors.append(("schema-violation", f"{rel_path}: unknown section kind {kind}"))
            continue
        section_path = (bundle_dir / rel_path).resolve()
        try:
            section = json.loads(section_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append(("invalid-json", f"{rel_path}: section file is not valid JSON"))
            continue
        if not isinstance(section, dict):
            errors.append(("invalid-json", f"{rel_path}: section file is not a JSON object"))
            continue
        for violation in schema_errors(section, section_schema, rel_path):
            errors.append(("schema-violation", violation))
        if str(section.get("section", "")) != kind:
            errors.append(
                ("schema-violation", f"{rel_path}: section discriminator {section.get('section')!r} does not match artifact kind {kind}")
            )
        for shape in scan_file_sensitive(section_path):
            errors.append(("sensitive-shape", f"{rel_path} contains {shape} shape"))

        # Metric completeness and unit bounds for retrieval sections.
        if kind == "retrieval":
            errors.extend(retrieval_metric_errors(section, rel_path))
            conditions = section.get("conditions")
            if isinstance(conditions, dict):
                corpus_version = conditions.get("corpus_version")
                if isinstance(corpus_version, str):
                    for item in corpora:
                        if item.get("corpus_id") == section.get("corpus_id") and item.get("version") != corpus_version:
                            errors.append(
                                (
                                    "provenance-reference",
                                    f"{rel_path}: conditions.corpus_version {corpus_version} does not match "
                                    f"provenance version {item.get('version')}",
                                )
                            )
                query_set_version = conditions.get("query_set_version")
                if isinstance(query_set_version, str):
                    for item in query_sets:
                        if item.get("query_set_id") == section.get("query_set_id") and item.get("version") != query_set_version:
                            errors.append(
                                (
                                    "provenance-reference",
                                    f"{rel_path}: conditions.query_set_version {query_set_version} does not match "
                                    f"provenance version {item.get('version')}",
                                )
                            )

        if kind == "prompt-injection":
            errors.extend(prompt_injection_consistency_errors(section, rel_path))

        # Cross-artifact references from sections into provenance.
        for run_id in section.get("run_ids", []):
            if run_id not in known_run_ids:
                errors.append(("provenance-reference", f"{rel_path}: run_id {run_id} not in provenance.runs"))
        corpus_id = section.get("corpus_id")
        if isinstance(corpus_id, str) and corpus_id not in known_corpus_ids:
            errors.append(("provenance-reference", f"{rel_path}: corpus_id {corpus_id} not in provenance.corpora"))
        query_set_id = section.get("query_set_id")
        if isinstance(query_set_id, str) and query_set_id not in known_query_set_ids:
            errors.append(("provenance-reference", f"{rel_path}: query_set_id {query_set_id} not in provenance.query_sets"))

    # --- Bilingual report pair --------------------------------------------------------
    errors.extend(_report_errors(bundle_dir / REPORT_EN_PATH, "report-en", declared_roles))
    errors.extend(_report_errors(bundle_dir / REPORT_ZH_PATH, "report-zh", declared_roles))
    report_en = bundle_dir / REPORT_EN_PATH
    report_zh = bundle_dir / REPORT_ZH_PATH
    if report_en.exists() and report_zh.exists():
        for violation in bilingual_parity_errors(report_en, report_zh):
            errors.append(("bilingual-parity", violation))
        for shape in scan_sensitive_shapes(report_en.read_text(encoding="utf-8")):
            errors.append(("sensitive-shape", f"{REPORT_EN_PATH} contains {shape} shape"))
        for shape in scan_sensitive_shapes(report_zh.read_text(encoding="utf-8")):
            errors.append(("sensitive-shape", f"{REPORT_ZH_PATH} contains {shape} shape"))

    return errors


def discover_bundles() -> list[Path]:
    """Example bundle plus every tracked release bundle under public-evidence/releases."""
    discovered: list[Path] = []
    example_dir = REPO_ROOT / "public-evidence" / "example"
    if (example_dir / "manifest.json").exists():
        discovered.append(example_dir)
    releases_dir = REPO_ROOT / "public-evidence" / "releases"
    if releases_dir.is_dir():
        for entry in sorted(releases_dir.iterdir()):
            if entry.is_dir() and (entry / "manifest.json").exists():
                discovered.append(entry)
    return discovered


def load_contract() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    manifest_schema = load_schema(MANIFEST_SCHEMA_PATH)
    section_schemas = {kind: load_schema(CONTRACT_DIR / "sections" / f"{kind}.schema.json") for kind in SECTION_KINDS}
    return manifest_schema, section_schemas


def bundle_id_of(bundle_dir: Path) -> str:
    manifest_path = bundle_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "<invalid>"
    if isinstance(manifest, dict) and isinstance(manifest.get("bundle_id"), str):
        return manifest["bundle_id"]
    return "<invalid>"


def _display(bundle_dir: Path) -> str:
    try:
        return bundle_dir.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(bundle_dir)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate-evidence-bundle.py",
        description="Validate Public Evidence Bundles against the versioned contract.",
    )
    parser.add_argument("--all", action="store_true", help="validate the example bundle and every tracked release bundle")
    parser.add_argument("--list", action="store_true", help="list discovered bundles without validating")
    parser.add_argument("bundle_dirs", nargs="*", help="bundle directories to validate")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for bundle_dir in discover_bundles():
            print(f"{_display(bundle_dir)} {bundle_id_of(bundle_dir)}")
        return 0
    if not args.all and not args.bundle_dirs:
        print("validate-evidence-bundle.py: error: provide bundle directories or --all", file=sys.stderr)
        return 2

    manifest_schema, section_schemas = load_contract()
    if args.all:
        bundle_dirs = discover_bundles()
        if not bundle_dirs:
            print("validate-evidence-bundle.py: no bundles discovered under public-evidence/", file=sys.stderr)
            return 1
    else:
        bundle_dirs = [Path(item) for item in args.bundle_dirs]

    failed = 0
    seen_bundle_ids: dict[str, str] = {}
    for bundle_dir in bundle_dirs:
        bundle_id = bundle_id_of(bundle_dir)
        if bundle_id != "<invalid>" and bundle_id in seen_bundle_ids:
            failed += 1
            print(
                f"ERROR {_display(bundle_dir)}: duplicate-bundle-id — bundle_id {bundle_id} "
                f"also used by {seen_bundle_ids[bundle_id]}"
            )
        seen_bundle_ids.setdefault(bundle_id, _display(bundle_dir))
        errors = validate_bundle(bundle_dir, manifest_schema, section_schemas)
        if errors:
            failed += 1
            for check, detail in errors:
                print(f"ERROR {_display(bundle_dir)}: {check} — {detail}")
        else:
            print(f"OK {_display(bundle_dir)}: bundle {bundle_id} schema 1.0.0 passes all contract checks")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
