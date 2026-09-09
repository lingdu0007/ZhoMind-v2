import re
from typing import Literal
from urllib.parse import unquote

from pydantic import BaseModel, ConfigDict, field_validator


class ContentAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    material_class: Literal["public_material", "team_shared_internal", "sanitized_bounded_internal_case"]
    audience: Literal["all_admitted_members"]
    sensitivity: Literal["restricted"]
    sanitized: Literal[True]

    @field_validator("sanitized", mode="before")
    @classmethod
    def require_true_boolean(cls, value: object) -> object:
        if value is not True:
            raise ValueError("sanitization requires an explicit true attestation")
        return value


def source_admission_allowed(source: dict) -> bool:
    try:
        admission = ContentAdmission.model_validate(source.get("content_admission"))
    except ValueError:
        return False
    if source.get("access_scope") == "public":
        return admission.material_class == "public_material"
    if source.get("access_scope") != "controlled_internal":
        return False
    expected = (
        "sanitized_bounded_internal_case" if source.get("source_tier") == "bounded_internal_case"
        else "team_shared_internal"
    )
    return admission.material_class == expected


_CREDENTIAL_PATTERNS = (
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai-api-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z0-9 ]* )?PRIVATE KEY-----")),
    ("database-url-with-password", re.compile(r"\b(?:postgres|postgresql|mysql)://[^/\s:@]+:[^@\s/]+@")),
    ("credential-assignment", re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|secret|token)\s*[:=]\s*\S+")),
)
_PII_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
)


def credential_findings(value: str) -> list[str]:
    """Return rule names only, never matching private values."""
    return [name for name, pattern in _CREDENTIAL_PATTERNS if pattern.search(value)]


def recognizable_private_material(value: str) -> bool:
    """Bounded pattern check, never a substitute for accountable content review."""
    # Inspect raw text plus three decoding rounds; deeper encodings fail closed.
    for _ in range(4):
        if credential_findings(value) or any(pattern.search(value) for pattern in _PII_PATTERNS):
            return True
        decoded = unquote(value)
        if decoded == value:
            return False
        value = decoded
    return True
