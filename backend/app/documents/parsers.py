from __future__ import annotations

import re
import socket
from collections.abc import Callable, Mapping
from datetime import date, datetime
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    PdfReader = None  # type: ignore[assignment]

from app.common.exceptions import AppError
from app.documents.types import ParsedDocument

_SUPPORTED_EXTENSIONS = {"txt", "md", "pdf"}
_AGENT_ENTRY_REQUIRED_FIELDS = {
    "entry_id",
    "title",
    "domain",
    "review_status",
    "applicable_versions",
    "review_date",
    "sources",
}
_SENSITIVE_QUERY_PARTS = ("credential", "password", "secret", "signature", "signed", "token")
_REDIRECT_QUERY_KEYS = {"continue", "next", "redirect", "redirect_uri", "return_to", "target", "url"}
_SOURCE_REQUIRED_FIELDS = {"authority", "availability", "title", "url", "version"}


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _probe_public_source(url: str) -> None:
    hostname = urlsplit(url).hostname or ""
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    }
    if not addresses or any(not ip_address(address).is_global for address in addresses):
        raise OSError("source hostname does not resolve exclusively to public addresses")

    request = Request(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "ZhoMind-Source-Validator/1.0"},
        method="GET",
    )
    opener = build_opener(_RejectRedirects())
    try:
        with opener.open(request, timeout=5) as response:
            if not 200 <= response.status < 300:
                raise OSError("source returned a non-success status")
    except HTTPError as exc:
        if 300 <= exc.code < 400:
            raise OSError("source URL redirects and is not canonical") from exc
        raise


def validate_canonical_source_url(url: str, *, source_probe: Callable[[str], None]) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise AppError(
            status_code=400,
            code="AGENT_SOURCE_URL_INVALID",
            message="source URL is invalid",
            detail={"url": url},
        ) from exc

    hostname = (parsed.hostname or "").rstrip(".").lower()
    unsafe = (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or bool(parsed.fragment)
        or hostname == "localhost"
        or hostname.endswith((".localhost", ".local", ".internal"))
        or "." not in hostname
    )
    try:
        address = ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        unsafe = True

    for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key in _REDIRECT_QUERY_KEYS or any(part in normalized_key for part in _SENSITIVE_QUERY_PARTS):
            unsafe = True

    if unsafe:
        raise AppError(
            status_code=400,
            code="AGENT_SOURCE_URL_INVALID",
            message="source URL must be a credential-free public HTTPS canonical URL",
            detail={"url": url},
        )

    try:
        source_probe(url)
    except Exception as exc:
        raise AppError(
            status_code=400,
            code="AGENT_SOURCE_INACCESSIBLE",
            message="source URL is not publicly accessible",
            detail={"url": url},
        ) from exc


def _metadata_value(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metadata_value(item) for item in value]
    return value


def parse_agent_entry(
    filename: str,
    content: bytes,
    *,
    source_probe: Callable[[str], None] | None = None,
) -> ParsedDocument:
    parsed = parse_document(filename, content)
    if parsed.file_type != "md":
        raise AppError(
            status_code=400,
            code="AGENT_ENTRY_MARKDOWN_REQUIRED",
            message="Agent entry must be a Markdown document",
        )

    lines = parsed.text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AppError(
            status_code=400,
            code="AGENT_ENTRY_FRONT_MATTER_REQUIRED",
            message="Agent entry must start with YAML front matter",
        )
    try:
        closing_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise AppError(
            status_code=400,
            code="AGENT_ENTRY_FRONT_MATTER_INVALID",
            message="Agent entry YAML front matter is not closed",
        ) from exc

    try:
        loaded = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as exc:
        raise AppError(
            status_code=400,
            code="AGENT_ENTRY_FRONT_MATTER_INVALID",
            message="Agent entry YAML front matter is invalid",
        ) from exc
    metadata = dict(loaded) if isinstance(loaded, Mapping) else {}
    invalid_fields = {
        field
        for field in _AGENT_ENTRY_REQUIRED_FIELDS
        if field not in metadata or metadata[field] in (None, "", [])
    }
    for field in ("entry_id", "title", "domain", "review_status"):
        if field in metadata and (not isinstance(metadata[field], str) or not metadata[field].strip()):
            invalid_fields.add(field)
    entry_id = metadata.get("entry_id")
    if isinstance(entry_id, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", entry_id) is None:
        invalid_fields.add("entry_id")
    applicable_versions = metadata.get("applicable_versions")
    if not isinstance(applicable_versions, list) or not applicable_versions or any(
        not isinstance(version, str) or not version.strip() for version in applicable_versions
    ):
        invalid_fields.add("applicable_versions")
    raw_review_date = metadata.get("review_date")
    if isinstance(raw_review_date, datetime):
        raw_review_date = raw_review_date.date()
    if not isinstance(raw_review_date, date):
        try:
            date.fromisoformat(raw_review_date) if isinstance(raw_review_date, str) else None
        except ValueError:
            invalid_fields.add("review_date")
        else:
            if not isinstance(raw_review_date, str):
                invalid_fields.add("review_date")

    raw_sources = metadata.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        invalid_fields.add("sources")
    elif isinstance(raw_sources, list):
        for index, source in enumerate(raw_sources):
            if not isinstance(source, Mapping):
                invalid_fields.add(f"sources[{index}]")
                continue
            for field in _SOURCE_REQUIRED_FIELDS:
                value = source.get(field)
                if not isinstance(value, (str, date, datetime)) or not str(value).strip():
                    invalid_fields.add(f"sources[{index}].{field}")

    if invalid_fields:
        raise AppError(
            status_code=400,
            code="AGENT_ENTRY_METADATA_INVALID",
            message="Agent entry metadata has missing or invalid fields",
            detail={"fields": sorted(invalid_fields)},
        )

    normalized = _metadata_value(metadata)
    assert isinstance(normalized, dict)
    sources = normalized.get("sources")
    probe = source_probe or _probe_public_source
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and isinstance(source.get("url"), str):
                validate_canonical_source_url(source["url"], source_probe=probe)

    body = "\n".join(lines[closing_index + 1 :]).lstrip()
    return ParsedDocument(
        source_file=parsed.source_file,
        file_type=parsed.file_type,
        text=body,
        metadata=normalized,
    )


def parse_document(filename: str, content: bytes) -> ParsedDocument:
    file_type = Path(filename).suffix.lower().lstrip(".")
    if file_type not in _SUPPORTED_EXTENSIONS:
        raise AppError(
            status_code=415,
            code="DOC_FILE_TYPE_NOT_SUPPORTED",
            message="document file type not supported",
            detail={"file_type": file_type or "unknown"},
        )

    if file_type in {"txt", "md"}:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AppError(
                status_code=400,
                code="DOC_TEXT_ENCODING_INVALID",
                message="text document must be valid UTF-8",
                detail={"file_type": file_type},
            ) from exc
    else:
        if PdfReader is None:
            raise AppError(
                status_code=500,
                code="DOC_PARSER_DEPENDENCY_MISSING",
                message="pdf parser dependency is not installed",
            )
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise AppError(
                status_code=400,
                code="DOC_PDF_INVALID",
                message="invalid or corrupt pdf document",
                detail={"file_type": file_type},
            ) from exc
        if not text.strip():
            raise AppError(
                status_code=400,
                code="DOC_PDF_TEXT_NOT_EXTRACTABLE",
                message="pdf document must contain extractable text",
                detail={"file_type": file_type},
            )

    return ParsedDocument(source_file=filename, file_type=file_type, text=text)
