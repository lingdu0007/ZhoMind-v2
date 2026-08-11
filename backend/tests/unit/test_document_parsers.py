from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from app.common.exceptions import AppError
from app.documents.parsers import parse_agent_entry, parse_document, validate_canonical_source_url
from app.documents.types import ParsedDocument


def test_parsed_document_is_dataclass() -> None:
    assert is_dataclass(ParsedDocument)


def test_parse_txt_document_success() -> None:
    parsed = parse_document("notes.txt", b"line 1\nline 2")

    assert parsed == ParsedDocument(source_file="notes.txt", file_type="txt", text="line 1\nline 2")


def test_parse_md_document_success() -> None:
    parsed = parse_document("README.MD", b"# Title\n\nBody")

    assert parsed.source_file == "README.MD"
    assert parsed.file_type == "md"
    assert parsed.text == "# Title\n\nBody"


@pytest.mark.parametrize("filename", ["notes.txt", "README.md"])
def test_parse_text_document_rejects_invalid_utf8(filename: str) -> None:
    with pytest.raises(AppError) as exc_info:
        parse_document(filename, b"line 1\xffline 2")

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "DOC_TEXT_ENCODING_INVALID"


def test_parse_pdf_document_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(self) -> str | None:
            return self._text

    class _FakePdfReader:
        def __init__(self, _: object) -> None:
            self.pages = [_FakePage("page one"), _FakePage(None), _FakePage("page three")]

    monkeypatch.setattr("app.documents.parsers.PdfReader", _FakePdfReader)

    parsed = parse_document("paper.pdf", b"%PDF-1.4\nstub")

    assert parsed.source_file == "paper.pdf"
    assert parsed.file_type == "pdf"
    assert parsed.text == "page one\n\npage three"


def test_parse_pdf_document_rejects_when_no_text_can_be_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePage:
        def extract_text(self) -> None:
            return None

    class _EmptyPdfReader:
        def __init__(self, _: object) -> None:
            self.pages = [_FakePage()]

    monkeypatch.setattr("app.documents.parsers.PdfReader", _EmptyPdfReader)

    with pytest.raises(AppError) as exc_info:
        parse_document("scanned.pdf", b"%PDF-1.4\nstub")

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "DOC_PDF_TEXT_NOT_EXTRACTABLE"


def test_parse_pdf_document_rejects_invalid_or_corrupt_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenPdfReader:
        def __init__(self, _: object) -> None:
            raise ValueError("malformed pdf")

    monkeypatch.setattr("app.documents.parsers.PdfReader", _BrokenPdfReader)

    with pytest.raises(AppError) as exc_info:
        parse_document("broken.pdf", b"definitely-not-a-real-pdf")

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "DOC_PDF_INVALID"


def test_parse_document_rejects_unsupported_extension() -> None:
    with pytest.raises(AppError) as exc_info:
        parse_document("data.csv", b"a,b,c")

    assert exc_info.value.status_code == 415
    assert exc_info.value.code == "DOC_FILE_TYPE_NOT_SUPPORTED"


_VALID_AGENT_ENTRY = b"""---
entry_id: pae-workflow-001
title: Prefer deterministic workflows when the path is known
domain: workflow-vs-agent
review_status: approved
applicable_versions:
  - framework-neutral
review_date: 2026-08-12
sources:
  - title: Building Effective Agents
    authority: Anthropic
    url: https://www.anthropic.com/research/building-effective-agents
    version: 2024-12-19
    availability: verified
---
# Decision Question

When should a deterministic workflow be preferred over an Agent?
"""


def test_parse_agent_entry_extracts_valid_front_matter_from_body() -> None:
    parsed = parse_agent_entry("workflow.md", _VALID_AGENT_ENTRY, source_probe=lambda _url: None)

    assert parsed.text.startswith("# Decision Question")
    assert parsed.metadata["entry_id"] == "pae-workflow-001"
    assert parsed.metadata["title"] == "Prefer deterministic workflows when the path is known"
    assert parsed.metadata["domain"] == "workflow-vs-agent"
    assert parsed.metadata["review_date"] == "2026-08-12"
    assert parsed.metadata["sources"] == [
        {
            "title": "Building Effective Agents",
            "authority": "Anthropic",
            "url": "https://www.anthropic.com/research/building-effective-agents",
            "version": "2024-12-19",
            "availability": "verified",
        }
    ]


def test_parse_agent_entry_reports_all_missing_required_metadata_fields() -> None:
    content = b"""---
title: Incomplete entry
domain: workflow-vs-agent
review_status: approved
---
# Decision Question
"""

    with pytest.raises(AppError) as exc_info:
        parse_agent_entry("incomplete.md", content, source_probe=lambda _url: None)

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "AGENT_ENTRY_METADATA_INVALID"
    assert exc_info.value.detail == {
        "fields": ["applicable_versions", "entry_id", "review_date", "sources"]
    }


def test_parse_agent_entry_reports_incomplete_source_and_version_metadata() -> None:
    content = b"""---
entry_id: pae-workflow-001
title: Incomplete source entry
domain: workflow-vs-agent
review_status: approved
applicable_versions: []
review_date: not-a-date
sources:
  - url: https://example.com/source
---
# Decision Question
"""

    with pytest.raises(AppError) as exc_info:
        parse_agent_entry("incomplete-source.md", content, source_probe=lambda _url: None)

    assert exc_info.value.code == "AGENT_ENTRY_METADATA_INVALID"
    assert exc_info.value.detail == {
        "fields": [
            "applicable_versions",
            "review_date",
            "sources[0].authority",
            "sources[0].availability",
            "sources[0].title",
            "sources[0].version",
        ]
    }


def test_parse_agent_entry_rejects_unstable_entry_identity_format() -> None:
    content = _VALID_AGENT_ENTRY.replace(b"pae-workflow-001", b"Mutable Entry ID!")

    with pytest.raises(AppError) as exc_info:
        parse_agent_entry("unstable-id.md", content, source_probe=lambda _url: None)

    assert exc_info.value.code == "AGENT_ENTRY_METADATA_INVALID"
    assert exc_info.value.detail == {"fields": ["entry_id"]}


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/source",
        "https://user:password@example.com/source",
        "https://localhost/source",
        "https://127.0.0.1/source",
        "https://10.2.3.4/source",
        "https://example.com/source?access_token=secret",
        "https://example.com/source?redirect=https%3A%2F%2Fevil.example",
    ],
)
def test_canonical_source_url_rejects_unsafe_url_before_network_probe(url: str) -> None:
    probed: list[str] = []

    with pytest.raises(AppError) as exc_info:
        validate_canonical_source_url(url, source_probe=probed.append)

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "AGENT_SOURCE_URL_INVALID"
    assert probed == []


def test_canonical_source_url_normalizes_inaccessible_source_failure() -> None:
    def inaccessible(_url: str) -> None:
        raise OSError("connection refused with implementation details")

    with pytest.raises(AppError) as exc_info:
        validate_canonical_source_url("https://example.com/source", source_probe=inaccessible)

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "AGENT_SOURCE_INACCESSIBLE"
    assert exc_info.value.detail == {"url": "https://example.com/source"}
