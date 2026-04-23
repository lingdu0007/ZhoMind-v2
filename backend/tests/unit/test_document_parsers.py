from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from app.common.exceptions import AppError
from app.documents.parsers import parse_document
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
