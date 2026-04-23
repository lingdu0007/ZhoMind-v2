from __future__ import annotations

from io import BytesIO
from pathlib import Path

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    PdfReader = None  # type: ignore[assignment]

from app.common.exceptions import AppError
from app.documents.types import ParsedDocument

_SUPPORTED_EXTENSIONS = {"txt", "md", "pdf"}


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

    return ParsedDocument(source_file=filename, file_type=file_type, text=text)
