"""
PDF text extraction.

Primary extractor : PyMuPDF  - fast and accurate on digital PDFs
Fallback extractor: pypdf           - used if PyMuPDF fails to parse the file

LIMITATION (stated up front, and repeated in the UI): this is *text* extraction,
not OCR. A scanned resume - i.e. a photo of a page wrapped in a PDF - contains
no text layer, so nothing can be pulled out of it. We detect that case and
return a clear error instead of silently matching an empty document.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.config import settings
from backend.text_utils import clean_text


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass
class ExtractionResult:
    """Outcome of extracting text from one PDF."""

    text: str
    page_count: int
    char_count: int
    warning: Optional[str] = None


def _extract_with_pymupdf(pdf_bytes: bytes) -> tuple[str, int]:
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PDFExtractionError(
                "This PDF is password protected. Please upload an unlocked copy."
            )
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages), len(pages)


def _extract_with_pypdf(pdf_bytes: bytes) -> tuple[str, int]:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - depends on the file
            raise PDFExtractionError("This PDF is password protected.") from exc
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages), len(pages)


def extract_text_from_pdf(pdf_bytes: bytes) -> ExtractionResult:
    """Extract clean text from PDF bytes.

    Raises PDFExtractionError with a human-readable message for: empty uploads,
    oversized uploads, non-PDF files, corrupt files, and image-only (scanned)
    PDFs where no text layer exists.
    """
    if not pdf_bytes:
        raise PDFExtractionError("The uploaded file is empty (0 bytes).")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise PDFExtractionError(
            f"File is too large ({len(pdf_bytes) / 1_048_576:.1f} MB). "
            f"Limit is {settings.max_upload_mb} MB."
        )

    # Every real PDF starts with the magic bytes "%PDF" (allow a small offset,
    # some generators put junk in front of the header).
    if b"%PDF" not in pdf_bytes[:1024]:
        raise PDFExtractionError(
            "This does not look like a PDF file. Please upload a .pdf resume."
        )

    raw_text, page_count, last_error = "", 0, None
    for extractor in (_extract_with_pymupdf, _extract_with_pypdf):
        try:
            raw_text, page_count = extractor(pdf_bytes)
            if raw_text.strip():
                break  # got something usable
        except PDFExtractionError:
            raise  # password protection etc. - do not retry with the other lib
        except Exception as exc:  # corrupt file, unsupported feature, ...
            last_error = exc

    if not raw_text.strip():
        if last_error is not None and page_count == 0:
            raise PDFExtractionError(
                "The PDF could not be parsed - it may be corrupted. "
                f"({type(last_error).__name__})"
            )
        raise PDFExtractionError(
            "No text could be extracted from this PDF. It is most likely a "
            "scanned/image-only document. This system does not include OCR, so "
            "please upload a text-based PDF (e.g. exported from Word or LaTeX)."
        )

    text = clean_text(raw_text)

    warning = None
    if len(text) < settings.min_extracted_chars:
        warning = (
            f"Only {len(text)} characters were extracted. The result may be "
            "unreliable - the PDF might be mostly images."
        )

    return ExtractionResult(
        text=text,
        page_count=page_count,
        char_count=len(text),
        warning=warning,
    )
