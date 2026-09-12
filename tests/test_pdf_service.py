"""Tests for PDF text extraction, including the failure cases."""

from __future__ import annotations

import pytest

from backend.services.pdf_service import PDFExtractionError, extract_text_from_pdf


def test_extracts_text_from_a_valid_pdf(text_pdf_bytes):
    result = extract_text_from_pdf(text_pdf_bytes)

    assert result.page_count == 1
    assert result.char_count > 0
    assert "Jane Doe" in result.text
    assert "PyTorch" in result.text


def test_extracted_text_is_cleaned(text_pdf_bytes):
    result = extract_text_from_pdf(text_pdf_bytes)

    assert "  " not in result.text  # runs of spaces collapsed
    assert result.text == result.text.strip()


def test_empty_upload_is_rejected():
    with pytest.raises(PDFExtractionError, match="empty"):
        extract_text_from_pdf(b"")


def test_non_pdf_file_is_rejected():
    with pytest.raises(PDFExtractionError, match="does not look like a PDF"):
        extract_text_from_pdf(b"this is a plain text file, not a pdf")


def test_corrupt_pdf_is_rejected():
    # Starts with the right magic bytes but the body is garbage.
    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(b"%PDF-1.4\n" + b"\x00\xff" * 200)


def test_image_only_pdf_reports_the_ocr_limitation(image_only_pdf_bytes):
    """A scanned resume has no text layer - we must say so, not return "" silently."""
    with pytest.raises(PDFExtractionError, match="scanned|image-only|OCR"):
        extract_text_from_pdf(image_only_pdf_bytes)


def test_oversized_upload_is_rejected(monkeypatch):
    from backend.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 0)  # everything is too big now
    with pytest.raises(PDFExtractionError, match="too large"):
        extract_text_from_pdf(b"%PDF-1.4 " + b"x" * 100)


def test_sample_resume_pdf_round_trips(tmp_path):
    """The generated demo PDF must be readable by the extractor."""
    from pathlib import Path

    sample = Path(__file__).resolve().parent.parent / "data" / "sample_resume.pdf"
    if not sample.exists():
        pytest.skip("run scripts/generate_sample_pdf.py first")

    result = extract_text_from_pdf(sample.read_bytes())
    assert "ALEX MORGAN" in result.text
    assert result.char_count > 500
