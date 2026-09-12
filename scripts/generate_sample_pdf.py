"""
Turn data/sample_resume.txt into data/sample_resume.pdf.

Only needed so the project ships with a real PDF to demo the upload flow.
Run with:  python scripts/generate_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "sample_resume.txt"
TARGET = ROOT / "data" / "sample_resume.pdf"

# A4 page geometry, in points.
PAGE_W, PAGE_H = 595, 842
MARGIN = 56
FONT_SIZE = 9.5
LINE_HEIGHT = 13


def main() -> int:
    if not SOURCE.exists():
        print(f"Source not found: {SOURCE}", file=sys.stderr)
        return 1

    lines = SOURCE.read_text(encoding="utf-8").split("\n")
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = MARGIN

    for line in lines:
        if y > PAGE_H - MARGIN:  # start a new page when we run out of room
            page = doc.new_page(width=PAGE_W, height=PAGE_H)
            y = MARGIN
        if line.strip():
            # Headings (ALL CAPS lines) get a bold font.
            is_heading = line.strip().isupper() and len(line.strip()) < 40
            page.insert_text(
                (MARGIN, y),
                line,
                fontsize=FONT_SIZE + (1 if is_heading else 0),
                fontname="helv" if not is_heading else "hebo",
            )
        y += LINE_HEIGHT

    doc.save(TARGET)
    doc.close()
    print(f"Wrote {TARGET} ({TARGET.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
