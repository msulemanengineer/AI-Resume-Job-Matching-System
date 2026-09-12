"""
Shared pytest fixtures.

The embedding model is downloaded on first use (~80 MB) and loading it takes a
few seconds, so tests that need it are marked `slow` and the model is loaded at
most once per test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `backend` importable when pytest is run from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: needs the embedding model (downloads ~80 MB on first run)"
    )


@pytest.fixture(scope="session")
def sample_resume_text() -> str:
    return (DATA_DIR / "sample_resume.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def sample_job_text() -> str:
    return (DATA_DIR / "sample_job_description.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def unrelated_job_text() -> str:
    """A job description from a completely different field.

    Used to check that the similarity score actually discriminates - a matcher
    that returns a high score for everything is useless.
    """
    return (
        "Head Pastry Chef. We are hiring an experienced pastry chef for our "
        "bakery. You will design seasonal dessert menus, laminate croissant "
        "dough daily, manage kitchen stock and food-safety paperwork, train "
        "junior bakers, and run the morning bread service. Requires a culinary "
        "diploma, at least five years in a professional kitchen, and a valid "
        "food hygiene certificate."
    )


@pytest.fixture(scope="session")
def text_pdf_bytes() -> bytes:
    """A small, valid, text-based PDF built on the fly with PyMuPDF."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Jane Doe - Machine Learning Engineer", fontsize=14)
    page.insert_text((72, 130), "Skills: Python, PyTorch, Docker, SQL", fontsize=11)
    page.insert_text((72, 150), "Built NLP models and deployed them with FastAPI.", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="session")
def image_only_pdf_bytes() -> bytes:
    """A valid PDF with a drawing but no text layer - stands in for a scan."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 300, 300), color=(0, 0, 0), fill=(0.6, 0.6, 0.6))
    data = doc.tobytes()
    doc.close()
    return data
