"""
API tests using FastAPI's TestClient (no running server required).

TestClient sends real HTTP requests through the ASGI app in-process, so these
tests exercise routing, Pydantic validation and error handling exactly as a
real client would.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

LONG_RESUME = (
    "Experienced Python developer. Built machine learning models with "
    "scikit-learn and PyTorch, served them behind a FastAPI service, and "
    "deployed with Docker. Comfortable with SQL, Pandas, NumPy and Git."
)
LONG_JOB = (
    "We need an AI/ML intern with Python, scikit-learn and NLP experience. "
    "You will build REST APIs with FastAPI and work with SQL databases. "
    "Docker and AWS knowledge is a plus."
)


# ---------------------------------------------------------------------------
# System endpoints (fast - no model)
# ---------------------------------------------------------------------------


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "all-MiniLM-L6-v2"
    assert body["skills_in_dictionary"] > 20


def test_skills_endpoint_exposes_the_dictionary():
    response = client.get("/skills")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["skills"])
    assert any(s["name"] == "Python" for s in body["skills"])


def test_openapi_schema_is_generated():
    assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# Validation (fast - rejected before the model is touched)
# ---------------------------------------------------------------------------


def test_match_rejects_a_missing_field():
    response = client.post("/match", json={"resume_text": LONG_RESUME})
    assert response.status_code == 422


def test_match_rejects_text_that_is_too_short():
    response = client.post(
        "/match", json={"resume_text": "hi", "job_description": "hi"}
    )
    assert response.status_code == 422


def test_extract_text_rejects_a_non_pdf():
    response = client.post(
        "/extract-text",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_extract_text_reports_the_scanned_pdf_limitation(image_only_pdf_bytes):
    response = client.post(
        "/extract-text",
        files={"file": ("scan.pdf", image_only_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert "OCR" in response.json()["detail"]


def test_extract_text_returns_text_for_a_valid_pdf(text_pdf_bytes):
    response = client.post(
        "/extract-text",
        files={"file": ("resume.pdf", text_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "Jane Doe" in body["text"]
    assert body["page_count"] == 1
    assert body["char_count"] > 0


# ---------------------------------------------------------------------------
# Matching (slow - loads the model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_match_returns_a_complete_report():
    response = client.post(
        "/match",
        json={
            "resume_text": LONG_RESUME,
            "job_description": LONG_JOB,
            "use_semantic_skills": True,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["overall_match_score"] <= 100
    assert "Python" in body["matching_skills"]
    assert "FastAPI" in body["matching_skills"]
    assert body["match_band"] in {
        "Low alignment",
        "Moderate alignment",
        "Good alignment",
        "Strong alignment",
    }
    assert "not a probability" in body["disclaimer"]


@pytest.mark.slow
def test_match_response_matches_the_declared_schema():
    """Pydantic would have raised on the way out, but assert the contract too."""
    body = client.post(
        "/match", json={"resume_text": LONG_RESUME, "job_description": LONG_JOB}
    ).json()

    breakdown = body["score_breakdown"]
    assert set(breakdown) == {
        "semantic_similarity_percent",
        "skill_coverage_percent",
        "semantic_weight",
        "skill_weight",
        "formula",
    }
    assert isinstance(body["resume_skills"], list)
    assert isinstance(body["missing_skills"], list)


@pytest.mark.slow
def test_match_upload_accepts_a_pdf(text_pdf_bytes):
    response = client.post(
        "/match/upload",
        files={"resume_file": ("resume.pdf", text_pdf_bytes, "application/pdf")},
        data={"job_description": LONG_JOB, "use_semantic_skills": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["overall_match_score"] >= 0
    assert "Python" in body["resume_skills"]


@pytest.mark.slow
def test_match_upload_rejects_a_too_short_job_description(text_pdf_bytes):
    response = client.post(
        "/match/upload",
        files={"resume_file": ("resume.pdf", text_pdf_bytes, "application/pdf")},
        data={"job_description": "short"},
    )
    assert response.status_code == 422


@pytest.mark.slow
def test_match_upload_rejects_a_scanned_pdf(image_only_pdf_bytes):
    response = client.post(
        "/match/upload",
        files={"resume_file": ("scan.pdf", image_only_pdf_bytes, "application/pdf")},
        data={"job_description": LONG_JOB},
    )
    assert response.status_code == 400
