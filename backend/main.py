"""
FastAPI application - the backend of the AI Resume-Job Matching System.

ENDPOINTS
    GET  /health         service + model status
    GET  /skills         the skill dictionary currently in use
    POST /match          JSON in  -> matching report out
    POST /match/upload   PDF resume (multipart) + job description -> report
    POST /extract-text   PDF -> plain text (used by the UI to preview)

WHY A SEPARATE BACKEND AT ALL?
    The Streamlit UI could import the services directly, but splitting them
    means the matching logic is reusable by any client (a React app, a CLI, a
    cron job), can be deployed and scaled on its own, and is testable through a
    stable HTTP contract. That separation is the point of the architecture.

PRIVACY
    Nothing is written to disk. Uploaded PDFs are read into memory, converted to
    text, used for one request and dropped when the function returns. Resume
    CONTENT is never logged - only lengths and counts.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models import (
    ExtractedTextResponse,
    HealthResponse,
    MatchRequest,
    MatchResponse,
)
from backend.services import embedding_service, skill_service
from backend.services.matching_service import match_resume_to_job
from backend.services.pdf_service import PDFExtractionError, extract_text_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Resume-Job Matching System",
    description=(
        "Educational NLP prototype. Compares a resume with a job description "
        "using sentence embeddings and cosine similarity, plus a transparent "
        "dictionary-based skill extractor.\n\n"
        "**The score is a text-similarity measure, not a hiring decision.**"
    ),
    version="1.0.0",
)

# The Streamlit UI runs on a different port, so the browser treats it as a
# different origin. In a real deployment this list would be restricted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "AI Resume-Job Matching System",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness probe. Does NOT force the model to load, so it stays instant."""
    loaded = embedding_service.is_model_loaded()
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model_name,
        model_loaded=loaded,
        embedding_dimension=(
            embedding_service.get_embedding_dimension() if loaded else None
        ),
        skills_in_dictionary=len(skill_service.load_skills()),
    )


@app.get("/skills", tags=["system"])
def list_skills() -> dict:
    """Expose the skill dictionary so users can see exactly what is searched for."""
    skills = skill_service.load_skills()
    return {
        "count": len(skills),
        "skills": [{"name": s.name, "aliases": list(s.aliases)} for s in skills],
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@app.post("/match", response_model=MatchResponse, tags=["matching"])
def match(request: MatchRequest) -> MatchResponse:
    """Match resume TEXT against a job description."""
    logger.info(
        "match: resume_chars=%d job_chars=%d semantic_skills=%s",
        len(request.resume_text),
        len(request.job_description),
        request.use_semantic_skills,
    )  # lengths only - never the content
    try:
        result = match_resume_to_job(
            resume_text=request.resume_text,
            job_description=request.job_description,
            use_semantic_skills=request.use_semantic_skills,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected model failure
        logger.exception("Matching failed")
        raise HTTPException(
            status_code=500, detail=f"Matching failed: {type(exc).__name__}"
        ) from exc
    return MatchResponse(**result)


@app.post("/match/upload", response_model=MatchResponse, tags=["matching"])
async def match_upload(
    resume_file: UploadFile = File(..., description="Resume as a text-based PDF"),
    job_description: str = Form(..., description="Job description text"),
    use_semantic_skills: bool = Form(True),
) -> MatchResponse:
    """Match an uploaded PDF resume against a pasted job description."""
    pdf_bytes = await resume_file.read()
    logger.info(
        "match/upload: file_size=%d bytes, job_chars=%d",
        len(pdf_bytes),
        len(job_description),
    )

    try:
        extracted = extract_text_from_pdf(pdf_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        del pdf_bytes  # drop the raw file from memory as early as possible

    if len(job_description.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Job description is too short (minimum 20 characters).",
        )

    try:
        result = match_resume_to_job(
            resume_text=extracted.text,
            job_description=job_description,
            use_semantic_skills=use_semantic_skills,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return MatchResponse(**result)


@app.post("/extract-text", response_model=ExtractedTextResponse, tags=["matching"])
async def extract_text(
    file: UploadFile = File(..., description="PDF to extract text from"),
) -> ExtractedTextResponse:
    """Extract plain text from a PDF (resume or job description)."""
    pdf_bytes = await file.read()
    logger.info("extract-text: file_size=%d bytes", len(pdf_bytes))
    try:
        extracted = extract_text_from_pdf(pdf_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExtractedTextResponse(
        text=extracted.text,
        char_count=extracted.char_count,
        page_count=extracted.page_count,
        warning=extracted.warning,
    )


if __name__ == "__main__":  # `python -m backend.main`
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
