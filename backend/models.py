"""
Pydantic models: the request/response contract of the API.

Pydantic validates incoming JSON automatically (types, required fields, length
limits) and FastAPI uses these classes to generate the OpenAPI docs at /docs.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class MatchRequest(BaseModel):
    """Plain-text matching request (no file upload)."""

    resume_text: str = Field(
        ...,
        min_length=20,
        description="Raw resume text (already extracted from a PDF, or typed).",
    )
    job_description: str = Field(
        ...,
        min_length=20,
        description="Raw job description text.",
    )
    use_semantic_skills: bool = Field(
        default=True,
        description=(
            "If true, job skills that are NOT found literally in the resume are "
            "additionally checked with embeddings and may be reported as "
            "'possibly related' instead of plainly missing."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "resume_text": "Final-year CS student. Built ML models with Python, "
                "scikit-learn and Pandas. Deployed a FastAPI service on Docker.",
                "job_description": "Looking for an AI/ML intern comfortable with "
                "Python, scikit-learn, NLP and REST APIs. Docker is a plus.",
                "use_semantic_skills": True,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class RelatedSkill(BaseModel):
    """A job skill that was not found literally but looks semantically close."""

    skill: str = Field(..., description="Job-description skill (canonical name).")
    similarity: float = Field(
        ..., description="Cosine similarity to the closest passage of the resume (0-1)."
    )
    evidence: str = Field(
        ..., description="The resume snippet that produced that similarity."
    )


class ScoreBreakdown(BaseModel):
    """Shows exactly how the headline number was produced - no black box."""

    semantic_similarity_percent: float = Field(
        ..., description="Cosine similarity between the two document embeddings, x100."
    )
    skill_coverage_percent: float = Field(
        ...,
        description="Share of job-description skills that were detected in the resume.",
    )
    semantic_weight: float = Field(..., description="Weight applied to the semantic part.")
    skill_weight: float = Field(..., description="Weight applied to the skill part.")
    formula: str = Field(..., description="Human-readable formula actually used.")


class MatchResponse(BaseModel):
    """Everything the UI needs to render a matching report."""

    overall_match_score: float = Field(
        ...,
        description=(
            "Weighted blend of semantic similarity and skill coverage, 0-100. "
            "This is a TEXT SIMILARITY measure, not a probability of being hired."
        ),
    )
    semantic_similarity_score: float = Field(
        ..., description="Pure cosine similarity x100, before any blending."
    )
    match_band: str = Field(
        ..., description="Heuristic label: Low / Moderate / Good / Strong alignment."
    )
    score_breakdown: ScoreBreakdown

    resume_skills: List[str] = Field(..., description="Skills detected in the resume.")
    job_skills: List[str] = Field(..., description="Skills detected in the job description.")
    matching_skills: List[str] = Field(
        ..., description="Job skills that were also found in the resume."
    )
    missing_skills: List[str] = Field(
        ...,
        description="Job skills not detected in the resume (possibly missing, possibly "
        "just worded differently).",
    )
    semantically_related_skills: List[RelatedSkill] = Field(
        default_factory=list,
        description="Subset of missing skills that look semantically close to the resume.",
    )
    extra_resume_skills: List[str] = Field(
        default_factory=list,
        description="Resume skills the job description did not ask for.",
    )

    resume_char_count: int = Field(..., description="Length of the processed resume text.")
    job_char_count: int = Field(..., description="Length of the processed job text.")
    disclaimer: str = Field(..., description="Scope/limitations reminder.")


class HealthResponse(BaseModel):
    """Answer of GET /health."""

    status: str
    embedding_model: str
    model_loaded: bool
    embedding_dimension: Optional[int] = None
    skills_in_dictionary: int


class ExtractedTextResponse(BaseModel):
    """Answer of POST /extract-text."""

    text: str
    char_count: int
    page_count: int
    warning: Optional[str] = None
