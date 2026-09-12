r"""
Matching service - orchestrates the whole pipeline and produces the score.

    resume text  -> clean -> chunk -> embed -> document vector
    job text     -> clean -> chunk -> embed -> document vector
                              |
                     cosine similarity
                              |
                    semantic similarity %          skill coverage %
                              \                   /
                               weighted blend -> overall match score

WHY COSINE SIMILARITY?
    Cosine measures the ANGLE between two vectors and ignores their length. For
    text that matters: a 3-page resume and a 1-paragraph job ad describing the
    same profile point in the same direction, but a length-sensitive metric
    (like Euclidean distance) would call them far apart simply because one
    document is longer.

WHAT THE SCORE IS NOT
    It is a measure of TEXT similarity. It is not a probability of being hired,
    not a competence rating, and not validated against any hiring outcome.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from backend.config import settings
from backend.services import skill_service
from backend.services.embedding_service import embed_long_text
from backend.text_utils import clean_text

DISCLAIMER = (
    "This score measures TEXT SIMILARITY between a resume and a job description. "
    "It is an educational prototype, not a hiring tool: it does not measure "
    "competence and it is not a probability of being hired or shortlisted. "
    "Always review the underlying skills and evidence yourself."
)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, in [-1, 1].

                        a . b
        cos(a, b) = ---------------
                     ||a|| * ||b||

    We call scikit-learn's implementation (rather than writing the formula by
    hand) because it handles the numerical edge cases; the maths is exactly the
    one above. Two zero vectors are defined as 0.0 similarity.
    """
    a = np.asarray(vec_a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(1, -1)

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Vector dimensions do not match: {a.shape[1]} vs {b.shape[1]}"
        )
    if not np.any(a) or not np.any(b):
        return 0.0

    return float(sk_cosine_similarity(a, b)[0][0])


def similarity_to_percentage(similarity: float) -> float:
    """Map cosine similarity onto a 0-100 display scale.

    Negative similarities (rare for this model, which almost always outputs
    values in [0, 1]) are clamped to 0 so the UI never shows a negative
    percentage. NOTE: this is a linear rescale for display only - it applies no
    calibration and adds no statistical meaning.
    """
    return round(max(0.0, min(1.0, similarity)) * 100, 2)


def score_band(score: float) -> str:
    """Heuristic, human-readable label for a 0-100 score.

    The thresholds are a readability aid chosen by hand. They are NOT derived
    from labelled hiring data, and a different embedding model would need
    different bands.
    """
    if score >= 75:
        return "Strong alignment"
    if score >= 60:
        return "Good alignment"
    if score >= 40:
        return "Moderate alignment"
    return "Low alignment"


def match_resume_to_job(
    resume_text: str,
    job_description: str,
    use_semantic_skills: bool = True,
    semantic_weight: Optional[float] = None,
    skill_weight: Optional[float] = None,
) -> dict:
    """Run the full matching pipeline and return a plain dict (-> MatchResponse)."""
    semantic_weight = (
        semantic_weight if semantic_weight is not None else settings.semantic_weight
    )
    skill_weight = skill_weight if skill_weight is not None else settings.skill_weight

    # --- 1. Preprocess -------------------------------------------------
    resume = clean_text(resume_text)
    job = clean_text(job_description)
    if not resume:
        raise ValueError("Resume text is empty after cleaning.")
    if not job:
        raise ValueError("Job description is empty after cleaning.")

    # --- 2. Embed both documents ---------------------------------------
    resume_vector = embed_long_text(resume)
    job_vector = embed_long_text(job)

    # --- 3. Semantic similarity ----------------------------------------
    raw_similarity = cosine_similarity(resume_vector, job_vector)
    semantic_percent = similarity_to_percentage(raw_similarity)

    # --- 4. Skill extraction (exact, dictionary based) -------------------
    resume_skills = skill_service.extract_skills(resume)
    job_skills = skill_service.extract_skills(job)
    comparison = skill_service.compare_skills(resume_skills, job_skills)
    coverage_percent = round(comparison["coverage"] * 100, 2)

    # --- 5. Optional semantic skill layer --------------------------------
    related = []
    if use_semantic_skills and comparison["missing"]:
        related = skill_service.find_semantically_related_skills(
            comparison["missing"], resume
        )

    # --- 6. Blend into one headline number --------------------------------
    # If the job description mentions no dictionary skill at all, the coverage
    # term is meaningless, so we fall back to pure semantic similarity rather
    # than punishing the candidate for our dictionary's blind spots.
    if job_skills:
        overall = semantic_weight * semantic_percent + skill_weight * coverage_percent
        formula = (
            f"{semantic_weight:g} x semantic({semantic_percent}%) + "
            f"{skill_weight:g} x skill_coverage({coverage_percent}%)"
        )
    else:
        overall = semantic_percent
        formula = (
            "semantic similarity only - no dictionary skills were detected in "
            "the job description, so skill coverage was not used"
        )
    overall = round(overall, 2)

    return {
        "overall_match_score": overall,
        "semantic_similarity_score": semantic_percent,
        "match_band": score_band(overall),
        "score_breakdown": {
            "semantic_similarity_percent": semantic_percent,
            "skill_coverage_percent": coverage_percent,
            "semantic_weight": semantic_weight,
            "skill_weight": skill_weight,
            "formula": formula,
        },
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": comparison["matching"],
        "missing_skills": comparison["missing"],
        "semantically_related_skills": related,
        "extra_resume_skills": comparison["extra"],
        "resume_char_count": len(resume),
        "job_char_count": len(job),
        "disclaimer": DISCLAIMER,
    }
