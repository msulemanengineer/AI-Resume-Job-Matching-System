"""Tests for cosine similarity and the end-to-end matching pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.matching_service import (
    cosine_similarity,
    match_resume_to_job,
    score_band,
    similarity_to_percentage,
)

# ---------------------------------------------------------------------------
# Pure maths - no model needed, so these run instantly
# ---------------------------------------------------------------------------


def test_identical_vectors_have_similarity_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_ignores_magnitude():
    """The property that makes cosine the right choice for documents of
    different lengths: scaling a vector must not change the score."""
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a * 100) == pytest.approx(1.0, abs=1e-6)


def test_zero_vector_returns_zero_not_nan():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_mismatched_dimensions_raise():
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity(np.zeros(3), np.zeros(4))


def test_percentage_conversion():
    assert similarity_to_percentage(1.0) == 100.0
    assert similarity_to_percentage(0.732) == 73.2
    assert similarity_to_percentage(0.0) == 0.0


def test_negative_similarity_is_clamped_to_zero():
    """The UI must never show a negative percentage."""
    assert similarity_to_percentage(-0.4) == 0.0


def test_score_bands_are_ordered():
    assert score_band(90) == "Strong alignment"
    assert score_band(65) == "Good alignment"
    assert score_band(45) == "Moderate alignment"
    assert score_band(10) == "Low alignment"


# ---------------------------------------------------------------------------
# Full pipeline - needs the embedding model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_match_returns_the_full_report(sample_resume_text, sample_job_text):
    result = match_resume_to_job(sample_resume_text, sample_job_text)

    expected_keys = {
        "overall_match_score",
        "semantic_similarity_score",
        "match_band",
        "score_breakdown",
        "resume_skills",
        "job_skills",
        "matching_skills",
        "missing_skills",
        "semantically_related_skills",
        "extra_resume_skills",
        "resume_char_count",
        "job_char_count",
        "disclaimer",
    }
    assert expected_keys <= set(result)
    assert 0 <= result["overall_match_score"] <= 100


@pytest.mark.slow
def test_a_relevant_resume_scores_higher_than_an_irrelevant_one(
    sample_resume_text, sample_job_text, unrelated_job_text
):
    """The single most important behavioural test: the score must discriminate."""
    relevant = match_resume_to_job(sample_resume_text, sample_job_text)
    irrelevant = match_resume_to_job(sample_resume_text, unrelated_job_text)

    assert relevant["overall_match_score"] > irrelevant["overall_match_score"]
    assert relevant["semantic_similarity_score"] > irrelevant["semantic_similarity_score"]


@pytest.mark.slow
def test_identical_documents_score_near_one_hundred(sample_job_text):
    result = match_resume_to_job(sample_job_text, sample_job_text)

    assert result["semantic_similarity_score"] > 99
    assert result["missing_skills"] == []


@pytest.mark.slow
def test_matching_and_missing_skills_partition_the_job_skills(
    sample_resume_text, sample_job_text
):
    result = match_resume_to_job(sample_resume_text, sample_job_text)

    assert set(result["matching_skills"]) | set(result["missing_skills"]) == set(
        result["job_skills"]
    )
    assert not set(result["matching_skills"]) & set(result["missing_skills"])


@pytest.mark.slow
def test_score_breakdown_matches_the_reported_score(
    sample_resume_text, sample_job_text
):
    """Guards the transparency promise: the published formula must reproduce
    the published number."""
    result = match_resume_to_job(sample_resume_text, sample_job_text)
    breakdown = result["score_breakdown"]

    recomputed = (
        breakdown["semantic_weight"] * breakdown["semantic_similarity_percent"]
        + breakdown["skill_weight"] * breakdown["skill_coverage_percent"]
    )
    assert result["overall_match_score"] == pytest.approx(recomputed, abs=0.01)


@pytest.mark.slow
def test_semantic_skill_layer_can_be_switched_off(sample_resume_text, sample_job_text):
    off = match_resume_to_job(sample_resume_text, sample_job_text, use_semantic_skills=False)
    assert off["semantically_related_skills"] == []


@pytest.mark.slow
def test_exact_skill_baseline_is_unaffected_by_the_semantic_layer(
    sample_resume_text, sample_job_text
):
    """The optional layer must only ADD explanation, never change the baseline."""
    on = match_resume_to_job(sample_resume_text, sample_job_text, use_semantic_skills=True)
    off = match_resume_to_job(sample_resume_text, sample_job_text, use_semantic_skills=False)

    assert on["matching_skills"] == off["matching_skills"]
    assert on["missing_skills"] == off["missing_skills"]
    assert on["overall_match_score"] == off["overall_match_score"]


@pytest.mark.slow
def test_empty_input_raises():
    with pytest.raises(ValueError):
        match_resume_to_job("   ", "a job description that is long enough")


@pytest.mark.slow
def test_falls_back_to_pure_semantic_score_when_the_job_lists_no_skills(
    sample_resume_text,
):
    job = (
        "We are looking for a friendly, motivated person to join our small team. "
        "You should communicate well and enjoy solving problems with colleagues."
    )
    result = match_resume_to_job(sample_resume_text, job)

    assert result["job_skills"] == []
    assert result["overall_match_score"] == result["semantic_similarity_score"]
    assert "semantic similarity only" in result["score_breakdown"]["formula"]
