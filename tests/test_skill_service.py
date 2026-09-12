"""Tests for the dictionary-based skill extractor (the explainable baseline)."""

from __future__ import annotations

import pytest

from backend.services.skill_service import (
    compare_skills,
    extract_skills,
    extract_skills_with_evidence,
    load_skills,
)


def test_dictionary_loads():
    skills = load_skills()
    assert len(skills) > 20
    names = {s.name for s in skills}
    assert {"Python", "Machine Learning", "FastAPI", "Docker"} <= names


def test_detects_plain_skills():
    found = extract_skills("I write Python and use Docker every day.")
    assert "Python" in found
    assert "Docker" in found


def test_detection_is_case_insensitive():
    assert extract_skills("PYTHON, python, PyThOn") == extract_skills("python")


def test_aliases_map_to_the_canonical_name():
    """'sklearn' and 'scikit-learn' must both report as "Scikit-learn"."""
    assert "Scikit-learn" in extract_skills("Modelled with sklearn.")
    assert "Scikit-learn" in extract_skills("Modelled with scikit-learn.")


def test_multi_word_skills_are_detected():
    found = extract_skills("Strong background in natural language processing.")
    assert "NLP" in found


def test_symbol_heavy_skill_names_survive_normalisation():
    found = extract_skills("Languages: C++, C#, Node.js. Pipelines with CI/CD.")
    assert {"C++", "C#", "Node.js", "CI/CD"} <= set(found)


def test_substrings_do_not_produce_false_positives():
    """The classic trap: 'Java' must not fire on 'JavaScript'."""
    found = extract_skills("I only know JavaScript.")
    assert "JavaScript" in found
    assert "Java" not in found


def test_no_skills_in_unrelated_text():
    assert extract_skills("I baked sourdough bread and croissants.") == []


def test_empty_text_returns_empty_list():
    assert extract_skills("") == []


def test_extraction_is_deterministic(sample_resume_text):
    assert extract_skills(sample_resume_text) == extract_skills(sample_resume_text)


def test_evidence_explains_each_detection():
    evidence = extract_skills_with_evidence("Trained models with sklearn.")
    assert evidence["Scikit-learn"] == "sklearn"


def test_compare_skills_splits_into_matching_missing_extra():
    result = compare_skills(
        resume_skills=["Python", "Docker", "React"],
        job_skills=["Python", "AWS", "Docker"],
    )
    assert result["matching"] == ["Python", "Docker"]
    assert result["missing"] == ["AWS"]
    assert result["extra"] == ["React"]
    assert result["coverage"] == pytest.approx(2 / 3)


def test_coverage_is_zero_when_the_job_lists_no_skills():
    result = compare_skills(resume_skills=["Python"], job_skills=[])
    assert result["coverage"] == 0.0
    assert result["missing"] == []


def test_sample_documents_produce_sensible_skills(sample_resume_text, sample_job_text):
    resume_skills = extract_skills(sample_resume_text)
    job_skills = extract_skills(sample_job_text)

    assert {"Python", "PyTorch", "FastAPI", "Pandas"} <= set(resume_skills)
    assert {"Python", "Machine Learning", "NLP", "SQL"} <= set(job_skills)

    comparison = compare_skills(resume_skills, job_skills)
    assert comparison["coverage"] > 0.5  # the sample pair is a deliberate good fit
