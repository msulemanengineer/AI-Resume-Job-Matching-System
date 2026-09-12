"""
Skill extraction - the transparent, explainable half of the system.

METHOD: dictionary lookup with aliases.
    For each skill in data/skills.json we search the normalised text for any of
    its aliases, anchored on word boundaries. "Python" matches "python" and
    "Python3" (an alias) but not "pythonic-sounding-word".

This is deliberately NOT a machine-learning method. It is a rule-based baseline
and it should be described that way in an interview:

  Strengths - fully explainable ("we matched the literal string 'FastAPI'"),
              zero training data, instant, easy for a user to audit and extend.
  Weaknesses- only knows the skills in the dictionary; cannot read context
              ("no experience with Java" still counts as a Java mention); misses
              unlisted synonyms; cannot tell a one-week course from three years
              of production work.

The optional semantic layer (`find_semantically_related_skills`) reduces the
"unlisted synonym" weakness, but it is layered ON TOP of the exact baseline so
both can be compared and explained side by side.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.config import settings
from backend.text_utils import normalize_for_matching, split_sentences


@dataclass(frozen=True)
class Skill:
    """One entry of the skill dictionary."""

    name: str          # canonical display name, e.g. "Scikit-learn"
    aliases: tuple     # every spelling we search for, e.g. ("sklearn", ...)


@lru_cache(maxsize=1)
def load_skills(skills_file: Optional[Path] = None) -> List[Skill]:
    """Load (and cache) the skill dictionary from JSON."""
    path = Path(skills_file or settings.skills_file)
    if not path.exists():
        raise FileNotFoundError(f"Skill dictionary not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    skills: List[Skill] = []
    for entry in data.get("skills", []):
        name = entry["name"]
        # The canonical name is always searchable, plus any declared aliases.
        aliases = {name.lower(), *(a.lower() for a in entry.get("aliases", []))}
        skills.append(Skill(name=name, aliases=tuple(sorted(aliases))))
    return skills


@lru_cache(maxsize=512)
def _alias_pattern(alias: str) -> re.Pattern:
    """Build a whole-word regex for one alias.

    `\b` does not work at the edge of symbols like "C++" or "C#" (those are not
    word characters), so we use look-around on whitespace instead: the alias
    must be surrounded by a space/start/end, never glued to another letter.
    """
    escaped = re.escape(alias)
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def extract_skills(text: str, skills: Optional[Sequence[Skill]] = None) -> List[str]:
    """Return the canonical names of every dictionary skill found in `text`.

    Order follows the dictionary, so results are deterministic and comparable
    between two documents.
    """
    if not text:
        return []

    skills = skills if skills is not None else load_skills()
    haystack = normalize_for_matching(text)

    found: List[str] = []
    for skill in skills:
        if any(_alias_pattern(alias).search(haystack) for alias in skill.aliases):
            found.append(skill.name)
    return found


def extract_skills_with_evidence(text: str) -> Dict[str, str]:
    """Map each detected skill -> the alias that triggered the detection.

    Useful for debugging false positives ("why did it think I know Torch?").
    """
    haystack = normalize_for_matching(text)
    evidence: Dict[str, str] = {}
    for skill in load_skills():
        for alias in skill.aliases:
            if _alias_pattern(alias).search(haystack):
                evidence[skill.name] = alias
                break
    return evidence


def compare_skills(resume_skills: Sequence[str], job_skills: Sequence[str]) -> dict:
    """Set comparison between the two skill lists.

    matching = job skills present in the resume
    missing  = job skills absent from the resume
    extra    = resume skills the job never asked for
    coverage = |matching| / |job skills|  (0.0 when the job lists no skills)
    """
    resume_set, job_set = set(resume_skills), set(job_skills)

    matching = [s for s in job_skills if s in resume_set]
    missing = [s for s in job_skills if s not in resume_set]
    extra = [s for s in resume_skills if s not in job_set]

    coverage = len(matching) / len(job_set) if job_set else 0.0
    return {
        "matching": matching,
        "missing": missing,
        "extra": extra,
        "coverage": coverage,
    }


def find_semantically_related_skills(
    missing_skills: Sequence[str],
    resume_text: str,
    threshold: Optional[float] = None,
) -> List[dict]:
    """OPTIONAL layer: are the 'missing' skills really absent?

    For each missing skill we embed the skill name and compare it against every
    sentence of the resume. If the closest sentence is similar enough, we report
    the skill as "possibly related" together with that sentence as evidence, so
    the user can judge for themselves.

    Example this catches: the job asks for "Machine Learning", the resume says
    "trained predictive models to forecast demand" - no literal match, but the
    sentence is semantically close.

    This is a *soft signal*, not proof. It does not change the exact-match
    baseline used for the score; it only adds explanation.
    """
    threshold = threshold if threshold is not None else settings.semantic_skill_threshold
    if not missing_skills or not resume_text.strip():
        return []

    # Imported here so that pure keyword-matching code paths never pay the cost
    # of loading a transformer model.
    from backend.services.embedding_service import embed_texts

    sentences = split_sentences(resume_text)
    if not sentences:
        return []

    skill_vectors = embed_texts(list(missing_skills))      # (n_skills, dim)
    sentence_vectors = embed_texts(sentences)              # (n_sentences, dim)

    # Vectors are unit length, so a dot product IS the cosine similarity.
    similarity = skill_vectors @ sentence_vectors.T        # (n_skills, n_sentences)

    related: List[dict] = []
    for i, skill in enumerate(missing_skills):
        best_idx = int(np.argmax(similarity[i]))
        best_score = float(similarity[i][best_idx])
        if best_score >= threshold:
            snippet = sentences[best_idx]
            related.append(
                {
                    "skill": skill,
                    "similarity": round(best_score, 4),
                    "evidence": snippet[:200] + ("..." if len(snippet) > 200 else ""),
                }
            )

    related.sort(key=lambda item: item["similarity"], reverse=True)
    return related
