"""
Small, dependency-free text helpers used by several services.

Deliberately light-touch preprocessing. Transformer models are trained on
normal human sentences, so aggressive cleaning (stop-word removal, stemming,
lower-casing everything) would REMOVE information the model relies on. We only
fix the mess that PDF extraction introduces.
"""

from __future__ import annotations

import re
from typing import List

# PDF extraction artefacts we want to normalise away.
_BULLETS = re.compile(r"[\u2022\u25cf\u25aa\u00b7\u2023\u2043\u2219\uf0b7]")
_MULTISPACE = re.compile(r"[ \t\u00a0]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
# A hyphen at the end of a line: "machine lear-\nning" -> "machine learning"
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalise raw text so the embedding model sees clean sentences.

    Steps (all conservative, nothing semantic is thrown away):
      1. normalise line endings
      2. re-join words split across a line break by a hyphen
      3. replace bullet glyphs with spaces
      4. collapse runs of spaces / blank lines
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _BULLETS.sub(" ", text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)

    # Strip trailing spaces on every line, then trim the whole block.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Lower-case + punctuation-flattened text used ONLY for keyword lookup.

    Note this is a *separate* pipeline from the embedding input: keyword search
    wants aggressive normalisation, the transformer does not.

    Characters that are part of real skill names (`+`, `#`, `.`, `/`, `-`) are
    kept so that "C++", "C#", "Node.js" and "CI/CD" remain findable.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word windows.

    Why: all-MiniLM-L6-v2 silently truncates anything past 256 word-pieces.
    A 2-page resume would lose most of its content. Splitting into windows and
    averaging the vectors keeps the whole document in play. The overlap stops a
    sentence that straddles a boundary from being cut in half in every chunk.
    """
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break  # last window already reached the end
    return chunks


def split_sentences(text: str) -> List[str]:
    """Very small sentence/line splitter used to show 'evidence' snippets.

    Not a linguistic sentence tokenizer - resumes are mostly bullet points, so
    splitting on newlines plus sentence-ending punctuation works well enough
    and adds no extra dependency.
    """
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in pieces if len(p.strip()) >= 15]
