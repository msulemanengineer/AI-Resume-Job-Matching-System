"""Tests for the preprocessing helpers."""

from __future__ import annotations

import pytest

from backend.text_utils import (
    chunk_words,
    clean_text,
    normalize_for_matching,
    split_sentences,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("Python     and    SQL") == "Python and SQL"


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert clean_text("machine lear-\nning") == "machine learning"


def test_clean_text_removes_bullet_glyphs():
    cleaned = clean_text("\u2022 Python\n\u2022 SQL")
    assert "\u2022" not in cleaned
    assert "Python" in cleaned and "SQL" in cleaned


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


def test_normalize_keeps_characters_that_belong_to_skill_names():
    normalized = normalize_for_matching("C++, C#, Node.js and CI/CD!")
    for token in ("c++", "c#", "node.js", "ci/cd"):
        assert token in normalized


def test_chunking_covers_every_word():
    text = " ".join(str(i) for i in range(500))
    chunks = chunk_words(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].split()[0] == "0"
    assert chunks[-1].split()[-1] == "499"


def test_chunks_overlap():
    text = " ".join(str(i) for i in range(300))
    chunks = chunk_words(text, chunk_size=100, overlap=20)

    tail = chunks[0].split()[-20:]
    head = chunks[1].split()[:20]
    assert tail == head


def test_short_text_is_a_single_chunk():
    assert chunk_words("only a few words here", 180, 30) == ["only a few words here"]


def test_chunking_empty_text_returns_nothing():
    assert chunk_words("", 180, 30) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_words("a b c", chunk_size=10, overlap=10)


def test_split_sentences_drops_tiny_fragments():
    sentences = split_sentences("Ok. I built a semantic search engine in Python.")
    assert any("semantic search" in s for s in sentences)
    assert all(len(s) >= 15 for s in sentences)
