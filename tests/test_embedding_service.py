"""
Tests for the embedding layer.

Marked `slow` because they download / load the transformer model.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.embedding_service import (
    embed_chunks,
    embed_long_text,
    embed_text,
    embed_texts,
    get_embedding_dimension,
)

pytestmark = pytest.mark.slow


def test_embedding_has_the_expected_shape():
    vector = embed_text("Python developer with machine learning experience.")

    assert vector.ndim == 1
    assert vector.shape[0] == get_embedding_dimension() == 384


def test_embeddings_are_unit_length():
    """normalize_embeddings=True means the dot product IS the cosine."""
    vector = embed_text("FastAPI and Docker.")
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic():
    a = embed_text("Data scientist with NLP experience.")
    b = embed_text("Data scientist with NLP experience.")
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_batch_embedding_shape():
    vectors = embed_texts(["Python", "Java", "SQL"])
    assert vectors.shape == (3, 384)


def test_empty_batch_returns_empty_array():
    assert embed_texts([]).shape == (0, 384)


def test_embedding_empty_text_raises():
    with pytest.raises(ValueError):
        embed_text("   ")


def test_semantically_similar_sentences_are_closer_than_unrelated_ones():
    """The core claim of the whole project, asserted directly."""
    a = embed_text("I build machine learning models in Python.")
    b = embed_text("I develop ML systems using the Python language.")  # same meaning
    c = embed_text("I bake sourdough bread every morning.")            # unrelated

    assert float(a @ b) > float(a @ c)
    assert float(a @ b) > 0.6


def test_long_document_embedding_is_one_normalised_vector(sample_resume_text):
    vector = embed_long_text(sample_resume_text)

    assert vector.shape == (384,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_long_text_is_split_into_several_chunks(sample_resume_text):
    chunks, vectors = embed_chunks(sample_resume_text)

    assert len(chunks) > 1  # otherwise MiniLM would have truncated the resume
    assert vectors.shape == (len(chunks), 384)


def test_chunked_embedding_differs_from_naive_truncation(sample_resume_text):
    """Shows chunking actually changes the result - i.e. it is not decoration."""
    chunked = embed_long_text(sample_resume_text)
    truncated = embed_text(sample_resume_text)  # model silently cuts at 256 tokens

    assert float(chunked @ truncated) < 0.999
