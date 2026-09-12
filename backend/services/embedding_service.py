"""
Embedding service - turns text into vectors.

WHAT IS AN EMBEDDING?
    An embedding is a list of numbers (here: 384 of them) that represents the
    *meaning* of a piece of text. The model was trained so that texts humans
    would call similar end up close together in this 384-dimensional space,
    and unrelated texts end up far apart. That is what lets us compare
    "built ML models in Python" with "experience developing machine learning
    solutions" and get a high score, even though they share almost no words.

MODEL: all-MiniLM-L6-v2
    - 6 transformer layers (a distilled, small BERT), ~22M parameters, ~80 MB
    - outputs 384 dimensions
    - trained with a contrastive objective on ~1B sentence pairs: similar pairs
      are pulled together, random pairs are pushed apart
    - the token vectors of the last layer are MEAN POOLED into one sentence
      vector, then L2-normalised

WHY NOT A BIGGER MODEL? It runs on a CPU in milliseconds and is enough to
demonstrate the concept. Bigger models would improve quality but make the demo
slow and heavy - an explicit trade-off, not an oversight.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

import numpy as np

from backend.config import settings
from backend.text_utils import chunk_words

logger = logging.getLogger(__name__)

# The model is loaded once and reused (loading takes a few seconds and ~100 MB
# of RAM - doing it per request would be very slow). A lock makes the lazy
# initialisation safe when several requests arrive at the same time.
_model = None
_model_lock = threading.Lock()


def get_model():
    """Return the shared SentenceTransformer instance, loading it on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check inside the lock
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model: %s", settings.embedding_model_name)
                _model = SentenceTransformer(settings.embedding_model_name)
                logger.info(
                    "Model loaded. Embedding dimension: %s",
                    _dimension_of(_model),
                )
    return _model


def is_model_loaded() -> bool:
    """True if the model is already in memory (used by /health)."""
    return _model is not None


def _dimension_of(model) -> int:
    """Read the model's output dimension across sentence-transformers versions.

    v5 renamed `get_sentence_embedding_dimension()` to `get_embedding_dimension()`;
    the old name still works but emits a FutureWarning. Prefer the new name and
    fall back, so the project runs on both v3/v4 and v5+.
    """
    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension"
    )
    return int(getter())


def get_embedding_dimension() -> int:
    """Size of the vectors this model produces (384 for all-MiniLM-L6-v2)."""
    return _dimension_of(get_model())


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed a list of short texts -> array of shape (len(texts), dim).

    `normalize_embeddings=True` rescales every vector to unit length, which
    makes the dot product identical to cosine similarity and keeps scores in a
    predictable range.
    """
    if not texts:
        return np.empty((0, get_embedding_dimension()), dtype=np.float32)

    model = get_model()
    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed a single short text -> 1-D vector of shape (dim,)."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    return embed_texts([text])[0]


def embed_long_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> np.ndarray:
    """Embed a *document* (resume / job description) into one vector.

    The model only reads the first 256 word-pieces of its input, so a whole
    resume cannot be embedded in one go without losing most of it. Instead:

        1. split the document into overlapping ~180-word chunks
        2. embed every chunk
        3. average the chunk vectors (mean pooling at the document level)
        4. re-normalise to unit length

    Averaging is the simple, standard choice: the document vector ends up in
    the "centre of mass" of its parts. Its weakness is dilution - a long resume
    with one relevant section gets averaged down by the irrelevant sections.
    That trade-off is documented rather than hidden.
    """
    chunk_size = chunk_size or settings.chunk_size_words
    overlap = overlap if overlap is not None else settings.chunk_overlap_words

    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    chunks = chunk_words(text, chunk_size, overlap)
    if not chunks:
        raise ValueError("Cannot embed empty text.")

    chunk_vectors = embed_texts(chunks)
    doc_vector = chunk_vectors.mean(axis=0)

    norm = np.linalg.norm(doc_vector)
    if norm > 0:
        doc_vector = doc_vector / norm
    return doc_vector.astype(np.float32)


def embed_chunks(text: str) -> tuple[List[str], np.ndarray]:
    """Return the chunks of a document *and* their individual vectors.

    Used by the semantic skill matcher, which needs to know *where* in the
    resume a match came from so it can show evidence to the user.
    """
    chunks = chunk_words(text, settings.chunk_size_words, settings.chunk_overlap_words)
    if not chunks:
        return [], np.empty((0, get_embedding_dimension()), dtype=np.float32)
    return chunks, embed_texts(chunks)
