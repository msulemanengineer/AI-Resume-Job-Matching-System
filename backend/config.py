"""
Central configuration for the AI Resume-Job Matching System.

Everything that a reviewer might want to tweak (model name, score weights,
thresholds, file size limits) lives here instead of being scattered as magic
numbers through the code. Values can be overridden with environment variables
or a `.env` file - see `.env.example`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains `backend/`, `frontend/`, `data/`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env with defaults."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow field names starting with "model_"
    )

    # --- Embedding model -----------------------------------------------
    # all-MiniLM-L6-v2: 6 transformer layers, 384-dimensional output,
    # ~80 MB on disk. Small and fast enough to run on a laptop CPU.
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # MiniLM truncates input at 256 word-pieces (~180-200 words). A resume is
    # much longer than that, so we split long text into word chunks, embed each
    # chunk, and average the vectors. See embedding_service.embed_long_text().
    chunk_size_words: int = 180
    chunk_overlap_words: int = 30

    # --- Scoring --------------------------------------------------------
    # The headline "overall match score" is a transparent weighted blend of:
    #   1. semantic similarity (cosine between the two document embeddings)
    #   2. skill coverage      (% of job skills that appear in the resume)
    # These weights are a design choice, NOT a validated/learned parameter.
    semantic_weight: float = 0.7
    skill_weight: float = 0.3

    # Optional semantic skill matching: a job skill that is not found literally
    # in the resume is flagged as "possibly related" if the cosine similarity
    # between the skill name and the closest resume chunk exceeds this value.
    semantic_skill_threshold: float = 0.45

    # --- Uploads / limits -----------------------------------------------
    max_upload_mb: int = 5
    min_extracted_chars: int = 50  # below this we assume extraction failed

    # --- Files -----------------------------------------------------------
    skills_file: Path = DATA_DIR / "skills.json"

    # --- API / frontend ---------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Used by the Streamlit app to reach the backend.
    api_base_url: str = "http://127.0.0.1:8000"


settings = Settings()
