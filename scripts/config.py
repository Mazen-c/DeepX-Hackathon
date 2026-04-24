"""Shared configuration for the Arabic ABSA hackathon workspace.

Phase 2 and later stages import this module for local paths and environment
defaults. The module is intentionally lightweight and has no heavy runtime
dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during very early setup
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_STORE_DIR = PROJECT_ROOT / "chroma_store"
SQLITE_DB_PATH = PROJECT_ROOT / "absa_phase2.db"

DEFAULT_TRAIN_CSV = PROCESSED_DIR / "train_augmented.csv"
DEFAULT_COLLECTION_NAME = "arabic_absa"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def apply_local_runtime_defaults() -> None:
    """Set local-only environment defaults used by the phase 2 pipeline."""

    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODELS_DIR))


def as_dict() -> dict[str, str]:
    """Return a serialisable snapshot of the core workspace paths."""

    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "models_dir": str(MODELS_DIR),
        "chroma_store_dir": str(CHROMA_STORE_DIR),
        "sqlite_db_path": str(SQLITE_DB_PATH),
        "default_train_csv": str(DEFAULT_TRAIN_CSV),
        "default_collection_name": DEFAULT_COLLECTION_NAME,
        "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
        "fallback_embedding_model": FALLBACK_EMBEDDING_MODEL,
    }
