"""Phase 2 local vector store and retrieval utilities.

This module intentionally keeps imports for optional heavy dependencies inside
functions so the module stays importable even before the local model cache or
vector store exists.
"""

from __future__ import annotations

import argparse
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np

from scripts.config import (
    CHROMA_STORE_DIR,
    DEFAULT_COLLECTION_NAME as CONFIG_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    MODELS_DIR,
    PROCESSED_DIR,
    apply_local_runtime_defaults,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROCESSED_DIR / "train_augmented.csv"
DEFAULT_STORE_PATH = CHROMA_STORE_DIR
DEFAULT_MODEL_CACHE = MODELS_DIR
DEFAULT_COLLECTION_NAME = CONFIG_COLLECTION_NAME
DEFAULT_MODEL_NAME = DEFAULT_EMBEDDING_MODEL
FALLBACK_MODEL_NAME = FALLBACK_EMBEDDING_MODEL
BATCH_SIZE = 128

_MODEL = None
_MODEL_NAME = None
_COLLECTION = None


class _HashingEmbedder:
    """CPU-only fallback embedder used when SentenceTransformer cannot load."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False, batch_size: Optional[int] = None):
        matrix = self._vectorizer.transform([str(text) for text in texts])
        return matrix.toarray()


class _TransformerEmbedder:
    """Direct Transformers embedding wrapper with mean pooling and L2 normalisation."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self._model.eval()

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False, batch_size: Optional[int] = None):
        batch_size = max(1, int(batch_size or len(texts) or 1))
        embeddings: List[np.ndarray] = []

        with self._torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [str(text) for text in texts[start : start + batch_size]]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                output = self._model(**encoded)
                last_hidden_state = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
                summed = (last_hidden_state * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype="float32")


def _warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def _ensure_local_cache_env() -> None:
    apply_local_runtime_defaults()
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_MODEL_CACHE))


def _safe_import_chromadb():
    try:
        import chromadb  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "chromadb is required for phase 2. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return chromadb


def _safe_import_sentence_transformers():
    raise RuntimeError(
        "sentence-transformers is no longer used by phase 2. The loader now uses transformers AutoTokenizer/AutoModel directly."
    )


def _load_model(model_name: Optional[str] = None):
    global _MODEL, _MODEL_NAME

    if _MODEL is not None:
        return _MODEL

    _ensure_local_cache_env()
    primary_model = model_name or os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    attempted = [primary_model]
    if primary_model != FALLBACK_MODEL_NAME:
        attempted.append(FALLBACK_MODEL_NAME)

    last_error: Optional[Exception] = None
    for candidate in attempted:
        try:
            _MODEL = _TransformerEmbedder(candidate, cache_dir=str(DEFAULT_MODEL_CACHE))
            _MODEL_NAME = candidate
            return _MODEL
        except Exception as exc:  # pragma: no cover - depends on local model availability
            last_error = exc

    _warn(
        f"Could not load Transformers model {attempted!r}; using local hashing embeddings instead. Last error: {last_error}"
    )
    _MODEL = _HashingEmbedder()
    _MODEL_NAME = "local-hashing-fallback"
    return _MODEL


def _load_collection(store_path: Path = DEFAULT_STORE_PATH, collection_name: str = DEFAULT_COLLECTION_NAME):
    global _COLLECTION

    if _COLLECTION is not None:
        return _COLLECTION

    chromadb = _safe_import_chromadb()
    client = chromadb.PersistentClient(path=str(store_path))
    _COLLECTION = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return _COLLECTION


def _read_source_rows(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "review_clean" not in df.columns:
        raise ValueError("Expected a review_clean column in the input CSV.")

    if "review_id" not in df.columns:
        df = df.copy()
        df["review_id"] = df.index.astype(str)

    if "aspect" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Expected aspect and sentiment columns in the input CSV.")

    df["review_clean"] = df["review_clean"].fillna("").astype(str).str.strip()
    df["review_text"] = (
        df["review_text"].fillna("").astype(str).str.strip()
        if "review_text" in df.columns
        else df["review_clean"]
    )
    df["aspect"] = df["aspect"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip()
    if "source" not in df.columns:
        df["source"] = "original"
    else:
        df["source"] = df["source"].fillna("original").astype(str)

    df = df[df["review_clean"] != ""].reset_index(drop=True)
    return df


def _make_metadata_row(row: pd.Series) -> Dict[str, Any]:
    metadata = {
        "review_id": str(row.get("review_id", "")),
        "aspect": str(row.get("aspect", "")),
        "sentiment": str(row.get("sentiment", "")),
        "source": str(row.get("source", "original")),
    }
    business_name = row.get("business_name", "")
    business_category = row.get("business_category", "")
    platform = row.get("platform", "")
    if pd.notna(business_name) and str(business_name).strip():
        metadata["business_name"] = str(business_name).strip()
    if pd.notna(business_category) and str(business_category).strip():
        metadata["business_category"] = str(business_category).strip()
    if pd.notna(platform) and str(platform).strip():
        metadata["platform"] = str(platform).strip()
    return metadata


def build_vector_store(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    store_path: Path | str = DEFAULT_STORE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
    model_name: Optional[str] = None,
) -> int:
    """Build and persist the ChromaDB index from the cleaned training data."""

    input_path = Path(input_path)
    store_path = Path(store_path)
    df = _read_source_rows(input_path)
    model = _load_model(model_name)

    chromadb = _safe_import_chromadb()
    client = chromadb.PersistentClient(path=str(store_path))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if len(df) == 0:
        raise ValueError(f"No usable rows found in {input_path}")

    print(f"Indexing {len(df):,} rows from {input_path} into {store_path} ...", flush=True)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    embeddings: List[List[float]] = []

    for index, row in df.iterrows():
        review_id = str(row.get("review_id", index))
        ids.append(str(uuid.uuid4()))
        documents.append(str(row["review_clean"]))
        metadatas.append(_make_metadata_row(row))

        if len(ids) >= batch_size:
            embeddings = model.encode(documents, show_progress_bar=False, batch_size=min(batch_size, len(documents)))
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
            ids, documents, metadatas = [], [], []

    if ids:
        embeddings = model.encode(documents, show_progress_bar=False, batch_size=min(batch_size, len(documents)))
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    count = collection.count()
    print(f"Indexed {count:,} documents using model {_MODEL_NAME or model_name or DEFAULT_MODEL_NAME}.", flush=True)
    return count


def retrieve_examples(
    text: str,
    n: int = 3,
    store_path: Path | str = DEFAULT_STORE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """Return the top-n semantically similar reviews for a query string."""

    if not text or not str(text).strip():
        return []

    model = _load_model()
    collection = _load_collection(Path(store_path), collection_name)
    query_vec = model.encode([str(text)], show_progress_bar=False)[0].tolist()
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=max(1, int(n)),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0] or []
    metadatas = result.get("metadatas", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []

    examples: List[Dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        examples.append(
            {
                "review": document,
                "review_id": metadata.get("review_id", ""),
                "aspect": metadata.get("aspect", ""),
                "sentiment": metadata.get("sentiment", ""),
                "source": metadata.get("source", ""),
                "distance": distance,
            }
        )
    return examples


def index_is_ready(store_path: Path | str = DEFAULT_STORE_PATH, collection_name: str = DEFAULT_COLLECTION_NAME) -> bool:
    """Check whether a persisted collection already exists and has rows."""

    store_path = Path(store_path)
    if not store_path.exists():
        return False

    chromadb = _safe_import_chromadb()
    client = chromadb.PersistentClient(path=str(store_path))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return False
    return collection.count() > 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the phase 2 ChromaDB vector store.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="CSV to index")
    parser.add_argument("--store", default=str(DEFAULT_STORE_PATH), help="ChromaDB persistence directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows to index per batch")
    parser.add_argument("--model", default=None, help="Embedding model name override")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    build_vector_store(
        input_path=Path(args.input),
        store_path=Path(args.store),
        collection_name=args.collection,
        batch_size=max(1, int(args.batch_size)),
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())