"""Train a lightweight local ABSA model and save reusable weights.

The model is a CPU-friendly scikit-learn One-vs-Rest linear classifier over
TF-IDF features. It predicts multi-label targets where each label is
"aspect|sentiment".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _prepare_examples(frame: pd.DataFrame) -> tuple[List[str], List[List[str]]]:
    required = {"review_id", "review_clean", "aspect", "sentiment"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = frame.copy()
    work["review_clean"] = work["review_clean"].fillna("").astype(str).str.strip()
    work["aspect"] = work["aspect"].fillna("").astype(str).str.strip().str.lower()
    work["sentiment"] = work["sentiment"].fillna("").astype(str).str.strip().str.lower()
    work = work[work["review_clean"] != ""]

    work["label"] = work["aspect"] + "|" + work["sentiment"]
    grouped = (
        work.groupby(["review_id", "review_clean"], as_index=False)["label"]
        .apply(lambda s: sorted(set(x for x in s if "|" in x and not x.startswith("|"))))
    )
    grouped = grouped[grouped["label"].map(len) > 0]

    texts = grouped["review_clean"].tolist()
    labels = grouped["label"].tolist()
    if not texts:
        raise ValueError("No usable training rows were found after preprocessing.")
    return texts, labels


def train_and_save(train_csv: Path, out_weights: Path, out_meta: Path) -> Dict[str, object]:
    frame = _read_csv(train_csv)
    texts, labels = _prepare_examples(frame)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=120_000,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        class_weight="balanced",
                    )
                ),
            ),
        ]
    )

    print(f"[INFO] Training on {len(texts):,} reviews with {len(mlb.classes_):,} labels...", flush=True)
    model.fit(texts, y)

    out_weights.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "label_binarizer": mlb}, out_weights)

    metadata: Dict[str, object] = {
        "train_csv": str(train_csv),
        "weights_path": str(out_weights),
        "samples": int(len(texts)),
        "labels": int(len(mlb.classes_)),
        "classes": [str(x) for x in mlb.classes_],
        "vectorizer": "TfidfVectorizer(char_wb, ngram_range=(3,5), max_features=120000)",
        "classifier": "OneVsRest(LogisticRegression(liblinear, class_weight=balanced))",
    }
    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train local ABSA model weights")
    parser.add_argument("--train", default="Data\\processed\\train_augmented.csv", help="Input long-format training CSV")
    parser.add_argument("--weights", default="models\\local_absa_weights.joblib", help="Output weights file path")
    parser.add_argument("--meta", default="models\\local_absa_weights.meta.json", help="Output metadata JSON path")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    metadata = train_and_save(Path(args.train), Path(args.weights), Path(args.meta))
    print("[INFO] Weights saved successfully.", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

