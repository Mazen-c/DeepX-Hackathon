"""Train a production-ready local multi-label ABSA model.

Expected input CSV defaults to Data/processed/train_augmented_wide.csv.
The script supports either:
1) Wide format: columns [text, labels]
2) Hackathon wide format: columns [review_clean/review_text, aspect_sentiments]
3) Long format fallback: columns [review_id, review_clean/review_text, aspect, sentiment]

Usage:
    python train_absa.py
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


CLASSES: List[str] = [
    "ambiance|negative",
    "ambiance|neutral",
    "ambiance|positive",
    "app_experience|negative",
    "app_experience|neutral",
    "app_experience|positive",
    "cleanliness|negative",
    "cleanliness|neutral",
    "cleanliness|positive",
    "delivery|negative",
    "delivery|neutral",
    "delivery|positive",
    "food|negative",
    "food|neutral",
    "food|positive",
    "general|negative",
    "general|neutral",
    "general|positive",
    "none|neutral",
    "price|negative",
    "price|neutral",
    "price|positive",
    "service|negative",
    "service|neutral",
    "service|positive",
]


def load_data(csv_path: Path | str = "Data/processed/train_augmented_wide.csv") -> pd.DataFrame:
    """Load input CSV and validate minimum required structure."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[INFO] Loaded data from {path} with shape={df.shape}", flush=True)

    columns = set(df.columns)
    has_label_wide = {"text", "labels"}.issubset(columns)
    has_absa_wide = "aspect_sentiments" in columns and ("review_clean" in columns or "review_text" in columns)
    has_long = {"aspect", "sentiment"}.issubset(columns) and (
        "review_clean" in columns or "review_text" in columns
    )

    if not has_label_wide and not has_absa_wide and not has_long:
        raise ValueError(
            "Expected columns ['text', 'labels'], wide ABSA columns including "
            "['aspect_sentiments', 'review_clean/review_text'], or long-format columns including "
            "['aspect', 'sentiment', 'review_clean/review_text']"
        )
    return df


def _clean_text(value: object) -> str:
    """Normalize text safely for Arabic and English without lossy transforms."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _parse_labels_value(raw: object) -> List[str]:
    """Parse a labels cell that may be list-like, JSON-like, or comma-separated."""

    if raw is None or pd.isna(raw):
        return []

    if isinstance(raw, (list, tuple, set)):
        tokens = [str(x).strip().lower() for x in raw]
        return [x for x in tokens if x]

    value = str(raw).strip()
    if not value:
        return []

    # Try literal list parsing first: ['food|positive', 'service|negative']
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                tokens = [str(x).strip().lower() for x in parsed]
                return [x for x in tokens if x]
        except (ValueError, SyntaxError):
            pass

    tokens = [part.strip().lower() for part in value.split(",")]
    return [x for x in tokens if x]


def _parse_jsonish_value(raw: object) -> object:
    """Parse JSON/literal cells emitted by the cleaning pipeline."""

    if raw is None or pd.isna(raw):
        return None

    value = str(raw).strip()
    if not value:
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def _labels_from_aspect_sentiments(raw: object) -> List[str]:
    parsed = _parse_jsonish_value(raw)
    labels: List[str] = []
    if isinstance(parsed, dict):
        for aspect, sentiment in parsed.items():
            aspect_text = str(aspect).strip().lower()
            sentiment_text = str(sentiment).strip().lower()
            if aspect_text and sentiment_text:
                labels.append(f"{aspect_text}|{sentiment_text}")
    return labels


def preprocess(df: pd.DataFrame, allowed_classes: Sequence[str] = CLASSES) -> Tuple[List[str], List[List[str]]]:
    """Clean texts and convert target labels into a multi-label list per sample."""

    allowed = set(allowed_classes)
    columns = set(df.columns)

    if {"text", "labels"}.issubset(columns):
        work = df[["text", "labels"]].copy()
        work["text"] = work["text"].map(_clean_text)
        work["labels"] = work["labels"].map(_parse_labels_value)
    elif "aspect_sentiments" in columns and ("review_clean" in columns or "review_text" in columns):
        text_col = "review_clean" if "review_clean" in columns else "review_text"
        work = df[[text_col, "aspect_sentiments"]].copy()
        work = work.rename(columns={text_col: "text"})
        work["text"] = work["text"].map(_clean_text)
        work["labels"] = work["aspect_sentiments"].map(_labels_from_aspect_sentiments)
    else:
        # Fallback for long-format rows: one aspect/sentiment per row.
        text_col = "review_clean" if "review_clean" in columns else "review_text"
        work = df.copy()
        work[text_col] = work[text_col].map(_clean_text)
        work["aspect"] = work["aspect"].fillna("").astype(str).str.strip().str.lower()
        work["sentiment"] = work["sentiment"].fillna("").astype(str).str.strip().str.lower()
        work["label"] = work["aspect"] + "|" + work["sentiment"]
        work = work[(work[text_col] != "") & (work["label"].str.contains(r"\|", regex=True))]
        grouped = work.groupby(text_col, as_index=False)["label"].apply(lambda s: sorted(set(s)))
        grouped = grouped.rename(columns={text_col: "text", "label": "labels"})
        work = grouped

    work = work[work["text"] != ""].copy()

    unknown_counter: Counter[str] = Counter()

    def filter_labels(labels: Iterable[str]) -> List[str]:
        cleaned: List[str] = []
        for label in labels:
            norm = str(label).strip().lower()
            if not norm:
                continue
            if norm in allowed:
                cleaned.append(norm)
            else:
                unknown_counter[norm] += 1
        return sorted(set(cleaned))

    work["labels"] = work["labels"].map(filter_labels)
    work = work[work["labels"].map(len) > 0].reset_index(drop=True)

    if len(work) == 0:
        raise ValueError("No valid rows left after preprocessing. Check text/labels formatting.")

    texts = work["text"].tolist()
    labels = work["labels"].tolist()

    label_distribution = Counter(label for row in labels for label in row)
    print(f"[INFO] Preprocessed samples={len(texts):,}", flush=True)
    print(f"[INFO] Unique labels present={len(label_distribution):,}/25", flush=True)
    print("[INFO] Label distribution:", flush=True)
    for cls in allowed_classes:
        print(f"  - {cls}: {label_distribution.get(cls, 0)}", flush=True)

    if unknown_counter:
        top_unknown = unknown_counter.most_common(10)
        print(f"[WARN] Found {sum(unknown_counter.values())} unknown labels. Top examples: {top_unknown}", flush=True)

    return texts, labels


def train_model(
    texts: Sequence[str],
    labels: Sequence[Sequence[str]],
    classes: Sequence[str] = CLASSES,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train vectorizer + OneVsRest LogisticRegression with reproducible split."""

    mlb = MultiLabelBinarizer(classes=list(classes))
    y = mlb.fit_transform(labels)

    x_train, x_val, y_train, y_val = train_test_split(
        list(texts),
        y,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )

    print(f"[INFO] Train size={len(x_train):,}, Validation size={len(x_val):,}", flush=True)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=120000,
        lowercase=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)

    print(f"[INFO] X_train shape={x_train_vec.shape}, X_val shape={x_val_vec.shape}", flush=True)

    classifier = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    )
    classifier.fit(x_train_vec, y_train)

    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "label_binarizer": mlb,
        "classes": list(classes),
        "x_val_vec": x_val_vec,
        "y_val": y_val,
    }


def evaluate(model_bundle: Dict[str, object]) -> Dict[str, object]:
    """Evaluate on validation split using F1 micro and per-class report."""

    classifier = model_bundle["classifier"]
    x_val_vec = model_bundle["x_val_vec"]
    y_val = model_bundle["y_val"]
    classes = model_bundle["classes"]

    y_pred = classifier.predict(x_val_vec)
    f1_micro = f1_score(y_val, y_pred, average="micro", zero_division=0)
    report_text = classification_report(
        y_val,
        y_pred,
        target_names=list(classes),
        zero_division=0,
    )

    print(f"[INFO] Validation F1 micro: {f1_micro:.4f}", flush=True)
    print("[INFO] Per-class classification report:", flush=True)
    print(report_text, flush=True)

    return {
        "f1_micro": float(f1_micro),
        "classification_report": report_text,
    }


def save_model(model_bundle: Dict[str, object], output_path: Path | str = "models/local_absa_weights.joblib") -> Path:
    """Persist exactly the required deployable model bundle."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "vectorizer": model_bundle["vectorizer"],
        "classifier": model_bundle["classifier"],
        "label_binarizer": model_bundle["label_binarizer"],
        "classes": model_bundle["classes"],
    }
    joblib.dump(artifact, path)
    print(f"[INFO] Saved model weights to: {path}", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train multi-label ABSA model and save local weights.")
    parser.add_argument("--input", default="Data/processed/train_augmented_wide.csv", help="Input CSV path")
    parser.add_argument("--output", default="models/local_absa_weights_v3_wide.joblib", help="Output .joblib file path")
    args = parser.parse_args()

    df = load_data(args.input)
    texts, labels = preprocess(df)
    model_bundle = train_model(texts, labels, classes=CLASSES, random_state=42)
    evaluate(model_bundle)
    save_model(model_bundle, args.output)
    print("[INFO] Training completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
