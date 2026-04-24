"""Phase 3 Groq inference engine for Arabic ABSA.

This module provides a single importable predict(review_text) function for later
Streamlit integration, plus batch/validation utilities that write JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from scripts.config import PROCESSED_DIR, SQLITE_DB_PATH, apply_local_runtime_defaults
from DataBase.db import get_connection, initialize_database
from DataBase.mongo_db import log_prediction as mongo_log_prediction, log_run_artifacts
from scripts.rag_utils import index_is_ready, retrieve_examples

VALID_ASPECTS = {
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
    "none",
}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

DEFAULT_PRIMARY_MODEL = "llama-3.3-70b-versatile"
DEFAULT_FALLBACK_MODEL = "llama-3.1-8b-instant"
DEFAULT_VAL_CSV = PROCESSED_DIR / "val_clean.csv"
DEFAULT_RESULTS_JSON = PROCESSED_DIR / "val_results.json"
DEFAULT_PREDICTIONS_JSON = PROCESSED_DIR / "val_predictions.json"

SYSTEM_PROMPT = """You are an expert in Aspect-Based Sentiment Analysis for Arabic customer reviews.

TASK: Extract ALL aspects mentioned in the review. For each aspect, classify sentiment.

VALID ASPECTS (use ONLY these exact strings):
food, service, price, cleanliness, delivery, ambiance, app_experience, general, none

VALID SENTIMENTS (use ONLY these exact strings):
positive, negative, neutral

OUTPUT FORMAT - return ONLY this JSON, no preamble, no explanation:
{"predictions": [{"aspect": "<aspect>", "sentiment": "<sentiment>"}]}

RULES:
- A review can have MULTIPLE aspects. Extract all of them.
- If no specific aspect is mentioned, use "general".
- Never invent aspects outside the valid list.
- Never output anything except the JSON object.
- Do not output any text before or after the JSON object.
"""

_CLIENT = None


class RateLimiter:
    """Token bucket limiter for Groq free-tier safety margins."""

    def __init__(self, requests_per_minute: int = 28, daily_quota: int = 14400) -> None:
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.daily_quota = max(1, int(daily_quota))
        self._tokens = float(self.requests_per_minute)
        self._refill_rate = self.requests_per_minute / 60.0
        self._last_refill = time.monotonic()
        self._calls_today = 0
        self._day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _reset_day_if_needed(self) -> None:
        current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if current_day != self._day_key:
            self._day_key = current_day
            self._calls_today = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(
            float(self.requests_per_minute),
            self._tokens + elapsed * self._refill_rate,
        )

    def acquire(self) -> None:
        while True:
            self._reset_day_if_needed()
            if self._calls_today >= self.daily_quota:
                raise RuntimeError("Daily Groq request quota reached.")

            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._calls_today += 1
                return

            wait_seconds = (1.0 - self._tokens) / self._refill_rate
            time.sleep(max(wait_seconds, 0.05))


def _warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def _info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def _default_prediction() -> Dict[str, Any]:
    return {"predictions": [{"aspect": "general", "sentiment": "neutral"}]}


def _parse_json_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    text_fixed = text.replace("'", '"')
    try:
        return json.loads(text_fixed)
    except Exception:
        return None


def _pairs_from_gold_row(row: pd.Series) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    if "aspect" in row and "sentiment" in row:
        aspect = str(row.get("aspect", "")).strip().lower()
        sentiment = str(row.get("sentiment", "")).strip().lower()
        if aspect in VALID_ASPECTS and sentiment in VALID_SENTIMENTS:
            return [(aspect, sentiment)]

    aspects_raw = _parse_json_cell(row.get("aspects"))
    sentiments_raw = _parse_json_cell(row.get("aspect_sentiments"))

    aspects: List[str] = []
    if isinstance(aspects_raw, list):
        aspects = [str(x).strip().lower() for x in aspects_raw if str(x).strip()]

    sentiment_map: Dict[str, str] = {}
    if isinstance(sentiments_raw, dict):
        sentiment_map = {
            str(k).strip().lower(): str(v).strip().lower()
            for k, v in sentiments_raw.items()
            if str(k).strip()
        }

    for aspect in aspects:
        sentiment = sentiment_map.get(aspect)
        if aspect in VALID_ASPECTS and sentiment in VALID_SENTIMENTS:
            pairs.append((aspect, sentiment))

    if not pairs:
        pairs.append(("general", "neutral"))

    return sorted(set(pairs))


def _build_rag_shots(review: str, rag_shots: int, max_distance: float) -> str:
    if rag_shots <= 0:
        return ""

    try:
        if not index_is_ready():
            _warn("RAG index is not ready; continuing without few-shot examples.")
            return ""
    except Exception as exc:
        _warn(f"RAG index check failed; continuing without few-shot examples. Reason: {exc}")
        return ""

    try:
        candidates = retrieve_examples(review, n=max(3, rag_shots * 2))
    except Exception as exc:
        _warn(f"Failed to retrieve RAG examples: {exc}")
        return ""

    shots: List[str] = []
    for ex in candidates:
        distance = ex.get("distance")
        try:
            if distance is not None and float(distance) > max_distance:
                continue
        except Exception:
            pass

        aspect = str(ex.get("aspect", "")).strip().lower()
        sentiment = str(ex.get("sentiment", "")).strip().lower()
        if aspect not in VALID_ASPECTS or sentiment not in VALID_SENTIMENTS:
            continue

        review_text = str(ex.get("review", "")).strip()
        if not review_text:
            continue

        payload = {"predictions": [{"aspect": aspect, "sentiment": sentiment}]}
        shots.append(f"Review: {review_text}\nOutput: {json.dumps(payload, ensure_ascii=False)}")
        if len(shots) >= rag_shots:
            break

    if not shots:
        return ""

    return "\n\n".join(shots)


def build_prompt(review: str, rag_shots: int = 3, max_distance: float = 0.7) -> str:
    shots_block = _build_rag_shots(review=review, rag_shots=rag_shots, max_distance=max_distance)
    if shots_block:
        return f"Few-shot examples:\n{shots_block}\n\nReview: {review}\nOutput:"
    return f"Review: {review}\nOutput:"


def parse_response(raw: str) -> Tuple[Dict[str, Any], str]:
    if not raw:
        return _default_prediction(), "fallback"

    # Layer 1: direct JSON parse.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, "direct"
    except Exception:
        pass

    # Layer 2: extract the first JSON object from mixed text.
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "regex"
        except Exception:
            pass

    # Layer 3: safe fallback.
    return _default_prediction(), "fallback"


def validate_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    raw_preds = prediction.get("predictions", []) if isinstance(prediction, dict) else []
    cleaned: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    if isinstance(raw_preds, list):
        for item in raw_preds:
            if not isinstance(item, dict):
                continue
            aspect = str(item.get("aspect", "")).strip().lower()
            sentiment = str(item.get("sentiment", "")).strip().lower()
            if aspect in VALID_ASPECTS and sentiment in VALID_SENTIMENTS:
                key = (aspect, sentiment)
                if key not in seen:
                    seen.add(key)
                    cleaned.append({"aspect": aspect, "sentiment": sentiment})

    if not cleaned:
        cleaned = _default_prediction()["predictions"]

    return {"predictions": cleaned}


def _ensure_phase3_tables(db_path: Path | str = SQLITE_DB_PATH) -> None:
    initialize_database(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                raw_response TEXT,
                parse_status TEXT NOT NULL,
                model_name TEXT,
                latency_ms REAL,
                error_text TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def _log_prediction(
    review_text: str,
    prediction: Dict[str, Any],
    raw_response: str,
    parse_status: str,
    model_name: str,
    latency_ms: float,
    error_text: str = "",
    db_path: Path | str = SQLITE_DB_PATH,
) -> None:
    _ensure_phase3_tables(db_path)
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs(
                review_text,
                prediction_json,
                raw_response,
                parse_status,
                model_name,
                latency_ms,
                error_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_text,
                json.dumps(prediction, ensure_ascii=False),
                raw_response,
                parse_status,
                model_name,
                latency_ms,
                error_text,
            ),
        )
        connection.commit()

    # Mirror logs to MongoDB when available; do not break inference if Mongo is offline.
    try:
        mongo_log_prediction(
            review_text=review_text,
            result=prediction,
            latency_ms=latency_ms,
            parse_status=parse_status,
            model_version=model_name,
            raw_response=raw_response,
            error_text=error_text,
        )
    except Exception:
        pass


def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    apply_local_runtime_defaults()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment or .env file.")

    try:
        from groq import Groq  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("groq package is required. Install dependencies with: pip install -r requirements.txt") from exc

    _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def _call_groq(
    review: str,
    rag_shots: int,
    max_distance: float,
    primary_model: str,
    fallback_model: str,
    limiter: RateLimiter,
) -> Tuple[Dict[str, Any], str, str, float]:
    client = _get_client()
    user_prompt = build_prompt(review=review, rag_shots=rag_shots, max_distance=max_distance)

    last_error = ""
    for model_name in (primary_model, fallback_model):
        if not model_name:
            continue
        try:
            limiter.acquire()
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw = response.choices[0].message.content or ""
            parsed, parse_status = parse_response(raw)
            validated = validate_prediction(parsed)
            return validated, raw, f"{model_name}|{parse_status}", latency_ms
        except Exception as exc:
            last_error = str(exc)
            _warn(f"Model {model_name} failed: {exc}")

    raise RuntimeError(f"All model attempts failed. Last error: {last_error}")


def predict(
    review: str,
    rag_shots: int = 3,
    max_distance: float = 0.7,
    primary_model: str = DEFAULT_PRIMARY_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    limiter: Optional[RateLimiter] = None,
) -> Dict[str, Any]:
    """Predict aspect-sentiment pairs for one review and return submission-ready JSON."""

    text = str(review or "").strip()
    if not text:
        return _default_prediction()

    limiter = limiter or RateLimiter()
    raw = ""
    parse_status = "fallback"
    model_used = ""
    latency_ms = 0.0

    try:
        output, raw, status_token, latency_ms = _call_groq(
            review=text,
            rag_shots=rag_shots,
            max_distance=max_distance,
            primary_model=primary_model,
            fallback_model=fallback_model,
            limiter=limiter,
        )
        model_used, parse_status = status_token.split("|", 1)
        _log_prediction(
            review_text=text,
            prediction=output,
            raw_response=raw,
            parse_status=parse_status,
            model_name=model_used,
            latency_ms=latency_ms,
        )
        return output
    except Exception as exc:
        fallback = _default_prediction()
        _log_prediction(
            review_text=text,
            prediction=fallback,
            raw_response=raw,
            parse_status="fallback",
            model_name=model_used,
            latency_ms=latency_ms,
            error_text=str(exc),
        )
        return fallback


def _prediction_pairs(prediction: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in prediction.get("predictions", []):
        aspect = str(item.get("aspect", "")).strip().lower()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if aspect in VALID_ASPECTS and sentiment in VALID_SENTIMENTS:
            pairs.append((aspect, sentiment))
    if not pairs:
        pairs = [("general", "neutral")]
    return sorted(set(pairs))


def _to_submission_row(review_id: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    pairs = _prediction_pairs(prediction)
    aspects = [aspect for aspect, _ in pairs]
    aspect_sentiments = {aspect: sentiment for aspect, sentiment in pairs}
    return {
        "review_id": review_id,
        "aspects": aspects,
        "aspect_sentiments": aspect_sentiments,
    }


def _all_labels() -> List[str]:
    labels: List[str] = []
    for aspect in sorted(VALID_ASPECTS):
        for sentiment in sorted(VALID_SENTIMENTS):
            labels.append(f"{aspect}|{sentiment}")
    return labels


def _compute_f1(
    gold_sets: Sequence[set[str]],
    pred_sets: Sequence[set[str]],
) -> Tuple[float, Dict[str, float]]:
    labels = _all_labels()
    per_tp = {label: 0 for label in labels}
    per_fp = {label: 0 for label in labels}
    per_fn = {label: 0 for label in labels}

    for gold, pred in zip(gold_sets, pred_sets):
        for label in labels:
            in_gold = label in gold
            in_pred = label in pred
            if in_gold and in_pred:
                per_tp[label] += 1
            elif (not in_gold) and in_pred:
                per_fp[label] += 1
            elif in_gold and (not in_pred):
                per_fn[label] += 1

    total_tp = sum(per_tp.values())
    total_fp = sum(per_fp.values())
    total_fn = sum(per_fn.values())
    micro_den = (2 * total_tp) + total_fp + total_fn
    micro_f1 = (2 * total_tp / micro_den) if micro_den else 0.0

    per_class_f1: Dict[str, float] = {}
    for label in labels:
        den = (2 * per_tp[label]) + per_fp[label] + per_fn[label]
        per_class_f1[label] = (2 * per_tp[label] / den) if den else 0.0

    return micro_f1, per_class_f1


def _iter_validation_rows(val_csv: Path | str, max_rows: Optional[int]) -> Iterable[pd.Series]:
    frame = pd.read_csv(Path(val_csv), encoding="utf-8-sig")
    if "review_clean" not in frame.columns and "review_text" not in frame.columns:
        raise ValueError("Validation CSV must contain review_clean or review_text column.")

    if max_rows is not None:
        frame = frame.head(max(1, int(max_rows))).copy()

    for _, row in frame.iterrows():
        yield row


def batch_predict(
    reviews: Sequence[str],
    rag_shots: int = 3,
    max_distance: float = 0.7,
    primary_model: str = DEFAULT_PRIMARY_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
) -> List[Dict[str, Any]]:
    """Run predict over a sequence with progress logs."""

    limiter = RateLimiter()
    outputs: List[Dict[str, Any]] = []
    total = len(reviews)
    for idx, review in enumerate(reviews, start=1):
        outputs.append(
            predict(
                review=review,
                rag_shots=rag_shots,
                max_distance=max_distance,
                primary_model=primary_model,
                fallback_model=fallback_model,
                limiter=limiter,
            )
        )
        if idx == 1 or idx % 10 == 0 or idx == total:
            _info(f"batch_predict progress: {idx}/{total}")
    return outputs


def run_validation(
    val_csv: Path | str = DEFAULT_VAL_CSV,
    output_json: Path | str = DEFAULT_RESULTS_JSON,
    predictions_json: Path | str = DEFAULT_PREDICTIONS_JSON,
    max_rows: int = 500,
    rag_shots: int = 3,
    max_distance: float = 0.7,
    primary_model: str = DEFAULT_PRIMARY_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
) -> Dict[str, Any]:
    """Run validation sweep, compute F1, and write JSON artifacts."""

    limiter = RateLimiter()
    output_json = Path(output_json)
    predictions_json = Path(predictions_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    predictions_json.parent.mkdir(parents=True, exist_ok=True)

    prediction_rows: List[Dict[str, Any]] = []
    gold_sets: List[set[str]] = []
    pred_sets: List[set[str]] = []
    fallback_count = 0

    rows = list(_iter_validation_rows(val_csv=val_csv, max_rows=max_rows))
    total = len(rows)
    _info(f"Running validation on {total} rows...")

    for idx, row in enumerate(rows, start=1):
        review_text = str(row.get("review_clean") or row.get("review_text") or "").strip()
        if not review_text:
            review_text = ""

        raw = ""
        parse_status = "fallback"
        model_used = ""
        latency_ms = 0.0
        prediction = _default_prediction()
        error_text = ""

        try:
            prediction, raw, status_token, latency_ms = _call_groq(
                review=review_text,
                rag_shots=rag_shots,
                max_distance=max_distance,
                primary_model=primary_model,
                fallback_model=fallback_model,
                limiter=limiter,
            )
            model_used, parse_status = status_token.split("|", 1)
        except Exception as exc:
            error_text = str(exc)
            parse_status = "fallback"
            prediction = _default_prediction()

        if parse_status == "fallback":
            fallback_count += 1

        _log_prediction(
            review_text=review_text,
            prediction=prediction,
            raw_response=raw,
            parse_status=parse_status,
            model_name=model_used,
            latency_ms=latency_ms,
            error_text=error_text,
        )

        gold_pairs = _pairs_from_gold_row(row)
        pred_pairs = _prediction_pairs(prediction)
        gold_set = {f"{a}|{s}" for a, s in gold_pairs}
        pred_set = {f"{a}|{s}" for a, s in pred_pairs}

        gold_sets.append(gold_set)
        pred_sets.append(pred_set)

        prediction_rows.append(_to_submission_row(str(row.get("review_id", idx)), prediction))

        if idx == 1 or idx % 10 == 0 or idx == total:
            _info(f"validation progress: {idx}/{total}")

    micro_f1, per_class_f1 = _compute_f1(gold_sets=gold_sets, pred_sets=pred_sets)
    parse_failure_rate = (fallback_count / total) if total else 1.0

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "rows_evaluated": total,
        "rag_shots": int(rag_shots),
        "max_distance": float(max_distance),
        "primary_model": primary_model,
        "fallback_model": fallback_model,
        "parse_failure_rate": round(parse_failure_rate, 6),
        "f1_micro": round(micro_f1, 6),
        "per_class_f1": {k: round(v, 6) for k, v in per_class_f1.items()},
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    with predictions_json.open("w", encoding="utf-8") as file:
        json.dump(prediction_rows, file, ensure_ascii=False, indent=2)

    # Save run summary and submission rows to MongoDB when available.
    try:
        log_run_artifacts(summary=results, submission_rows=prediction_rows)
    except Exception:
        pass

    _info(f"Saved validation summary to {output_json}")
    _info(f"Saved per-row predictions to {predictions_json}")
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 Groq ABSA inference engine")
    parser.add_argument("--review", default=None, help="Single review text to predict")
    parser.add_argument("--val-csv", default=str(DEFAULT_VAL_CSV), help="Validation CSV path")
    parser.add_argument("--rows", type=int, default=500, help="Rows for validation run")
    parser.add_argument("--rag-shots", type=int, default=3, help="RAG few-shot examples count")
    parser.add_argument("--max-distance", type=float, default=0.7, help="RAG distance threshold")
    parser.add_argument("--model", default=DEFAULT_PRIMARY_MODEL, help="Primary Groq model")
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL, help="Fallback Groq model")
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_JSON), help="Validation summary JSON output")
    parser.add_argument(
        "--predictions-output",
        default=str(DEFAULT_PREDICTIONS_JSON),
        help="Per-row predictions JSON output",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.review:
        result = predict(
            review=args.review,
            rag_shots=max(0, int(args.rag_shots)),
            max_distance=float(args.max_distance),
            primary_model=str(args.model),
            fallback_model=str(args.fallback_model),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0

    summary = run_validation(
        val_csv=Path(args.val_csv),
        output_json=Path(args.output),
        predictions_json=Path(args.predictions_output),
        max_rows=max(1, int(args.rows)),
        rag_shots=max(0, int(args.rag_shots)),
        max_distance=float(args.max_distance),
        primary_model=str(args.model),
        fallback_model=str(args.fallback_model),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())