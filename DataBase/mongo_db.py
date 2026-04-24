"""MongoDB helpers for Phase 4 dashboard and prediction logging."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "arabic_absa")


def _get_collection(name: str):
    from pymongo import MongoClient  # type: ignore

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.admin.command("ping")
    db = client[MONGO_DB]
    return client, db[name]


def init_db() -> bool:
    """Ensure MongoDB collections and indexes exist."""

    try:
        client, predictions = _get_collection("predictions")
        _, runs = _get_collection("runs")
        predictions.create_index("created_at")
        predictions.create_index("review_id")
        runs.create_index("created_at")
        client.close()
        return True
    except Exception:
        return False


def log_prediction(
    review_text: str,
    result: Dict[str, Any],
    latency_ms: float,
    parse_status: str = "ok",
    model_version: str = "",
    review_id: str = "",
    raw_response: str = "",
    error_text: str = "",
) -> bool:
    try:
        client, predictions = _get_collection("predictions")
        doc = {
            "review_id": str(review_id or ""),
            "review_text": str(review_text or ""),
            "predictions_json": result,
            "latency_ms": float(latency_ms or 0.0),
            "parse_status": str(parse_status or "fallback"),
            "model_version": str(model_version or ""),
            "raw_response": str(raw_response or ""),
            "error_text": str(error_text or ""),
            "created_at": datetime.now(timezone.utc),
        }
        predictions.insert_one(doc)
        client.close()
        return True
    except Exception:
        return False


def log_run_artifacts(summary: Dict[str, Any], submission_rows: List[Dict[str, Any]]) -> bool:
    """Store validation summary and generated submission rows in MongoDB."""

    try:
        client, runs = _get_collection("runs")
        runs.insert_one(
            {
                "created_at": datetime.now(timezone.utc),
                "summary": summary,
                "submission_rows": submission_rows,
            }
        )
        client.close()
        return True
    except Exception:
        return False


def get_recent(n: int = 20) -> pd.DataFrame:
    try:
        client, predictions = _get_collection("predictions")
        rows = list(
            predictions.find(
                {},
                {
                    "_id": 0,
                    "review_text": 1,
                    "predictions_json": 1,
                    "latency_ms": 1,
                    "parse_status": 1,
                    "model_version": 1,
                    "created_at": 1,
                },
            )
            .sort("created_at", -1)
            .limit(max(1, int(n)))
        )
        client.close()
        if not rows:
            return pd.DataFrame(columns=["review_text", "predictions_json", "latency_ms", "parse_status"])

        for row in rows:
            value = row.get("predictions_json")
            row["predictions_json"] = json.dumps(value, ensure_ascii=False)
            created = row.get("created_at")
            row["created_at"] = created.isoformat() if hasattr(created, "isoformat") else str(created or "")

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["review_text", "predictions_json", "latency_ms", "parse_status"])


def get_stats() -> pd.DataFrame:
    try:
        client, predictions = _get_collection("predictions")
        rows = list(
            predictions.find(
                {},
                {
                    "_id": 0,
                    "predictions_json": 1,
                },
            )
        )
        client.close()

        flattened: List[Dict[str, Any]] = []
        for row in rows:
            payload = row.get("predictions_json", {})
            predictions_list = payload.get("predictions", []) if isinstance(payload, dict) else []
            if isinstance(predictions_list, list):
                for item in predictions_list:
                    if isinstance(item, dict):
                        aspect = str(item.get("aspect", "")).strip()
                        sentiment = str(item.get("sentiment", "")).strip()
                        if aspect and sentiment:
                            flattened.append({"aspect": aspect, "sentiment": sentiment})

        if not flattened:
            return pd.DataFrame(columns=["aspect", "sentiment", "count"])

        frame = pd.DataFrame(flattened)
        grouped = (
            frame.groupby(["aspect", "sentiment"])
            .size()
            .reset_index(name="count")
            .sort_values(["aspect", "sentiment"])
        )
        return grouped
    except Exception:
        return pd.DataFrame(columns=["aspect", "sentiment", "count"])


def get_quality_metrics() -> Dict[str, float]:
    try:
        client, predictions = _get_collection("predictions")
        rows = list(
            predictions.find(
                {},
                {
                    "_id": 0,
                    "latency_ms": 1,
                    "parse_status": 1,
                },
            )
        )
        client.close()
        total = len(rows)
        if total == 0:
            return {"total_predictions": 0.0, "avg_latency_ms": 0.0, "parse_success_rate": 0.0}

        avg_latency = sum(float(r.get("latency_ms") or 0.0) for r in rows) / total
        ok_count = sum(1 for r in rows if str(r.get("parse_status", "")).lower() != "fallback")
        success_rate = ok_count / total
        return {
            "total_predictions": float(total),
            "avg_latency_ms": float(round(avg_latency, 2)),
            "parse_success_rate": float(round(success_rate, 4)),
        }
    except Exception:
        return {"total_predictions": 0.0, "avg_latency_ms": 0.0, "parse_success_rate": 0.0}
