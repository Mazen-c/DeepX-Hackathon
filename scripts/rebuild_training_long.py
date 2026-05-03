"""Rebuild the long augmented training CSV from the clean wide CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIDE_PATH = PROJECT_ROOT / "Data" / "processed" / "train_augmented_wide.csv"
LONG_PATH = PROJECT_ROOT / "Data" / "processed" / "train_augmented.csv"

LONG_COLUMNS = [
    "review_id",
    "review_text",
    "review_clean",
    "star_rating",
    "date",
    "business_name",
    "business_category",
    "platform",
    "aspect",
    "sentiment",
    "source",
]


def _clean_cell(value: object) -> str:
    return " ".join(str(value or "").replace("???", " ").split())


def rebuild() -> tuple[int, int]:
    rows_written = 0
    wide_rows = 0
    with WIDE_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        with LONG_PATH.open("w", encoding="utf-8-sig", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=LONG_COLUMNS)
            writer.writeheader()

            for row in reader:
                wide_rows += 1
                try:
                    aspect_sentiments = json.loads(row.get("aspect_sentiments") or "{}")
                except json.JSONDecodeError:
                    aspect_sentiments = {}

                if not isinstance(aspect_sentiments, dict) or not aspect_sentiments:
                    aspect_sentiments = {"general": "neutral"}

                for aspect, sentiment in aspect_sentiments.items():
                    writer.writerow(
                        {
                            "review_id": _clean_cell(row.get("review_id", "")),
                            "review_text": _clean_cell(row.get("review_text", "")),
                            "review_clean": _clean_cell(row.get("review_clean", "")),
                            "star_rating": _clean_cell(row.get("star_rating", "")),
                            "date": _clean_cell(row.get("date", "")),
                            "business_name": _clean_cell(row.get("business_name", "")),
                            "business_category": _clean_cell(row.get("business_category", "")),
                            "platform": _clean_cell(row.get("platform", "")),
                            "aspect": _clean_cell(str(aspect).lower()),
                            "sentiment": _clean_cell(str(sentiment).lower()),
                            "source": "wide_rebuilt",
                        }
                    )
                    rows_written += 1

    return wide_rows, rows_written


def main() -> int:
    wide_rows, rows_written = rebuild()
    print(
        f"[INFO] Rebuilt {LONG_PATH} with {rows_written:,} long rows from {wide_rows:,} wide rows.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
