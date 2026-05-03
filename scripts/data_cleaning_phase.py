"""Lightweight Arabic text cleaning helpers used by inference.

The original phase-1 data artifacts are already stored under Data/processed.
This module keeps the shared `clean_text` function importable for the UI and
hidden-test prediction paths.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

try:
    from pyarabic import araby
except ImportError:  # pragma: no cover - pyarabic is listed in requirements
    araby = None


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL_RE = re.compile(r"\u0640+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_MARK_RE = re.compile(r"#")
_SPACES_RE = re.compile(r"\s+")


def _is_missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def clean_text(value: Any) -> str:
    """Return a safe, CPU-only normalized review string.

    The function is intentionally conservative: it removes obvious noise, keeps
    Arabic/English words and punctuation, and avoids lossy translation or model
    calls.
    """

    if _is_missing(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_MARK_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)

    if araby is not None:
        text = araby.strip_tashkeel(text)
        text = araby.normalize_ligature(text)
        text = araby.normalize_hamza(text)
    else:
        text = _ARABIC_DIACRITICS_RE.sub("", text)

    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = _SPACES_RE.sub(" ", text)
    return text.strip().lower()

