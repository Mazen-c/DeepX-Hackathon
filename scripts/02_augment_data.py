"""Phase 1 Data Augmentation: Groq synthetic generation + Helsinki-NLP back-translation."""
from __future__ import annotations

import html
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# ── configuration ─────────────────────────────────────────────────────────────
TARGET_COUNT = 100           # bring each pair up to this
MAX_PER_PAIR = 30            # hard cap on Groq-generated reviews per pair
BATCH_SIZE = 5               # reviews requested per API call
RATE_LIMIT_SLEEP = 2.1       # seconds between every Groq API call
BACK_TRANSLATE_SAMPLE = 20   # existing reviews to back-translate per run
GROQ_MODEL = "openai/gpt-oss-120b"

INPUT_PATH = Path("data/processed/train_clean.csv")
OUTPUT_PATH = Path("data/processed/train_augmented.csv")

# ── text normalisation ────────────────────────────────────────────────────────
_ARABIC_DIACRITICS_RE = re.compile(r"[ؗ-ًؚ-ْ]")
_WHITESPACE_RE = re.compile(r"\s+")
_ALEF_RE = re.compile(r"[أإآ]")


def _clean(text: str) -> str:
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = _ALEF_RE.sub("ا", text)
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ── Groq helpers ──────────────────────────────────────────────────────────────
_ASPECT_AR: Dict[str, str] = {
    "food": "الطعام",
    "service": "الخدمة",
    "price": "السعر",
    "cleanliness": "النظافة",
    "delivery": "التوصيل",
    "ambiance": "الأجواء",
    "app_experience": "تجربة التطبيق",
    "general": "التجربة العامة",
    "none": "عام",
}
_SENTIMENT_AR: Dict[str, str] = {
    "positive": "إيجابية",
    "negative": "سلبية",
    "neutral": "محايدة",
}


def _build_prompt(aspect: str, sentiment: str, n: int) -> str:
    aspect_ar = _ASPECT_AR.get(aspect, aspect)
    sentiment_ar = _SENTIMENT_AR.get(sentiment, sentiment)
    return (
        f"اكتب {n} تعليقات قصيرة لعملاء مطاعم بالعامية المصرية أو السعودية. "
        f"كل تعليق يجب أن يتحدث فقط عن {aspect_ar} بشعور {sentiment_ar}. "
        "أرجع النتيجة كـ JSON array من النصوص فقط، بدون أي شرح إضافي. "
        'مثال: ["التعليق الأول", "التعليق الثاني"]'
    )


def _parse_groq_response(content: str) -> List[str]:
    """Extract a list of strings from the Groq response with robust fallback parsing."""
    # Strip markdown fences
    content = re.sub(r"```(?:json)?", "", content).strip("`").strip()

    # Direct JSON parse
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # Find JSON array anywhere in the response
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: extract numbered/bulleted lines
    lines = []
    for line in content.splitlines():
        line = re.sub(r'^\d+[\.\)]\s*["\']?', "", line).rstrip('",\'').strip()
        if len(line) > 10:
            lines.append(line)
    return lines


def _groq_batch(client, aspect: str, sentiment: str, n: int) -> List[str]:
    """Single Groq API call. Returns up to n Arabic reviews."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _build_prompt(aspect, sentiment, n)}],
            temperature=0.9,
            max_tokens=600,
        )
        content = response.choices[0].message.content
        return _parse_groq_response(content)[:n]
    except Exception as exc:
        print(f"    [WARN] Groq API error: {exc}", flush=True)
        return []


# ── back-translation ──────────────────────────────────────────────────────────

def back_translate(texts: List[str]) -> List[str]:
    """
    Translate Arabic -> English -> Arabic using 48 MB Helsinki-NLP models.
    Returns an empty list on any failure so the caller can skip gracefully.
    """
    print("\n[Step 2] Back-translation (Helsinki-NLP/opus-mt-ar-en + opus-mt-en-ar)...", flush=True)
    try:
        from transformers import pipeline  # type: ignore

        print("  Loading Helsinki-NLP/opus-mt-ar-en ...", flush=True)
        ar_to_en = pipeline(
            "translation", model="Helsinki-NLP/opus-mt-ar-en", device=-1
        )
        print("  Loading Helsinki-NLP/opus-mt-en-ar ...", flush=True)
        en_to_ar = pipeline(
            "translation", model="Helsinki-NLP/opus-mt-en-ar", device=-1
        )

        print(f"  Translating {len(texts)} reviews ar->en ...", flush=True)
        en_texts = [r["translation_text"] for r in ar_to_en(texts, batch_size=8)]

        print(f"  Translating {len(texts)} reviews en->ar ...", flush=True)
        back_ar = [r["translation_text"] for r in en_to_ar(en_texts, batch_size=8)]

        print(f"  Back-translation produced {len(back_ar)} reviews.", flush=True)
        return back_ar

    except Exception as exc:
        print(f"  [WARN] Back-translation failed and will be skipped: {exc}", flush=True)
        return []


# ── core augmentation logic ───────────────────────────────────────────────────

def identify_deficits(df: pd.DataFrame) -> List[Tuple[str, str, int]]:
    """Return (aspect, sentiment, current_count) for pairs below TARGET_COUNT."""
    counts = df.groupby(["aspect", "sentiment"]).size().reset_index(name="count")
    deficit_rows = counts[counts["count"] < TARGET_COUNT]
    return [
        (row["aspect"], row["sentiment"], int(row["count"]))
        for _, row in deficit_rows.iterrows()
    ]


def _make_row(
    text: str,
    aspect: str,
    sentiment: str,
    uid: int,
    source: str,
) -> Dict:
    return {
        "review_id": f"aug_{source}_{aspect}_{sentiment}_{uid}",
        "review_text": text,
        "review_clean": _clean(text),
        "star_rating": "",
        "date": "",
        "business_name": "synthetic",
        "business_category": "synthetic",
        "platform": "synthetic",
        "aspect": aspect,
        "sentiment": sentiment,
        "source": source,
    }


def augment(df: pd.DataFrame, groq_client: Optional[object]) -> pd.DataFrame:
    deficits = identify_deficits(df)

    if not deficits:
        print("All pairs already meet the TARGET_COUNT threshold. Nothing to augment.")
        return df

    print(
        f"\n[Step 1] Groq synthetic generation — {len(deficits)} deficient pairs "
        f"(< {TARGET_COUNT} samples each):"
    )
    for aspect, sentiment, count in deficits:
        to_gen = min(TARGET_COUNT - count, MAX_PER_PAIR)
        print(f"  ({aspect:<16}, {sentiment:<9}): {count:>4} samples -> generate up to {to_gen}")

    synthetic_rows: List[Dict] = []
    uid = 0

    # ── Groq synthetic generation ─────────────────────────────────────────────
    if groq_client is not None:
        total_calls = sum(
            -(-min(TARGET_COUNT - c, MAX_PER_PAIR) // BATCH_SIZE)  # ceiling division
            for _, _, c in deficits
        )
        print(
            f"\n  Total estimated Groq API calls: {total_calls} "
            f"(sleep {RATE_LIMIT_SLEEP}s between each call)",
            flush=True,
        )

        first_call = True
        for aspect, sentiment, current_count in deficits:
            needed = min(TARGET_COUNT - current_count, MAX_PER_PAIR)
            if needed <= 0:
                continue

            n_calls = -(-needed // BATCH_SIZE)  # ceiling division
            generated: List[str] = []

            for call_idx in range(n_calls):
                # ── RATE LIMIT: sleep between every API call ──────────────────
                if not first_call:
                    time.sleep(RATE_LIMIT_SLEEP)
                first_call = False

                remaining = needed - len(generated)
                batch_n = min(BATCH_SIZE, remaining)
                if batch_n <= 0:
                    break

                print(
                    f"  [{aspect}/{sentiment}] call {call_idx+1}/{n_calls} "
                    f"— requesting {batch_n} reviews...",
                    flush=True,
                )
                batch = _groq_batch(groq_client, aspect, sentiment, batch_n)
                generated.extend(batch)

            for text in generated[:needed]:
                if text.strip():
                    uid += 1
                    synthetic_rows.append(_make_row(text, aspect, sentiment, uid, "groq"))

        groq_added = sum(1 for r in synthetic_rows if r["source"] == "groq")
        print(f"\n  Groq generated {groq_added} synthetic reviews total.", flush=True)

    else:
        print("  Skipping Groq generation (GROQ_API_KEY not set or groq package missing).")

    # ── Back-translation ──────────────────────────────────────────────────────
    deficit_set = {(a, s) for a, s, _ in deficits}
    # Filter df to rows belonging to deficient pairs
    df_filtered = df[
        df.apply(lambda r: (r["aspect"], r["sentiment"]) in deficit_set, axis=1)
    ]
    sample_df = (
        df_filtered.dropna(subset=["review_clean"])
        .loc[df_filtered["review_clean"].str.strip() != ""]
        .head(BACK_TRANSLATE_SAMPLE)
    )

    if not sample_df.empty:
        sample_texts = sample_df["review_clean"].tolist()
        sample_aspects = sample_df["aspect"].tolist()
        sample_sentiments = sample_df["sentiment"].tolist()

        bt_texts = back_translate(sample_texts)

        bt_added = 0
        for bt_text, aspect, sentiment in zip(bt_texts, sample_aspects, sample_sentiments):
            if bt_text and bt_text.strip():
                uid += 1
                bt_added += 1
                synthetic_rows.append(_make_row(bt_text.strip(), aspect, sentiment, uid, "backtrans"))
        print(f"  Back-translation added {bt_added} reviews.", flush=True)
    else:
        print("\n[Step 2] No deficient-class samples available for back-translation. Skipping.")

    # ── merge and return ──────────────────────────────────────────────────────
    if not synthetic_rows:
        print("No synthetic rows generated. Returning original data unchanged.")
        return df

    synth_df = pd.DataFrame(synthetic_rows)

    original_df = df.copy()
    if "source" not in original_df.columns:
        original_df["source"] = "original"

    return pd.concat([original_df, synth_df], ignore_index=True)


# ── reporting ─────────────────────────────────────────────────────────────────

def print_distribution(df: pd.DataFrame, label: str = "") -> None:
    dist = (
        df.groupby(["aspect", "sentiment"])
        .size()
        .reset_index(name="count")
        .sort_values(["aspect", "sentiment"])
    )
    header = f"Class Distribution — {label}" if label else "Class Distribution:"
    print(f"\n{'-' * 50}")
    print(header)
    print(f"{'-' * 50}")
    print(f"{'Aspect':<20} {'Sentiment':<12} {'Count':>6}")
    print(f"{'-' * 50}")
    for _, row in dist.iterrows():
        flag = " <" if row["count"] < TARGET_COUNT else ""
        print(f"{row['aspect']:<20} {row['sentiment']:<12} {row['count']:>6}{flag}")
    print(f"{'-' * 50}")
    print(f"Total rows: {len(df)}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run the data cleaning pipeline first to produce data/processed/train_clean.csv"
        )

    print(f"Loading {INPUT_PATH} ...", flush=True)
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Defensive normalisation on required columns
    df["aspect"] = df["aspect"].fillna("none").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("neutral").astype(str).str.strip()
    df["review_clean"] = df["review_clean"].fillna("").astype(str)

    print(f"Loaded {len(df):,} rows.")
    print_distribution(df, "BEFORE augmentation")

    # ── initialise Groq client ────────────────────────────────────────────────
    groq_client = None
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("\n[WARN] GROQ_API_KEY not found in environment. Groq generation will be skipped.")
    else:
        try:
            from groq import Groq  # type: ignore

            groq_client = Groq(api_key=api_key)
            print(f"Groq client ready  (model: {GROQ_MODEL}).", flush=True)
        except ImportError:
            print("[WARN] `groq` package not installed. Run: pip install groq")

    # ── run augmentation ──────────────────────────────────────────────────────
    augmented_df = augment(df, groq_client)

    # ── save output ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved augmented dataset -> {OUTPUT_PATH}", flush=True)

    print_distribution(augmented_df, "AFTER augmentation")

    # ── summary ───────────────────────────────────────────────────────────────
    original_count = len(df)
    total_count = len(augmented_df)
    source_counts = (
        augmented_df["source"].value_counts().to_dict()
        if "source" in augmented_df.columns
        else {}
    )

    print(f"\n{'=' * 50}")
    print("Augmentation Summary")
    print(f"{'=' * 50}")
    print(f"  Original rows     : {original_count:>6,}")
    print(f"  Groq generated    : {source_counts.get('groq', 0):>6,}")
    print(f"  Back-translated   : {source_counts.get('backtrans', 0):>6,}")
    print(f"  Total added       : {total_count - original_count:>6,}")
    print(f"  Final row count   : {total_count:>6,}")
    print(f"{'=' * 50}")

    # Verify no pair is still critically under-represented
    remaining_deficits = identify_deficits(augmented_df)
    if remaining_deficits:
        print("\n[INFO] Pairs still below target (expected if cap was hit):")
        for a, s, c in remaining_deficits:
            print(f"  ({a}, {s}): {c} (cap hit — original count was too low)")
    else:
        print("\nAll aspect/sentiment pairs now meet the target count.")


if __name__ == "__main__":
    main()
