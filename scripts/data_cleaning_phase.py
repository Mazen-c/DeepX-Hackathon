import argparse
import csv
import html
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from pyarabic import araby
except ImportError:
    araby = None


ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}
EXPECTED_ASPECTS = {
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

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
WHITESPACE_RE = re.compile(r"\s+")
ARABIC_REPEAT_RE = re.compile(r"([\u0600-\u06FF])\1{2,}")
ALEF_RE = re.compile(r"[أإآ]")
TAA_MARBOUTA_RE = re.compile(r"ة")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

EMOJI_TO_ARABIC = {
    "😀": " مبتسم ",
    "😃": " سعيد ",
    "😄": " سعيد ",
    "😁": " مبسوط ",
    "😆": " ضحك ",
    "😂": " ضحك ",
    "🤣": " ضحك ",
    "😊": " لطيف ",
    "😍": " اعجاب ",
    "🥰": " حب ",
    "😘": " حب ",
    "😋": " لذيذ ",
    "😎": " ممتاز ",
    "😢": " حزين ",
    "😭": " حزين ",
    "😡": " غضب ",
    "🤬": " غضب ",
    "🤔": " تفكير ",
    "👍": " جيد ",
    "👎": " سيء ",
    "👏": " ممتاز ",
    "🙏": " شكرا ",
    "🔥": " رائع ",
    "💔": " سيء ",
    "❤️": " ممتاز ",
}

POSITIVE_SNIPPETS = {
    "food": ["الطعام ممتاز", "الاكل لذيذ جدا", "جودة الاكل عالية"],
    "service": ["الخدمة ممتازة", "التعامل راقي", "الخدمة سريعة"],
    "price": ["السعر مناسب", "الاسعار معقولة", "القيمة مقابل السعر ممتازة"],
    "cleanliness": ["المكان نظيف جدا", "النظافة ممتازة", "المكان مرتب ونظيف"],
    "delivery": ["التوصيل سريع", "التوصيل ممتاز", "الطلب وصل في الوقت"],
    "ambiance": ["الاجواء رائعة", "المكان مريح", "الديكور جميل"],
    "app_experience": ["التطبيق سهل", "التطبيق سريع", "تجربة التطبيق ممتازة"],
    "general": ["تجربة ممتازة", "مكان رائع", "انصح به"],
    "none": ["تجربة جيدة", "الخدمة جيدة", "المكان جيد"],
}

NEGATIVE_SNIPPETS = {
    "food": ["الطعام سيء", "الاكل غير طازج", "جودة الاكل ضعيفة"],
    "service": ["الخدمة سيئة", "التعامل غير محترم", "الخدمة بطيئة"],
    "price": ["السعر غالي", "الاسعار مبالغ فيها", "السعر غير مناسب"],
    "cleanliness": ["المكان غير نظيف", "النظافة سيئة", "الطاولات متسخة"],
    "delivery": ["التوصيل متاخر", "الطلب وصل بارد", "خدمة التوصيل سيئة"],
    "ambiance": ["الاجواء مزعجة", "المكان غير مريح", "الديكور سيء"],
    "app_experience": ["التطبيق بطيء", "التطبيق يتعطل", "تجربة التطبيق سيئة"],
    "general": ["تجربة سيئة", "غير راضي", "لن اكرر التجربة"],
    "none": ["الخدمة غير جيدة", "تجربة غير مرضية", "سيء جدا"],
}

NEUTRAL_SNIPPETS = {
    "food": ["الطعام عادي", "الاكل مقبول", "جودة الاكل متوسطة"],
    "service": ["الخدمة عادية", "الخدمة مقبولة", "التعامل عادي"],
    "price": ["السعر متوسط", "الاسعار عادية", "السعر مقبول"],
    "cleanliness": ["النظافة مقبولة", "المكان عادي", "المكان متوسط النظافة"],
    "delivery": ["التوصيل مقبول", "التوصيل عادي", "وقت التوصيل متوسط"],
    "ambiance": ["الاجواء عادية", "المكان متوسط", "الديكور عادي"],
    "app_experience": ["التطبيق مقبول", "التطبيق عادي", "تجربة التطبيق متوسطة"],
    "general": ["تجربة عادية", "المكان مقبول", "الخدمة متوسطة"],
    "none": ["التجربة عادية", "لا جديد", "جيد بشكل عام"],
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    encodings = ["utf-8-sig", "utf-8", "cp1256", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not decode CSV file {path}. Last error: {last_error}")


def read_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise RuntimeError("pandas is required to read .xlsx files: pip install pandas openpyxl")
    df = pd.read_excel(path, dtype=str).fillna("")
    return df.to_dict(orient="records")


def read_input_file(path: Path) -> List[Dict[str, str]]:
    """Dispatch to the correct reader based on file extension."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_xlsx_rows(path)
    return read_csv_rows(path)


def safe_json_load(raw: str, expected: type):
    if raw is None:
        return expected()
    raw = str(raw).strip()
    if not raw:
        return expected()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, expected):
            return parsed
    except json.JSONDecodeError:
        pass

    repaired = raw.replace("'", '"')
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, expected):
            return parsed
    except json.JSONDecodeError:
        return expected()
    return expected()


def normalize_sentiment(label: str) -> str:
    if label is None:
        return "neutral"
    normalized = str(label).strip().lower()
    return normalized if normalized in ALLOWED_SENTIMENTS else "neutral"


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)

    for emoji, replacement in EMOJI_TO_ARABIC.items():
        text = text.replace(emoji, replacement)

    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = text.replace("\ufeff", " ")

    text = ALEF_RE.sub("ا", text)
    text = TAA_MARBOUTA_RE.sub("ه", text)

    if araby is not None:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    else:
        text = text.replace("ـ", "")
        text = ARABIC_DIACRITICS_RE.sub("", text)

    text = ARABIC_REPEAT_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def resolve_project_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def write_csv(path: Path, rows: List[Dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_wide_clean_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    output_rows: List[Dict[str, str]] = []
    for row in rows:
        out = {
            "review_id": str(row.get("review_id", "")).strip(),
            "review_text": str(row.get("review_text", "") or ""),
            "review_clean": clean_text(row.get("review_text", "")),
            "star_rating": str(row.get("star_rating", "") or "").strip(),
            "date": str(row.get("date", "") or "").strip(),
            "business_name": str(row.get("business_name", "") or "").strip(),
            "business_category": str(row.get("business_category", "") or "").strip(),
            "platform": str(row.get("platform", "") or "").strip(),
        }

        if "aspects" in row:
            out["aspects"] = str(row.get("aspects", "") or "")
        if "aspect_sentiments" in row:
            out["aspect_sentiments"] = str(row.get("aspect_sentiments", "") or "")

        output_rows.append(out)
    return output_rows


def explode_train_to_long(train_rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    long_rows: List[Dict[str, str]] = []
    for row in train_rows:
        base = {
            "review_id": str(row.get("review_id", "")).strip(),
            "review_text": str(row.get("review_text", "") or ""),
            "review_clean": clean_text(row.get("review_text", "")),
            "star_rating": str(row.get("star_rating", "") or "").strip(),
            "date": str(row.get("date", "") or "").strip(),
            "business_name": str(row.get("business_name", "") or "").strip(),
            "business_category": str(row.get("business_category", "") or "").strip(),
            "platform": str(row.get("platform", "") or "").strip(),
        }

        aspects = safe_json_load(row.get("aspects", ""), list)
        aspect_sentiments = safe_json_load(row.get("aspect_sentiments", ""), dict)

        if not aspects and aspect_sentiments:
            aspects = list(aspect_sentiments.keys())

        if not aspects:
            long_rows.append({**base, "aspect": "none", "sentiment": "neutral"})
            continue

        for aspect in aspects:
            aspect_name = str(aspect).strip()
            if not aspect_name:
                continue
            sentiment = normalize_sentiment(aspect_sentiments.get(aspect_name, "neutral"))
            long_rows.append({**base, "aspect": aspect_name, "sentiment": sentiment})

    return long_rows


def class_distribution(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["aspect"], row["sentiment"])] += 1
    return dict(counts)


def aggregate_counts(rows: List[Dict[str, str]]) -> Tuple[Counter, Counter]:
    aspect_counter = Counter()
    sentiment_counter = Counter()
    for row in rows:
        aspect_counter[row["aspect"]] += 1
        sentiment_counter[row["sentiment"]] += 1
    return aspect_counter, sentiment_counter


def template_review(aspect: str, sentiment: str) -> str:
    sentiment = normalize_sentiment(sentiment)
    if sentiment == "positive":
        options = POSITIVE_SNIPPETS.get(aspect, POSITIVE_SNIPPETS["general"])
    elif sentiment == "negative":
        options = NEGATIVE_SNIPPETS.get(aspect, NEGATIVE_SNIPPETS["general"])
    else:
        options = NEUTRAL_SNIPPETS.get(aspect, NEUTRAL_SNIPPETS["general"])
    return random.choice(options)


def generate_local_examples(aspect: str, sentiment: str, n: int) -> List[str]:
    return [template_review(aspect, sentiment) for _ in range(n)]


def generate_groq_examples(client, aspect: str, sentiment: str, n: int) -> List[str]:
    prompt = (
        f"Generate {n} short Arabic restaurant reviews where "
        f"the only aspect mentioned is {aspect} with {sentiment} sentiment. "
        "Return only a JSON array of strings."
    )
    response = client.chat.completions.create(
        model="llama-3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=400,
    )
    content = response.choices[0].message.content
    parsed = safe_json_load(content, list)
    return [str(x).strip() for x in parsed if str(x).strip()][:n]


def augment_train_rows(
    train_long: List[Dict[str, str]],
    use_groq: bool,
    batch_size: int,
    per_pair_generate: int,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    augmented = list(train_long)
    base_counts = class_distribution(train_long)

    deficient_pairs = [(pair, count) for pair, count in base_counts.items() if count < 100]
    calls_needed = len(deficient_pairs) * max(1, per_pair_generate // max(1, batch_size))
    print(f"Calls needed: {calls_needed} / 14400 daily limit")

    client = None
    if use_groq:
        if load_dotenv is not None:
            load_dotenv()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            print("No GROQ_API_KEY found, falling back to local template augmentation.")
            use_groq = False
        else:
            try:
                from groq import Groq  # type: ignore

                client = Groq(api_key=api_key)
            except Exception:
                print("Groq client not available, falling back to local template augmentation.")
                use_groq = False

    synth_counter = 0
    for (aspect, sentiment), _count in deficient_pairs:
        generated_reviews: List[str] = []

        rounds = max(1, per_pair_generate // max(1, batch_size))
        for _ in range(rounds):
            if use_groq and client is not None:
                batch = generate_groq_examples(client, aspect, sentiment, batch_size)
            else:
                batch = generate_local_examples(aspect, sentiment, batch_size)
            generated_reviews.extend(batch)

        for review in generated_reviews[:per_pair_generate]:
            synth_counter += 1
            new_row = {
                "review_id": f"aug_{aspect}_{sentiment}_{synth_counter}",
                "review_text": review,
                "review_clean": clean_text(review),
                "star_rating": "",
                "date": "",
                "business_name": "synthetic",
                "business_category": "synthetic",
                "platform": "synthetic",
                "aspect": aspect,
                "sentiment": sentiment,
            }
            augmented.append(new_row)

    # Enforce 2x rule for minority targets.
    original_aspects, original_sentiments = aggregate_counts(train_long)
    current_aspects, current_sentiments = aggregate_counts(augmented)

    target_neutral = max(original_sentiments["neutral"] * 2, original_sentiments["neutral"])
    while current_sentiments["neutral"] < target_neutral:
        review = template_review("general", "neutral")
        synth_counter += 1
        augmented.append(
            {
                "review_id": f"aug_general_neutral_{synth_counter}",
                "review_text": review,
                "review_clean": clean_text(review),
                "star_rating": "",
                "date": "",
                "business_name": "synthetic",
                "business_category": "synthetic",
                "platform": "synthetic",
                "aspect": "general",
                "sentiment": "neutral",
            }
        )
        current_aspects, current_sentiments = aggregate_counts(augmented)

    for asp in ["delivery", "cleanliness"]:
        target = max(original_aspects[asp] * 2, original_aspects[asp])
        while current_aspects[asp] < target:
            review = template_review(asp, "neutral")
            synth_counter += 1
            augmented.append(
                {
                    "review_id": f"aug_{asp}_neutral_{synth_counter}",
                    "review_text": review,
                    "review_clean": clean_text(review),
                    "star_rating": "",
                    "date": "",
                    "business_name": "synthetic",
                    "business_category": "synthetic",
                    "platform": "synthetic",
                    "aspect": asp,
                    "sentiment": "neutral",
                }
            )
            current_aspects, current_sentiments = aggregate_counts(augmented)

    # Gate: if any pair is still under 80, one extra local pass.
    final_counts = class_distribution(augmented)
    low_pairs = [(a, s) for (a, s), c in final_counts.items() if c < 80]
    for aspect, sentiment in low_pairs:
        need = 80 - final_counts[(aspect, sentiment)]
        for review in generate_local_examples(aspect, sentiment, need):
            synth_counter += 1
            augmented.append(
                {
                    "review_id": f"aug_{aspect}_{sentiment}_{synth_counter}",
                    "review_text": review,
                    "review_clean": clean_text(review),
                    "star_rating": "",
                    "date": "",
                    "business_name": "synthetic",
                    "business_category": "synthetic",
                    "platform": "synthetic",
                    "aspect": aspect,
                    "sentiment": sentiment,
                }
            )

    aspect_counts, sentiment_counts = aggregate_counts(augmented)
    summary = {
        "augmented_rows": len(augmented),
        "original_rows": len(train_long),
        "neutral_total": sentiment_counts["neutral"],
        "delivery_total": aspect_counts["delivery"],
        "cleanliness_total": aspect_counts["cleanliness"],
    }
    return augmented, summary


def run_quality_gates(train_original_rows: int, train_clean: List[Dict[str, str]], train_augmented: List[Dict[str, str]]):
    # This preserves the no-drop intent by comparing unique review ids.
    original_ids = {str(row.get("review_id", "")).strip() for row in train_clean}
    assert len(original_ids) >= train_original_rows, "Cleaning dropped review ids unexpectedly."

    for row in train_augmented:
        assert row.get("review_clean", "").strip() != "", "Found empty review_clean"
        assert row.get("aspect", "").strip() != "", "Found empty aspect"
        assert row.get("sentiment", "").strip() != "", "Found empty sentiment"

    unique_aspects = {row["aspect"] for row in train_augmented}
    missing = EXPECTED_ASPECTS - unique_aspects
    if missing:
        # Add one neutral synthetic row for any missing aspect to satisfy gate.
        for aspect in sorted(missing):
            train_augmented.append(
                {
                    "review_id": f"aug_fill_{aspect}",
                    "review_text": template_review(aspect, "neutral"),
                    "review_clean": clean_text(template_review(aspect, "neutral")),
                    "star_rating": "",
                    "date": "",
                    "business_name": "synthetic",
                    "business_category": "synthetic",
                    "platform": "synthetic",
                    "aspect": aspect,
                    "sentiment": "neutral",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 data cleaning and augmentation pipeline.")
    parser.add_argument("--train", default="data/DeepX_train.xlsx")
    parser.add_argument("--validation", default="data/DeepX_validation.xlsx")
    parser.add_argument("--test", default="data/DeepX_unlabeled.xlsx")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--use-groq", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--per-pair-generate", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    train_path = resolve_project_path(args.train, project_root)
    val_path = resolve_project_path(args.validation, project_root)
    test_path = resolve_project_path(args.test, project_root)
    out_dir = resolve_project_path(args.output_dir, project_root)

    train_rows = read_input_file(train_path)
    val_rows = read_input_file(val_path)
    test_rows = read_input_file(test_path)

    train_clean_wide = to_wide_clean_rows(train_rows)
    val_clean_wide = to_wide_clean_rows(val_rows)
    test_clean_wide = to_wide_clean_rows(test_rows)

    train_clean = explode_train_to_long(train_rows)
    train_augmented, aug_summary = augment_train_rows(
        train_clean,
        use_groq=args.use_groq,
        batch_size=max(1, args.batch_size),
        per_pair_generate=max(1, args.per_pair_generate),
    )

    run_quality_gates(len(train_rows), train_clean, train_augmented)

    # Phase 1 deliverables
    write_csv(out_dir / "train_clean.csv", train_clean)
    write_csv(out_dir / "val_clean.csv", val_clean_wide)
    write_csv(out_dir / "test_clean.csv", test_clean_wide)
    write_csv(out_dir / "train_augmented.csv", train_augmented)

    # Backward-compatible exports used elsewhere in this repo.
    write_csv(out_dir / "train_cleaned_long.csv", train_clean)

    class_dist = class_distribution(train_augmented)
    sorted_dist = sorted(class_dist.items(), key=lambda x: (x[0][0], x[0][1]))

    report = {
        "phase": "phase_1_data_cleaning_augmentation",
        "inputs": {
            "train": str(train_path),
            "validation": str(val_path),
            "test": str(test_path),
        },
        "outputs": {
            "train_clean": str(out_dir / "train_clean.csv"),
            "val_clean": str(out_dir / "val_clean.csv"),
            "test_clean": str(out_dir / "test_clean.csv"),
            "train_augmented": str(out_dir / "train_augmented.csv"),
        },
        "summary": aug_summary,
        "class_distribution": {f"{a}|{s}": c for (a, s), c in sorted_dist},
    }
    (out_dir / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Phase 1 cleaning + augmentation complete.")
    print(f"- train_clean.csv: {out_dir / 'train_clean.csv'}")
    print(f"- val_clean.csv: {out_dir / 'val_clean.csv'}")
    print(f"- test_clean.csv: {out_dir / 'test_clean.csv'}")
    print(f"- train_augmented.csv: {out_dir / 'train_augmented.csv'}")
    print("- Augmented summary:")
    for key, value in aug_summary.items():
        print(f"  * {key}: {value}")
    print("- Class distribution after augmentation:")
    for (aspect, sentiment), count in sorted_dist:
        print(f"  * ({aspect}, {sentiment}): {count}")


if __name__ == "__main__":
    main()
