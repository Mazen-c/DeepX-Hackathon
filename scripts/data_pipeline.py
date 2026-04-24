"""
Unified Data Pipeline — Phase 1
================================
Combines:
  • Full text cleaning  (Arabic normalisation, Arabizi/Franko detection & transliteration,
                         emoji-to-text, URL / e-mail stripping, diacritics removal …)
  • Advanced augmentation
      ① Groq LLM synthetic generation  (Arabic-dialect prompts, rate-limited)
      ② Helsinki-NLP back-translation  (Arabic → English → Arabic paraphrasing)
      ③ Template fallback              (no API key required)
  • Quality gates
  • Wide-format export

Usage
-----
  # basic run (template augmentation only, no API key needed)
  python scripts/data_pipeline.py

  # with Groq + back-translation
  python scripts/data_pipeline.py --use-groq

  # custom paths
  python scripts/data_pipeline.py --train data/DeepX_train.xlsx \
                                   --validation data/DeepX_validation.xlsx \
                                   --test data/DeepX_unlabeled.xlsx \
                                   --output-dir data/processed

  # skip the (slow) back-translation step
  python scripts/data_pipeline.py --use-groq --skip-backtranslate
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

try:
    from pyarabic import araby
except ImportError:
    araby = None

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    from langdetect import detect as _langdetect
    from langdetect.lang_detect_exception import LangDetectException as _LangDetectException
except ImportError:
    _langdetect = None  # type: ignore
    _LangDetectException = Exception  # type: ignore

try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
except ImportError:
    _GoogleTranslator = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}
EXPECTED_ASPECTS = {
    "food", "service", "price", "cleanliness",
    "delivery", "ambiance", "app_experience", "general", "none",
}

# Augmentation targets
TARGET_COUNT      = 100  # bring every (aspect, sentiment) pair up to this
MAX_PER_PAIR      = 30   # hard cap on Groq-generated reviews per deficient pair
BATCH_SIZE        = 5    # reviews requested per Groq API call
RATE_LIMIT_SLEEP  = 2.1  # seconds between Groq API calls
BACK_TRANSLATE_SAMPLE = 20  # existing reviews to back-translate per run
GROQ_MODEL        = "llama3-70b-8192"


# ── precompiled regexes ───────────────────────────────────────────────────────
URL_RE               = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE             = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
WHITESPACE_RE        = re.compile(r"\s+")
ARABIC_REPEAT_RE     = re.compile(r"([\u0600-\u06FF])\1{2,}")
ALEF_RE              = re.compile(r"[أإآ]")
TAA_MARBOUTA_RE      = re.compile(r"ة")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_LATIN_CHARS_RE      = re.compile(r"[A-Za-z]")
_ARABIZI_NUM_LETTER_RE = re.compile(r"(?<=[A-Za-z])[375]|[375](?=[A-Za-z])")
_REMAINING_LATIN_RE  = re.compile(r"[A-Za-z]")
# Mixed-script: text contains BOTH Arabic and Latin characters
_HAS_ARABIC_RE       = re.compile(r"[\u0600-\u06FF]")
_HAS_LATIN_RE        = re.compile(r"[A-Za-z]")
# Arabizi number digits that signal the word is Arabizi
_ARABIZI_NUMS_RE     = re.compile(r"[23578]")
# Lone Arabic letter (not و which means 'and') surrounded by whitespace or boundaries
_LONE_ARABIC_LETTER_RE = re.compile(r"(?<![\u0600-\u06FF])[\u0600-\u06FF](?![\u0600-\u06FF])")
# English common-word prefixes/suffixes heuristic: if a word is almost all vowels it's English
_ENGLISH_VOWELS_RE   = re.compile(r"^[aeiouAEIOU]+$")


# ── emoji → Arabic ───────────────────────────────────────────────────────────
EMOJI_TO_ARABIC = {
    "😀": " مبتسم ", "😃": " سعيد ", "😄": " سعيد ", "😁": " مبسوط ",
    "😆": " ضحك ",   "😂": " ضحك ",   "🤣": " ضحك ",   "😊": " لطيف ",
    "😍": " اعجاب ", "🥰": " حب ",    "😘": " حب ",    "😋": " لذيذ ",
    "😎": " ممتاز ", "😢": " حزين ",  "😭": " حزين ",  "😡": " غضب ",
    "🤬": " غضب ",   "🤔": " تفكير ", "👍": " جيد ",   "👎": " سيء ",
    "👏": " ممتاز ", "🙏": " شكرا ",  "🔥": " رائع ",  "💔": " سيء ",
    "❤️": " ممتاز ",
}

# ── template snippets for offline augmentation ────────────────────────────────
POSITIVE_SNIPPETS = {
    "food":           ["الطعام ممتاز", "الاكل لذيذ جدا", "جودة الاكل عالية"],
    "service":        ["الخدمة ممتازة", "التعامل راقي", "الخدمة سريعة"],
    "price":          ["السعر مناسب", "الاسعار معقولة", "القيمة مقابل السعر ممتازة"],
    "cleanliness":    ["المكان نظيف جدا", "النظافة ممتازة", "المكان مرتب ونظيف"],
    "delivery":       ["التوصيل سريع", "التوصيل ممتاز", "الطلب وصل في الوقت"],
    "ambiance":       ["الاجواء رائعة", "المكان مريح", "الديكور جميل"],
    "app_experience": ["التطبيق سهل", "التطبيق سريع", "تجربة التطبيق ممتازة"],
    "general":        ["تجربة ممتازة", "مكان رائع", "انصح به"],
    "none":           ["تجربة جيدة", "الخدمة جيدة", "المكان جيد"],
}
NEGATIVE_SNIPPETS = {
    "food":           ["الطعام سيء", "الاكل غير طازج", "جودة الاكل ضعيفة"],
    "service":        ["الخدمة سيئة", "التعامل غير محترم", "الخدمة بطيئة"],
    "price":          ["السعر غالي", "الاسعار مبالغ فيها", "السعر غير مناسب"],
    "cleanliness":    ["المكان غير نظيف", "النظافة سيئة", "الطاولات متسخة"],
    "delivery":       ["التوصيل متاخر", "الطلب وصل بارد", "خدمة التوصيل سيئة"],
    "ambiance":       ["الاجواء مزعجة", "المكان غير مريح", "الديكور سيء"],
    "app_experience": ["التطبيق بطيء", "التطبيق يتعطل", "تجربة التطبيق سيئة"],
    "general":        ["تجربة سيئة", "غير راضي", "لن اكرر التجربة"],
    "none":           ["الخدمة غير جيدة", "تجربة غير مرضية", "سيء جدا"],
}
NEUTRAL_SNIPPETS = {
    "food":           ["الطعام عادي", "الاكل مقبول", "جودة الاكل متوسطة"],
    "service":        ["الخدمة عادية", "الخدمة مقبولة", "التعامل عادي"],
    "price":          ["السعر متوسط", "الاسعار عادية", "السعر مقبول"],
    "cleanliness":    ["النظافة مقبولة", "المكان عادي", "المكان متوسط النظافة"],
    "delivery":       ["التوصيل مقبول", "التوصيل عادي", "وقت التوصيل متوسط"],
    "ambiance":       ["الاجواء عادية", "المكان متوسط", "الديكور عادي"],
    "app_experience": ["التطبيق مقبول", "التطبيق عادي", "تجربة التطبيق متوسطة"],
    "general":        ["تجربة عادية", "المكان مقبول", "الخدمة متوسطة"],
    "none":           ["التجربة عادية", "لا جديد", "جيد بشكل عام"],
}

# ── Groq prompt helpers ───────────────────────────────────────────────────────
_ASPECT_AR: Dict[str, str] = {
    "food":           "الطعام",
    "service":        "الخدمة",
    "price":          "السعر",
    "cleanliness":    "النظافة",
    "delivery":       "التوصيل",
    "ambiance":       "الأجواء",
    "app_experience": "تجربة التطبيق",
    "general":        "التجربة العامة",
    "none":           "عام",
}
_SENTIMENT_AR: Dict[str, str] = {
    "positive": "إيجابية",
    "negative": "سلبية",
    "neutral":  "محايدة",
}

# ── Arabizi transliteration maps ──────────────────────────────────────────────
ARBIZI_NUMBERS = {"2": "ء", "3": "ع", "3'": "غ", "5": "خ",
                  "6": "ط", "7": "ح", "8": "ق", "9": "ص"}
ARBIZI_PHONETICS = {"sh": "ش", "kh": "خ", "gh": "غ", "th": "ث",
                    "ou": "و", "ee": "ي", "oo": "و"}
ARBIZI_COMMON_WORDS = {
    "mesh": "مش", "bas": "بس", "7elw": "حلو", "kwayes": "كويس",
    "bgad": "بجد", "el": "ال", "3ayesh": "عايش", "tabee3": "طبيع",
    "sha3r": "شاعر", "ba3d": "بعد", "9oor": "صور", "makan": "مكان",
}
_ARBIZI_COMMON_WORDS_PATTERNS = sorted(ARBIZI_COMMON_WORDS.items(), key=lambda x: len(x[0]), reverse=True)
_ARBIZI_PHONETICS_PATTERNS    = sorted(ARBIZI_PHONETICS.items(),    key=lambda x: len(x[0]), reverse=True)
_ARBIZI_NUMBERS_PATTERNS      = sorted(ARBIZI_NUMBERS.items(),      key=lambda x: len(x[0]), reverse=True)

# Comprehensive char-by-char Arabizi map used for the aggressive post-clean pass
ARABIZI_COMPREHENSIVE: Dict[str, str] = {
    '0': '٠', '1': '١', '2': 'ء', '3': 'ع', '4': 'أ', '5': 'خ', '6': 'ط', '7': 'ح', '8': 'غ', '9': 'ق',
    'a': 'ا', 'b': 'ب', 'c': 'ك', 'd': 'د', 'e': 'ا', 'f': 'ف', 'g': 'ج',
    'h': 'ه', 'i': 'ي', 'j': 'ج', 'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن', 'o': 'و',
    'p': 'پ', 'q': 'ق', 'r': 'ر', 's': 'س', 't': 'ت', 'u': 'و', 'v': 'ف',
    'w': 'و', 'x': 'ك', 'y': 'ي', 'z': 'ز', "'": 'ء', '"': '', '-': '', '_': '',
}
_ARABIZI_DIGRAPHS: List[Tuple[str, str]] = [
    ('dh', 'ظ'), ('kh', 'خ'), ('gh', 'غ'), ('sh', 'ش'), ('th', 'ث'), ('zh', 'ز'),
]
# Languages that should never be treated as Arabizi
_NON_ARABIZI_LANGS = {'en', 'fr', 'es', 'it', 'de', 'pt', 'ru', 'zh', 'ja', 'ur', 'fa', 'ar'}


# ══════════════════════════════════════════════════════════════════════════════
#  FILE I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    encodings = ["utf-8-sig", "utf-8", "cp1256", "latin-1"]
    last_error = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode CSV file {path}. Last error: {last_error}")


def read_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    if pd is None:
        raise RuntimeError("pandas is required to read .xlsx files: pip install pandas openpyxl")
    df = pd.read_excel(path, dtype=str).fillna("")
    return df.to_dict(orient="records")


def read_input_file(path: Path) -> List[Dict[str, str]]:
    """Dispatch to the correct reader based on file extension."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_xlsx_rows(path)
    return read_csv_rows(path)


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_project_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

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


# ── language detection & translation ────────────────────────────────────────

_TRANSLATION_CACHE: dict = {}  # simple in-process cache to avoid duplicate API calls

def detect_language(text: str) -> Optional[str]:
    """
    Return a BCP-47 language code (e.g. 'en', 'fr', 'ar') or None on failure.
    Requires the langdetect package.
    """
    if _langdetect is None or not text or not text.strip():
        return None
    try:
        return _langdetect(text.strip())
    except Exception:
        return None


def translate_to_arabic(text: str) -> str:
    """
    Translate *text* to Arabic using Google Translate (deep-translator).
    Returns the original text unchanged if the library is not installed,
    the text is already Arabic, or translation fails.
    """
    if _GoogleTranslator is None:
        return text

    key = text.strip()
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]

    try:
        translated = _GoogleTranslator(source="auto", target="ar").translate(key)
        result = translated if translated and translated.strip() else text
    except Exception as exc:
        print(f"    [WARN] Translation failed: {exc}", flush=True)
        result = text

    _TRANSLATION_CACHE[key] = result
    return result


def is_arabizi_text(text: str) -> bool:
    """
    Detect Arabizi (Franko-Arabic).
    Returns True if >20% of chars are Latin OR Arabizi numbers appear
    adjacent to Latin letters (e.g. "3ayesh", "7elw").
    """
    if not text:
        return False
    latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    if (latin_count / len(text)) > 0.2:
        return True
    return bool(_ARABIZI_NUM_LETTER_RE.search(text))


def _is_arabizi_word(word: str) -> bool:
    """
    Return True only if a word looks like Arabizi (not plain English).

    Uses two safe, explicit signals only:
      1. The word contains an Arabizi number digit (2, 3, 5, 7, 8)
         e.g. '7elw', '3ayesh', 'ba3d'  →  True
      2. The word (lowercased) is an exact match in ARBIZI_COMMON_WORDS

    Deliberately conservative: plain English words ('The', 'Service',
    'Chef', 'Nawaf', 'great') return False so they are NEVER corrupted.
    """
    if not word:
        return False
    # Signal 1: Arabizi numerals embedded in the word
    if _ARABIZI_NUMS_RE.search(word):
        return True
    # Signal 2: exact match in common-word dictionary
    if word.lower() in ARBIZI_COMMON_WORDS:
        return True
    return False


def transliterate_arabizi(text: str) -> str:
    """
    Transliterate Arabizi tokens to Arabic script.
    Phase 1: common words matched whole-word (safe, always applied)
    Phase 2: phonetic digraphs (sh, kh, gh, th, ou, ee, oo) — applied
             ONLY within words identified as Arabizi
    Phase 3: Arabizi numbers adjacent to Latin letters only

    KEY SAFETY RULE: single-letter and phonetic swaps are scoped
    per-word so pure-English words like "The", "Service", "Chef"
    are never corrupted.
    """
    if not text:
        return text

    # Phase 1: whole-word common Arabizi vocabulary (longest match first, safe)
    for word, repl in _ARBIZI_COMMON_WORDS_PATTERNS:
        text = re.sub(r"\b" + re.escape(word) + r"\b", repl, text, flags=re.IGNORECASE)

    # Phase 2 & 3: process token-by-token to avoid corrupting English words
    tokens = re.split(r"(\s+)", text)  # keep whitespace tokens for round-trip
    result_tokens = []
    for token in tokens:
        if not token or not token.strip():  # whitespace – keep as-is
            result_tokens.append(token)
            continue
        # Check if this token (word) looks like Arabizi
        # Strip punctuation for the check
        bare = re.sub(r"[^\w]", "", token)
        if _is_arabizi_word(bare):
            # Apply phonetic digraph replacements
            processed = token
            for phonetic, repl in _ARBIZI_PHONETICS_PATTERNS:
                processed = re.sub(re.escape(phonetic), repl, processed, flags=re.IGNORECASE)
            # Apply Arabizi number substitutions
            for num, repl in _ARBIZI_NUMBERS_PATTERNS:
                pattern_a = r"(?<=[A-Za-z])" + re.escape(num)
                pattern_b = re.escape(num) + r"(?=[A-Za-z])"
                processed = re.sub(f"(?:{pattern_a}|{pattern_b})", repl, processed)
            result_tokens.append(processed)
        else:
            # Plain English (or already Arabic) — leave completely untouched
            result_tokens.append(token)

    return "".join(result_tokens)


def clean_text(text: str, groq_client=None) -> str:
    """
    Full text cleaning pipeline:
      1. HTML unescape + Unicode normalisation
      2. Emoji → Arabic word
      3. Mixed-script detection: if the review contains BOTH Arabic and Latin,
         only run Arabizi transliteration on Latin tokens that look like Arabizi.
         Plain English words ("The", "Chef", "Service") are left intact.
      4. Strip URLs, e-mails, invisible chars
      5. Arabic-specific normalisation (Alef, Taa Marbouta, diacritics, elongation)
      6. Repeated-character collapsing
      7. Whitespace normalisation
      8. Lone single Arabic-letter cleanup (except و)
    """
    if text is None:
        return ""

    raw_text = html.unescape(str(text))
    result   = unicodedata.normalize("NFKC", raw_text)

    # Emoji → Arabic
    for emoji, replacement in EMOJI_TO_ARABIC.items():
        result = result.replace(emoji, replacement)

    # ── Detect script composition ─────────────────────────────────────────────
    has_arabic = bool(_HAS_ARABIC_RE.search(result))
    has_latin  = bool(_HAS_LATIN_RE.search(result))
    is_mixed   = has_arabic and has_latin
    is_arabizi = is_arabizi_text(raw_text)

    # ── Translate non-Arabic, non-Arabizi text to Arabic ─────────────────────
    # Only fires when the review has NO Arabic characters and is not Arabizi
    # (i.e. it is plain English, French, Hindi, etc.)
    if not has_arabic and not is_arabizi and _GoogleTranslator is not None:
        lang = detect_language(result)
        # Translate everything that is not already Arabic
        if lang is not None and lang != "ar":
            result = translate_to_arabic(result)
            # After translation refresh script flags
            has_arabic = bool(_HAS_ARABIC_RE.search(result))
            has_latin  = bool(_HAS_LATIN_RE.search(result))
            is_mixed   = has_arabic and has_latin

    # ── Arabizi / transliteration block ──────────────────────────────────────
    if is_arabizi and len(raw_text.split()) > 10 and groq_client is not None:
        # Long Arabizi text: try Groq LLM for best quality
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        "Convert this Arabizi/Franko review to clean Arabic script. "
                        "Keep the dialect. Do not translate to English.\n\n" + raw_text
                    ),
                }],
            )
            result = response.choices[0].message.content
            # After Groq, strip any residual Latin
            if _HAS_ARABIC_RE.search(result):
                result = _REMAINING_LATIN_RE.sub(" ", result)
        except Exception:
            result = transliterate_arabizi(result)
    else:
        # Rule-based transliteration (word-aware, preserves English words)
        result = transliterate_arabizi(result)

    # ── Latin stripping logic ─────────────────────────────────────────────────
    # Re-check script composition AFTER transliteration
    has_arabic_now = bool(_HAS_ARABIC_RE.search(result))
    has_latin_now  = bool(_HAS_LATIN_RE.search(result))

    if has_arabic_now and has_latin_now:
        if is_mixed:
            # The original text already had Arabic — any Latin that survived
            # transliterate_arabizi was deliberately left intact (it's real English).
            # DO NOT strip it: "The place is great... المكان رائع" should stay mixed.
            pass
        else:
            # Original was pure Latin/Arabizi; now it has Arabic — strip residual Latin noise
            result = _REMAINING_LATIN_RE.sub(" ", result)
    elif has_arabic_now and not has_latin_now:
        pass  # already clean Arabic, nothing to do

    result = URL_RE.sub(" ", result)
    result = EMAIL_RE.sub(" ", result)
    result = result.replace("\u200f", " ").replace("\u200e", " ").replace("\ufeff", " ")

    # Arabic letter normalisation
    result = ALEF_RE.sub("ا", result)
    result = TAA_MARBOUTA_RE.sub("ه", result)

    if araby is not None:
        result = araby.strip_tashkeel(result)
        result = araby.strip_tatweel(result)
    else:
        result = result.replace("ـ", "")
        result = ARABIC_DIACRITICS_RE.sub("", result)

    result = ARABIC_REPEAT_RE.sub(r"\1", result)
    result = WHITESPACE_RE.sub(" ", result).strip()

    # ── Final lone-Arabic-letter cleanup ─────────────────────────────────────
    # Remove standalone single Arabic letters that are NOT و (connector 'and')
    # Pattern: a lone Arabic char (not و) that is not adjacent to another Arabic char
    result = re.sub(
        r"(?<![\u0600-\u06FF])(?!\u0648)[\u0600-\u06FF](?![\u0600-\u06FF])",
        " ",
        result,
    )
    result = WHITESPACE_RE.sub(" ", result).strip()

    # Safety net: if cleaning produced an empty string, return raw
    if len(result) < 3 and raw_text.strip():
        result = WHITESPACE_RE.sub(" ", raw_text).strip()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ NEURAL ARABIZI TRANSLITERATION  (post-clean pass)
# ══════════════════════════════════════════════════════════════════════════════

_GROQ_ARABIZI_SYSTEM = (
    "You are an expert Arabic linguist specializing in Arabizi/Franko. "
    "Transliterate the following Latin-script Arabic (Franko) text into clean, "
    "Egyptian/Gulf Dialectal Arabic script.\n"
    "- DO NOT translate into English.\n"
    "- Keep the sentiment, slang, and dialect exactly the same.\n"
    "- Return only the transliterated Arabic text, one per line.\n"
    "- If a word is standard English (e.g., 'service'), leave it as English."
)

_ARABIZI_BATCH_SIZE = 10


def _arabic_post_clean(text: str) -> str:
    """
    Apply only the Arabic-script normalisation steps from the main pipeline
    (Alef, Taa Marbouta, diacritics, elongation, repeated chars, whitespace,
    lone-letter cleanup) so that Groq transliteration output stays consistent
    with rule-based output.
    """
    result = ALEF_RE.sub("\u0627", text)
    result = TAA_MARBOUTA_RE.sub("\u0647", result)

    if araby is not None:
        result = araby.strip_tashkeel(result)
        result = araby.strip_tatweel(result)
    else:
        result = result.replace("\u0640", "")
        result = ARABIC_DIACRITICS_RE.sub("", result)

    result = ARABIC_REPEAT_RE.sub(r"\1", result)
    result = WHITESPACE_RE.sub(" ", result).strip()

    # Lone single Arabic letter cleanup (keep \u0648 = و)
    result = re.sub(
        r"(?<![\u0600-\u06FF])(?!\u0648)[\u0600-\u06FF](?![\u0600-\u06FF])",
        " ",
        result,
    )
    result = WHITESPACE_RE.sub(" ", result).strip()
    return result


def _groq_transliterate_batch(groq_client, texts: List[str]) -> List[str]:
    """
    Send up to _ARABIZI_BATCH_SIZE Arabizi texts to Groq for neural
    transliteration.  Returns a list of Arabic strings (same length as
    `texts`), or an empty list on any failure so the caller preserves originals.
    """
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _GROQ_ARABIZI_SYSTEM},
                {"role": "user",   "content": numbered},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"    [WARN] Groq transliteration batch failed: {exc}", flush=True)
        return []

    # Strip leading "1. ", "2. " numbering the model may echo back
    lines = [
        re.sub(r"^\d+\.\s*", "", ln).strip()
        for ln in raw.splitlines()
        if ln.strip()
    ]

    if len(lines) != len(texts):
        print(
            f"    [WARN] Groq returned {len(lines)} lines for {len(texts)} inputs "
            "— falling back to originals for this batch.",
            flush=True,
        )
        return []

    return lines


def apply_groq_arabizi_transliteration(
    rows: List[Dict[str, str]],
    groq_client,
) -> List[Dict[str, str]]:
    """
    Post-clean pass: for every row flagged as Arabizi (is_arabizi == 'true')
    replace review_clean with a Groq-neural-transliterated version, then
    re-apply Arabic normalisation (_arabic_post_clean) for consistency.

    Rows are processed in batches of _ARABIZI_BATCH_SIZE with a
    RATE_LIMIT_SLEEP pause between calls.  On any API failure the existing
    review_clean is preserved unchanged.
    """
    if groq_client is None:
        return rows

    arabizi_indices = [
        i for i, r in enumerate(rows)
        if str(r.get("is_arabizi", "")).lower() == "true"
    ]

    if not arabizi_indices:
        print("  [Groq Transliteration] No Arabizi rows found — skipping.", flush=True)
        return rows

    print(
        f"\n[Groq Transliteration] Re-transliterating {len(arabizi_indices)} "
        "Arabizi rows …",
        flush=True,
    )

    total_updated = 0
    batches = [
        arabizi_indices[i: i + _ARABIZI_BATCH_SIZE]
        for i in range(0, len(arabizi_indices), _ARABIZI_BATCH_SIZE)
    ]

    for batch_num, idx_batch in enumerate(batches, 1):
        texts = [rows[i]["review_text"] for i in idx_batch]
        print(
            f"  Batch {batch_num}/{len(batches)}  ({len(texts)} reviews) …",
            flush=True,
        )

        if batch_num > 1:
            time.sleep(RATE_LIMIT_SLEEP)

        transliterated = _groq_transliterate_batch(groq_client, texts)
        if not transliterated:          # API failure — keep originals
            continue

        for row_idx, ar_text in zip(idx_batch, transliterated):
            if ar_text and ar_text.strip():
                rows[row_idx]["review_clean"] = _arabic_post_clean(ar_text)
                total_updated += 1

    print(
        f"  [Groq Transliteration] Updated {total_updated}/{len(arabizi_indices)} rows.",
        flush=True,
    )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  RESIDUAL ARABIZI FIX  (merged post-processing pass)
# ══════════════════════════════════════════════════════════════════════════════

def _is_residual_arabizi(text: str) -> bool:
    """
    Detect rows that are still Arabizi after the main cleaning pass.

    Smart two-stage check (merged from fix_arabizi_rows + fix_arabizi_smart):
      Stage 1 — quick ratio gate: >50% of alphabetic chars are Latin.
      Stage 2 — langdetect guard: skip text that is confidently identified as a
                real non-Arabic language (English, French, …) to avoid converting
                genuine code-switched or English-only reviews.
    True Arabizi has a high Latin ratio AND contains Arabic-digit signals
    (2,3,5,6,7,8,9) when langdetect is uncertain.
    """
    if not text or not isinstance(text, str):
        return False
    total = len(text)
    if total == 0:
        return False
    latin_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    if (latin_count / total) <= 0.50:
        return False

    # langdetect guard — only runs if the library is available
    if _langdetect is not None:
        try:
            lang = _langdetect(text.strip())
        except _LangDetectException:
            lang = None
        if lang in _NON_ARABIZI_LANGS:
            return False
        # If langdetect is uncertain, require an Arabic-digit signal
        if lang is None:
            return bool(re.search(r'[23567890]', text))

    return True


def _convert_arabizi_aggressive(text: str) -> str:
    """Char-by-char Arabizi → Arabic using ARABIZI_COMPREHENSIVE."""
    t = str(text).lower()
    for latin, arabic in _ARABIZI_DIGRAPHS:
        t = t.replace(latin, arabic)
    for latin, arabic in ARABIZI_COMPREHENSIVE.items():
        if len(latin) == 1:
            t = t.replace(latin, arabic)
    return re.sub(r'\s+', ' ', t).strip()


def fix_residual_arabizi_rows(
    wide_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Scan wide-format rows for residual Arabizi in review_clean and convert them.
    Only review_clean is modified; all other columns are untouched.
    Prints before/after for the first 5 conversions and a final summary.
    """
    arabizi_indices = [
        i for i, r in enumerate(wide_rows)
        if _is_residual_arabizi(str(r.get('review_clean', '')))
    ]
    print(
        f"\n[Arabizi Fix] Found {len(arabizi_indices)} residual Arabizi rows in review_clean.",
        flush=True,
    )

    if not arabizi_indices:
        return wide_rows

    print("  Before / After samples (first 5):", flush=True)
    for idx in arabizi_indices[:5]:
        before = wide_rows[idx]['review_clean']
        after  = _convert_arabizi_aggressive(before)
        print(f"    [{idx}] BEFORE: {before[:120]}", flush=True)
        print(f"           AFTER:  {after[:120]}", flush=True)

    for idx in arabizi_indices:
        wide_rows[idx]['review_clean'] = _convert_arabizi_aggressive(
            wide_rows[idx]['review_clean']
        )

    still = [
        i for i in arabizi_indices
        if _is_residual_arabizi(str(wide_rows[i].get('review_clean', '')))
    ]
    if still:
        print(f"  [WARN] {len(still)} row(s) still flagged after conversion.", flush=True)
    else:
        print("  Verification: all residual Arabizi rows converted ✓", flush=True)

    print(f"  {len(arabizi_indices)} Arabizi rows fixed.", flush=True)
    return wide_rows


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════════════

def to_wide_clean_rows(rows: Iterable[Dict[str, str]], groq_client=None) -> List[Dict[str, str]]:
    """Clean rows and return in wide format (one row per review)."""
    output: List[Dict[str, str]] = []
    for row in rows:
        review_text = row.get("review_text", "") or ""
        out = {
            "review_id":          str(row.get("review_id", "")).strip(),
            "review_text":        str(review_text),
            "review_clean":       clean_text(review_text, groq_client),
            "is_arabizi":         "true" if is_arabizi_text(review_text) else "false",
            "star_rating":        str(row.get("star_rating", "") or "").strip(),
            "date":               str(row.get("date", "") or "").strip(),
            "business_name":      str(row.get("business_name", "") or "").strip(),
            "business_category":  str(row.get("business_category", "") or "").strip(),
            "platform":           str(row.get("platform", "") or "").strip(),
        }
        if "aspects" in row:
            out["aspects"] = str(row.get("aspects", "") or "")
        if "aspect_sentiments" in row:
            out["aspect_sentiments"] = str(row.get("aspect_sentiments", "") or "")
        output.append(out)
    return output


def explode_train_to_long(rows: Iterable[Dict[str, str]], groq_client=None) -> List[Dict[str, str]]:
    """
    Convert training rows to long format: one row per (review, aspect, sentiment) triple.
    """
    long_rows: List[Dict[str, str]] = []
    for row in rows:
        review_text = row.get("review_text", "") or ""
        base = {
            "review_id":         str(row.get("review_id", "")).strip(),
            "review_text":       str(review_text),
            "review_clean":      clean_text(review_text, groq_client),
            "is_arabizi":        "true" if is_arabizi_text(review_text) else "false",
            "star_rating":       str(row.get("star_rating", "") or "").strip(),
            "date":              str(row.get("date", "") or "").strip(),
            "business_name":     str(row.get("business_name", "") or "").strip(),
            "business_category": str(row.get("business_category", "") or "").strip(),
            "platform":          str(row.get("platform", "") or "").strip(),
        }
        aspects          = safe_json_load(row.get("aspects", ""), list)
        aspect_sentiments = safe_json_load(row.get("aspect_sentiments", ""), dict)

        if not aspects and aspect_sentiments:
            aspects = list(aspect_sentiments.keys())

        if not aspects:
            long_rows.append({**base, "aspect": "none", "sentiment": "neutral", "source": "original"})
            continue

        for aspect in aspects:
            aspect_name = str(aspect).strip()
            if not aspect_name:
                continue
            sentiment = normalize_sentiment(aspect_sentiments.get(aspect_name, "neutral"))
            long_rows.append({**base, "aspect": aspect_name, "sentiment": sentiment, "source": "original"})

    return long_rows


def wide_from_long(
    train_rows: List[Dict[str, str]],
    long_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Re-collapse long-format rows back to one wide row per review_id."""
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in long_rows:
        rid = str(row["review_id"])
        if not rid.startswith("aug_"):
            grouped.setdefault(rid, []).append(row)

    wide_rows: List[Dict[str, str]] = []

    # Real reviews
    _WIDE_BASE_COLS = {"review_id", "review_text", "review_clean", "star_rating", "business_category", "platform"}

    seen_real: set = set()
    for row in long_rows:
        rid = str(row["review_id"])
        if rid.startswith("aug_") or rid in seen_real:
            continue
        seen_real.add(rid)
        group = grouped[rid]
        aspects_list    = [r["aspect"] for r in group]
        sentiments_dict = {r["aspect"]: r["sentiment"] for r in group}
        base = {k: v for k, v in group[0].items() if k in _WIDE_BASE_COLS}
        base["aspects"]           = json.dumps(aspects_list, ensure_ascii=False)
        base["aspect_sentiments"] = json.dumps(sentiments_dict, ensure_ascii=False)
        base["n_aspects"]         = str(len(aspects_list))
        wide_rows.append(base)

    # Synthetic rows
    for row in long_rows:
        if not str(row["review_id"]).startswith("aug_"):
            continue
        base = {k: v for k, v in row.items() if k in _WIDE_BASE_COLS}
        base["aspects"]           = json.dumps([row["aspect"]], ensure_ascii=False)
        base["aspect_sentiments"] = json.dumps({row["aspect"]: row["sentiment"]}, ensure_ascii=False)
        base["n_aspects"]         = "1"
        wide_rows.append(base)

    return wide_rows


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION — STEP 1: GROQ SYNTHETIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_groq_prompt(aspect: str, sentiment: str, n: int) -> str:
    aspect_ar   = _ASPECT_AR.get(aspect, aspect)
    sentiment_ar = _SENTIMENT_AR.get(sentiment, sentiment)
    return (
        f"اكتب {n} تعليقات قصيرة لعملاء مطاعم بالعامية المصرية أو السعودية. "
        f"كل تعليق يجب أن يتحدث فقط عن {aspect_ar} بشعور {sentiment_ar}. "
        "أرجع النتيجة كـ JSON array من النصوص فقط، بدون أي شرح إضافي. "
        'مثال: ["التعليق الأول", "التعليق الثاني"]'
    )


def _parse_groq_response(content: str) -> List[str]:
    """Extract a list of strings from a Groq response with robust fallback parsing."""
    content = re.sub(r"```(?:json)?", "", content).strip("`").strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*?\]", content, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Last resort: numbered/bulleted lines
    lines = []
    for line in content.splitlines():
        line = re.sub(r'^\d+[.)]\s*["\']?', "", line).rstrip('",\'').strip()
        if len(line) > 10:
            lines.append(line)
    return lines


def _groq_batch(groq_client, aspect: str, sentiment: str, n: int) -> List[str]:
    """Single Groq API call. Returns up to n Arabic reviews."""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _build_groq_prompt(aspect, sentiment, n)}],
            temperature=0.9,
            max_tokens=600,
        )
        return _parse_groq_response(response.choices[0].message.content)[:n]
    except Exception as exc:
        print(f"    [WARN] Groq API error: {exc}", flush=True)
        return []


def template_review(aspect: str, sentiment: str) -> str:
    sentiment = normalize_sentiment(sentiment)
    if sentiment == "positive":
        options = POSITIVE_SNIPPETS.get(aspect, POSITIVE_SNIPPETS["general"])
    elif sentiment == "negative":
        options = NEGATIVE_SNIPPETS.get(aspect, NEGATIVE_SNIPPETS["general"])
    else:
        options = NEUTRAL_SNIPPETS.get(aspect, NEUTRAL_SNIPPETS["general"])
    return random.choice(options)


def _generate_reviews(groq_client, aspect: str, sentiment: str, needed: int) -> List[str]:
    """
    Generate up to `needed` reviews for a given (aspect, sentiment) pair.
    Uses Groq if available, falls back to local templates.
    """
    if groq_client is None:
        return [template_review(aspect, sentiment) for _ in range(needed)]

    n_calls   = -(-min(needed, MAX_PER_PAIR) // BATCH_SIZE)  # ceiling division
    generated: List[str] = []
    first_call = True

    for call_idx in range(n_calls):
        if not first_call:
            time.sleep(RATE_LIMIT_SLEEP)
        first_call = False

        remaining = min(needed, MAX_PER_PAIR) - len(generated)
        batch_n   = min(BATCH_SIZE, remaining)
        if batch_n <= 0:
            break

        print(
            f"    [{aspect}/{sentiment}] call {call_idx + 1}/{n_calls} "
            f"— requesting {batch_n} reviews …",
            flush=True,
        )
        batch = _groq_batch(groq_client, aspect, sentiment, batch_n)
        generated.extend(batch)

    # Fill any remaining with templates if Groq didn't produce enough
    still_needed = needed - len(generated)
    if still_needed > 0:
        generated.extend([template_review(aspect, sentiment) for _ in range(still_needed)])

    return generated[:needed]


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION — STEP 2: BACK-TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

def back_translate(texts: List[str]) -> List[str]:
    """
    Paraphrase Arabic reviews via Arabic → English → Arabic translation
    using the lightweight Helsinki-NLP/opus-mt models (≈48 MB total).
    Returns an empty list on any failure so the caller can skip gracefully.
    """
    print("\n[Step 2] Back-translation (Helsinki-NLP opus-mt-ar-en + opus-mt-en-ar) …", flush=True)
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        print("  Loading Helsinki-NLP/opus-mt-ar-en …", flush=True)
        ar_to_en = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en", device=-1)
        print("  Loading Helsinki-NLP/opus-mt-en-ar …", flush=True)
        en_to_ar = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar", device=-1)

        print(f"  Translating {len(texts)} reviews ar→en …", flush=True)
        en_texts = [r["translation_text"] for r in ar_to_en(texts, batch_size=8)]

        print(f"  Translating {len(texts)} reviews en→ar …", flush=True)
        back_ar  = [r["translation_text"] for r in en_to_ar(en_texts, batch_size=8)]

        print(f"  Back-translation produced {len(back_ar)} reviews.", flush=True)
        return back_ar

    except Exception as exc:
        print(f"  [WARN] Back-translation failed and will be skipped: {exc}", flush=True)
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_row(text: str, aspect: str, sentiment: str, uid: int, source: str) -> Dict:
    return {
        "review_id":         f"aug_{source}_{aspect}_{sentiment}_{uid}",
        "review_text":       text,
        "review_clean":      clean_text(text),
        "is_arabizi":        "false",
        "star_rating":       "",
        "date":              "",
        "business_name":     "synthetic",
        "business_category": "synthetic",
        "platform":          "synthetic",
        "aspect":            aspect,
        "sentiment":         sentiment,
        "source":            source,
    }


def _class_distribution(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["aspect"], row["sentiment"])] += 1
    return dict(counts)


def _aggregate_counts(rows: List[Dict[str, str]]) -> Tuple[Counter, Counter]:
    aspect_ctr    = Counter()
    sentiment_ctr = Counter()
    for row in rows:
        aspect_ctr[row["aspect"]] += 1
        sentiment_ctr[row["sentiment"]] += 1
    return aspect_ctr, sentiment_ctr


def augment(
    train_long: List[Dict[str, str]],
    groq_client,
    skip_backtranslate: bool = False,
) -> Tuple[List[Dict[str, str]], Dict]:
    """
    Full augmentation pipeline:
      ① Groq synthetic generation (or template fallback)
      ② Helsinki-NLP back-translation
      ③ Extra template passes to cover rare minority pairs
    Returns (augmented_long_rows, summary_dict).
    """

    # ── identify deficient pairs ──────────────────────────────────────────────
    base_dist = _class_distribution(train_long)
    deficits  = [(pair, cnt) for pair, cnt in base_dist.items() if cnt < TARGET_COUNT]

    if not deficits:
        print("All (aspect, sentiment) pairs already meet TARGET_COUNT. Nothing to augment.")
        total_calls = 0
    else:
        total_calls = sum(
            -(-min(TARGET_COUNT - c, MAX_PER_PAIR) // BATCH_SIZE)
            for _, c in deficits
        )

    print(
        f"\n[Step 1] Synthetic generation — {len(deficits)} deficient pairs "
        f"(< {TARGET_COUNT} samples each)."
    )
    print(f"  Estimated Groq API calls: {total_calls}  (limit ≈ 14 400/day)")
    for (aspect, sentiment), count in sorted(deficits):
        to_gen = min(TARGET_COUNT - count, MAX_PER_PAIR)
        print(f"    ({aspect:<16}, {sentiment:<9}): {count:>4} samples → generate up to {to_gen}")

    augmented = list(train_long)
    uid       = 0

    # ── Step 1: Groq / template generation ───────────────────────────────────
    for (aspect, sentiment), current_count in deficits:
        needed  = min(TARGET_COUNT - current_count, MAX_PER_PAIR)
        if needed <= 0:
            continue
        reviews = _generate_reviews(groq_client, aspect, sentiment, needed)
        for text in reviews:
            if text.strip():
                uid += 1
                augmented.append(_make_synthetic_row(text, aspect, sentiment, uid, "groq" if groq_client else "template"))

    groq_added = sum(1 for r in augmented if r.get("source") in ("groq", "template") and str(r["review_id"]).startswith("aug_"))
    print(f"\n  Synthetic reviews added in Step 1: {groq_added}", flush=True)

    # ── Step 2: Back-translation ──────────────────────────────────────────────
    bt_added = 0
    if not skip_backtranslate and deficits:
        deficit_set = {pair for pair, _ in deficits}
        bt_candidates = [
            r for r in augmented
            if (r["aspect"], r["sentiment"]) in deficit_set
            and r.get("review_clean", "").strip()
            and not str(r["review_id"]).startswith("aug_")
        ][:BACK_TRANSLATE_SAMPLE]

        if bt_candidates:
            bt_texts      = back_translate([r["review_clean"] for r in bt_candidates])
            for bt_text, base_row in zip(bt_texts, bt_candidates):
                if bt_text and bt_text.strip():
                    uid += 1
                    bt_added += 1
                    augmented.append(
                        _make_synthetic_row(bt_text.strip(), base_row["aspect"], base_row["sentiment"], uid, "backtrans")
                    )
            print(f"  Back-translation added {bt_added} reviews.", flush=True)
        else:
            print("\n[Step 2] No deficient-class samples available for back-translation. Skipping.")
    else:
        print("\n[Step 2] Back-translation skipped.")

    # ── Step 3: Balance minority targets (2× rule) ───────────────────────────
    original_aspects, original_sentiments = _aggregate_counts(train_long)
    current_aspects,  current_sentiments  = _aggregate_counts(augmented)

    target_neutral = original_sentiments["neutral"] * 3
    while current_sentiments["neutral"] < target_neutral:
        uid += 1
        augmented.append(_make_synthetic_row(template_review("general", "neutral"), "general", "neutral", uid, "balance"))
        _, current_sentiments = _aggregate_counts(augmented)

    for asp in ["delivery", "cleanliness"]:
        target = original_aspects[asp] * 3
        current_aspects, _ = _aggregate_counts(augmented)
        while current_aspects[asp] < target:
            uid += 1
            augmented.append(_make_synthetic_row(template_review(asp, "neutral"), asp, "neutral", uid, "balance"))
            current_aspects, _ = _aggregate_counts(augmented)

    # ── Step 4: Final safety pass — ensure every pair hits at least 80 ───────
    final_dist = _class_distribution(augmented)
    low_pairs  = [(a, s) for (a, s), c in final_dist.items() if c < 80]
    for aspect, sentiment in low_pairs:
        need = 80 - final_dist[(aspect, sentiment)]
        for text in [template_review(aspect, sentiment) for _ in range(need)]:
            uid += 1
            augmented.append(_make_synthetic_row(text, aspect, sentiment, uid, "fill"))

    aspect_ctr, sentiment_ctr = _aggregate_counts(augmented)
    summary = {
        "original_rows":   len(train_long),
        "augmented_rows":  len(augmented),
        "groq_added":      groq_added,
        "bt_added":        bt_added,
        "total_added":     len(augmented) - len(train_long),
        "neutral_total":   sentiment_ctr["neutral"],
        "delivery_total":  aspect_ctr["delivery"],
        "cleanliness_total": aspect_ctr["cleanliness"],
    }
    return augmented, summary


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY GATES
# ══════════════════════════════════════════════════════════════════════════════

def run_quality_gates(
    original_row_count: int,
    train_clean: List[Dict[str, str]],
    train_augmented: List[Dict[str, str]],
) -> None:
    """Raise AssertionError (or patch data) if quality constraints are not met."""

    # No review ids should have been dropped
    original_ids = {str(r.get("review_id", "")).strip() for r in train_clean}
    assert len(original_ids) >= original_row_count, "Cleaning dropped review ids unexpectedly."

    # Every augmented row must have non-empty required fields
    for row in train_augmented:
        assert row.get("review_clean", "").strip(), "Found empty review_clean"
        assert row.get("aspect",       "").strip(), "Found empty aspect"
        assert row.get("sentiment",    "").strip(), "Found empty sentiment"

    # All expected aspects must be represented
    unique_aspects = {row["aspect"] for row in train_augmented}
    missing        = EXPECTED_ASPECTS - unique_aspects
    if missing:
        print(f"  [QA] Adding synthetic fill rows for missing aspects: {missing}")
        for aspect in sorted(missing):
            review = template_review(aspect, "neutral")
            train_augmented.append({
                "review_id":         f"aug_fill_{aspect}",
                "review_text":       review,
                "review_clean":      clean_text(review),
                "is_arabizi":        "false",
                "star_rating":       "",
                "date":              "",
                "business_name":     "synthetic",
                "business_category": "synthetic",
                "platform":          "synthetic",
                "aspect":            aspect,
                "sentiment":         "neutral",
                "source":            "fill",
            })


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_distribution(rows: List[Dict[str, str]], label: str = "") -> None:
    dist: Dict[Tuple[str, str], int] = defaultdict(int)
    for row in rows:
        dist[(row["aspect"], row["sentiment"])] += 1
    header = f"Class Distribution — {label}" if label else "Class Distribution"
    print(f"\n{'-' * 52}")
    print(header)
    print(f"{'-' * 52}")
    print(f"{'Aspect':<20} {'Sentiment':<12} {'Count':>6}")
    print(f"{'-' * 52}")
    for (aspect, sentiment), count in sorted(dist.items()):
        flag = " <" if count < TARGET_COUNT else ""
        print(f"{aspect:<20} {sentiment:<12} {count:>6}{flag}")
    print(f"{'-' * 52}")
    print(f"Total rows: {len(rows)}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ARG PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Phase 1 data cleaning + augmentation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",            default="data/DeepX_train.xlsx",      help="Path to training Excel file")
    parser.add_argument("--validation",       default="data/DeepX_validation.xlsx", help="Path to validation Excel file")
    parser.add_argument("--test",             default="data/DeepX_unlabeled.xlsx",  help="Path to unlabeled/test Excel file")
    parser.add_argument("--output-dir",       default="data/processed",             help="Output directory for CSVs and report")
    parser.add_argument("--use-groq",         action="store_true",                  help="Enable Groq LLM augmentation (requires GROQ_API_KEY in .env)")
    parser.add_argument("--skip-backtranslate", action="store_true",               help="Skip Helsinki-NLP back-translation step")
    parser.add_argument("--batch-size",       type=int, default=BATCH_SIZE,         help="Reviews per Groq API call")
    parser.add_argument("--per-pair-generate", type=int, default=MAX_PER_PAIR,      help="Max reviews to generate per deficient pair")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    if load_dotenv is not None:
        load_dotenv()

    project_root = Path(__file__).resolve().parent.parent
    train_path   = resolve_project_path(args.train,      project_root)
    val_path     = resolve_project_path(args.validation, project_root)
    test_path    = resolve_project_path(args.test,       project_root)
    out_dir      = resolve_project_path(args.output_dir, project_root)

    # ── initialise Groq client ────────────────────────────────────────────────
    groq_client = None
    if args.use_groq:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            print("[WARN] --use-groq specified but GROQ_API_KEY not set. Falling back to templates.")
        elif Groq is None:
            print("[WARN] groq package not installed. Run: pip install groq. Falling back to templates.")
        else:
            groq_client = Groq(api_key=api_key)
            print(f"Groq client ready (model: {GROQ_MODEL}).", flush=True)

    # ── read raw inputs ───────────────────────────────────────────────────────
    print(f"\nReading training data  : {train_path}")
    train_rows = read_input_file(train_path)
    print(f"Reading validation data: {val_path}")
    val_rows   = read_input_file(val_path)
    print(f"Reading test data      : {test_path}")
    test_rows  = read_input_file(test_path)
    print(f"  Loaded {len(train_rows):,} train | {len(val_rows):,} val | {len(test_rows):,} test rows.")

    # ── clean ─────────────────────────────────────────────────────────────────
    print("\n[Cleaning] Processing all splits …", flush=True)
    train_clean_wide = to_wide_clean_rows(train_rows, groq_client)
    val_clean_wide   = to_wide_clean_rows(val_rows,   groq_client)
    test_clean_wide  = to_wide_clean_rows(test_rows,  groq_client)
    train_clean_long = explode_train_to_long(train_rows, groq_client)
    print(f"  train_clean_long: {len(train_clean_long):,} rows")

    # ── Groq neural Arabizi transliteration (post-clean pass) ─────────────────
    if groq_client is not None:
        print("\n[Groq Transliteration] Applying neural Arabizi transliteration …", flush=True)
        train_clean_wide = apply_groq_arabizi_transliteration(train_clean_wide, groq_client)
        train_clean_long = apply_groq_arabizi_transliteration(train_clean_long, groq_client)
        val_clean_wide   = apply_groq_arabizi_transliteration(val_clean_wide,   groq_client)
        test_clean_wide  = apply_groq_arabizi_transliteration(test_clean_wide,  groq_client)

    # ── Validation: Review ID 1482 ────────────────────────────────────────────
    _TARGET_ID = "1482"
    print(f"\n--- Validation: Review ID {_TARGET_ID} (Before → After) ---")
    _found_1482 = False
    for _r in train_clean_long:
        if str(_r.get("review_id", "")).strip() == _TARGET_ID:
            print(f"  Before (review_text) : {_r['review_text'][:200]}")
            print(f"  After  (review_clean): {_r['review_clean'][:200]}")
            _found_1482 = True
            break
    if not _found_1482:
        print(f"  [INFO] Review ID {_TARGET_ID} not found in training set.")

    print_distribution(train_clean_long, "BEFORE augmentation")

    # ── augment ───────────────────────────────────────────────────────────────
    train_augmented, aug_summary = augment(
        train_clean_long,
        groq_client=groq_client,
        skip_backtranslate=args.skip_backtranslate,
    )

    # ── quality gates ─────────────────────────────────────────────────────────
    print("\n[QA] Running quality gates …", flush=True)
    run_quality_gates(len(train_rows), train_clean_long, train_augmented)
    print("  Quality gates passed ✓")

    print_distribution(train_augmented, "AFTER augmentation")

    # ── write outputs ─────────────────────────────────────────────────────────
    print(f"\n[Output] Writing CSVs to {out_dir} …", flush=True)
    write_csv(out_dir / "train_clean.csv",         train_clean_long)
    write_csv(out_dir / "train_cleaned_long.csv",  train_clean_long)   # backward-compat alias
    write_csv(out_dir / "val_clean.csv",           val_clean_wide)
    write_csv(out_dir / "test_clean.csv",          test_clean_wide)
    write_csv(out_dir / "train_augmented.csv",     train_augmented)

    train_augmented_wide = wide_from_long(train_rows, train_augmented)
    train_augmented_wide = fix_residual_arabizi_rows(train_augmented_wide)
    write_csv(out_dir / "train_augmented_wide.csv", train_augmented_wide)

    # ── JSON report ───────────────────────────────────────────────────────────
    class_dist  = _class_distribution(train_augmented)
    sorted_dist = sorted(class_dist.items(), key=lambda x: (x[0][0], x[0][1]))

    arabizi_count = sum(1 for r in train_augmented if r.get("is_arabizi") == "true")
    arabizi_pct   = round(100 * arabizi_count / max(1, len(train_augmented)), 2)

    report = {
        "phase":   "phase_1_cleaning_augmentation",
        "inputs":  {"train": str(train_path), "validation": str(val_path), "test": str(test_path)},
        "outputs": {
            "train_clean":          str(out_dir / "train_clean.csv"),
            "val_clean":            str(out_dir / "val_clean.csv"),
            "test_clean":           str(out_dir / "test_clean.csv"),
            "train_augmented":      str(out_dir / "train_augmented.csv"),
            "train_augmented_wide": str(out_dir / "train_augmented_wide.csv"),
        },
        "summary":       aug_summary,
        "arabizi_stats": {
            "total_reviews_scanned":  len(train_augmented),
            "arabizi_detected_count": arabizi_count,
            "arabizi_detected_pct":   arabizi_pct,
        },
        "class_distribution": {f"{a}|{s}": c for (a, s), c in sorted_dist},
    }
    (out_dir / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── final console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("Pipeline Complete")
    print("=" * 52)
    for key, value in aug_summary.items():
        print(f"  {key:<25}: {value:>8,}" if isinstance(value, int) else f"  {key:<25}: {value}")
    print(f"  arabizi_detected_pct    : {arabizi_pct:>7}%")
    print("=" * 52)

    # Arabizi diagnostic sample
    arabizi_sample = [r for r in train_augmented if r.get("is_arabizi") == "true"][:3]
    if arabizi_sample:
        print("\n--- Arabizi Transliteration Sample (Before → After) ---")
        for i, row in enumerate(arabizi_sample, 1):
            print(f"  [{i}] Before: {row['review_text'][:120]}")
            print(f"  [{i}] After : {row['review_clean'][:120]}")
            print()


if __name__ == "__main__":
    main()
