# Feature Blueprint: Phase 1 Data Validation (The Gatekeeper)

## Context
* **Role:** QA Data Engineer.
* **Goal:** Create a script (`src/data/01b_validate_cleaning.py`) that strictly audits the cleaned data produced by the upstream pipeline before it is allowed into the Data Augmentation phase.

## Inputs
* Read from: `data/processed/train_clean.csv`

## Processing Requirements (The Audit)
Write a Python script that runs the following strict tests and prints a "PASS/FAIL" terminal report.

1. **Data Loss Check:**
    * Count total rows.
    * **FAIL condition:** If total rows are less than 1,870 (meaning the cleaning script dropped more than 5% of the original 1,971 rows).
2. **Null Value Check:**
    * **FAIL condition:** If there is a single NaN or Null value in `review_text`, `aspects`, or `aspect_sentiments`.
3. **JSON Schema Audit (CRITICAL):**
    * Iterate through every single row in the `aspect_sentiments` column.
    * Attempt to parse it using `json.loads()`.
    * **FAIL condition:** If the code throws a JSONDecodeError. This means the cleaning script failed to remove the double-escaped quotes (e.g., `"{""service"": ""positive""}"`).
4. **Taxonomy & Strict Typo Audit:**
    * Extract every aspect and sentiment from the successfully parsed JSONs.
    * **FAIL condition:** If any aspect is NOT exactly one of: `['food', 'service', 'price', 'cleanliness', 'delivery', 'ambiance', 'app_experience', 'general', 'none']`.
    * **FAIL condition:** If any sentiment is NOT exactly one of: `['positive', 'negative', 'neutral']`.

## Output
* Print a beautiful, color-coded terminal report summarizing the checks.
* If all tests PASS, print the exact distribution of `(aspect, sentiment)` pairs so the Data Augmentation script knows exactly how many synthetic rows to generate.
* **Do not modify or overwrite the CSV.** This is a read-only audit.