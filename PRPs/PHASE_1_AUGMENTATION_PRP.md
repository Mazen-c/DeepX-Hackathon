# Feature Blueprint: Phase 1 Data Augmentation

## Context
* **Role:** Data Engineer (Person B).
* **Goal:** Create a Python script (`src/data/02_augment_data.py`) to fix class imbalance by generating synthetic data for rare aspect/sentiment pairs.

## Inputs
* Read from: `data/processed/train_clean.csv`

## Processing Requirements
1. **Identify Deficits:** Group the dataset by `aspect` and `sentiment`. Identify all pairs with a count of `< 100` (e.g., `cleanliness: neutral`).
2. **Groq Synthetic Generation:**
    * Use the `groq` library and the `llama3-70b-8192` model.
    * For each deficient pair, ask the model to generate realistic Arabic customer reviews (Egyptian or Saudi dialect).
    * Target: Generate enough reviews to bring the pair count up to 100, capped at a maximum of 30 generated reviews per pair to save time.
    * **Batching Rule:** Request exactly 5 reviews per API call.
    * **Rate Limiting Rule:** You MUST implement `time.sleep(2.1)` between every API call to respect Groq's free tier limits.
3. **Back-Translation (Additive):**
    * Take up to 20 existing reviews from the deficient classes.
    * Use `Helsinki-NLP/opus-mt-ar-en` to translate to English, then `Helsinki-NLP/opus-mt-en-ar` to translate back to Arabic. 
    * Wrap this block in a `try/except`. If it fails due to network/memory, print a warning and gracefully skip it without crashing the script.

## Output Validation & Saving
* Append the newly generated Groq text and Back-Translated text to the main dataframe.
* Save to: `data/processed/train_augmented.csv`.
* Print the new class distribution to the console to prove the imbalance is fixed.