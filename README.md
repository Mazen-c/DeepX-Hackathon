# DeepX-Hackathon

## Data Cleaning Phase (Aspect-Based Sentiment)

This project includes a cleaning pipeline for the objective of extracting multiple aspects from each review and assigning sentiment per aspect.

### What the cleaning script does

- Loads train, validation, and unlabeled CSV files with encoding fallback.
- Cleans review text (HTML unescape, URL/email removal, whitespace cleanup, repeated-char normalization).
- Applies optional Arabic normalization (remove diacritics and unify common letter variants).
- Parses `aspects` and `aspect_sentiments` JSON fields safely.
- Normalizes sentiment labels to `positive`, `negative`, or `neutral`.
- Creates:
	- Wide files (one row per review)
	- Long files (one row per review-aspect pair), which are ideal for aspect-level modeling
- Writes a `cleaning_report.json` summary.

### Run

```powershell
python scripts/data_cleaning_phase.py
```

Optional flags:

```powershell
python scripts/data_cleaning_phase.py --drop-empty-text
python scripts/data_cleaning_phase.py --disable-arabic-normalization
python scripts/data_cleaning_phase.py --output-dir Data/processed
```

### Output files

Under `Data/processed`:

- `train_cleaned_wide.csv`
- `train_cleaned_long.csv`
- `validation_cleaned_wide.csv`
- `validation_cleaned_long.csv`
- `unlabeled_cleaned.csv`
- `cleaning_report.json`