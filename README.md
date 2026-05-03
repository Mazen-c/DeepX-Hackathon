# DeepX Hackathon - Arabic ABSA System

Aspect-Based Sentiment Analysis (ABSA) for Arabic customer reviews. The system extracts one or more aspect/sentiment pairs from each review and returns a strict JSON-compatible schema.

This repository is designed for the hackathon constraints:

- Zero financial cost: no paid APIs.
- CPU-only local development.
- Low bandwidth: no model downloads above 500 MB.
- Secrets loaded from `.env`; API keys are never hardcoded.

## Current Architecture

The current runnable system has four main layers:

```text
Data/processed CSV artifacts
        |
        v
Local model training
train_absa.py
TF-IDF char n-grams + OneVsRest Logistic Regression
        |
        v
models/local_absa_weights_v3_wide.joblib
        |
        v
Inference layer
scripts/groq_engine.py
local model -> heuristic fallback
optional Groq + optional RAG for batch prediction
        |
        v
Custom web dashboard
app.py
SQLite logging + optional MongoDB mirror
```

### What Each Layer Does

| Layer | Files | Purpose |
|---|---|---|
| Processed data | `Data/processed/*.csv` | Cleaned, augmented, validation, and prediction artifacts used by training and evaluation. |
| Text cleaning | `scripts/data_cleaning_phase.py` | Lightweight `clean_text` helper used by inference and hidden-test export. |
| Local model | `train_absa.py`, `models/local_absa_weights_v3_wide.joblib` | CPU-friendly multi-label classifier for aspect/sentiment labels. |
| RAG index | `scripts/rag_utils.py`, `chroma_store/` | Optional ChromaDB retrieval of similar labeled examples for Groq prompting. |
| Inference | `scripts/groq_engine.py` | Validates JSON output, runs Groq when enabled, falls back to local/heuristic predictions. |
| Web UI | `app.py` | Custom HTTP dashboard, not Streamlit. Uses `predict_local()` for fast free demos. |
| Databases | `DataBase/db.py`, `DataBase/mongo_db.py` | SQLite is local/default. MongoDB is optional for mirrored prediction logs. |

## Repository Structure

```text
DeepX Hackathon/
|-- app.py                         # Custom local web dashboard
|-- train_absa.py                  # Main local model trainer
|-- requirements.txt               # Python dependencies
|-- AGENTS.md                      # Project rules and constraints
|-- README.md
|
|-- scripts/
|   |-- config.py                  # Shared paths and environment defaults
|   |-- data_cleaning_phase.py     # Shared clean_text helper
|   |-- groq_engine.py             # Inference, validation, hidden-test export
|   |-- rag_utils.py               # ChromaDB build/retrieval utilities
|   |-- rebuild_training_long.py   # Rebuild long training CSV from wide CSV
|   |-- train_local_weights.py     # Lightweight long-format trainer
|   |-- train_on_predictions.py    # Pseudo-label training helper
|   |-- run_ui.bat / run_ui.ps1    # UI launch helpers
|   `-- run_ui_detached.py         # Detached UI launcher
|
|-- Data/
|   `-- processed/
|       |-- train_augmented_wide.csv
|       |-- train_augmented.csv
|       |-- train_clean.csv
|       |-- val_clean.csv
|       |-- val_results*.json
|       `-- deepx_hidden_predictions*.json
|
|-- models/
|   |-- local_absa_weights_v3_wide.joblib
|   `-- local_absa_weights_v3_wide.meta.json
|
|-- DataBase/
|   |-- db.py                      # SQLite helpers
|   `-- mongo_db.py                # Optional MongoDB helpers
|
|-- chroma_store/                  # Persisted ChromaDB vector store
|-- Notebook/                      # Exploration notebook
|-- phases/                        # Phase documentation
`-- PRPs/                          # Planning docs
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=arabic_absa
```

`GROQ_API_KEY` is only required for Groq-backed batch prediction. The local dashboard can run without Groq. MongoDB is optional; SQLite logging works by default.

## Run The System

### Start the local dashboard

```bash
python app.py --host 127.0.0.1 --port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

On Windows you can also use:

```powershell
.\scripts\run_ui.ps1
```

or:

```bat
scripts\run_ui.bat
```

### Train the main local model

```bash
python train_absa.py --input Data/processed/train_augmented_wide.csv --output models/local_absa_weights_v3_wide.joblib
```

The current checked-in metadata reports:

| Metric | Value |
|---|---:|
| Training samples | 2,815 |
| Best threshold | 0.55 |
| Max aspects per review | 4 |
| Holdout micro F1 | 0.692 |

### Build or rebuild the RAG vector store

```bash
python scripts/rag_utils.py --input Data/processed/train_augmented.csv
```

This uses `transformers` with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` from the local cache when available. If the model cannot load, the code falls back to a local hashing embedder so the pipeline stays CPU-only.

### Run a single Groq-backed prediction

```bash
python scripts/groq_engine.py --review "sample Arabic review text"
```

### Export hidden-test predictions

Local-only, no Groq API calls:

```bash
python scripts/groq_engine.py --mode hidden --disable-groq --input "Data/DeepX_hidden_test .xlsx" --predictions-output Data/processed/deepx_hidden_predictions.json
```

Groq + RAG enabled:

```bash
python scripts/groq_engine.py --mode hidden --input "Data/DeepX_hidden_test .xlsx" --predictions-output Data/processed/deepx_hidden_predictions.json
```

### Run validation

```bash
python scripts/groq_engine.py --mode validation --val-csv Data/processed/val_clean.csv --output Data/processed/val_results.json --predictions-output Data/processed/val_predictions.json
```

## Output Schema

The live predictor returns:

```json
{
  "predictions": [
    {
      "aspect": "food",
      "sentiment": "positive"
    },
    {
      "aspect": "service",
      "sentiment": "negative"
    }
  ]
}
```

Hidden-test exports use submission rows:

```json
{
  "review_id": 1,
  "aspects": ["food", "service"],
  "aspect_sentiments": {
    "food": "positive",
    "service": "negative"
  }
}
```

## Labels

Valid aspects:

- `food`
- `service`
- `price`
- `cleanliness`
- `delivery`
- `ambiance`
- `app_experience`
- `general`
- `none`

Valid sentiments:

- `positive`
- `negative`
- `neutral`

The production label space is 25 classes: every aspect/sentiment combination used by the trainer, with `none|neutral` as the only `none` class.

## Inference Behavior

`app.py` uses `predict_local()` so demos are fast, free, and CPU-only:

```text
review text
   -> clean_text()
   -> local_absa_weights_v3_wide.joblib
   -> validated JSON
   -> SQLite log
   -> optional MongoDB mirror
```

`scripts/groq_engine.py` can also use Groq for batch or CLI prediction:

```text
review text
   -> optional RAG few-shot examples
   -> Groq llama-3.3-70b-versatile
   -> Groq llama-3.1-8b-instant fallback
   -> local model fallback
   -> heuristic fallback
   -> validated JSON
```

The Groq rate limiter is set to 28 requests per minute and 14,400 requests per day.

## Tech Stack

| Area | Technology |
|---|---|
| Data processing | `pandas`, `re`, `pyarabic` |
| Local ML | `scikit-learn`, `joblib` |
| Optional LLM | `groq`, `python-dotenv` |
| Optional RAG | `chromadb`, `transformers`, `torch` |
| Web UI | Python `http.server`, HTML, CSS, JavaScript |
| Local logging | `sqlite3` |
| Optional logging mirror | `pymongo` |

## Important Notes

- The README previously mentioned Streamlit and Plotly, but the current app is a custom HTTP dashboard.
- The README previously listed Phase 1 scripts that are not present in this workspace. The processed CSV artifacts are present under `Data/processed/`.
- `scripts/data_pipeline.py` is currently an empty placeholder.
- The checked-in UI and model path target `models/local_absa_weights_v3_wide.joblib`.

## License

Built for the DeepX Hackathon. All rights reserved by the respective authors.
