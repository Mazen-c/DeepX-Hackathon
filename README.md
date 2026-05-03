# DeepX Arabic ABSA Hackathon

Arabic Aspect-Based Sentiment Analysis (ABSA) system for customer feedback.

The app takes an Arabic review, extracts one or more aspect-sentiment pairs, returns strict JSON, and logs every prediction to SQLite with MongoDB mirroring for Compass.

## Current Features

- Custom local web UI at `http://127.0.0.1:8501`
- CPU-friendly local prediction path using saved model weights
- Strict JSON output:

```json
{
  "predictions": [
    {"aspect": "service", "sentiment": "positive"},
    {"aspect": "price", "sentiment": "negative"}
  ]
}
```

- SQLite logging in `absa_phase2.db`
- MongoDB logging in `arabic_absa.predictions`
- Dashboard metrics for prediction count, latency, parse success, and aspect distribution
- Data cleaning, augmentation, validation, and hidden-test prediction scripts

## Project Constraints

- No paid APIs are required for the UI demo path.
- Local development is CPU-only.
- Heavy Hugging Face downloads should be avoided.
- API keys must stay in `.env`; never hardcode them.

## Tech Stack

- Python
- pandas, numpy, scikit-learn
- pyarabic
- SQLite
- MongoDB + PyMongo
- Groq free API for optional LLM/RAG workflows
- Plain HTML/CSS/JavaScript UI served by `app.py`

## Important Files

```text
app.py                                  # Main web UI and API server
scripts/config.py                       # Shared paths and environment loading
scripts/groq_engine.py                  # Prediction, validation, and DB logging logic
DataBase/db.py                          # SQLite helper
DataBase/mongo_db.py                    # MongoDB helper
scripts/run_ui.bat                      # Windows UI launcher
scripts/run_ui.ps1                      # PowerShell UI launcher
models/local_absa_weights_v3_wide.joblib
models/local_absa_weights_v3_wide.meta.json
Data/processed/train_augmented_wide.csv
absa_phase2.db
```

## Setup

From PowerShell:

```powershell
cd "C:\DeepX Hackathon"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create or update `.env`:

```env
GROQ_API_KEY=your_groq_key_here
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=arabic_absa
```

The local UI can run without calling Groq, but the key is used by optional Groq validation/export workflows.

## Run The UI

Recommended:

```powershell
cd "C:\DeepX Hackathon"
.\.venv\Scripts\python.exe app.py --host 127.0.0.1 --port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

Or run:

```powershell
.\scripts\run_ui.bat
```

## MongoDB Compass

Start MongoDB if needed:

```powershell
Get-Service MongoDB
Start-Service MongoDB
```

In MongoDB Compass, connect to:

```text
mongodb://localhost:27017/
```

Prediction documents are written to:

```text
Database: arabic_absa
Collection: predictions
```

Validation/submission run artifacts are written to:

```text
Database: arabic_absa
Collection: runs
```

## API Endpoints

Prediction:

```http
POST /api/predict
Content-Type: application/json

{"review": "الخدمة ممتازة والسعر مناسب"}
```

Response includes the prediction and database write confirmation:

```json
{
  "result": {
    "predictions": [
      {"aspect": "price", "sentiment": "positive"}
    ]
  },
  "database": {
    "sqlite_ready": true,
    "sqlite_inserted": true,
    "mongo_ready": true,
    "mongo_database": "arabic_absa"
  }
}
```

Dashboard data:

```http
GET /api/dashboard
```

## Data Pipeline

Clean raw data:

```powershell
.\.venv\Scripts\python.exe scripts\data_cleaning_phase.py
```

Augment data:

```powershell
.\.venv\Scripts\python.exe scripts\02_augment_data.py
```

Train local weights:

```powershell
.\.venv\Scripts\python.exe scripts\train_local_weights.py
```

Run validation or hidden-test prediction workflows:

```powershell
.\.venv\Scripts\python.exe scripts\groq_engine.py --help
```

## Main Outputs

Under `Data/processed`:

- `train_clean.csv`
- `val_clean.csv`
- `test_clean.csv`
- `train_augmented.csv`
- `train_augmented_wide.csv`
- `val_predictions_v3_wide.json`
- `val_results_v3_wide.json`
- `deepx_hidden_predictions*.json`
- `cleaning_report.json`

## Notes

- The UI uses the local model path first, so demo predictions are fast and free.
- Every UI prediction is saved to SQLite.
- MongoDB mirroring happens automatically when `pymongo` is installed and MongoDB is running.
- If Compass is not updating, make sure the UI is launched with `.\.venv\Scripts\python.exe`, not system `python`.
