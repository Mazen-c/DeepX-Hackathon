# 🔍 DeepX Hackathon — Arabic ABSA System

> **Aspect-Based Sentiment Analysis for Arabic Customer Reviews**  
> Extract structured, multi-aspect sentiment from Arabic text — zero cost, CPU-only, production-ready.

---

## 📌 Overview

This project is an end-to-end **Aspect-Based Sentiment Analysis (ABSA)** pipeline built for the DeepX Hackathon. Given an Arabic customer review, the system identifies every mentioned aspect and classifies the sentiment toward each one, returning a strict JSON response.

**Example Input:**
```
الطعام كان ممتازا لكن الخدمة بطيئة جدا
```

**Example Output:**
```json
{
  "predictions": [
    { "aspect": "food",    "sentiment": "positive" },
    { "aspect": "service", "sentiment": "negative" }
  ]
}
```

---

## ⚡ Key Constraints

| Constraint | Requirement |
|---|---|
| 💸 Cost | Zero — Groq free tier only, no paid APIs |
| 💻 Compute | CPU-only, no GPU assumed |
| 📦 Bandwidth | Models ≤ 500 MB |
| 🔑 API Keys | Via `.env` only — never hardcoded |

---

## 🏗️ Architecture

The system is organized into **5 sequential phases**:

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────┐
│  Phase 1: Data Cleaning &   │  ← pyarabic, pandas, Groq (augmentation)
│           Augmentation      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 2: Vector Store /    │  ← ChromaDB + multilingual MiniLM
│           RAG Index         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 3: Groq LLM Engine   │  ← llama-3.3-70b → llama-3.1-8b → local fallback
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 4: Streamlit         │  ← Real-time dashboard + MongoDB logging
│           Dashboard         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Phase 5: Evaluation &      │  ← Hidden test predictions + submission JSON
│           Submission        │
└─────────────────────────────┘
```

---

## 📁 Repository Structure

```
DeepX-Hackathon/
├── app.py                          # Phase 4 — Streamlit dashboard entry point
├── train_absa.py                   # Production multi-label ABSA model trainer
├── requirements.txt                # Python dependencies
├── CLAUDE.md                       # Project rules & constraints for AI assistance
│
├── scripts/
│   ├── config.py                   # Centralized workspace path configuration
│   ├── groq_engine.py              # Phase 3 — Groq LLM inference + fallback logic
│   ├── rag_utils.py                # Phase 2 — ChromaDB indexing & retrieval
│   ├── train_local_weights.py      # Lightweight TF-IDF + OvR classifier trainer
│   ├── train_on_predictions.py     # Semi-supervised pseudo-label training
│   └── data_pipeline.py           # (Phase 1 data pipeline)
│
├── DataBase/
│   ├── db.py                       # SQLite helpers (retrieval cache, app state)
│   └── mongo_db.py                 # MongoDB helpers (prediction logging, metrics)
│
├── Data/
│   ├── DeepX_hidden_test.xlsx      # Competition hidden test set
│   └── processed/                  # Cleaned & augmented CSVs (generated)
│
├── Notebook/
│   └── Data-exploring.ipynb        # Exploratory data analysis
│
├── PRPs/
│   ├── PHASE_1_CLEANING_PRP.md     # Data cleaning blueprint
│   └── PHASE_1_AUGMENTATION_PRP.md # Data augmentation blueprint
│
└── phases/                         # Phase documentation (Word files)
    ├── Phase_2_Local_Vector_Store_RAG.docx
    ├── Phase_3_Groq_LLM_Engine.docx
    ├── Phase_4_Streamlit_Dashboard.docx
    └── Phase_5_Evaluation_Submission.docx
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
MONGO_URI=mongodb://localhost:27017/    # optional
MONGO_DB=arabic_absa                   # optional
```

> **MongoDB is optional.** The dashboard works without it — only prediction logging is disabled.

### 3. Run the Pipeline

Execute phases in order:

```bash
# Phase 1a — Clean raw data
python scripts/data_cleaning_phase.py

# Phase 1b — Validate cleaned data (QA gatekeeper)
python scripts/validate_cleaning.py

# Phase 1c — Augment with synthetic data
python scripts/augment_data.py

# Phase 2 — Build ChromaDB vector store
python scripts/rag_utils.py

# Phase 3a — Train local fallback classifier
python scripts/train_local_weights.py

# Phase 3b — Train full production model
python train_absa.py

# Phase 4 — Launch dashboard
streamlit run app.py
```

### Cleaning Flags

```bash
# Drop reviews with empty text
python scripts/data_cleaning_phase.py --drop-empty-text

# Disable Arabic normalization
python scripts/data_cleaning_phase.py --disable-arabic-normalization

# Custom output directory
python scripts/data_cleaning_phase.py --output-dir Data/processed
```

---

## 🧠 Model Details

### Aspect Taxonomy

The system recognizes exactly **9 aspects**:

| Aspect | Description |
|---|---|
| `food` | Food quality, taste, freshness |
| `service` | Staff, responsiveness, attitude |
| `price` | Value for money, cost |
| `cleanliness` | Hygiene, tidiness |
| `delivery` | Speed, packaging, accuracy |
| `ambiance` | Atmosphere, decor, noise |
| `app_experience` | Mobile/web app usability |
| `general` | Overall impression (no specific aspect) |
| `none` | No meaningful sentiment expressed |

### Sentiment Labels

`positive` · `negative` · `neutral`

---

### Inference Chain

```
Input Review
     │
     ├─► Groq llama-3.3-70b-versatile  ──► Parse JSON ──► ✅ Return
     │         (primary, rate-limited)         │
     │                                         │ parse fail / timeout
     ├─► Groq llama-3.1-8b-instant     ──► Parse JSON ──► ✅ Return
     │         (fast fallback)                 │
     │                                         │ Groq unavailable
     └─► Local TF-IDF + OvR Ensemble   ──────────────────► ✅ Return
               (5 versioned weight files, CPU-only)
```

### Local Ensemble Weights

| Model File | Weight |
|---|---|
| `local_absa_weights_v3_wide.joblib` | 0.70 |
| `local_absa_weights_v2.joblib` | 0.50 |
| `local_absa_weights.joblib` | 0.40 |
| `local_absa_weights_v4_meta.joblib` | 0.30 |
| `pseudo_labels_weights.joblib` | 0.30 |

> Threshold: `0.55` · Max aspects per review: `4` · Top-gap filter: `0.2`

---

## 📊 Dashboard

Launch with `streamlit run app.py` to access:

- **Live Prediction** — enter any Arabic review and analyze it
- **Demo Reviews** — 5 pre-loaded example reviews to explore quickly
- **Metrics Bar** — total predictions, average latency (ms), parse success rate
- **Recent Predictions Log** — last 20 predictions from MongoDB
- **Class Distribution Chart** — aspect × sentiment breakdown (Plotly)

---

## 🗄️ Data Outputs

All generated files appear under `Data/processed/`:

| File | Description |
|---|---|
| `train_cleaned_wide.csv` | One row per review |
| `train_cleaned_long.csv` | One row per aspect-review pair |
| `validation_cleaned_wide.csv` | Cleaned validation set (wide) |
| `validation_cleaned_long.csv` | Cleaned validation set (long) |
| `unlabeled_cleaned.csv` | Cleaned unlabeled data |
| `train_augmented.csv` | Final training set after augmentation |
| `cleaning_report.json` | Summary stats from the cleaning run |
| `val_predictions.json` | Validation set predictions |
| `deepx_hidden_predictions.json` | Hidden test set submission |

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | `pandas`, `pyarabic`, `re` |
| LLM Inference | `groq` (Groq API — free tier) |
| RAG / Vector Store | `chromadb`, `sentence-transformers` |
| Local ML | `scikit-learn` (TF-IDF + One-vs-Rest LR) |
| Translation (augment) | `Helsinki-NLP/opus-mt-ar-en` + `opus-mt-en-ar` |
| Dashboard | `streamlit`, `plotly` |
| Operational DB | `pymongo` (MongoDB) |
| Local Cache DB | `sqlite3` (built-in) |
| Config | `python-dotenv` |

---

## ⚙️ Configuration Reference

All paths are resolved in `scripts/config.py`:

| Variable | Default |
|---|---|
| `GROQ_API_KEY` | *(required, from `.env`)* |
| `MONGO_URI` | `mongodb://localhost:27017/` |
| `MONGO_DB` | `arabic_absa` |
| `RAG_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` |
| `CHROMA_STORE_DIR` | `<root>/chroma_store/` |
| `SQLITE_DB_PATH` | `<root>/absa_phase2.db` |

---

## 📋 Coding Standards

- Clean, modular Python with **type hints** throughout
- Missing values handled gracefully at every step
- No hardcoded API keys — always use `os.environ.get("GROQ_API_KEY")`
- Console progress logs on all long-running operations
- Rate limiter enforces **28 req/min** and **14,400 req/day** against Groq's free tier

---

## 📄 License

This project was built for the DeepX Hackathon. All rights reserved by the respective authors.
