# Global Project Rules: Arabic ABSA Hackathon

## 1. Project Context
We are building an Aspect-Based Sentiment Analysis (ABSA) system for Arabic customer feedback. The final goal is to extract multiple aspects and sentiments from a single review and output them as a strict JSON schema. 

## 2. STRICT CONSTRAINTS (DO NOT VIOLATE)
* **Zero Financial Cost:** Do NOT write code that requires paid APIs (OpenAI, Anthropic, etc.). 
* **Low Bandwidth:** Do NOT write code that downloads Hugging Face models larger than 500MB. 
* **Compute Limit:** All processing must run on a standard local CPU. Do not assume access to a GPU for local development.

## 3. Approved Tech Stack
* **Data Processing:** `pandas`, `re`, `pyarabic`
* **LLM Engine:** Groq Free API (`groq` Python library). The key is stored in a `.env` file and loaded via `python-dotenv`.
* **Lightweight NLP:** `transformers` (ONLY for the approved 48MB Helsinki-NLP translation models).

## 4. Coding Standards
* Write clean, modular Python code with type hints.
* Always handle missing values gracefully.
* Do not hardcode API keys. Use `os.environ.get("GROQ_API_KEY")`.
* Print clear console logs (e.g., progress bars or status messages) so we can track the script's progress during the hackathon.