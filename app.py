"""Standalone web UI for the Arabic ABSA hackathon project.

Run:
    python app.py

The app intentionally avoids Streamlit so the interface can be fully custom
while still using the local CPU-friendly ABSA weights and database logging.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd

from DataBase.db import get_connection
from DataBase.mongo_db import get_mongo_settings, init_db as init_mongo_db
from scripts.config import PROCESSED_DIR, SQLITE_DB_PATH
from scripts.groq_engine import _ensure_phase3_tables, predict_local

APP_TITLE = "Arabic ABSA Command Center"
DATASET_PATH = PROCESSED_DIR / "train_augmented_wide.csv"
MODEL_META_PATH = Path("models") / "local_absa_weights_v3_wide.meta.json"
MODEL_WEIGHTS_PATH = Path("models") / "local_absa_weights_v3_wide.joblib"
_MONGO_STATUS_CACHE: Dict[str, Any] = {"checked_at": 0.0, "ready": False}

DEMO_REVIEWS = [
    "الطعام كان ممتازا لكن الخدمة بطيئة جدا",
    "التطبيق صعب الاستخدام والدفع لا يعمل عند الاستلام",
    "السعر مناسب والمكان نظيف والموظفين محترمين",
    "التوصيل تأخر ساعتين والطلب وصل بارد",
    "الأجواء رائعة والطعام لذيذ لكن السعر مرتفع",
]

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Arabic ABSA Command Center</title>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #111822;
      --panel-2: #151f2c;
      --ink: #eef5f8;
      --muted: #9fb0bd;
      --line: #243141;
      --teal: #2dd4bf;
      --amber: #f59e0b;
      --rose: #fb7185;
      --violet: #a78bfa;
      --green: #34d399;
      --red: #f87171;
      --blue: #60a5fa;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.36);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 18% 0%, rgba(45, 212, 191, 0.18), transparent 28rem),
        radial-gradient(circle at 88% 18%, rgba(251, 113, 133, 0.12), transparent 24rem),
        linear-gradient(135deg, #0b0f14 0%, #101722 54%, #0d131a 100%);
      min-height: 100vh;
    }

    button, textarea { font: inherit; }

    .shell {
      width: min(1380px, calc(100% - 32px));
      margin: 0 auto;
      padding: 22px 0 32px;
    }

    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 12px 0 20px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .mark {
      width: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(45, 212, 191, 0.35);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(45, 212, 191, 0.24), rgba(96, 165, 250, 0.14));
      box-shadow: 0 0 0 5px rgba(45, 212, 191, 0.08);
      color: var(--teal);
      font-weight: 900;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.45rem, 2vw, 2.1rem);
      letter-spacing: 0;
      line-height: 1.1;
    }

    .subtitle {
      color: var(--muted);
      margin-top: 4px;
      font-size: 0.95rem;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 38px;
      padding: 0 13px;
      border: 1px solid rgba(52, 211, 153, 0.28);
      border-radius: 999px;
      color: #d8fff3;
      background: rgba(52, 211, 153, 0.08);
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 16px rgba(52, 211, 153, 0.8);
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(340px, 0.85fr);
      gap: 18px;
      align-items: stretch;
    }

    .workbench, .side-panel, .metric, .recent-panel {
      border: 1px solid rgba(159, 176, 189, 0.16);
      background: linear-gradient(180deg, rgba(17, 24, 34, 0.94), rgba(13, 19, 26, 0.92));
      box-shadow: var(--shadow);
    }

    .workbench {
      min-height: 520px;
      padding: 22px;
      border-radius: 8px;
    }

    .section-label {
      margin: 0 0 10px;
      color: var(--teal);
      text-transform: uppercase;
      font-weight: 800;
      font-size: 0.78rem;
      letter-spacing: 0.14em;
    }

    .headline {
      margin: 0 0 12px;
      font-size: clamp(2rem, 5vw, 4.7rem);
      line-height: 0.97;
      letter-spacing: 0;
      max-width: 11ch;
    }

    .copy {
      margin: 0 0 20px;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.65;
      font-size: 1rem;
    }

    .input-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      gap: 14px;
      align-items: stretch;
    }

    textarea {
      width: 100%;
      min-height: 178px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #081019;
      color: var(--ink);
      padding: 18px;
      line-height: 1.7;
      direction: rtl;
      outline: none;
    }

    textarea:focus {
      border-color: rgba(45, 212, 191, 0.72);
      box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.11);
    }

    .actions {
      display: grid;
      grid-template-rows: 1fr 1fr;
      gap: 12px;
    }

    .primary, .secondary, .chip {
      cursor: pointer;
      border: 1px solid transparent;
      border-radius: 8px;
      transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
    }

    .primary {
      color: #031311;
      background: linear-gradient(135deg, var(--teal), #8df4cf);
      font-weight: 900;
      font-size: 1.02rem;
      box-shadow: 0 16px 36px rgba(45, 212, 191, 0.22);
    }

    .secondary {
      color: var(--ink);
      background: rgba(255, 255, 255, 0.04);
      border-color: var(--line);
      font-weight: 800;
    }

    .primary:hover, .secondary:hover, .chip:hover {
      transform: translateY(-1px);
      border-color: rgba(45, 212, 191, 0.44);
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .chip {
      color: var(--muted);
      background: rgba(255, 255, 255, 0.035);
      border-color: rgba(159, 176, 189, 0.16);
      padding: 9px 11px;
      font-size: 0.88rem;
    }

    .output {
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      min-height: 132px;
    }

    .result {
      border: 1px solid rgba(159, 176, 189, 0.16);
      border-radius: 8px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.04);
      position: relative;
      overflow: hidden;
    }

    .result::before {
      content: "";
      position: absolute;
      inset: 0 0 auto;
      height: 4px;
      background: var(--blue);
    }

    .result.positive::before { background: var(--green); }
    .result.negative::before { background: var(--red); }
    .result.neutral::before { background: var(--amber); }

    .aspect {
      font-size: 1.06rem;
      font-weight: 900;
      text-transform: capitalize;
      margin-bottom: 8px;
    }

    .sentiment {
      display: inline-flex;
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--muted);
      font-weight: 800;
      font-size: 0.82rem;
      text-transform: uppercase;
    }

    .side-panel {
      border-radius: 8px;
      padding: 18px;
      display: grid;
      gap: 14px;
    }

    .model-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background:
        linear-gradient(135deg, rgba(245, 158, 11, 0.10), transparent 38%),
        rgba(255, 255, 255, 0.035);
    }

    .model-card h2, .recent-panel h2 {
      margin: 0 0 10px;
      font-size: 1.05rem;
    }

    .kv {
      display: grid;
      gap: 9px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.93rem;
    }

    .kv div {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid rgba(159, 176, 189, 0.12);
      padding-bottom: 8px;
    }

    .kv strong {
      color: var(--ink);
      text-align: right;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }

    .metric {
      border-radius: 8px;
      padding: 16px;
    }

    .metric span {
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.09em;
    }

    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: clamp(1.35rem, 3vw, 2.1rem);
    }

    .lower {
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
      gap: 18px;
      margin-top: 18px;
    }

    .recent-panel {
      border-radius: 8px;
      padding: 18px;
      min-height: 300px;
    }

    .bars {
      display: grid;
      gap: 10px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 130px 1fr 44px;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .track {
      height: 11px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 999px;
      overflow: hidden;
    }

    .fill {
      height: 100%;
      width: 0;
      background: linear-gradient(90deg, var(--teal), var(--amber));
      border-radius: inherit;
      transition: width 300ms ease;
    }

    .recent-list {
      display: grid;
      gap: 10px;
      max-height: 410px;
      overflow: auto;
      padding-right: 4px;
    }

    .recent-item {
      border: 1px solid rgba(159, 176, 189, 0.14);
      border-radius: 8px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.035);
    }

    .recent-text {
      color: var(--ink);
      line-height: 1.55;
      direction: rtl;
      text-align: right;
      margin-bottom: 8px;
    }

    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .tag {
      padding: 4px 7px;
      border-radius: 999px;
      background: rgba(96, 165, 250, 0.14);
      color: #d9eaff;
      font-size: 0.78rem;
      font-weight: 800;
    }

    .toast {
      min-height: 22px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .toast.error { color: var(--rose); }

    @media (max-width: 960px) {
      .hero, .lower, .input-grid { grid-template-columns: 1fr; }
      .actions { grid-template-columns: 1fr 1fr; grid-template-rows: auto; }
      .metrics { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .headline { max-width: none; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="mark">AB</div>
        <div>
          <h1>Arabic ABSA Command Center</h1>
          <div class="subtitle">Local CPU model trained on the clean wide augmented dataset</div>
        </div>
      </div>
      <div class="status-pill"><span class="dot"></span><span id="dbStatus">Database logging online</span></div>
    </header>

    <section class="hero">
      <div class="workbench">
        <p class="section-label">Live Prediction</p>
        <h2 class="headline">Extract every aspect. Judge every sentiment.</h2>
        <p class="copy">Paste an Arabic customer review and get strict ABSA JSON plus a visual breakdown. Every run is written to the local database, with MongoDB mirroring when your local service is available.</p>

        <div class="input-grid">
          <textarea id="review" spellcheck="false">الطعام كان ممتازا لكن الخدمة بطيئة جدا</textarea>
          <div class="actions">
            <button class="primary" id="analyze">Analyze Review</button>
            <button class="secondary" id="clear">Clear</button>
          </div>
        </div>

        <div class="chips" id="demos"></div>
        <div class="toast" id="toast"></div>
        <div class="output" id="output"></div>
      </div>

      <aside class="side-panel">
        <div class="model-card">
          <p class="section-label">Model Source</p>
          <h2 id="modelName">local_absa_weights_v3_wide.joblib</h2>
          <div class="kv">
            <div><span>Dataset</span><strong id="datasetName">train_augmented_wide.csv</strong></div>
            <div><span>Rows</span><strong id="datasetRows">...</strong></div>
            <div><span>Holdout F1</span><strong id="modelF1">...</strong></div>
            <div><span>Question marks</span><strong id="questionMarks">...</strong></div>
          </div>
        </div>
        <div class="model-card">
          <p class="section-label">JSON Contract</p>
          <pre id="jsonPreview">{"predictions":[]}</pre>
        </div>
      </aside>
    </section>

    <section class="metrics">
      <div class="metric"><span>Total Predictions</span><strong id="totalPredictions">0</strong></div>
      <div class="metric"><span>Average Latency</span><strong id="avgLatency">0 ms</strong></div>
      <div class="metric"><span>Parse Success</span><strong id="successRate">0%</strong></div>
    </section>

    <section class="lower">
      <div class="recent-panel">
        <h2>Aspect Distribution</h2>
        <div class="bars" id="bars"></div>
      </div>
      <div class="recent-panel">
        <h2>Recent Database Writes</h2>
        <div class="recent-list" id="recent"></div>
      </div>
    </section>
  </main>

  <script>
    const demos = __DEMOS__;
    const review = document.querySelector("#review");
    const output = document.querySelector("#output");
    const toast = document.querySelector("#toast");
    const jsonPreview = document.querySelector("#jsonPreview");

    function showToast(message, isError = false) {
      toast.textContent = message;
      toast.className = isError ? "toast error" : "toast";
    }

    function renderPredictions(payload) {
      const predictions = payload.predictions || [];
      jsonPreview.textContent = JSON.stringify(payload, null, 2);
      output.innerHTML = predictions.map((item) => `
        <div class="result ${item.sentiment}">
          <div class="aspect">${item.aspect}</div>
          <span class="sentiment">${item.sentiment}</span>
        </div>
      `).join("");
    }

    async function analyze() {
      const text = review.value.trim().replace(/\?{3,}/g, " ").replace(/\s+/g, " ");
      if (!text) {
        showToast("Enter a review first.", true);
        return;
      }
      showToast("Analyzing with local wide-trained weights...");
      document.querySelector("#analyze").disabled = true;
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({review: text})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed");
        renderPredictions(data.result);
        const db = data.database || {};
        const mirror = db.mongo_ready ? " + MongoDB mirror" : "";
        const saveMessage = db.sqlite_inserted
          ? `Saved to SQLite${mirror} in ${data.latency_ms.toFixed(1)} ms.`
          : `Prediction finished, but the database write was not confirmed.`;
        showToast(saveMessage, !db.sqlite_inserted);
        await loadDashboard();
      } catch (error) {
        showToast(error.message, true);
      } finally {
        document.querySelector("#analyze").disabled = false;
      }
    }

    function renderBars(distribution) {
      const bars = document.querySelector("#bars");
      const top = distribution.slice(0, 10);
      const max = Math.max(1, ...top.map((item) => item.count));
      bars.innerHTML = top.length ? top.map((item) => `
        <div class="bar-row">
          <span>${item.aspect} · ${item.sentiment}</span>
          <div class="track"><div class="fill" style="width:${Math.round((item.count / max) * 100)}%"></div></div>
          <strong>${item.count}</strong>
        </div>
      `).join("") : `<div class="toast">No database predictions yet.</div>`;
    }

    function renderRecent(rows) {
      const recent = document.querySelector("#recent");
      recent.innerHTML = rows.length ? rows.map((row) => {
        const preds = (row.predictions_json.predictions || []).map((item) => `<span class="tag">${item.aspect}: ${item.sentiment}</span>`).join("");
        return `<article class="recent-item">
          <div class="recent-text">${row.review_text}</div>
          <div class="tags">${preds}</div>
        </article>`;
      }).join("") : `<div class="toast">Predictions will appear here after your first run.</div>`;
    }

    async function loadDashboard() {
      const response = await fetch("/api/dashboard");
      const data = await response.json();
      document.querySelector("#totalPredictions").textContent = data.metrics.total_predictions;
      document.querySelector("#avgLatency").textContent = `${data.metrics.avg_latency_ms.toFixed(1)} ms`;
      document.querySelector("#successRate").textContent = `${(data.metrics.parse_success_rate * 100).toFixed(1)}%`;
      document.querySelector("#datasetRows").textContent = data.dataset.rows.toLocaleString();
      document.querySelector("#modelF1").textContent = data.model.holdout_f1_micro || "n/a";
      document.querySelector("#questionMarks").textContent = data.dataset.question_mark_cells;
      const db = data.database || {};
      document.querySelector("#dbStatus").textContent = db.sqlite_ready
        ? (db.mongo_ready ? "SQLite + MongoDB logging online" : "SQLite logging online; MongoDB offline")
        : "Database unavailable";
      renderBars(data.distribution);
      renderRecent(data.recent);
    }

    document.querySelector("#demos").innerHTML = demos.map((text, index) => `<button class="chip" data-index="${index}">Demo ${index + 1}</button>`).join("");
    document.querySelector("#demos").addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      review.value = demos[Number(button.dataset.index)];
      review.focus();
    });
    document.querySelector("#analyze").addEventListener("click", analyze);
    document.querySelector("#clear").addEventListener("click", () => {
      review.value = "";
      output.innerHTML = "";
      jsonPreview.textContent = '{"predictions":[]}';
      showToast("");
    });
    review.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
    });

    loadDashboard();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, text: str, content_type: str = "text/html; charset=utf-8") -> None:
    body = text.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _sanitize_review(value: object) -> str:
    text = str(value or "").replace("???", " ")
    return " ".join(text.split())


def _prediction_log_count() -> int:
    _ensure_phase3_tables(SQLITE_DB_PATH)
    with get_connection(SQLITE_DB_PATH) as connection:
        row = connection.execute("SELECT COUNT(*) AS total FROM prediction_logs").fetchone()
    return int(row["total"] if row is not None else 0)


def _mongo_ready(ttl_seconds: float = 10.0, force: bool = False) -> bool:
    now = time.monotonic()
    if not force and now - float(_MONGO_STATUS_CACHE["checked_at"]) < ttl_seconds:
        return bool(_MONGO_STATUS_CACHE["ready"])

    ready = init_mongo_db()
    _MONGO_STATUS_CACHE["checked_at"] = now
    _MONGO_STATUS_CACHE["ready"] = ready
    return ready


def _load_dataset_info() -> Dict[str, Any]:
    if not DATASET_PATH.exists():
        return {"path": str(DATASET_PATH), "rows": 0, "question_mark_cells": "missing"}

    frame = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    question_mark_cells = 0
    for column in frame.select_dtypes(include=["object"]).columns:
        question_mark_cells += int(frame[column].fillna("").astype(str).str.contains(r"\?\?\?", regex=True).sum())

    return {
        "path": str(DATASET_PATH),
        "name": DATASET_PATH.name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "question_mark_cells": int(question_mark_cells),
    }


def _load_model_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "weights": str(MODEL_WEIGHTS_PATH),
        "weights_exists": MODEL_WEIGHTS_PATH.exists(),
        "updated_at": "",
        "holdout_f1_micro": "",
    }
    if MODEL_WEIGHTS_PATH.exists():
        info["updated_at"] = datetime.fromtimestamp(MODEL_WEIGHTS_PATH.stat().st_mtime).isoformat(timespec="seconds")
    if MODEL_META_PATH.exists():
        try:
            meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            info.update(meta)
        except Exception:
            pass
    return info


def _read_prediction_logs(limit: int = 20) -> List[Dict[str, Any]]:
    _ensure_phase3_tables(SQLITE_DB_PATH)
    with get_connection(SQLITE_DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT review_text, prediction_json, parse_status, model_name, latency_ms, created_at
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()

    normalized: List[Dict[str, Any]] = []
    for row in rows:
        payload = row["prediction_json"]
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {"predictions": []}
        normalized.append(
            {
                "review_text": row["review_text"],
                "predictions_json": payload,
                "parse_status": row["parse_status"],
                "model_name": row["model_name"],
                "latency_ms": float(row["latency_ms"] or 0.0),
                "created_at": row["created_at"],
            }
        )
    return normalized


def _dashboard_payload() -> Dict[str, Any]:
    _ensure_phase3_tables(SQLITE_DB_PATH)
    ready = True
    try:
        with get_connection(SQLITE_DB_PATH) as connection:
            rows = connection.execute(
                "SELECT prediction_json, parse_status, latency_ms FROM prediction_logs"
            ).fetchall()
    except sqlite3.Error:
        rows = []
        ready = False

    total = len(rows)
    avg_latency = sum(float(row["latency_ms"] or 0.0) for row in rows) / total if total else 0.0
    success = sum(1 for row in rows if str(row["parse_status"] or "").lower() != "fallback")
    distribution: Counter[tuple[str, str]] = Counter()

    for row in rows:
        try:
            payload = json.loads(row["prediction_json"])
        except Exception:
            payload = {}
        for item in payload.get("predictions", []):
            aspect = str(item.get("aspect", "")).strip()
            sentiment = str(item.get("sentiment", "")).strip()
            if aspect and sentiment:
                distribution[(aspect, sentiment)] += 1

    mongo_settings = get_mongo_settings()

    return {
        "database": {
            "path": str(SQLITE_DB_PATH),
            "ready": ready,
            "sqlite_ready": ready,
            "mongo_ready": _mongo_ready(),
            "mongo_database": mongo_settings["database"],
        },
        "dataset": _load_dataset_info(),
        "model": _load_model_info(),
        "metrics": {
            "total_predictions": total,
            "avg_latency_ms": round(avg_latency, 2),
            "parse_success_rate": round(success / total, 4) if total else 0.0,
        },
        "distribution": [
            {"aspect": aspect, "sentiment": sentiment, "count": count}
            for (aspect, sentiment), count in distribution.most_common()
        ],
        "recent": _read_prediction_logs(15),
    }


class ABSAHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        print(f"[UI] {self.address_string()} - {format % args}", flush=True)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html = HTML.replace("__DEMOS__", json.dumps(DEMO_REVIEWS, ensure_ascii=False))
            _text_response(self, html)
            return
        if path == "/api/dashboard":
            _json_response(self, _dashboard_payload())
            return
        _json_response(self, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            _json_response(self, {"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
        except Exception:
            _json_response(self, {"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        review = _sanitize_review(payload.get("review"))
        if not review:
            _json_response(self, {"error": "Review text is required"}, status=HTTPStatus.BAD_REQUEST)
            return

        before_count = _prediction_log_count()
        start = time.perf_counter()
        result = predict_local(review, log=True)
        latency_ms = (time.perf_counter() - start) * 1000.0
        after_count = _prediction_log_count()
        sqlite_inserted = after_count > before_count
        _json_response(
            self,
            {
                "result": result,
                "latency_ms": round(latency_ms, 2),
                "database": {
                    "path": str(SQLITE_DB_PATH),
                    "sqlite_ready": True,
                    "sqlite_inserted": sqlite_inserted,
                    "rows_before": before_count,
                    "rows_after": after_count,
                    "mongo_ready": _mongo_ready(),
                    "mongo_database": get_mongo_settings()["database"],
                },
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    _ensure_phase3_tables(SQLITE_DB_PATH)
    server = ThreadingHTTPServer((args.host, args.port), ABSAHandler)
    print(f"[INFO] {APP_TITLE} running at http://{args.host}:{args.port}", flush=True)
    print(f"[INFO] Using dataset: {DATASET_PATH}", flush=True)
    print(f"[INFO] Using local weights: {MODEL_WEIGHTS_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down UI server.", flush=True)
    except BaseException as exc:
        print(f"[ERROR] UI server stopped unexpectedly: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        print("[INFO] UI server closing socket.", flush=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
