"""Phase 4 Streamlit dashboard for Arabic ABSA with MongoDB logging."""

from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from DataBase.mongo_db import get_quality_metrics, get_recent, get_stats, init_db, log_prediction
from scripts.groq_engine import predict

DEMOS: List[str] = [
    "الطعام كان ممتازا لكن الخدمة بطيئة جدا",
    "التطبيق صعب الاستخدام وبطيء",
    "السعر مناسب والمكان نظيف",
    "التوصيل تأخر ساعتين والطلب وصل بارد",
    "الأجواء رائعة والطعام لذيذ لكن السعر مرتفع",
]

SENTIMENT_COLORS: Dict[str, str] = {
    "positive": "#1D9E75",
    "negative": "#E24B4A",
    "neutral": "#888780",
}


def _style_prediction_table(df: pd.DataFrame):
    def colorize(series: pd.Series):
        color = SENTIMENT_COLORS.get(str(series.get("sentiment", "")).lower(), "#444444")
        return [f"background-color: {color}; color: white" if col == "sentiment" else "" for col in series.index]

    return df.style.apply(colorize, axis=1)


def _app() -> None:
    st.set_page_config(page_title="ABSA · Arabic Reviews", layout="wide")

    mongo_ok = init_db()
    st.title("Arabic Aspect-Based Sentiment Analysis")
    st.caption("Powered by Groq + Dynamic RAG")
    if mongo_ok:
        st.success("MongoDB connected: mongodb://localhost:27017/")
    else:
        st.warning("MongoDB not reachable. UI still works, but Mongo logging is unavailable.")

    metrics = get_quality_metrics()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Predictions", int(metrics.get("total_predictions", 0)))
    m2.metric("Avg Latency (ms)", f"{metrics.get('avg_latency_ms', 0):.2f}")
    m3.metric("Parse Success Rate", f"{metrics.get('parse_success_rate', 0) * 100:.1f}%")

    if "_review_text" not in st.session_state:
        st.session_state["_review_text"] = ""

    cols = st.columns(len(DEMOS))
    for idx, demo in enumerate(DEMOS):
        if cols[idx].button(f"Demo {idx + 1}"):
            st.session_state["_review_text"] = demo
            st.rerun()

    review_input = st.text_area(
        "Enter Arabic review:",
        value=st.session_state["_review_text"],
        height=120,
        key="review_input_widget",
    )
    # Sync typed value back to backing state so it persists across reruns
    st.session_state["_review_text"] = review_input

    if st.button("Analyze", type="primary"):
        review = review_input.strip()
        if not review:
            st.error("Please enter a review first.")
        else:
            with st.spinner("Analyzing..."):
                t0 = time.time()
                result = predict(review)
                latency_ms = int((time.time() - t0) * 1000)

            predictions_df = pd.DataFrame(result.get("predictions", []))
            if predictions_df.empty:
                predictions_df = pd.DataFrame([{"aspect": "general", "sentiment": "neutral"}])

            st.subheader("Prediction Output")
            st.dataframe(_style_prediction_table(predictions_df), use_container_width=True)
            st.caption(f"Latency: {latency_ms} ms")

            parse_status = "ok" if len(predictions_df) > 0 else "fallback"
            model_version = "llama-3.3-70b-versatile"
            log_prediction(
                review_text=review,
                result=result,
                latency_ms=latency_ms,
                parse_status=parse_status,
                model_version=model_version,
            )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Recent Predictions")
        recent_df = get_recent(20)
        st.dataframe(recent_df, use_container_width=True)

    with right:
        st.subheader("Class Distribution")
        stats_df = get_stats()
        if stats_df.empty:
            st.info("No prediction stats yet.")
        else:
            fig = px.bar(
                stats_df,
                x="aspect",
                y="count",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    _app()
