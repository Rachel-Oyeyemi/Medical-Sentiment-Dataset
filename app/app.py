"""Streamlit portfolio application for mental-health status text classification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_status
from preprocess import preprocess_dataframe
from train_model import build_advanced

st.set_page_config(page_title="Medical Sentiment NLP", page_icon="🧠", layout="wide")


@st.cache_resource
def load_or_train() -> tuple[Path, str]:
    """Load a generated demo model or train a deterministic local fallback."""
    model_path = ROOT / "models/advanced_tfidf_linear_svm.joblib"
    if model_path.exists():
        return model_path, "Generated deterministic demo artifact"
    sample = pd.read_csv(ROOT / "data/sample_data/medical_sentiment_sample.csv")
    clean, _ = preprocess_dataframe(sample)
    model = build_advanced().fit(clean["clean_text"], clean["status"])
    cached = ROOT / "models/_cached_advanced.joblib"
    cached.parent.mkdir(exist_ok=True)
    joblib.dump(model, cached)
    return cached, "Cached sample fallback"


MODEL_PATH, SCOPE = load_or_train()
METRICS_PATH = ROOT / "reports/benchmark_metrics.json"
METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Project Overview",
        "Prediction Interface",
        "Model Performance",
        "Visualizations",
        "Business Insights",
        "About",
    ],
)
st.sidebar.error("Educational portfolio demo only — not diagnosis, treatment, or crisis triage.")
st.sidebar.caption(SCOPE)

if page == "Home":
    st.title("Medical Sentiment Dataset: Mental-Health Status Classification")
    columns = st.columns(4)
    columns[0].metric("Official raw rows", "53,043")
    columns[1].metric("Target classes", "7")
    columns[2].metric("Primary metric", "Macro F1")
    columns[3].metric("Deployment", "Human review support")
    st.warning(
        "This application cannot determine whether someone has a mental-health condition or is in immediate danger. "
        "If there is an urgent safety concern, contact local emergency services or a qualified human professional."
    )

elif page == "Project Overview":
    st.header("Project Overview")
    st.markdown(
        """
        The Kaggle dataset combines user-generated statements labeled as **Normal, Depression, Suicidal,
        Anxiety, Bipolar, Stress,** or **Personality disorder**. Despite the dataset title, this is a
        seven-class status-label classification task—not ordinary positive/negative sentiment analysis.

        The project compares an interpretable word TF-IDF Logistic Regression baseline with a calibrated
        word-and-character Linear SVM. All outputs are framed as research and portfolio demonstrations.
        """
    )

elif page == "Prediction Interface":
    st.header("Text Classification Demonstration")
    st.caption("Do not enter private, identifying, or real patient information.")
    text = st.text_area(
        "Enter synthetic or non-identifying example text",
        height=170,
        value="I have felt overwhelmed and my thoughts keep racing this week.",
    )
    if st.button("Classify text", use_container_width=True):
        try:
            result = predict_status(text, MODEL_PATH)
            st.metric("Predicted dataset label", result["prediction"])
            st.metric("Model confidence", f"{result['confidence']:.1%}")
            if result["review_recommended"]:
                st.warning("Low confidence: abstain from automated use and request qualified human review.")
            if result["prediction"] == "Suicidal":
                st.error(
                    "The model matched the dataset's Suicidal label. This is not a risk assessment. "
                    "Any real safety concern requires immediate qualified human support."
                )
            st.dataframe(
                pd.DataFrame(result["probabilities"].items(), columns=["Class", "Probability"]),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(result["safety"])
        except ValueError as exc:
            st.warning(str(exc))

elif page == "Model Performance":
    st.header("Model Performance")
    rows = METRICS.get("models", [])
    if rows:
        display = pd.DataFrame(
            [{key: value for key, value in row.items() if key not in {"confusion_matrix", "labels"}} for row in rows]
        )
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.markdown(f"**Validation-selected model:** {METRICS.get('recommended_model')}")
        st.caption(METRICS.get("benchmark_scope", ""))
    st.image(str(ROOT / "visuals/model_comparison.svg"), use_container_width=True)
    st.image(str(ROOT / "visuals/per_class_f1.svg"), use_container_width=True)

elif page == "Visualizations":
    st.header("Exploratory and Evaluation Visualizations")
    for name in [
        "class_distribution.svg",
        "text_length_distribution.svg",
        "word_count_distribution.svg",
        "missing_values.svg",
        "text_feature_correlation.svg",
        "top_terms.svg",
        "confusion_matrix.svg",
        "safety_workflow.svg",
    ]:
        path = ROOT / "visuals" / name
        if path.exists():
            st.image(path, caption=name.replace("_", " ").replace(".svg", "").title(), use_container_width=True)

elif page == "Business Insights":
    st.header("Business and Responsible-AI Insights")
    st.markdown(
        """
        1. Use macro recall and macro F1 so smaller classes are not hidden by majority-class accuracy.
        2. Provide an abstain/review state for low-confidence predictions.
        3. Never use model labels as diagnoses or autonomous crisis decisions.
        4. Test source leakage, duplicates, demographic bias, calibration, and drift before any real deployment.
        5. Minimize retained text and apply strict privacy, access, and consent controls.
        """
    )

else:
    st.header("About")
    st.markdown(
        "Built by **Rachel Oyeyemi** as a recruiter-ready Data Analytics & AI portfolio project demonstrating "
        "NLP engineering, multiclass evaluation, deployment, testing, executive communication, and responsible AI."
    )
