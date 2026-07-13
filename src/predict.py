"""Prediction helper for the Medical Sentiment text-classification demo."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocess import clean_text


def predict_status(
    text: str,
    model_path: str | Path = "models/advanced_tfidf_linear_svm.joblib",
) -> dict:
    """Return a predicted class, confidence, and probability distribution."""
    normalized = clean_text(text)
    if len(normalized.split()) < 2:
        raise ValueError("Please enter at least two meaningful words.")
    model = joblib.load(model_path)
    probabilities = model.predict_proba(pd.Series([normalized]))[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda row: row[1], reverse=True)
    confidence = float(ranked[0][1])
    return {
        "prediction": str(ranked[0][0]),
        "confidence": confidence,
        "review_recommended": confidence < 0.60,
        "probabilities": {str(label): float(value) for label, value in ranked},
        "safety": "This result is not a diagnosis or crisis assessment. Sensitive content requires qualified human review.",
    }
