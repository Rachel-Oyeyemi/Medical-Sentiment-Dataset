"""Train baseline and advanced multiclass mental-health text classifiers."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from feature_engineering import TARGET, TEXT_FEATURE
from utils import configure_logging, save_json

LOGGER = logging.getLogger(__name__)


def split_data(frame: pd.DataFrame):
    """Create reproducible 70/15/15 stratified train, validation, and test splits."""
    train, temporary = train_test_split(
        frame, test_size=0.30, random_state=42, stratify=frame[TARGET]
    )
    validation, test = train_test_split(
        temporary, test_size=0.50, random_state=42, stratify=temporary[TARGET]
    )
    return train, validation, test


def build_baseline() -> Pipeline:
    """Return an interpretable word TF-IDF plus balanced Logistic Regression baseline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=25_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    C=2.0,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def build_advanced() -> Pipeline:
    """Return a word-and-character TF-IDF calibrated Linear SVM classifier."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.99,
                    max_features=35_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=25_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("select", SelectKBest(chi2, k="all")),
            (
                "model",
                CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", C=1.2, random_state=42),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/clean_text.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    configure_logging()

    frame = pd.read_csv(args.input)
    train, validation, test = split_data(frame)
    output = Path(args.models_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline = build_baseline().fit(train[TEXT_FEATURE], train[TARGET])
    advanced = build_advanced().fit(train[TEXT_FEATURE], train[TARGET])
    joblib.dump(baseline, output / "baseline_tfidf_logistic_regression.joblib", compress=3)
    joblib.dump(advanced, output / "advanced_tfidf_linear_svm.joblib", compress=3)
    joblib.dump(
        {
            "X_validation": validation[TEXT_FEATURE],
            "y_validation": validation[TARGET],
            "X_test": test[TEXT_FEATURE],
            "y_test": test[TARGET],
        },
        output / "evaluation_split.joblib",
        compress=3,
    )
    save_json(
        {
            "target": TARGET,
            "text_feature": TEXT_FEATURE,
            "classes": sorted(frame[TARGET].unique().tolist()),
            "split_strategy": "stratified 70/15/15",
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "random_seed": 42,
            "responsible_use": "Educational research and portfolio demonstration only; not diagnostic or crisis triage.",
        },
        output / "model_metadata.json",
    )
    LOGGER.info("Saved baseline and advanced models to %s", output)


if __name__ == "__main__":
    main()
