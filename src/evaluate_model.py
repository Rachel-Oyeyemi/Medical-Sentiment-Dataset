"""Evaluate multiclass NLP classifiers on validation and untouched test data."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from utils import save_json


def evaluate(model, X: pd.Series, y: pd.Series, model_name: str, split: str) -> tuple[dict, pd.DataFrame]:
    """Return aggregate metrics and a per-class classification report."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    labels = list(model.classes_)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y, predictions, average="weighted", zero_division=0
    )[2]
    y_binary = label_binarize(y, classes=labels)
    metrics = {
        "model": model_name,
        "split": split,
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc_ovr_macro": float(
            roc_auc_score(y_binary, probabilities, multi_class="ovr", average="macro")
        ),
        "log_loss": float(log_loss(y, probabilities, labels=labels)),
        "low_confidence_rate_below_060": float((probabilities.max(axis=1) < 0.60).mean()),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "labels": labels,
    }
    report = pd.DataFrame(
        classification_report(y, predictions, labels=labels, output_dict=True, zero_division=0)
    ).T.reset_index(names="class")
    report.insert(0, "model", model_name)
    report.insert(1, "split", split)
    return metrics, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    directory = Path(args.models_dir)
    split = joblib.load(directory / "evaluation_split.joblib")
    models = {
        "TF-IDF + Logistic Regression": joblib.load(
            directory / "baseline_tfidf_logistic_regression.joblib"
        ),
        "TF-IDF + Calibrated Linear SVM": joblib.load(
            directory / "advanced_tfidf_linear_svm.joblib"
        ),
    }

    aggregate: list[dict] = []
    reports: list[pd.DataFrame] = []
    for name, model in models.items():
        for split_name in ["validation", "test"]:
            metrics, report = evaluate(
                model,
                split[f"X_{split_name}"],
                split[f"y_{split_name}"],
                name,
                split_name,
            )
            aggregate.append(metrics)
            reports.append(report)

    validation = [row for row in aggregate if row["split"] == "validation"]
    recommended = max(validation, key=lambda row: row["macro_f1"])["model"]
    save_json(
        {
            "benchmark_scope": "Deterministic overlapping synthetic demo corpus; rerun on the official Kaggle CSV for final results.",
            "primary_metric": "macro_f1",
            "models": aggregate,
            "recommended_model": recommended,
            "safety_decision": "Educational text-classification demonstration only; never use for diagnosis, treatment, or crisis triage.",
        },
        "reports/benchmark_metrics.json",
    )
    pd.concat(reports, ignore_index=True).to_csv(
        "reports/classification_reports.csv", index=False
    )
    pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"confusion_matrix", "labels"}} for row in aggregate]
    ).to_csv("reports/model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
