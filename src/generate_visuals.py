"""Generate EDA and model-evaluation visualizations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VISUALS = ROOT / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(VISUALS / name, format="svg", bbox_inches="tight")
    plt.close()


def main() -> None:
    frame = pd.read_csv(ROOT / "data/processed/clean_text.csv")
    metrics = json.loads((ROOT / "reports/benchmark_metrics.json").read_text())
    test_rows = [row for row in metrics["models"] if row["split"] == "test"]

    frame["status"].value_counts().sort_values().plot.barh(title="Demo Class Distribution")
    plt.xlabel("Rows")
    _save("class_distribution.svg")

    frame["text_length"].plot.hist(bins=35, title="Text Length Distribution")
    plt.xlabel("Characters")
    _save("text_length_distribution.svg")

    frame["word_count"].plot.hist(bins=30, title="Word Count Distribution")
    plt.xlabel("Words")
    _save("word_count_distribution.svg")

    pd.Series(
        {
            "statement": int(frame["statement"].isna().sum()),
            "status": int(frame["status"].isna().sum()),
        }
    ).plot.bar(title="Missing Values After Preprocessing")
    plt.ylabel("Missing rows")
    _save("missing_values.svg")

    numeric = frame[["text_length", "word_count", "digit_count", "unique_word_ratio"]].corr()
    plt.imshow(numeric, aspect="auto")
    plt.xticks(range(len(numeric)), numeric.columns, rotation=35, ha="right")
    plt.yticks(range(len(numeric)), numeric.columns)
    plt.colorbar(label="Correlation")
    plt.title("Engineered Text Feature Correlation")
    _save("text_feature_correlation.svg")

    comparison = pd.DataFrame(test_rows).set_index("model")
    comparison[["macro_f1", "macro_recall", "accuracy"]].plot.bar(title="Test Model Comparison")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xticks(rotation=10, ha="right")
    _save("model_comparison.svg")

    advanced = next(row for row in test_rows if "SVM" in row["model"])
    matrix = np.array(advanced["confusion_matrix"])
    labels = advanced["labels"]
    plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Advanced Model Confusion Matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
    _save("confusion_matrix.svg")

    report = pd.read_csv(ROOT / "reports/classification_reports.csv")
    per_class = report[(report["model"].str.contains("SVM")) & (report["split"] == "test")]
    per_class = per_class[per_class["class"].isin(labels)].set_index("class")
    per_class["f1-score"].sort_values().plot.barh(title="Advanced Model F1 by Class")
    plt.xlim(0, 1)
    plt.xlabel("F1 score")
    _save("per_class_f1.svg")

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english", min_df=4, max_features=1000)
    term_matrix = vectorizer.fit_transform(frame["clean_text"])
    scores = np.asarray(term_matrix.mean(axis=0)).ravel()
    names = np.asarray(vectorizer.get_feature_names_out())
    pd.Series(scores, index=names).nlargest(15).sort_values().plot.barh(
        title="Most Frequent Demo TF-IDF Terms"
    )
    plt.xlabel("Mean TF-IDF")
    _save("top_terms.svg")

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.axis("off")
    boxes = [
        (0.02, "Text received"),
        (0.22, "Model score"),
        (0.42, "Confidence check"),
        (0.64, "Human review"),
        (0.84, "Non-clinical action"),
    ]
    for x, label in boxes:
        axis.text(
            x,
            0.5,
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white"},
        )
    for start, end in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(end[0] - 0.08, 0.5),
            xytext=(start[0] + 0.08, 0.5),
            xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->"},
        )
    axis.set_title("Responsible Human-in-the-Loop Workflow")
    _save("safety_workflow.svg")


if __name__ == "__main__":
    main()
