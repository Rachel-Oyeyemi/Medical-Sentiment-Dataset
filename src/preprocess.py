"""Validate and preprocess mental-health status text data."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from utils import configure_logging, save_json

LOGGER = logging.getLogger(__name__)
TEXT_COLUMN = "statement"
TARGET_COLUMN = "status"
EXPECTED_LABELS = {
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Bipolar",
    "Stress",
    "Personality disorder",
}


def find_input(raw_dir: str | Path = "data/raw") -> Path:
    """Locate the official Combined_Data CSV or the deterministic demo file."""
    candidates = sorted(Path(raw_dir).glob("*.csv"))
    official = [path for path in candidates if "combined" in path.name.lower()]
    if official:
        return official[0]
    demo = [path for path in candidates if "demo" in path.name.lower()]
    if demo:
        return demo[0]
    if candidates:
        return candidates[0]
    sample = Path("data/sample_data/medical_sentiment_sample.csv")
    if sample.exists():
        return sample
    raise FileNotFoundError("No CSV found. Run download_data.py or generate_demo_data.py.")


def clean_text(value: object) -> str:
    """Normalize URLs, markup, whitespace, and casing while retaining negation."""
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"u/[A-Za-z0-9_-]+|r/[A-Za-z0-9_-]+", " USER ", text)
    text = re.sub(r"[^A-Za-z0-9\s'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {str(column).strip().lower(): column for column in frame.columns}
    if TEXT_COLUMN not in mapping or TARGET_COLUMN not in mapping:
        raise ValueError(f"Expected columns 'statement' and 'status'; received {list(frame.columns)}")
    return frame.rename(columns={mapping[TEXT_COLUMN]: TEXT_COLUMN, mapping[TARGET_COLUMN]: TARGET_COLUMN})


def preprocess_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean text, remove unusable records, deduplicate normalized text, and report quality."""
    source = _normalize_columns(frame.copy())
    input_rows = len(source)
    missing_statement = int(source[TEXT_COLUMN].isna().sum())
    missing_status = int(source[TARGET_COLUMN].isna().sum())
    source = source.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    source[TARGET_COLUMN] = source[TARGET_COLUMN].astype(str).str.strip()
    source["clean_text"] = source[TEXT_COLUMN].map(clean_text)
    source["text_length"] = source["clean_text"].str.len()
    source["word_count"] = source["clean_text"].str.split().str.len()
    source["digit_count"] = source["clean_text"].str.count(r"\d")
    source["unique_word_ratio"] = source["clean_text"].map(
        lambda text: len(set(text.split())) / max(len(text.split()), 1)
    )

    empty_after_cleaning = int((source["word_count"] < 2).sum())
    source = source[source["word_count"] >= 2].copy()
    duplicate_rows = int(source.duplicated(subset=["clean_text", TARGET_COLUMN]).sum())
    conflicting_texts = int((source.groupby("clean_text")[TARGET_COLUMN].nunique() > 1).sum())
    source = source.sort_index().drop_duplicates(subset=["clean_text"], keep="first")
    source = source[source[TARGET_COLUMN].isin(EXPECTED_LABELS)].reset_index(drop=True)

    q1 = float(source["text_length"].quantile(0.25)) if len(source) else 0.0
    q3 = float(source["text_length"].quantile(0.75)) if len(source) else 0.0
    upper = q3 + 1.5 * (q3 - q1)
    long_text_outliers = int((source["text_length"] > upper).sum())

    quality = {
        "input_rows": int(input_rows),
        "input_columns": int(frame.shape[1]),
        "missing_statement": missing_statement,
        "missing_status": missing_status,
        "empty_after_cleaning": empty_after_cleaning,
        "normalized_duplicate_rows_removed": duplicate_rows,
        "conflicting_label_texts": conflicting_texts,
        "output_rows": int(len(source)),
        "labels": {str(key): int(value) for key, value in source[TARGET_COLUMN].value_counts().items()},
        "text_length_iqr_upper": upper,
        "long_text_outliers": long_text_outliers,
    }
    return source, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="data/processed/clean_text.csv")
    args = parser.parse_args()
    configure_logging()
    input_path = Path(args.input) if args.input else find_input()
    frame = pd.read_csv(input_path, low_memory=False)
    clean, quality = preprocess_dataframe(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    save_json(quality, "reports/data_quality.json")
    LOGGER.info("Saved %d cleaned rows from %s", len(clean), input_path)


if __name__ == "__main__":
    main()
