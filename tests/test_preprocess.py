import sys

import pandas as pd

sys.path.append("src")
from preprocess import clean_text, preprocess_dataframe


def test_clean_text_preserves_negation():
    assert clean_text("I can't cope!!!") == "i can't cope"


def test_preprocess_removes_normalized_duplicates():
    frame = pd.DataFrame(
        {
            "statement": ["I feel okay!", "I feel okay", "I worry a lot"],
            "status": ["Normal", "Normal", "Anxiety"],
        }
    )
    clean, quality = preprocess_dataframe(frame)
    assert len(clean) == 2
    assert quality["normalized_duplicate_rows_removed"] == 1


def test_preprocess_rejects_wrong_schema():
    frame = pd.DataFrame({"text": ["hello"], "label": ["Normal"]})
    try:
        preprocess_dataframe(frame)
    except ValueError as exc:
        assert "statement" in str(exc)
    else:
        raise AssertionError("Expected schema validation error")
