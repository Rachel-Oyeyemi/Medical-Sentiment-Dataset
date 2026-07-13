import sys

import pandas as pd

sys.path.append("src")
from train_model import build_baseline, split_data


def test_baseline_fits_multiclass_text():
    rows = []
    for label, phrase in [
        ("Normal", "calm ordinary day"),
        ("Anxiety", "racing worry panic"),
        ("Stress", "deadlines workload pressure"),
    ]:
        for index in range(10):
            rows.append({"clean_text": f"{phrase} example {index}", "status": label})
    frame = pd.DataFrame(rows)
    train, validation, test = split_data(frame)
    model = build_baseline().fit(train["clean_text"], train["status"])
    assert set(model.classes_) == {"Normal", "Anxiety", "Stress"}
    assert len(model.predict(test["clean_text"])) == len(test)
