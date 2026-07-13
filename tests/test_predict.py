import sys

sys.path.append("src")
from train_model import build_baseline


def test_model_pipeline_has_probability_output():
    model = build_baseline()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")
