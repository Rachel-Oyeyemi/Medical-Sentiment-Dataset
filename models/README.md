# Model Artifacts

Run `python run_pipeline.py` to generate:

- `baseline_tfidf_logistic_regression.joblib`
- `advanced_tfidf_linear_svm.joblib`
- `evaluation_split.joblib`

Binary artifacts are reproducible from the committed deterministic demo generator and are excluded from the public Git tree to keep the repository lightweight. `model_metadata.json` documents the benchmark configuration. The Streamlit application trains a cached fallback when a generated model is not present.
