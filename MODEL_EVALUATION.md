# Model Evaluation

## Evaluation Design
The deterministic demonstration uses a stratified 70% training, 15% validation, and 15% test split after normalized-text deduplication. Validation macro F1 selects the preferred model; the test split is retained for final reporting.

## Metrics
- Accuracy
- Macro precision
- Macro recall
- Macro F1 — primary
- Weighted F1
- One-vs-rest macro ROC-AUC
- Log loss
- Percentage of predictions below 60% confidence
- Per-class precision, recall, F1, and support
- Confusion matrix

| Model | Split | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC | Log Loss | Low Confidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | validation | 0.859 | 0.795 | 0.816 | 0.805 | 0.860 | 0.923 | 0.627 | 24.4% |
| TF-IDF + Logistic Regression | test | 0.872 | 0.820 | 0.822 | 0.816 | 0.873 | 0.931 | 0.589 | 22.4% |
| TF-IDF + Calibrated Linear SVM | validation | 0.921 | 0.924 | 0.842 | 0.874 | 0.918 | 0.920 | 0.486 | 3.1% |
| TF-IDF + Calibrated Linear SVM | test | 0.932 | 0.938 | 0.865 | 0.893 | 0.930 | 0.932 | 0.452 | 3.2% |

**Validation-selected model:** TF-IDF + Calibrated Linear SVM

## Interpretation
The demo corpus intentionally contains overlapping wording and 7% annotation noise, so perfect performance is neither expected nor desirable. Results demonstrate comparative model behavior and the need for abstention and human review. Official Kaggle metrics must be regenerated after authenticated download.

## Safety Decision
No model in this repository is approved for diagnosis, treatment, emergency response, or autonomous crisis triage.
