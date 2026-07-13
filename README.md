# Medical Sentiment Dataset — Responsible Mental-Health NLP

[![CI](https://github.com/Rachel-Oyeyemi/Medical-Sentiment-Dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/Rachel-Oyeyemi/Medical-Sentiment-Dataset/actions)

A recruiter-ready machine-learning portfolio project for seven-class mental-health status text classification using leakage-aware preprocessing, word and character TF-IDF, Logistic Regression, calibrated Linear SVM, multiclass evaluation, Streamlit, automated tests, executive reporting, and responsible-AI guardrails.

> **Responsible-use notice:** This project is educational. It is not a medical device, diagnosis system, treatment tool, crisis service, or substitute for qualified human professionals.

## Project Overview
Despite the Kaggle title, this is not ordinary positive/negative sentiment analysis. The target is one of seven labels: **Normal, Depression, Suicidal, Anxiety, Bipolar, Stress,** or **Personality disorder**.

The portfolio question is: **Can a reproducible sparse-NLP pipeline classify the dataset's broad labels while clearly communicating uncertainty, leakage risk, privacy limits, and non-clinical use?**

## Dataset Source and Profile
- Kaggle dataset: `suchintikasarkar/sentiment-analysis-for-mental-health`
- Raw file commonly named `Combined_Data.csv`
- Published reproduction: 53,043 rows and 3 columns
- Useful columns: `statement` and `status`; the source index is excluded
- Published missing statements: 362
- Published usable rows after null and empty-text handling: 52,543
- Seven-class target with substantial imbalance

| Label | Raw rows | Share |
|---|---:|---:|
| Normal | 16,351 | 30.83% |
| Depression | 15,404 | 29.04% |
| Suicidal | 10,653 | 20.08% |
| Anxiety | 3,888 | 7.33% |
| Bipolar | 2,877 | 5.42% |
| Stress | 2,669 | 5.03% |
| Personality disorder | 1,201 | 2.26% |

The official CSV is not committed. The repository includes authenticated Kaggle download support and a deterministic overlapping synthetic demo corpus so the pipeline and app run immediately.

## Business Problem
Large volumes of mental-health-related text can be difficult to summarize and prioritize consistently. This project demonstrates non-clinical content analytics and human-review support—not automated diagnosis or crisis triage.

## Methodology
1. Download the official data or generate the deterministic demo corpus.
2. Validate schema, missing text, label values, normalized duplicates, conflicting labels, and text-length outliers.
3. Normalize URLs and markup while preserving negation.
4. Remove normalized-text duplicates before splitting to reduce leakage.
5. Create stratified 70% training, 15% validation, and 15% test partitions.
6. Train a word TF-IDF Logistic Regression baseline.
7. Train an advanced word/character TF-IDF calibrated Linear SVM.
8. Select the preferred model using validation macro F1.
9. Report untouched test accuracy, macro precision/recall/F1, weighted F1, ROC-AUC, log loss, low-confidence rate, and confusion matrices.
10. Package the Streamlit app, tests, CI, notebooks, reports, visuals, and executive presentation.

## Models
### Baseline
**Word TF-IDF + class-weighted Logistic Regression** provides a transparent and efficient sparse-text benchmark.

### Advanced
**Word/character TF-IDF + calibrated Linear SVM** captures phrase structure and spelling variation while producing probability estimates for confidence-based abstention.

## Demonstration Results
The committed metrics come from a deliberately overlapping synthetic corpus with class imbalance and annotation noise. They prove that the pipeline runs; they are not official Kaggle or clinical performance.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | Log Loss | Low Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.872 | 0.820 | 0.822 | 0.816 | 0.931 | 0.589 | 22.4% |
| TF-IDF + Calibrated Linear SVM | 0.932 | 0.938 | 0.865 | 0.893 | 0.932 | 0.452 | 3.2% |

**Validation-selected demo model:** TF-IDF + Calibrated Linear SVM. These are synthetic demonstration results, not official Kaggle or clinical metrics.

## Business Impact
Potential value is limited to properly governed research analytics, content trend summaries, and qualified human-review prioritization. Real deployment requires prospective validation, privacy controls, source-aware evaluation, calibration, subgroup testing, monitoring, and human override.

## Repository Structure
```text
├── app/                       # Multi-page Streamlit app
├── data/raw/                  # Authenticated Kaggle CSV location
├── data/processed/            # Generated clean dataset
├── data/sample_data/          # Small synthetic preview
├── docs/                      # Architecture and model card
├── models/                    # Reproducible artifact instructions and metadata
├── notebooks/                 # Four end-to-end notebooks
├── presentation/              # 10-slide deck, generator, and notes
├── reports/                   # Quality, aggregate, and class-level metrics
├── src/                       # Production-style pipeline modules
├── tests/                     # Unit and smoke tests
├── visuals/                   # EDA, evaluation, and governance charts
├── PROJECT_CHARTER.md
├── EDA_REPORT.md
├── MODEL_COMPARISON.md
├── MODEL_EVALUATION.md
├── BUSINESS_RECOMMENDATIONS.md
├── RESUME_LINKEDIN_INTERVIEW.md
├── run_pipeline.py
└── requirements.txt
```

## How to Run
```bash
git clone https://github.com/Rachel-Oyeyemi/Medical-Sentiment-Dataset.git
cd Medical-Sentiment-Dataset
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Immediate deterministic demonstration
python run_pipeline.py

# Official Kaggle dataset
python src/download_data.py --source auto
python src/preprocess.py --input data/raw/Combined_Data.csv
python src/train_model.py
python src/evaluate_model.py
python src/generate_visuals.py

streamlit run app/app.py
pytest -q
```

## Responsible Use
Do not enter private or identifying patient information into the demonstration. Never interpret a predicted label as a diagnosis or safety determination. Maintain a separate qualified-human process for real urgent concerns.

## Future Improvements
- Source-aware and group-aware validation
- Transformer challenger with calibrated probabilities
- Selective classification and abstention optimization
- Out-of-distribution detection
- Bias and subgroup evaluation
- Privacy-preserving training and access controls
- Drift and annotation-quality monitoring
