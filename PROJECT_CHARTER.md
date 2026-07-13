# Project Charter — Medical Sentiment Dataset

## Business Problem
High-volume user-generated mental-health text is difficult to summarize consistently. This project demonstrates how broad dataset labels can support research analytics and qualified human-review queues without claiming clinical diagnosis or autonomous crisis detection.

## Objectives
- Build a reproducible seven-class NLP pipeline.
- Compare an interpretable baseline with a calibrated advanced classifier.
- Prioritize macro F1, macro recall, and class-level confusion patterns.
- Demonstrate abstention, human review, privacy, and responsible-use controls.
- Produce recruiter-ready code, documentation, Streamlit, tests, and executive communication.

## Stakeholders
Data-science hiring managers, NLP engineers, responsible-AI teams, trust-and-safety reviewers, digital-health analytics teams, product managers, and qualified human reviewers.

## Success Metrics
Macro F1 is primary. Supporting measures include accuracy, macro precision, macro recall, weighted F1, one-vs-rest macro ROC-AUC, log loss, per-class metrics, low-confidence rate, reproducibility, and documentation quality.

## Expected Business Impact
Potential non-clinical value includes content trend analysis, quality-assurance sampling, and prioritization for human review. The project must not be used for diagnosis, treatment recommendations, eligibility decisions, or crisis intervention.

## Technical Architecture
Kaggle CSV or deterministic demo → schema and quality validation → privacy-conscious normalization → duplicate-leakage controls → TF-IDF features → Logistic Regression / calibrated Linear SVM → stratified validation and test → metrics and error analysis → Streamlit demonstration → governance recommendations.

## End-to-End Workflow
1. Download or generate data.
2. Validate labels, nulls, duplicates, conflicts, and text length.
3. Clean text while preserving negation.
4. Create stratified train, validation, and test partitions.
5. Train baseline and advanced models.
6. Select by validation macro F1 and report untouched test metrics.
7. Generate reports, visualizations, app, tests, and presentation.
