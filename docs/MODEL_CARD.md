# Model Card — Medical Sentiment NLP Classifier

## Intended Use
Educational portfolio demonstration, NLP research workflow, and non-clinical human-review prototyping using synthetic or properly governed text.

## Prohibited Use
Diagnosis, treatment decisions, crisis assessment, emergency response, insurance/employment/education eligibility, autonomous moderation, or any use involving identifiable patient information.

## Model Family
Baseline word TF-IDF Logistic Regression and advanced calibrated Linear SVM using word and character TF-IDF features.

## Metrics
Macro F1 is primary. Macro recall, accuracy, weighted F1, multiclass ROC-AUC, log loss, low-confidence rate, per-class metrics, and confusion matrices are reported.

## Limitations
Source aggregation, uncertain annotation practices, class imbalance, duplicate language, demographic and platform bias, domain shift, indirect labels, probability miscalibration, and lack of clinical validation.

## Human Oversight
Low-confidence predictions should abstain. Sensitive content requires qualified human review through a separate governed workflow.
