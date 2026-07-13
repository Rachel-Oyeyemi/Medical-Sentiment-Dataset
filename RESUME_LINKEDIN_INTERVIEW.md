# Resume, LinkedIn, and Interview Materials

## Resume Bullets
- Built an end-to-end seven-class mental-health text classification pipeline using word/character TF-IDF, class-weighted Logistic Regression, calibrated Linear SVM, stratified validation, macro-F1 evaluation, and leakage-aware deduplication.
- Developed a production-style Streamlit application, modular data and modeling code, automated tests, CI, notebooks, model reports, visualizations, and a 10-slide executive presentation.
- Translated a sensitive NLP use case into responsible-AI controls including non-diagnostic positioning, low-confidence abstention, human-review routing, privacy safeguards, calibration monitoring, and bias/drift recommendations.

## LinkedIn Project Description
I built a recruiter-ready Medical Sentiment NLP portfolio project using the Kaggle Sentiment Analysis for Mental Health dataset. The project treats the task as seven-class status-label classification rather than simple positive/negative sentiment. It compares TF-IDF Logistic Regression with a calibrated word-and-character Linear SVM, emphasizes macro F1 and class-level errors, and includes a complete Streamlit app, testing, CI, notebooks, executive documentation, and responsible-use guardrails. The application is explicitly educational and non-diagnostic.

## Interview Talking Points
- Why macro F1 and per-class recall matter for imbalanced multiclass NLP.
- How normalized-text deduplication prevents train/test leakage.
- Why character n-grams help with informal and misspelled language.
- Why calibrated probabilities still require abstention and monitoring.
- How source aggregation and labeling inconsistency limit generalization.
- Why a simpler sparse linear model can be preferable to a transformer in a portfolio deployment.

## Common Interview Questions and Sample Answers

**Why not optimize accuracy alone?**  
Normal and Depression dominate the source data, while Personality disorder is much smaller. Accuracy can look strong while minority-class recall is weak, so macro F1 and the confusion matrix are more informative.

**Can this model diagnose a mental-health condition?**  
No. It predicts dataset labels from language patterns. The labels are not a clinical assessment, and the source data does not support diagnostic use.

**How did you control leakage?**  
I normalize text before deduplication and remove repeated normalized statements prior to the stratified split. I also report identical text associated with conflicting labels.

**Why use a calibrated Linear SVM?**  
Linear SVM performs well on sparse high-dimensional text. Calibration provides probabilities for confidence thresholds, while the app still abstains and requests human review when confidence is low.

**What would you do next?**  
Run source-aware validation on the official data, evaluate a transformer challenger, test calibration and subgroup fairness, add out-of-distribution detection, and conduct prospective human-reviewed evaluation.
