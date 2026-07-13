# Technical Architecture

```text
Kaggle authentication / deterministic demo
                  ↓
Schema, label, null, duplicate, and conflict validation
                  ↓
Privacy-conscious text normalization preserving negation
                  ↓
Stratified 70% train / 15% validation / 15% test
                  ↓
Baseline: word TF-IDF + Logistic Regression
Advanced: word + character TF-IDF + calibrated Linear SVM
                  ↓
Macro metrics, ROC-AUC, log loss, confidence, confusion matrix
                  ↓
Streamlit demonstration + abstention + human-review messaging
                  ↓
Monitoring, privacy, bias, and governance recommendations
```
