# Model Comparison

## Baseline — Word TF-IDF + Logistic Regression
A strong sparse-text baseline that is fast, class-weighted, easy to inspect, and suitable for explaining coefficient-based language associations. It establishes whether a more complex text representation creates meaningful improvement.

## Advanced — Word/Character TF-IDF + Calibrated Linear SVM
The advanced model combines word n-grams with character n-grams, improving robustness to spelling variation, informal writing, and short phrases. Linear SVM is effective for high-dimensional sparse text, while sigmoid calibration supplies probabilities for confidence thresholds and abstention.

## Selection Method
Models are trained on 70% of the data. Validation macro F1 selects the preferred model. Final metrics are reported on a separate 15% test split. Exact normalized text is deduplicated before splitting to reduce leakage.

## Recommended Metrics
- Primary: macro F1
- Safety-sensitive support: macro recall and per-class recall
- Overall context: accuracy and weighted F1
- Ranking: multiclass one-vs-rest macro ROC-AUC
- Calibration: log loss and low-confidence rate
- Error analysis: confusion matrix and class-level report

## Advanced Transformer Path
A carefully governed transformer such as DistilBERT or DeBERTa is a reasonable future challenger. It is not committed as the default model because reproducibility, resource cost, calibration, privacy review, and responsible deployment matter more than maximizing a single score.
