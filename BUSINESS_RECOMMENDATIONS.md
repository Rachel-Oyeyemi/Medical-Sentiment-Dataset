# Business Recommendations

## Executive Summary
The project demonstrates a robust seven-class NLP workflow, but the dataset labels must be treated as broad content categories—not diagnoses. Any real application should be human-in-the-loop, privacy-preserving, and capable of abstaining.

## Key Findings
- Class imbalance makes macro metrics essential.
- Duplicate text can create severe train/test leakage.
- Character features can improve resilience to informal language.
- Probability output is useful only when calibration and abstention are monitored.
- Source aggregation creates uncertainty about label consistency and generalization.

## Recommendations
1. Position the system as research analytics or reviewer prioritization, never diagnosis.
2. Route low-confidence and sensitive predictions to qualified human review.
3. Maintain a separate safety workflow outside the classifier for real urgent concerns.
4. Minimize text retention and prohibit identifiable patient data in the demo.
5. Evaluate calibration, subgroup performance, source leakage, and temporal drift.
6. Use prospective human-labeled validation before any operational pilot.

## Risk Assessment
Major risks include false reassurance, unnecessary escalation, privacy exposure, demographic bias, domain shift, annotation inconsistency, and overreliance on confidence scores. Controls include access restrictions, data minimization, abstention, audits, model cards, monitoring, and human override.

## Future Opportunities
Transformer benchmarking, out-of-distribution detection, selective classification, source-aware validation, active learning, explainability for reviewers, and privacy-preserving model training.
