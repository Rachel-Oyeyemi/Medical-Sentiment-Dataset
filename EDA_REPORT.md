# Exploratory Data Analysis Report

## Official Dataset Profile
Published reproductions of the Kaggle file report **53,043 rows and three raw columns**: an index field, `statement`, and the seven-class target `status`. They report 362 null statements and 52,543 usable rows after null and post-cleaning empty-text removal.

| Label | Raw rows | Share |
|---|---:|---:|
| Normal | 16,351 | 30.83% |
| Depression | 15,404 | 29.04% |
| Suicidal | 10,653 | 20.08% |
| Anxiety | 3,888 | 7.33% |
| Bipolar | 2,877 | 5.42% |
| Stress | 2,669 | 5.03% |
| Personality disorder | 1,201 | 2.26% |

## Data Types and Target
The useful predictor is unstructured text; `status` is a categorical multiclass target. The source index is not predictive and must be removed.

## Missing Values and Duplicates
Missing or empty statements cannot be modeled. Exact and normalized-text duplicates must be removed before splitting to prevent inflated test performance. Identical text carrying different labels is reported as a label-conflict quality issue.

## Distribution and Outlier Analysis
Text length and word count are right-skewed. Long posts are not automatically invalid; they are flagged using the IQR rule for review rather than removed. Numeric correlation analysis is limited to engineered text-length features because TF-IDF dimensions are sparse and high-dimensional.

## Target Analysis
The official target is imbalanced: the two smallest classes account for far fewer rows than Normal and Depression. Accuracy therefore cannot be the sole metric. Macro recall, macro F1, per-class confusion matrices, and low-confidence coverage are emphasized.

## Business and Ethical Context
The dataset aggregates material from multiple public datasets with different communities and labeling practices. Labels may reflect source-specific annotation conventions rather than clinical diagnoses. Potential risks include source leakage, duplicate content, stigmatizing labels, privacy concerns, domain shift, and harm from false confidence.

## Visual Assets
The `visuals/` directory contains class balance, length distributions, engineered-feature correlation, common terms, model comparison, per-class F1, confusion matrix, and a human-review workflow.
