# Data Dictionary

| Field | Type | Description |
|---|---|---|
| `Unnamed: 0` | Integer | Source index present in the raw Kaggle CSV; excluded from modeling. |
| `statement` | Text | User-generated or social-media-style statement aggregated from public datasets. |
| `status` | Category | One of seven dataset labels: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, or Personality disorder. |
| `clean_text` | Text | Normalized statement used by TF-IDF models. |
| `text_length` | Integer | Character count after normalization. |
| `word_count` | Integer | Token count after normalization. |
| `digit_count` | Integer | Count of digits in normalized text. |
| `unique_word_ratio` | Float | Unique tokens divided by total tokens. |
