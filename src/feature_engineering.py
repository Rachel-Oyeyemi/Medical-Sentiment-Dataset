"""Feature and target definitions for the NLP classification pipeline."""
TEXT_FEATURE = "clean_text"
TARGET = "status"
AUXILIARY_TEXT_FEATURES = ["text_length", "word_count", "digit_count", "unique_word_ratio"]
