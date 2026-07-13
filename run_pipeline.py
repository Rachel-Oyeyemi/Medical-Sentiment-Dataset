"""Run the complete reproducible Medical Sentiment demo pipeline."""
from __future__ import annotations

import subprocess
import sys

STEPS = [
    ["src/generate_demo_data.py"],
    ["src/preprocess.py"],
    ["src/train_model.py"],
    ["src/evaluate_model.py"],
    ["src/generate_visuals.py"],
]

for step in STEPS:
    subprocess.run([sys.executable, *step], check=True)
print("Medical Sentiment pipeline complete")
