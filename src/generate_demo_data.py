"""Generate a deterministic, overlapping seven-class mental-health text demo corpus.

The generated text is synthetic and intentionally ambiguous. It exists only so the
repository runs without Kaggle credentials; it must never be presented as clinical
or official dataset evidence.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

LABEL_COUNTS = {
    "Normal": 1635,
    "Depression": 1540,
    "Suicidal": 1065,
    "Anxiety": 389,
    "Bipolar": 288,
    "Stress": 267,
    "Personality disorder": 120,
}

CORE = {
    "Normal": [
        "I feel steady and able to handle my day",
        "my mood has been mostly calm and balanced",
        "I enjoyed time with friends and felt present",
        "I am sleeping normally and keeping up with routines",
        "today felt ordinary and manageable",
    ],
    "Depression": [
        "I feel low and disconnected from things I used to enjoy",
        "it is hard to find energy or motivation lately",
        "I feel empty and have been withdrawing from people",
        "most days feel heavy and hopeless",
        "I struggle to get out of bed and start anything",
    ],
    "Suicidal": [
        "I do not want to be here anymore and I feel unsafe with myself",
        "I keep thinking about ending my life",
        "I wish I would not wake up and I cannot see another way",
        "I am afraid I may hurt myself if I stay alone",
        "I feel like I cannot keep going",
    ],
    "Anxiety": [
        "my thoughts keep racing and I cannot stop worrying",
        "I feel panicked and my heart pounds for no clear reason",
        "I keep expecting something bad to happen",
        "I cannot relax because every small problem feels dangerous",
        "worry is making it difficult to sleep and focus",
    ],
    "Stress": [
        "deadlines and responsibilities are piling up around me",
        "I feel overloaded by work and family demands",
        "there is too much pressure and not enough time",
        "I am exhausted from trying to keep up with everything",
        "my workload is intense and I need a break",
    ],
    "Bipolar": [
        "I have periods of very high energy and little sleep followed by crashes",
        "my mood shifts from unusually energized to deeply low",
        "I have been impulsive and intensely confident before suddenly slowing down",
        "some days I talk fast and start many plans then lose all energy",
        "my sleep and activity change sharply with my mood",
    ],
    "Personality disorder": [
        "I fear people will leave and my relationships become very intense",
        "my sense of who I am changes depending on who I am with",
        "small signs of rejection trigger overwhelming emotions",
        "I switch between idealizing people and feeling betrayed by them",
        "my emotions and relationships feel unstable and hard to control",
    ],
}

SHARED = [
    "I have been tired and distracted",
    "I do not feel like myself",
    "sleep has been difficult",
    "I am struggling to focus",
    "I feel overwhelmed by my thoughts",
    "I have been avoiding people",
    "my mood has changed recently",
    "I am trying to understand what is happening",
]

CONTEXTS = [
    "after a difficult week",
    "when I am at home",
    "during classes",
    "at work",
    "late at night",
    "around other people",
    "since my routine changed",
    "even when nothing obvious is wrong",
    "and it has started affecting my daily routine",
    "but I am still trying to ask for support",
]

OPENERS = ["lately", "recently", "today", "for several weeks", "some mornings", "on and off"]
CONNECTORS = ["and", "while", "but", "because", "so"]


def _light_noise(text: str, rng: random.Random) -> str:
    """Add mild user-generated-text noise without inserting label identifiers."""
    if rng.random() < 0.16:
        text = text.replace("I ", "i ", 1)
    if rng.random() < 0.12:
        text = text.replace("cannot", "can't")
    if rng.random() < 0.10:
        text += rng.choice(["...", "", " honestly", " right now"])
    if rng.random() < 0.08:
        text = text.replace(" and ", " & ", 1)
    return text


def generate_demo(seed: int = 42) -> pd.DataFrame:
    """Return a reproducible synthetic corpus with class imbalance and ambiguity."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    labels = list(LABEL_COUNTS)
    for label, count in LABEL_COUNTS.items():
        for _ in range(count):
            opener = rng.choice(OPENERS)
            primary = rng.choice(CORE[label])
            secondary = rng.choice(SHARED)
            context = rng.choice(CONTEXTS)
            if rng.random() < 0.34:
                other = rng.choice([value for value in labels if value != label])
                secondary = rng.choice(CORE[other])
            statement = f"{opener}, {primary} {rng.choice(CONNECTORS)} {secondary} {context}."
            statement = _light_noise(statement, rng)
            rows.append({"statement": statement, "status": label})

    rng.shuffle(rows)
    noisy_count = int(len(rows) * 0.07)
    for index in range(noisy_count):
        current = rows[index]["status"]
        rows[index]["status"] = rng.choice([label for label in labels if label != current])

    return pd.DataFrame(rows).drop_duplicates(subset=["statement"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/medical_sentiment_demo.csv")
    parser.add_argument("--preview", default="data/sample_data/medical_sentiment_sample.csv")
    args = parser.parse_args()
    frame = generate_demo()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    preview = Path(args.preview)
    preview.parent.mkdir(parents=True, exist_ok=True)
    frame.groupby("status", group_keys=False).head(30).to_csv(preview, index=False)
    print(f"Saved {len(frame):,} synthetic demo rows to {output}")


if __name__ == "__main__":
    main()
