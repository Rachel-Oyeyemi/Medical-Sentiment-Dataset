"""Download the Kaggle Sentiment Analysis for Mental Health dataset.

Authentication is required through KaggleHub or the Kaggle CLI. Credentials are
never read from or written to the repository by this module.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from utils import configure_logging

LOGGER = logging.getLogger(__name__)
DATASET_HANDLE = "suchintikasarkar/sentiment-analysis-for-mental-health"


def _copy_dataset_files(source: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for file in source.rglob("*"):
        if file.is_file():
            destination = output / file.name
            shutil.copy2(file, destination)
            copied.append(destination)
    if not copied:
        raise FileNotFoundError(f"No files were returned from {source}")
    return copied


def download_with_kagglehub(output: Path) -> list[Path]:
    """Download using KaggleHub and copy files into the requested directory."""
    import kagglehub

    cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    return _copy_dataset_files(cache_path, output)


def download_with_cli(output: Path) -> list[Path]:
    """Download and unzip using the Kaggle command-line client."""
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_HANDLE, "-p", str(output), "--unzip"],
        check=True,
    )
    files = [path for path in output.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError("Kaggle CLI completed without creating dataset files")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--source", choices=["auto", "kagglehub", "cli"], default="auto")
    args = parser.parse_args()
    configure_logging()
    output = Path(args.output)
    errors: list[str] = []

    if args.source in {"auto", "kagglehub"}:
        try:
            files = download_with_kagglehub(output)
            LOGGER.info("Downloaded %s file(s) with KaggleHub", len(files))
            return
        except Exception as exc:
            errors.append(f"KaggleHub: {exc}")

    if args.source in {"auto", "cli"}:
        try:
            files = download_with_cli(output)
            LOGGER.info("Downloaded %s file(s) with Kaggle CLI", len(files))
            return
        except Exception as exc:
            errors.append(f"Kaggle CLI: {exc}")

    raise RuntimeError(
        "Dataset download failed. Authenticate to Kaggle and accept any dataset terms. "
        + " | ".join(errors)
    )


if __name__ == "__main__":
    main()
