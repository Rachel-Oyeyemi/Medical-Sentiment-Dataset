"""Shared file, logging, and serialization utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent project-wide log format."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the normalized path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_json(payload: Any, path: str | Path) -> None:
    """Serialize JSON with stable indentation and UTF-8 encoding."""
    output = ensure_parent(path)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
