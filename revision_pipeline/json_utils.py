"""Strict JSON serialization helpers for experiment artifacts."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any


def sanitize_non_finite(value: Any) -> Any:
    """Return a JSON-safe copy, mapping NaN and +/-Inf to JSON null."""
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, dict):
        return {key: sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_non_finite(item) for item in value]
    return value


def dump_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    """Write strict JSON after sanitizing non-finite floating-point values."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(sanitize_non_finite(value), indent=indent, allow_nan=False) + "\n"
    )
