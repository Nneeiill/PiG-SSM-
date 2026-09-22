"""Project-root resolution for the revision-v2 pipeline.

Historical code hard-coded a legacy host path (removed in revision-v2).
All revision-v2 entry points resolve the root from (in priority order) an explicit
argument, the ``PIG_SSM_PROJECT_ROOT`` environment variable, or this file's location.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["PIG_SSM_PROJECT_ROOT"])
        if "PIG_SSM_PROJECT_ROOT" in os.environ
        else None,
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.expanduser().resolve()
        if (root / "organized_pig_ssm" / "pig_ssm_model.py").is_file():
            return root
    raise FileNotFoundError("Cannot locate decoder-only-gpt project root")