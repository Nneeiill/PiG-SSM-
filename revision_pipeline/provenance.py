"""Portable runtime provenance for revision-v2 experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def runtime_provenance(
    *,
    started_at: str,
    ended_at: str | None,
    exit_code: int | None,
    gpu_index: int | None,
    cuda_visible_devices: str | None,
    gpu_name: str | None,
) -> dict[str, Any]:
    """Build the stable runtime block written into every run configuration."""
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "exit_code": exit_code,
        "gpu_index": gpu_index,
        "cuda_visible_devices": cuda_visible_devices,
        "gpu_name": gpu_name,
    }
