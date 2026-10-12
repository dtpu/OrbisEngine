"""Paths shared by the API, supervisor, and local pipeline commands."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_runs_root(value: str | Path | None = None) -> Path:
    """Resolve a configured runs directory consistently, regardless of process cwd."""
    candidate = Path(value or os.environ.get("WANDER_RUNS_ROOT") or ".context/runs").expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def resolve_stage_ledger() -> Path:
    """Direct and supervised commands must use the same source-keyed paid history."""
    candidate = Path(
        os.environ.get("WANDER_STAGE_LEDGER") or REPOSITORY_ROOT / ".context/pipeline-attempts.json"
    ).expanduser()
    return (candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate).resolve()
