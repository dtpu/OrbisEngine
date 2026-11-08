"""Reopen an existing upload without creating attempts or changing its outcome."""

from __future__ import annotations

import fcntl
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from orchestrator.api.progress import find_run_directory, read_json
from orchestrator.repository import PipelineRepository
from orchestrator.steps import RunOptions


@contextmanager
def upload_lock(root: Path, source_sha256: str):
    """Serialize matching and publication across API processes sharing this runs root."""
    directory = root / ".upload-locks"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(directory / source_sha256, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def processing_options(value: dict) -> dict:
    # hands_off controls questions/retries, not the requested reconstruction. Reopening
    # a manual run does not convert it to automatic mode or resume it.
    return RunOptions.model_validate(value).model_dump(exclude={"hands_off"})


def matching_upload(
    repository: PipelineRepository,
    root: Path,
    source_sha256: str,
    size: int,
    options: RunOptions,
) -> dict | None:
    expected = processing_options(options.model_dump())
    for summary in repository.runs_for_source(source_sha256):
        directory = find_run_directory(root, summary["id"])
        if directory is None:
            continue
        manifest = read_json(directory / "run.json")
        try:
            if (
                manifest.get("sourceSha256") != source_sha256
                or processing_options(manifest["options"]) != expected
                or processing_options(summary["configuration"]["options"]) != expected
            ):
                continue
            source = Path(manifest["source"])
            if not source.is_absolute():
                source = directory / source
            if (
                source.is_symlink()
                or not source.resolve().is_relative_to(directory.resolve())
                or not source.is_file()
                or source.stat().st_size != size
            ):
                continue
            # A database digest alone cannot establish that the retained run still owns
            # these bytes. Corrupt/missing input must not redirect the user to another clip.
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest == source_sha256:
                return summary
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None
