"""Retained full-video clean candidates and a hash-pinned downstream selection."""

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "wander.clean-selection/1"
OUTPUTS = {
    "video": "clean.mp4",
    "frame": "clean-f0.png",
    "masks": "masks.npz",
    "report": "clean.json",
    "log": "clean.log",
}
INPUTS = {"inputMasks": "track-clean-masks.npz", "inputReport": "track-clean-masks.json"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_slug(slug):
    if not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", slug):
        raise ValueError(
            "--clean-candidate must be a safe 1–64 character slug using letters, digits, _ or -"
        )
    return slug


def candidate_directory(ctx, slug):
    validate_slug(slug)
    ctx = Path(ctx).resolve()
    parent, directory = ctx / "clean-candidates", ctx / "clean-candidates" / slug
    if parent.is_symlink() or directory.is_symlink() or not directory.resolve().is_relative_to(ctx):
        raise ValueError("Clean candidate paths must stay inside the run without symlinks")
    return directory


@contextmanager
def selection_lock(ctx):
    """Coordinate clean selection with the runner's Marble submission."""
    path = Path(ctx) / "clean-selection.lock"
    if path.is_symlink():
        raise ValueError("Clean selection lock cannot be a symlink")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another clean selection or Marble submission owns this run") from None
        yield


def _write_json(path, document):
    if path.is_symlink():
        raise ValueError("Clean selection receipts cannot be symlinks")
    descriptor, temporary = tempfile.mkstemp(prefix=".clean-selection-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_selection(ctx, clip, run_name):
    """None means legacy paths; malformed or changed selections never fall back."""
    ctx = Path(ctx).resolve()
    receipt = ctx / "clean-selection.json"
    if receipt.is_symlink():
        raise ValueError("Clean selection receipt cannot be a symlink")
    if not receipt.exists():
        if any(ctx.glob("clean-candidates/*/selection.json")):
            raise ValueError(
                "Clean selection receipt is missing despite a retained successful candidate"
            )
        return None
    try:
        doc = json.loads(receipt.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Clean selection receipt is unreadable") from error
    if (
        not isinstance(doc, dict)
        or set(doc) != {"schema", "candidate", "sourceSha256", "runName", "files"}
        or doc["schema"] != SCHEMA
        or doc["runName"] != run_name
        or doc["sourceSha256"] != sha256(clip)
        or not isinstance(doc["files"], dict)
        or not set(OUTPUTS) <= set(doc["files"]) <= set(OUTPUTS) | set(INPUTS)
    ):
        raise ValueError("Clean selection receipt has invalid source, scope or files")
    directory = candidate_directory(ctx, doc["candidate"])
    history = directory / "selection.json"
    try:
        if history.is_symlink() or json.loads(history.read_text()) != doc:
            raise ValueError("Clean selection does not match its retained success receipt")
    except (OSError, ValueError) as error:
        raise ValueError("Clean selection does not match its retained success receipt") from error
    for key, pin in doc["files"].items():
        expected = directory / (OUTPUTS | INPUTS)[key]
        if (
            not isinstance(pin, dict)
            or set(pin) != {"path", "sha256"}
            or pin["path"] != expected.relative_to(ctx).as_posix()
            or expected.is_symlink()
            or not expected.is_file()
            or expected.stat().st_size == 0
            or pin["sha256"] != sha256(expected)
        ):
            raise ValueError(
                f"Selected clean candidate file changed or escaped its directory: {key}"
            )
    return doc


def publish_selection(ctx, clip, run_name, slug):
    """Call only after paid clean completes, while holding selection_lock."""
    ctx = Path(ctx).resolve()
    directory = candidate_directory(ctx, slug)
    document = {
        "schema": SCHEMA,
        "candidate": slug,
        "sourceSha256": sha256(clip),
        "runName": run_name,
        "files": {},
    }
    for key, name in (OUTPUTS | INPUTS).items():
        path = directory / name
        if key in INPUTS and not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Clean candidate is missing a complete retained file: {key}")
        document["files"][key] = {"path": path.relative_to(ctx).as_posix(), "sha256": sha256(path)}
    report = json.loads((directory / OUTPUTS["report"]).read_text())
    if (
        report.get("error")
        or report.get("sourceSha256") != document["sourceSha256"]
        or report.get("outputVideoSha256") != document["files"]["video"]["sha256"]
    ):
        raise ValueError("Clean candidate report does not confirm its source and output video")
    # Retain a success receipt with each candidate before replacing the active selection.
    history = directory / "selection.json"
    if history.exists() or history.is_symlink():
        raise ValueError("Clean candidate already has a retained selection receipt")
    _write_json(history, document)
    _write_json(ctx / "clean-selection.json", document)
    return document
