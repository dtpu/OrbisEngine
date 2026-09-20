"""Isolated run and immutable attempt filesystem layout."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


# Files a stage's own tooling writes to coordinate itself, never outputs of the stage. A lock
# is empty by design, and the emptiness check exists to catch an output that was truncated.
INCIDENTAL_SUFFIXES = (".lock",)


def is_output_file(path: Path) -> bool:
    """Whether a file under an attempt can carry one of the stage's declared roles.

    Both the QA validator and the freezer ask this, because they have to agree: a file one
    counts and the other does not is how a stage passes its own checks and then hands the next
    stage nothing (or, here, fails on a lock file the stage never meant to produce).
    """
    return path.is_file() and not path.is_symlink() and path.suffix not in INCIDENTAL_SUFFIXES


def safe_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe workspace identifier: {value!r}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class AttemptWorkspace:
    run_id: str
    node_id: str
    attempt_id: str
    root: Path

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def qa(self) -> Path:
        return self.root / "qa"

    @property
    def request(self) -> Path:
        return self.root / "request.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def stdout(self) -> Path:
        return self.root / "stdout.log"

    @property
    def stderr(self) -> Path:
        return self.root / "stderr.log"

    def finalize(self, value: dict[str, Any]) -> None:
        if self.result.exists():
            raise FileExistsError(f"attempt is already finalized: {self.attempt_id}")
        atomic_json(self.result, value)


class RunWorkspace:
    def __init__(self, base: Path, run_id: str):
        self.base = Path(base).resolve()
        self.run_id = safe_id(run_id)
        self.root = self.base / self.run_id
        self.input = self.root / "input"
        self.agent = self.root / "agent"
        self.attempts = self.root / "attempts"
        self.manifests = self.root / "manifests"
        self.outbox = self.root / "outbox"

    def initialize(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("run workspace cannot be a symlink")
        self.root.mkdir(mode=0o700, exist_ok=True)
        for directory in (
            self.input,
            self.agent,
            self.attempts,
            self.manifests,
            self.outbox,
        ):
            if directory.is_symlink():
                raise ValueError(f"workspace directory cannot be a symlink: {directory}")
            directory.mkdir(mode=0o700, exist_ok=True)

    def hydrate_input(self, role: str, source: Path) -> Path:
        role = safe_id(role)
        source = Path(source)
        if source.is_symlink():
            raise ValueError("workspace inputs must be regular files")
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError("workspace inputs must be regular files")
        destination = self.input / role / source.name
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise FileExistsError(f"input role already has different bytes: {role}")
            return destination
        temporary = destination.with_name(f".{destination.name}.part")
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o400)
        return destination

    def create_attempt(
        self,
        node_id: str,
        attempt_id: str,
        request: dict[str, Any],
    ) -> AttemptWorkspace:
        node_id, attempt_id = safe_id(node_id), safe_id(attempt_id)
        root = self.attempts / node_id / attempt_id
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        (root / "outputs").mkdir(mode=0o700)
        (root / "qa").mkdir(mode=0o700)
        atomic_json(root / "request.json", request)
        return AttemptWorkspace(self.run_id, node_id, attempt_id, root)

    def open_attempt(self, node_id: str, attempt_id: str) -> AttemptWorkspace:
        root = self.attempts / safe_id(node_id) / safe_id(attempt_id)
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(f"attempt workspace does not exist: {attempt_id}")
        return AttemptWorkspace(self.run_id, node_id, attempt_id, root)
