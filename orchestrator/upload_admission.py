"""Durable, root-wide admission of a finite number of automatic uploads.

This counts uploads, not provider requests or dollars. The supervisor dispatch lock is
independent of per-run stage locks: detached children never acquire this lock.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path

from orchestrator.journal import KINDS
from orchestrator.run_policy import hands_off

LEDGER_FILE = ".upload-admission.json"
DISPATCH_LOCK_FILE = ".upload-dispatch.lock"
OWNER_LOCK_FILE = ".upload-owner.lock"
SCHEMA = "wander.upload-admission/1"
MAX_BYTES = 16 * 1024 * 1024


class AdmissionError(RuntimeError):
    """Uncertain admission state must never authorize more work."""


def _decode(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AdmissionError("Duplicate admission or journal field")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _invalid())
    except (ValueError, RecursionError) as error:
        raise AdmissionError("Malformed upload admission state") from error


def _invalid():
    raise AdmissionError("Non-finite upload admission state")


def _read(directory: int, name: str, *, missing: bool = False) -> bytes | None:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except FileNotFoundError:
        if missing:
            return None
        raise AdmissionError(f"Missing upload admission evidence: {name}") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BYTES:
            raise AdmissionError(f"Invalid upload admission evidence: {name}")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            raw = source.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise AdmissionError(f"Oversized upload admission evidence: {name}")
        return raw
    finally:
        os.close(descriptor)


def _open_lock(directory: int, name: str) -> int:
    descriptor = os.open(
        name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise AdmissionError(f"Invalid upload admission lock: {name}")
    return descriptor


def _identity(directory: str, manifest: dict) -> dict:
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {"directory": directory, "runId": manifest["runId"], "manifestSha256": digest}


def _history(directory: int, *, required: bool) -> tuple[bool, bool, bool]:
    """Return (terminal, unresolved, started), refusing every unreadable journal line."""
    raw = _read(directory, "journal.jsonl", missing=not required)
    if raw is None:
        return False, False, False
    terminal = started = False
    active = set()
    for line in raw.splitlines():
        entry = _decode(line)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"kind", "at", "data"}
            or entry["kind"] not in KINDS
            or not isinstance(entry["at"], str)
            or not isinstance(entry["data"], dict)
        ):
            raise AdmissionError("Malformed upload journal")
        kind, data = entry["kind"], entry["data"]
        if kind.startswith("step."):
            started = True
            step, attempt = data.get("step"), data.get("attempt", "")
            if not isinstance(step, str) or not step or not isinstance(attempt, str):
                raise AdmissionError("Malformed upload attempt identity")
            key = (step, attempt)
            if kind in {"step.started", "step.executing"}:
                active.add(key)
                started = True
            else:
                if data.get("status") not in (
                    "succeeded",
                    "failed",
                    "canceled",
                    "blocked",
                    "skipped",
                ):
                    raise AdmissionError("Malformed upload attempt outcome")
                active.discard(key)
        elif kind == "run.finished":
            if data.get("status") not in {"succeeded", "failed", "canceled"}:
                raise AdmissionError("Malformed upload terminal outcome")
            terminal = True
        elif kind == "run.resumed":
            terminal = False
    return terminal, bool(active), started


class UploadAdmission:
    """One lifetime owner for a bound; all supervisors serialize automatic dispatch here."""

    def __init__(self, root: Path, limit: int | None, *, uploads_only: bool = False):
        if limit is not None and (type(limit) is not int or limit <= 0 or not uploads_only):
            raise AdmissionError("--max-uploads must be positive and requires --uploads-only")
        self.root = root.resolve()
        self.limit = limit
        self.uploads_only = uploads_only
        self.directory = self.lock = self.owner = None
        self.state = None

    def __enter__(self):
        try:
            self.directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self.lock = _open_lock(self.directory, DISPATCH_LOCK_FILE)
            if self.limit is not None:
                self.owner = _open_lock(self.directory, OWNER_LOCK_FILE)
                try:
                    fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise AdmissionError(
                        "A bounded upload supervisor already owns this root"
                    ) from error
            with self.dispatch():
                if self.limit is not None and self.state is None:
                    self._check_unadmitted([])
                    self.state = {
                        "schema": SCHEMA,
                        "root": str(self.root),
                        "maxUploads": self.limit,
                        "admissions": [],
                        "active": None,
                    }
                    # The stable lock contains an immutable policy anchor. A crash here
                    # leaves a missing ledger error, never an unbounded or reset policy.
                    anchor = json.dumps(self._anchor()).encode()
                    os.pwrite(self.lock, anchor, 0)
                    os.fsync(self.lock)
                    os.fsync(self.directory)
                    self._save()
                if self.state is not None and self.limit is None and self.uploads_only:
                    raise AdmissionError("This root requires its original --max-uploads limit")
                if self.limit is not None:
                    self.refresh()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for name in ("owner", "lock", "directory"):
            descriptor = getattr(self, name)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)

    def _anchor(self):
        return {key: self.state[key] for key in ("schema", "root", "maxUploads")}

    @contextmanager
    def dispatch(self):
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        try:
            metadata = os.stat(DISPATCH_LOCK_FILE, dir_fd=self.directory, follow_symlinks=False)
            held = os.fstat(self.lock)
            if (metadata.st_dev, metadata.st_ino) != (
                held.st_dev,
                held.st_ino,
            ) or held.st_nlink != 1:
                raise AdmissionError("Upload dispatch lock was replaced")
            if self.owner is not None:
                owner = os.stat(OWNER_LOCK_FILE, dir_fd=self.directory, follow_symlinks=False)
                held_owner = os.fstat(self.owner)
                if (owner.st_dev, owner.st_ino) != (
                    held_owner.st_dev,
                    held_owner.st_ino,
                ) or held_owner.st_nlink != 1:
                    raise AdmissionError("Upload owner lock was replaced")
            self._load()
            yield
        finally:
            fcntl.flock(self.lock, fcntl.LOCK_UN)

    def _load(self):
        anchor_raw = os.pread(self.lock, MAX_BYTES + 1, 0)
        raw = _read(self.directory, LEDGER_FILE, missing=True)
        if not anchor_raw and raw is None:
            if self.state is not None:
                raise AdmissionError("Upload admission policy disappeared")
            return
        if not anchor_raw or raw is None:
            raise AdmissionError("Missing upload admission ledger or policy anchor")
        anchor, state = _decode(anchor_raw), _decode(raw)
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "root", "maxUploads", "admissions", "active"}
            or state["schema"] != SCHEMA
            or state["root"] != str(self.root)
            or type(state["maxUploads"]) is not int
            or state["maxUploads"] <= 0
            or not isinstance(state["admissions"], list)
            or len(state["admissions"]) > state["maxUploads"]
            or not isinstance(anchor, dict)
            or type(anchor.get("maxUploads")) is not int
            or anchor != {key: state[key] for key in ("schema", "root", "maxUploads")}
            or (self.limit is not None and state["maxUploads"] != self.limit)
        ):
            raise AdmissionError("Invalid or mismatched upload admission policy")
        directories, ids = set(), set()
        for entry in state["admissions"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"directory", "runId", "manifestSha256"}
                or not isinstance(entry["directory"], str)
                or entry["directory"] in {"", ".", ".."}
                or "/" in entry["directory"]
                or not isinstance(entry["runId"], str)
                or not 0 < len(entry["runId"]) <= 512
                or not isinstance(entry["manifestSha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["manifestSha256"]) is None
                or entry["directory"] in directories
                or entry["runId"] in ids
            ):
                raise AdmissionError("Invalid or duplicate upload admission identity")
            directories.add(entry["directory"])
            ids.add(entry["runId"])
        if state["active"] is not None and (
            not isinstance(state["active"], str) or state["active"] not in directories
        ):
            raise AdmissionError("Invalid active upload admission")
        self.state = state

    def _save(self):
        temporary = f".{uuid.uuid4()}.upload-admission.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.directory,
        )
        try:
            with os.fdopen(descriptor, "w") as target:
                json.dump(self.state, target, sort_keys=True, allow_nan=False)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, LEDGER_FILE, src_dir_fd=self.directory, dst_dir_fd=self.directory)
            os.fsync(self.directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.directory)
            except FileNotFoundError:
                pass

    def _inspect(self, name, *, identity=None):
        from orchestrator.supervisor import run_manifest

        directory = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.directory
        )
        try:
            manifest = run_manifest(self.root / name)
            if manifest is None:
                raise AdmissionError(f"Unreadable upload manifest: {name}")
            if identity is not None and _identity(name, manifest) != identity:
                raise AdmissionError(f"Admitted upload identity changed: {name}")
            terminal, unresolved, started = _history(directory, required=identity is not None)
            for evidence in ("BRIEF.md", "session.id", "transcript.jsonl"):
                try:
                    os.stat(evidence, dir_fd=directory, follow_symlinks=False)
                    started = True
                except FileNotFoundError:
                    pass
            return manifest, (terminal, unresolved, started)
        finally:
            os.close(directory)

    def _check_unadmitted(self, admissions):
        from orchestrator.supervisor import run_manifest

        admitted = {item["directory"] for item in admissions}
        for described in sorted(self.root.glob("*/run.json")):
            if described.parent.name in admitted:
                continue
            manifest = run_manifest(described.parent)
            if manifest is None or not hands_off(manifest.get("options", {})):
                continue
            _, (terminal, unresolved, started) = self._inspect(described.parent.name)
            if unresolved or (started and not terminal):
                raise AdmissionError(
                    f"Unadmitted automatic work already started: {described.parent.name}"
                )

    def refresh(self):
        """Reclaim prior admissions before considering a fresh upload; caller holds dispatch."""
        if self.limit is None:
            return
        unfinished = []
        for identity in self.state["admissions"]:
            _, (terminal, unresolved, _) = self._inspect(identity["directory"], identity=identity)
            if not terminal or unresolved:
                unfinished.append(identity["directory"])
        self._check_unadmitted(self.state["admissions"])
        active = self.state["active"]
        next_active = active if active in unfinished else next(iter(unfinished), None)
        if active != next_active:
            self.state["active"] = next_active
            self._save()

    @property
    def active(self):
        return self.state["active"] if self.state else None

    @property
    def remaining(self):
        return self.limit - len(self.state["admissions"]) if self.limit is not None else None

    def claim(self, run_dir: Path, manifest: dict) -> bool:
        """Called immediately before a harness dispatch, under the common root lock."""
        if manifest is None:
            raise AdmissionError("Unreadable upload manifest before dispatch")
        if not hands_off(manifest.get("options", {})):
            return self.limit is None
        if self.limit is None:
            return self.state is None
        self.refresh()
        name = run_dir.name
        identity = _identity(name, manifest)
        admitted = self.state["admissions"]
        if any(item["directory"] == name for item in admitted):
            return self.active == name
        if self.active is not None or not self.remaining:
            return False
        if any(item["runId"] == manifest["runId"] for item in admitted):
            raise AdmissionError("Upload runId already belongs to an admitted directory")
        _, (terminal, unresolved, started) = self._inspect(name)
        if terminal:
            return False
        if unresolved or started:
            raise AdmissionError("Cannot adopt previously started automatic work")
        directory = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.directory
        )
        try:
            descriptor = os.open(
                "journal.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=directory,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(directory)
        finally:
            os.close(directory)
        admitted.append(identity)
        self.state["active"] = name
        self._save()
        return True
