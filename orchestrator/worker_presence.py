"""Bounded, local supervisor liveness; presence does not promise successful work."""

from __future__ import annotations

import json
import logging
import math
import os
import stat
import threading
import time
import uuid
from pathlib import Path

SCHEMA = "wander.worker-presence"
VERSION = 2
FRESHNESS_SECONDS = 45.0
REFRESH_SECONDS = 5.0
MAX_RECORD_BYTES = 16_384


def _run_id(value: object) -> bool:
    return value is None or (isinstance(value, str) and 0 < len(value) <= 512)


def _open_directory(root: Path) -> int:
    # A symlink must not move registration or discovery outside this runs root.
    return os.open(root / ".workers", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def worker_status(root: Path | None, run_id: str | None = None, *, hands_off: bool = False) -> dict:
    """Count fresh matching supervisors without disclosing local process information."""
    matching = busy = 0
    if root is not None:
        directory = None
        try:
            resolved = root.resolve()
            directory = _open_directory(resolved)
            for name in os.listdir(directory):
                try:
                    if not name.endswith(".json") or str(uuid.UUID(name[:-5])) != name[:-5]:
                        continue
                    descriptor = os.open(
                        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
                    )
                    with os.fdopen(descriptor, "rb") as record_file:
                        metadata = os.fstat(record_file.fileno())
                        if (
                            not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_size > MAX_RECORD_BYTES
                        ):
                            continue
                        record = json.loads(record_file.read(MAX_RECORD_BYTES + 1))
                    if not isinstance(record, dict):
                        continue
                    seen = record.get("last_seen")
                    if (
                        record.get("schema") != SCHEMA
                        or type(record.get("version")) is not int
                        or record["version"] not in (1, VERSION)
                        or (
                            record["version"] == VERSION
                            and type(record.get("uploads_only")) is not bool
                        )
                        or record.get("runs_root") != str(resolved)
                        or type(record.get("watcher")) is not bool
                        or "run_id" not in record
                        or not _run_id(record["run_id"])
                        or "active_run_id" not in record
                        or not _run_id(record["active_run_id"])
                        or type(seen) not in (int, float)
                        or not math.isfinite(seen)
                        or not 0 <= time.time() - seen <= FRESHNESS_SECONDS
                    ):
                        continue
                    # Version 1 predates restricted workers. Version 2 is deliberately
                    # incompatible with old readers, which only accepted version 1.
                    if (
                        record["version"] == VERSION
                        and record["uploads_only"]
                        and hands_off is not True
                    ):
                        continue
                    selected, active = record["run_id"], record["active_run_id"]
                    if selected is not None and active not in (None, selected):
                        continue
                    if run_id is None:
                        matches = record["watcher"] and selected is None
                    elif record["watcher"]:
                        matches = selected in (None, run_id)
                    else:
                        matches = active == run_id
                    if matches:
                        matching += 1
                        busy += active is not None
                except (OSError, ValueError, TypeError, OverflowError, RecursionError):
                    continue
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            if directory is not None:
                os.close(directory)
    return {
        "status": "available" if matching else "unavailable",
        "matchingWorkers": matching,
        "busyWorkers": busy,
        "detail": (
            "A matching supervisor has a recent heartbeat."
            if matching
            else "No matching supervisor has a recent heartbeat."
        ),
    }


class WorkerPresence:
    """Keep one independent record fresh even while a supervisor's agent blocks."""

    def __init__(
        self,
        root: Path,
        *,
        run_id: str | None = None,
        watcher: bool = True,
        uploads_only: bool = False,
        heartbeat: Path | None = None,
        refresh_interval: float = REFRESH_SECONDS,
    ):
        if not _run_id(run_id) or type(watcher) is not bool or type(uploads_only) is not bool:
            raise ValueError("Invalid supervisor scope")
        if not math.isfinite(refresh_interval) or not 0 < refresh_interval < FRESHNESS_SECONDS:
            raise ValueError("Heartbeat interval must be positive and below the freshness limit")
        self.root = root.resolve()
        self.run_id = run_id
        self.watcher = watcher
        self.uploads_only = uploads_only
        self.heartbeat = heartbeat
        self.refresh_interval = refresh_interval
        self.name = f"{uuid.uuid4()}.json"
        self._active: str | None = None
        self._directory: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: OSError | None = None

    def __enter__(self) -> WorkerPresence:
        try:
            (self.root / ".workers").mkdir(exist_ok=True)
            self._directory = _open_directory(self.root)
            self._write()
            self._thread = threading.Thread(
                target=self._refresh, name="worker-heartbeat", daemon=True
            )
            self._thread.start()
        except (OSError, RuntimeError) as error:
            self._remove()
            self._close_directory()
            raise RuntimeError("Could not register supervisor heartbeat") from error
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._remove()
        self._close_directory()
        if self._error is not None and exc_type is None:
            raise RuntimeError("Supervisor heartbeat stopped refreshing") from self._error

    def set_active(self, run_id: str | None) -> None:
        if not _run_id(run_id) or (self.run_id is not None and run_id not in (None, self.run_id)):
            raise ValueError("Active run does not match supervisor scope")
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Supervisor heartbeat stopped refreshing") from self._error
            self._active = run_id
            try:
                self._write()
            except OSError as error:
                self._error = error
                self._stop.set()
                self._remove()
                raise RuntimeError("Supervisor heartbeat stopped refreshing") from error

    def _write(self) -> None:
        if self._directory is None:
            raise RuntimeError("Supervisor heartbeat is not registered")
        now = time.time()
        record = {
            "schema": SCHEMA,
            "version": VERSION,
            "runs_root": str(self.root),
            "watcher": self.watcher,
            "uploads_only": self.uploads_only,
            "run_id": self.run_id,
            "active_run_id": self._active,
            "last_seen": now,
        }
        # Preserve the legacy timestamp format and refresh it on the same background timer.
        if self.heartbeat is not None:
            self.heartbeat.write_text(f"{now}\n")
        temporary = f".{self.name}.{uuid.uuid4()}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._directory,
            )
            with os.fdopen(descriptor, "w") as record_file:
                json.dump(record, record_file)
                record_file.write("\n")
            os.replace(temporary, self.name, src_dir_fd=self._directory, dst_dir_fd=self._directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=self._directory)
            except FileNotFoundError:
                pass

    def _refresh(self) -> None:
        while not self._stop.wait(self.refresh_interval):
            with self._lock:
                try:
                    self._write()
                except OSError as error:
                    self._error = error
                    self._remove()
                    logging.getLogger(__name__).error("Supervisor heartbeat stopped refreshing")
                    return

    def _remove(self) -> None:
        if self._directory is not None:
            try:
                os.unlink(self.name, dir_fd=self._directory)
            except OSError:
                pass  # Expiry bounds stale presence if cleanup is unavailable.

    def _close_directory(self) -> None:
        if self._directory is not None:
            os.close(self._directory)
            self._directory = None
