"""The record of what a run actually did, kept beside the run itself.

``journal.jsonl`` in the run directory is the truth: one line per thing that happened, appended
and never rewritten. The database rows are a projection of it, so the dashboard can show a run
without being the only place the run exists. A journal and its run directory are enough to say
what has been done and what it produced -- which is what makes a run survive this process
dying, rather than a durable-execution engine replaying a workflow.

Nothing here decides anything. The agent decides; this writes down what it decided and what
came of it.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from orchestrator.steps import base_step, needs_of

JOURNAL_FILE = "journal.jsonl"
ADMISSION_LOCK_FILE = ".step-admission.lock"

# What a line can be. A step is work that ran; the rest is the agent or an operator speaking.
KINDS = (
    "step.started",
    "step.executing",
    "step.finished",
    "note",
    "question",
    "answer",
    "run.finished",
    "run.resumed",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Entry:
    kind: str
    at: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, line: str) -> Entry | None:
        try:
            raw = json.loads(line)
        except ValueError:
            return None
        if not isinstance(raw, dict) or raw.get("kind") not in KINDS:
            return None
        return cls(kind=raw["kind"], at=raw.get("at", ""), data=raw.get("data") or {})


class Journal:
    """Append-only history of one run, in its own directory.

    The writes are O_APPEND and fsynced, so two processes writing at once interleave whole
    lines rather than corrupting each other: the agent's session and an operator's command can
    both be talking to a run at the same time.
    """

    def __init__(self, run_dir: Path, *, projection: RunProjection | None = None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / JOURNAL_FILE
        self.projection = projection

    def append(self, kind: str, **data: Any) -> Entry:
        if kind not in KINDS:
            raise ValueError(f"not a kind of thing that happens: {kind}")
        if kind == "question":
            # Older question projections overwrote node status. New questions are journal
            # holds, so readers can distinguish them when repairing legacy rows.
            data["statusSource"] = "journal"
        entry = Entry(kind=kind, at=_now(), data=data)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"kind": entry.kind, "at": entry.at, "data": entry.data}, separators=(",", ":")
        )
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, (line + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if self.projection is not None:
            self.projection.apply(entry)
        return entry

    def entries(self) -> list[Entry]:
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return []
        return [entry for entry in (Entry.parse(line) for line in lines) if entry is not None]

    @contextmanager
    def admission_lock(self) -> Iterator[None]:
        """Serialize a short check, journal append, and child-spawn admission.

        The file survives process restarts, but the operating system releases its lock if
        the holder dies. Callers must release it before waiting for stage execution.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.run_dir / ADMISSION_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def active_attempts(self) -> dict[str, dict[str, Entry]]:
        """Return unresolved starts, preserving each attempt independently.

        An empty attempt id represents legacy entries without an attempt field. Such a
        finish cannot clear a modern attempt whose identity it does not establish.
        """
        active: dict[str, dict[str, Entry]] = {}
        for entry in self.entries():
            step = entry.data.get("step")
            if not isinstance(step, str) or not step:
                continue
            attempt = entry.data.get("attempt")
            key = attempt if isinstance(attempt, str) and attempt else ""
            if entry.kind == "step.started":
                active.setdefault(step, {})[key] = entry
            elif entry.kind == "step.finished":
                attempts = active.get(step)
                if attempts is not None:
                    attempts.pop(key, None)
                    if not attempts:
                        active.pop(step)
        return active

    def execution_claimed(self, step: str, attempt: str) -> bool:
        """Whether this exact attempt has ever claimed stage execution."""
        return any(
            entry.kind == "step.executing"
            and entry.data.get("step") == step
            and entry.data.get("attempt") == attempt
            for entry in self.entries()
        )

    def attempt_number(self, step: str) -> int:
        """How many times this step has been started, plus one."""
        started = sum(
            1 for e in self.entries() if e.kind == "step.started" and e.data.get("step") == step
        )
        return started + 1

    def last_status(self, step: str) -> str | None:
        """What became of this step the last time it ran, if it has."""
        for entry in reversed(self.entries()):
            if entry.kind == "step.finished" and entry.data.get("step") == step:
                return entry.data.get("status")
        return None

    def done(self) -> set[str]:
        """Steps whose most recent run succeeded."""
        return {
            entry.data["step"]
            for entry in self.entries()
            if entry.kind == "step.finished"
            and entry.data.get("step")
            and self.last_status(entry.data["step"]) == "succeeded"
        }

    def pending_question(self) -> Entry | None:
        """The latest question not yet cleared by an answer or run finish."""
        pending: Entry | None = None
        for entry in self.entries():
            if entry.kind == "question":
                pending = entry
            if entry.kind in {"answer", "run.finished"}:
                pending = None
        return pending

    def unanswered(self) -> str | None:
        """The question this run is waiting on, if it is waiting on one."""
        pending = self.pending_question()
        return pending.data.get("question") if pending is not None else None

    def control_status(self) -> str | None:
        """An operator cancellation stays in effect until an explicit resume."""
        for entry in reversed(self.entries()):
            if entry.kind == "run.resumed":
                break
            if entry.kind == "run.finished" and entry.data.get("status") == "canceled":
                return "canceled"
        return "paused" if (self.run_dir / "paused").exists() else None

    def finished(self) -> str | None:
        if self.control_status() == "canceled":
            return "canceled"
        for entry in reversed(self.entries()):
            if entry.kind == "run.resumed":
                return None
            if entry.kind == "run.finished":
                return entry.data.get("status")
        return None


class RunProjection:
    """Mirrors a journal into the database rows the dashboard reads.

    This is one-way and derived. If it falls behind, or a run is recovered onto a fresh
    database, :meth:`replay` rebuilds it from the journal: nothing here is the only copy of
    anything. Failing to project must never fail the step that was being recorded, so every
    apply is guarded -- the journal line is already on disk by the time this runs.
    """

    def __init__(self, repository, run_id: str, *, catalogue=None):
        self.repository = repository
        self.run_id = run_id
        self.catalogue = catalogue or {}

    def apply(self, entry: Entry) -> None:
        try:
            self._apply(entry)
        except Exception as error:  # noqa: BLE001 - a projection may not break the run
            print(f"journal projection skipped {entry.kind}: {error}", flush=True)

    def replay(self, journal: Journal) -> int:
        """Rebuild the rows from the journal. Returns how many entries were applied."""
        applied = 0
        for entry in journal.entries():
            self.apply(entry)
            applied += 1
        return applied

    def _apply(self, entry: Entry) -> None:
        data = entry.data
        if entry.kind == "run.resumed":
            self.repository.reopen_run(self.run_id, resume_canceled=True)
        elif entry.kind == "step.started":
            self.repository.reopen_run(self.run_id)
            step = data["step"]
            self._ensure_node(step)
            self.repository.set_node_status(self.run_id, step, "running")
            self.repository.record_attempt_started(
                run_id=self.run_id,
                node_id=step,
                attempt_id=data["attempt"],
                number=int(data.get("number", 1)),
                command=list(data.get("command") or ()),
                parameters=dict(data.get("parameters") or {}),
                started_at=entry.at,
            )
        elif entry.kind == "step.finished":
            step = data["step"]
            self.repository.record_attempt_finished(
                run_id=self.run_id,
                attempt_id=data["attempt"],
                status=data["status"],
                error=data.get("error"),
                finished_at=entry.at,
            )
            self.repository.set_node_status(
                self.run_id, step, data["status"], data.get("error") or None
            )
        elif entry.kind == "note":
            self.repository.append_event(
                self.run_id, "run", data.get("step") or self.run_id, {"note": data.get("text", "")}
            )
        elif entry.kind == "question":
            step = data.get("step")
            if step:
                self._ensure_node(step, refresh_existing=False)
            self.repository.add_message(
                run_id=self.run_id,
                author="agent",
                message=data.get("question", ""),
                node_id=step,
            )
        elif entry.kind == "answer":
            self.repository.add_message(
                run_id=self.run_id,
                author=data.get("author", "operator"),
                message=data.get("text", ""),
                node_id=data.get("step"),
            )
        elif entry.kind == "run.finished":
            self.repository.finish_run(self.run_id, data.get("status", "succeeded"))

    def _ensure_node(self, step: str, *, refresh_existing: bool = True) -> None:
        """Make a row for a step the run is doing but did not plan.

        The plan is a suggestion, so the agent may run something that was never in it. A step
        the dashboard has no row for would otherwise be invisible.
        """
        base = base_step(step)
        described = self.catalogue.get(base)
        dependencies = list(getattr(described, "after", ()) or ())
        if base != step:
            dependencies = needs_of(step, int(step.rsplit("_", 1)[1]) + 1, True)
        self.repository.ensure_node(
            run_id=self.run_id,
            node_id=step,
            stage_type=step,
            definition={
                "id": step,
                "title": getattr(described, "summary", step),
                "writes": getattr(described, "writes", ""),
                "paid": bool(getattr(described, "paid", False)),
                "parameter_schema": getattr(described, "parameters", {}) or {},
            },
            dependencies=dependencies,
            refresh_existing=refresh_existing,
        )
