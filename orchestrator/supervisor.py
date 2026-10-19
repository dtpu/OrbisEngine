"""Give every run that needs one an agent, and stay out of the way.

This replaces the Temporal worker. It has no schedule, no queues and no graph: it looks for
runs that are not finished and not waiting on a person, and gives each one a session. What
happens inside a session is the agent's business, and what happened is in the run's journal.

Losing this process loses nothing. A run's state is its directory, so starting it again picks
every run up where it was -- including one whose session died halfway through a step, because
the journal says the step started and the directory says what it left behind.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from orchestrator.agent.harness import HarnessAgent, HarnessPolicy
from orchestrator.journal import Journal
from orchestrator.run_policy import hands_off
from orchestrator.session import RunSession
from orchestrator.config import REPOSITORY_ROOT, resolve_runs_root
from orchestrator.worker_presence import WorkerPresence
from orchestrator.upload_admission import AdmissionError, UploadAdmission
from orchestrator.steps import (
    STEPS,
    RunOptions,
    base_step,
    dependencies_within,
    people_steps,
    planned_steps,
)

RUN_FILE = "run.json"


def repository_from_environment():
    """The database mirror, when one is configured. Runs work without it."""
    url = os.environ.get("WANDER_DATABASE_URL")
    if not url:
        return None
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from orchestrator.repository import PipelineRepository

    return PipelineRepository(sessionmaker(create_engine(url), expire_on_commit=False))


def run_manifest(run_dir: Path, *, uploads_only: bool = False) -> dict | None:
    """Read a well-formed run without following a run or manifest symlink."""
    directory = None
    try:
        directory = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptor = os.open(
            RUN_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor) as manifest_file:
            if not stat.S_ISREG(os.fstat(manifest_file.fileno()).st_mode):
                return None
            manifest = json.load(manifest_file)
        if not isinstance(manifest, dict):
            return None
        run_id = manifest.get("runId")
        options = manifest.get("options", {})
        if (
            not isinstance(run_id, str)
            or not 0 < len(run_id) <= 512
            or not isinstance(manifest.get("name"), str)
            or not manifest["name"]
            or not isinstance(manifest.get("source"), str)
            or (manifest.get("goal") is not None and not isinstance(manifest["goal"], str))
            or (
                manifest.get("retainedDelivery") is not None
                and not isinstance(manifest["retainedDelivery"], dict)
            )
            or not isinstance(options, dict)
            or (uploads_only and not hands_off(options))
        ):
            return None
        automatic = hands_off(options)
        repository = manifest.get("repository")
        if automatic or manifest.get("retainedDelivery") is not None:
            if not isinstance(repository, str) or not repository:
                return None
        elif repository is not None and not isinstance(repository, str):
            return None
        # Ordinary legacy sessions fall back to the current repository and accept
        # integer strings for people. Automatic manifests require exact typed options.
        # Check the raw conversion used by RunSession too: Pydantic accepts "2.0",
        # but int("2.0") cannot be consumed by session planning.
        RunOptions.model_validate(options, strict=automatic)
        int(options.get("people") or 1)
        # Missing source files (including open_run's empty source) remain stage preflight.
        return manifest
    except (OSError, ValueError, RecursionError):
        return None
    finally:
        if directory is not None:
            os.close(directory)


def runs_needing_work(root: Path, *, uploads_only: bool = False) -> list[Path]:
    """Eligible run directories that are neither finished nor waiting on an answer."""
    pending = []
    for described in sorted(root.glob(f"*/{RUN_FILE}")):
        run_dir = described.parent
        manifest = run_manifest(run_dir, uploads_only=uploads_only)
        if manifest is None:
            continue
        journal = Journal(run_dir)
        if journal.finished():
            continue
        if journal.unanswered() and not hands_off(manifest.get("options", {})):
            continue
        if journal.control_status():
            continue
        pending.append(run_dir)
    return pending


def due(run_dir: Path, pushed: dict[str, tuple[float, int]], push_after: float) -> bool:
    """Whether to give this run another agent.

    Two things wake a run. Something happened -- a step it left running finished, an operator
    answered -- which is a new line in the journal and means there is now something to react
    to. Or the timer: a run that has gone quiet gets a fresh agent anyway, because a new
    session reads the journal rather than the conversation that gave up, and that is what
    eventually shifts something stuck.

    Without the first, an agent that correctly started a ten-minute step and ended its turn
    would sit until the timer. Without the second, a run whose agent simply stopped would sit
    for ever.
    """
    last = pushed.get(str(run_dir))
    if last is None:
        return True
    when, seen = last
    happened = len(Journal(run_dir).entries())
    return happened != seen or (time.monotonic() - when) >= push_after


@dataclass(frozen=True)
class WorkOutcome:
    message: str
    observed_entries: int
    resume_next_pass: bool = False


def work_on(run_dir: Path, repository, policy: HarnessPolicy, turns: int) -> WorkOutcome:
    import json

    described = json.loads((run_dir / RUN_FILE).read_text())
    run_id = described["runId"]
    journal = Journal(run_dir)
    observed_entries = len(journal.entries())
    if status := journal.control_status():
        return WorkOutcome(f"{run_id}: {status} by the operator", observed_entries)
    if status := journal.finished():
        return WorkOutcome(f"{run_id}: finished {status}", observed_entries)
    if repository is not None:
        # A run being picked up again is a run being worked on, whatever it last said.
        try:
            repository.reopen_run(run_id)
        except KeyError:
            pass
    session = RunSession(
        run_id,
        run_dir,
        HarnessAgent(policy),
        repository=repository,
        goal=described.get("goal"),
    )
    outcome = session.work(turns=turns)
    if outcome.finished:
        message = f"{run_id}: finished {outcome.finished}"
    elif outcome.waiting:
        message = f"{run_id}: waiting on an operator — {outcome.waiting}"
    elif outcome.error:
        message = f"{run_id}: session ended without deciding — {outcome.error}"
    else:
        message = f"{run_id}: still going after {turns} turns"
    # A fast detached step may finish before the harness's final message. If the
    # turn budget ends there, its finish is already in observed_entries, but the
    # next agent turn still needs to react to it. This is progress, not idle time.
    resume_next_pass = outcome.wants_another_turn and outcome.steps_run > 0 and outcome.running == 0
    return WorkOutcome(message, outcome.observed_entries, resume_next_pass)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        default=str(resolve_runs_root()),
        help="directory holding one folder per run",
    )
    parser.add_argument(
        "--run-id", help="select one run; stop after one pass unless --watch is set"
    )
    parser.add_argument(
        "--uploads-only",
        action="store_true",
        help="process only runs explicitly submitted with options.hands_off true",
    )
    parser.add_argument(
        "--max-uploads",
        type=int,
        help="persist a cumulative positive upload admission limit for this runs root",
    )
    parser.add_argument("--turns", type=int, default=12, help="turns per run before moving on")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once", action="store_true", help="make one pass over the runs instead of watching"
    )
    mode.add_argument(
        "--watch",
        action="store_true",
        help="keep watching, including when --run-id selects one run",
    )
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between passes")
    parser.add_argument(
        "--push-after",
        type=float,
        default=300.0,
        help="seconds before a run that went nowhere is given a fresh agent",
    )
    args = parser.parse_args(argv)
    if args.uploads_only and "WANDER_STAGE_AUTHORIZATION" in os.environ:
        parser.error(
            "--uploads-only cannot inherit WANDER_STAGE_AUTHORIZATION: "
            "a source-specific paid-stage permit must not be reused across uploads"
        )
    if args.max_uploads is not None and (args.max_uploads <= 0 or not args.uploads_only):
        parser.error("--max-uploads must be positive and requires --uploads-only")
    single_pass = args.once or (args.run_id and not args.watch)

    root = resolve_runs_root(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    try:
        with UploadAdmission(root, args.max_uploads, uploads_only=args.uploads_only) as admission:
            return supervise(args, root, bool(single_pass), admission)
    except (AdmissionError, OSError) as error:
        parser.error(f"Upload admission stopped: {error}")


def supervise(args, root: Path, single_pass: bool, admission: UploadAdmission) -> int:
    repository = repository_from_environment()
    policy = HarnessPolicy.from_environment()
    print(f"supervising {root} with {policy.kind}:{policy.model or 'default'}", flush=True)
    heartbeat = os.environ.get("WANDER_HEARTBEAT")
    pushed: dict[str, tuple[float, int]] = {}

    def scope():
        if args.max_uploads is not None:
            active = next(
                (
                    item["runId"]
                    for item in admission.state["admissions"]
                    if item["directory"] == admission.active
                ),
                None,
            )
            if active is not None and args.run_id is not None and active != args.run_id:
                raise AdmissionError("--run-id conflicts with the active upload admission")
            return {
                "run_id": active or args.run_id,
                "watcher": not single_pass and (active is not None or bool(admission.remaining)),
                "active_run_id": active,
                "admission": "occupied"
                if active is not None
                else ("exhausted" if not admission.remaining else None),
            }
        return {
            "run_id": args.run_id,
            "watcher": not single_pass,
            "manual_only": admission.state is not None,
            "active_run_id": None,
        }

    initial = scope()
    with WorkerPresence(
        root,
        run_id=initial["run_id"],
        watcher=initial["watcher"],
        uploads_only=args.uploads_only,
        manual_only=initial.get("manual_only", False),
        heartbeat=Path(heartbeat) if heartbeat else None,
    ) as presence:
        presence.set_scope(**initial)
        while True:
            with admission.dispatch():
                admission.refresh()
                presence.set_scope(**scope())
            pending = runs_needing_work(root, uploads_only=args.uploads_only)
            if admission.active is not None:
                pending.sort(key=lambda path: path.name != admission.active)
            for run_dir in pending:
                if not single_pass and not due(run_dir, pushed, args.push_after):
                    continue
                # Policy installation, claims, and automatic harness dispatch share this
                # root lock even for unrestricted workers. Detached stages never need it.
                with admission.dispatch():
                    manifest = run_manifest(run_dir, uploads_only=args.uploads_only)
                    if manifest is None or (args.run_id and manifest["runId"] != args.run_id):
                        continue
                    if not admission.claim(run_dir, manifest):
                        presence.set_scope(**scope())
                        continue
                    presence.set_scope(**scope())
                    run_id = manifest["runId"]
                    presence.set_active(run_id)
                    try:
                        current = run_manifest(run_dir, uploads_only=args.uploads_only)
                        if current != manifest:
                            if args.max_uploads is not None:
                                raise AdmissionError("Run manifest changed before dispatch")
                            continue
                        # A bounded claim is durable before the harness can start any work.
                        outcome = work_on(run_dir, repository, policy, args.turns)
                    finally:
                        admission.refresh()
                        presence.set_scope(**scope())
                print(outcome.message, flush=True)
                if outcome.resume_next_pass:
                    pushed.pop(str(run_dir), None)
                else:
                    pushed[str(run_dir)] = (time.monotonic(), outcome.observed_entries)
            if single_pass:
                return 0
            time.sleep(args.interval)


def owner_of(run_dir: Path) -> str | None:
    """The run that already holds this directory, if one does and says so."""
    import json

    try:
        return json.loads((run_dir / RUN_FILE).read_text()).get("runId")
    except (OSError, ValueError):
        return None


def name_for(root: Path, name: str, run_id: str) -> str:
    """A directory name of this run's own, keeping the clip's name where it is free.

    A run directory is named for the clip because that is what someone reading the runs root is
    looking for, and run_clip.py works in `<runs root>/<name>`, so the name and the directory
    are the same thing. Two uploads called `clip.mp4` are ordinary, though, and sharing one
    directory between them is not a collision that resolves itself: the second run reads the
    first one's bytes, the second run's `run.json` overwrites the first one's, and the first
    run can no longer be found by id at all. So a name another run holds is qualified with the
    run id, which is unique.
    """
    if owner_of(root / name) in (None, run_id) and not (root / name).exists():
        return name
    if owner_of(root / name) == run_id:
        return name
    qualified = f"{name}-{run_id}"
    held = owner_of(root / qualified)
    if held not in (None, run_id):
        raise FileExistsError(f"{root / qualified} already belongs to {held}")
    return qualified


def open_run(
    root: Path,
    *,
    run_id: str,
    name: str,
    source: Path | None,
    options: dict | None = None,
    goal: str | None = None,
    repository=None,
    source_artifact_id: str | None = None,
) -> Path:
    """Make the directory a run lives in, and the record that it exists.

    The source is copied in rather than referenced, so the run holds everything it needs and
    a run directory remains meaningful when the file it came from has moved.
    """
    import hashlib
    import json
    import shutil

    name = name_for(root, name, run_id)
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    held = run_dir / f"source{Path(source).suffix.lower()}" if source else None
    if held is not None and not held.exists():
        shutil.copy2(source, held)
    digest = None
    if held is not None:
        with held.open("rb") as copied_source:
            digest = hashlib.file_digest(copied_source, "sha256").hexdigest()
    (run_dir / RUN_FILE).write_text(
        json.dumps(
            {
                "runId": run_id,
                "name": name,
                "source": str(held) if held else "",
                "sourceSha256": digest,
                **({"sourceArtifactId": source_artifact_id} if source_artifact_id else {}),
                "repository": os.environ.get("WANDER_REPOSITORY", str(REPOSITORY_ROOT)),
                "options": options or {},
                **({"goal": goal} if goal else {}),
            },
            indent=2,
        )
    )
    if repository is not None:
        # Show the plan before anything has run, so a new run is not an empty page. These are
        # a suggestion: the agent may run something else, and the journal adds a row when it
        # does.
        # Seed the plan in the names this run's graph will use. How many people there are is
        # not known until tracking runs, but the shape is known now: a run that asked for the
        # multiperson graph gets `person_prep_00` even with one actor, and seeding the
        # single-person name leaves a row that can never run sitting beside the one that does.
        # What a step waits on has to be named the way this graph names it too. `lhm_frozen_00`
        # waits on `person_prep_00`, not on a `person_prep` that the multiperson graph never
        # has; an edge to a name that is not there is an edge the dashboard drops, and the
        # run then draws as a row of disconnected stages.
        settings = options or {}
        many = bool(settings.get("all_people")) or int(settings.get("people") or 1) > 1
        plan = people_steps(planned_steps(settings), 1, multiperson=many)
        for step in plan:
            described = STEPS[base_step(step)]
            repository.ensure_node(
                run_id=run_id,
                node_id=step,
                stage_type=step,
                definition={
                    "id": step,
                    "title": described.summary,
                    "writes": described.writes,
                    "paid": described.paid,
                    "parameter_schema": described.parameters or {},
                },
                dependencies=dependencies_within(step, plan, 1, many, settings),
            )
    return run_dir


if __name__ == "__main__":
    raise SystemExit(main())
