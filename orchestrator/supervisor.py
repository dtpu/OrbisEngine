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
import os
import sys
import time
from pathlib import Path

from orchestrator.agent.harness import HarnessAgent, HarnessPolicy
from orchestrator.journal import Journal
from orchestrator.session import RunSession
from orchestrator.steps import STEPS, base_step, people_steps, planned_steps

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


def runs_needing_work(root: Path) -> list[Path]:
    """Run directories that are neither finished nor waiting on an answer."""
    pending = []
    for described in sorted(root.glob(f"*/{RUN_FILE}")):
        run_dir = described.parent
        journal = Journal(run_dir)
        if journal.finished() or journal.unanswered():
            continue
        if (run_dir / "paused").exists():
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


def work_on(run_dir: Path, repository, policy: HarnessPolicy, turns: int) -> str:
    import json

    described = json.loads((run_dir / RUN_FILE).read_text())
    run_id = described["runId"]
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
        return f"{run_id}: finished {outcome.finished}"
    if outcome.waiting:
        return f"{run_id}: waiting on an operator — {outcome.waiting}"
    if outcome.error:
        return f"{run_id}: session ended without deciding — {outcome.error}"
    return f"{run_id}: still going after {turns} turns"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        default=os.environ.get("WANDER_RUNS_ROOT", ".context/runs"),
        help="directory holding one folder per run",
    )
    parser.add_argument("--run-id", help="work on one run and stop")
    parser.add_argument("--turns", type=int, default=12, help="turns per run before moving on")
    parser.add_argument(
        "--once", action="store_true", help="make one pass over the runs instead of watching"
    )
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between passes")
    parser.add_argument(
        "--push-after",
        type=float,
        default=300.0,
        help="seconds before a run that went nowhere is given a fresh agent",
    )
    args = parser.parse_args(argv)

    root = Path(args.runs).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = repository_from_environment()
    policy = HarnessPolicy.from_environment()
    print(f"supervising {root} with {policy.kind}:{policy.model or 'default'}", flush=True)

    while True:
        pending = runs_needing_work(root)
        if args.run_id:
            import json

            pending = [
                d for d in pending if json.loads((d / RUN_FILE).read_text())["runId"] == args.run_id
            ]
        for run_dir in pending:
            print(work_on(run_dir, repository, policy, args.turns), flush=True)
        if args.once or args.run_id:
            return 0
        time.sleep(args.interval)


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
    import json
    import shutil

    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    held = run_dir / f"source{Path(source).suffix.lower()}" if source else None
    if held is not None and not held.exists():
        shutil.copy2(source, held)
    (run_dir / RUN_FILE).write_text(
        json.dumps(
            {
                "runId": run_id,
                "name": name,
                "source": str(held) if held else "",
                **({"sourceArtifactId": source_artifact_id} if source_artifact_id else {}),
                "repository": os.environ.get("WANDER_REPOSITORY", str(Path.cwd())),
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
        settings = options or {}
        many = bool(settings.get("all_people")) or int(settings.get("people") or 1) > 1
        for step in people_steps(planned_steps(settings), 1, multiperson=many):
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
                dependencies=list(described.after),
            )
    return run_dir


if __name__ == "__main__":
    raise SystemExit(main())
