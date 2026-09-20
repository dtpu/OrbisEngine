"""`wander` -- what the agent uses to do things to a run.

The agent has a shell, and most of what it does with it needs no help from us: reading a JSON
file, decoding a frame, measuring a mask. This is for the few things that have to be written
down -- running a pipeline step, saying what it saw, asking a question, calling the run done --
so that a run's history survives the session that made it.

Everything here works on one run, named by ``WANDER_RUN_ID`` and ``WANDER_RUN_DIR`` in the
environment the agent was started with. Commands are deliberately few and hard to misuse: the
interesting decisions are the agent's, and this is only how it records them.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.journal import Journal, RunProjection
from orchestrator.steps import (
    SAME_AS,
    STEPS,
    base_step,
    needs_of,
    people_steps,
    planned_steps,
)

RUN_FILE = "run.json"
LOG_DIR = "logs"


@dataclass(frozen=True)
class RunContext:
    """Everything a command needs to act on a run, read from its own directory."""

    run_id: str
    run_dir: Path
    name: str
    source: Path
    repository: Path
    options: dict[str, Any]

    @classmethod
    def load(cls) -> RunContext:
        run_dir = Path(os.environ.get("WANDER_RUN_DIR", "")).resolve()
        if not run_dir.is_dir():
            raise SystemExit("WANDER_RUN_DIR is not set to a run directory")
        try:
            described = json.loads((run_dir / RUN_FILE).read_text())
        except OSError as error:
            raise SystemExit(f"{run_dir / RUN_FILE} is missing or unreadable: {error}") from error
        return cls(
            run_id=os.environ.get("WANDER_RUN_ID") or described["runId"],
            run_dir=run_dir,
            name=described["name"],
            source=Path(described["source"]),
            repository=Path(
                described.get("repository") or os.environ.get("WANDER_REPOSITORY", ".")
            ),
            options=described.get("options") or {},
        )

    def journal(self) -> Journal:
        return Journal(self.run_dir, projection=projection(self.run_id))


def projection(run_id: str) -> RunProjection | None:
    """The database mirror, when there is a database to mirror into.

    The agent can work on a run with no database at all -- the journal and the directory are
    the run. A missing or broken database must not stop a step from running.
    """
    url = os.environ.get("WANDER_DATABASE_URL")
    if not url:
        return None
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from orchestrator.repository import PipelineRepository

        sessions = sessionmaker(create_engine(url), expire_on_commit=False)
        return RunProjection(PipelineRepository(sessions), run_id, catalogue=STEPS)
    except Exception as error:  # noqa: BLE001 - the run matters more than its mirror
        print(f"not recording to the database: {error}", file=sys.stderr)
        return None


def run_clip_command(context: RunContext, step: str, extra: list[str]) -> list[str]:
    """The legacy command for one step, in this run's own directory.

    run_clip.py keeps its own state in that directory and checks its own dependencies against
    it, so the order the agent chooses is checked against what has really been done rather than
    against anything this process believes.
    """
    command = [
        sys.executable,
        str(context.repository / "scripts/run_clip.py"),
        "--clip",
        str(context.source),
        "--name",
        context.name,
        "--only",
        step,
        "--no-publish",
        "--stage-ledger",
        str(context.run_dir / "paid-ledger.json"),
    ]
    marble = context.options.get("marble")
    if marble:
        command += ["--marble", str(marble)]
    people = context.options.get("people")
    if isinstance(people, int) and people > 1:
        command += ["--people", str(people)]
    elif context.options.get("all_people"):
        command.append("--all-people")
    if context.options.get("one_shot", True):
        command.append("--one-shot")
    return command + extra


def step_environment(context: RunContext) -> dict[str, str]:
    """Point the legacy pipeline at this run's directory and nowhere else."""
    root = context.run_dir
    return {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "WANDER_RUNS_DIR": str(root.parent),
        "WANDER_PUBLIC_DIR": str(root / "public"),
        "WANDER_CLIPS_DIR": str(root / "clips"),
        "WANDER_MARBLE_DIR": str(root / "marble"),
        "WANDER_SHARE_DIR": str(root / "share"),
    }


def missing_credentials(step: str) -> list[str]:
    """Environment a step needs that this process does not have."""
    described = STEPS.get(base_step(step))
    if described is None:
        return []
    return [name for name in described.credentials if not os.environ.get(name)]


# Flags that belong to `wander step` rather than to the stage it runs. `rest` is a REMAINDER,
# so anything after the step name lands in it -- including these, which then reach run_clip.py
# and are rejected. Writing them after the name is the natural thing to do and what the
# standing instructions show, so take them back rather than failing on them.
OURS = {"--wait": "wait", "--stream": "stream"}


def take_our_flags(args: argparse.Namespace) -> None:
    kept = []
    for token in args.rest:
        if token in OURS:
            setattr(args, OURS[token], True)
        else:
            kept.append(token)
    args.rest = kept


def command_step(context: RunContext, args: argparse.Namespace) -> int:
    take_our_flags(args)
    """Start a step. By default it runs on its own and this returns at once.

    A step can take ten minutes on a GPU, and an agent that sits watching one is an agent
    holding a session open to do nothing. So the work is detached and journalled by the child
    that runs it: start it, stop your turn, and you will be given another when it lands.
    """
    absent = missing_credentials(args.step)
    if absent:
        # Running it anyway spends a turn to be told by the stage itself, and for a paid step
        # an ambiguous failure is worse than none.
        print(
            f"{args.step} needs {', '.join(absent)} and this run does not have it. "
            "That is an operator's to fix, not yours: `wander ask` for it.",
            file=sys.stderr,
        )
        return 2
    if not args.wait:
        return detach(context, args)
    step = args.step
    journal = context.journal()
    number = args.number or journal.attempt_number(step)
    attempt = args.attempt or f"{context.run_id}:{step}:{number}"
    command = run_clip_command(context, step, args.rest)
    described = STEPS.get(step)

    logs = context.run_dir / LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{step}.{number}.log"

    if not args.attempt:
        # A detached step's start is written by whoever detached it, so the run records the
        # step as under way the moment it is asked for rather than once the child gets going.
        journal.append(
            "step.started",
            step=step,
            attempt=attempt,
            number=number,
            command=command,
            parameters=parse_flags(args.rest),
            paid=bool(described and described.paid),
            log=str(log.relative_to(context.run_dir)),
        )
    started = time.monotonic()
    status, error = "succeeded", None
    try:
        with log.open("wb") as handle:
            process = subprocess.Popen(
                command,
                cwd=context.repository,
                env=step_environment(context),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for line in process.stdout:  # type: ignore[union-attr]
                handle.write(line)
                handle.flush()
                if args.stream:
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()
            code = process.wait()
        if code != 0:
            status, error = "failed", f"exited {code}"
    except BaseException as caught:  # noqa: BLE001 - the journal must say what happened
        status, error = "failed", f"{type(caught).__name__}: {caught}"
    finally:
        journal.append(
            "step.finished",
            step=step,
            attempt=attempt,
            status=status,
            error=error,
            seconds=round(time.monotonic() - started, 1),
            log=str(log.relative_to(context.run_dir)),
        )
    return 0 if status == "succeeded" else 1


def detach(context: RunContext, args: argparse.Namespace) -> int:
    """Hand the step to a child that outlives this command, and say where to watch it."""
    if os.environ.get("WANDER_STEP_CHILD"):
        raise SystemExit("a detached step tried to detach again; refusing to spawn")
    journal = context.journal()
    number = journal.attempt_number(args.step)
    attempt = f"{context.run_id}:{args.step}:{number}"
    log = Path(LOG_DIR) / f"{args.step}.{number}.log"
    described = STEPS.get(args.step)
    journal.append(
        "step.started",
        step=args.step,
        attempt=attempt,
        number=number,
        command=run_clip_command(context, args.step, args.rest),
        parameters=parse_flags(args.rest),
        paid=bool(described and described.paid),
        log=str(log),
    )
    # The flags go before the step name: `rest` is a REMAINDER, so anything after the name is
    # swallowed as a flag for run_clip.py. Putting --wait after it meant the child never saw
    # it, detached again, and spawned for ever.
    command = [
        sys.executable,
        "-m",
        "orchestrator.cli",
        "step",
        "--wait",
        "--number",
        str(number),
        "--attempt",
        attempt,
        args.step,
        *args.rest,
    ]
    subprocess.Popen(
        command,
        cwd=context.repository,
        env={
            **os.environ,
            "PYTHONPATH": str(context.repository),
            "WANDER_STEP_CHILD": "1",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    described = STEPS.get(args.step)
    print(
        f"{args.step} started"
        + ("  (this one costs money)" if described and described.paid else "")
    )
    print(f"  log      {log}   `wander log {args.step}` once it has written some")
    print("  you do not have to wait for it: end your turn, and you will be given another")
    print("  when it finishes. `wander ready` will show what it unblocked.")
    return 0


def parse_flags(rest: list[str]) -> dict[str, Any]:
    """The extra flags, as a mapping, for the record rather than for re-running anything."""
    flags: dict[str, Any] = {}
    index = 0
    while index < len(rest):
        token = rest[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:].replace("-", "_")
        if index + 1 < len(rest) and not rest[index + 1].startswith("--"):
            flags[name] = rest[index + 1]
            index += 2
        else:
            flags[name] = True
            index += 1
    return flags


def completed(context: RunContext) -> set[str]:
    """Steps the run directory says are finished.

    run_clip.py records each stage in state.json as it completes, so this is what the pipeline
    itself believes rather than what the journal remembers. The two differ when a stage was run
    outside `wander step`, when a run directory was recovered from somewhere else, or when a
    step half-finished: the directory is the thing later stages actually read, so it wins.
    """
    done: set[str] = set()
    try:
        stages = json.loads((context.run_dir / "state.json").read_text()).get("stages") or {}
    except (OSError, ValueError):
        stages = {}
    for name, record in stages.items():
        if name.startswith("_"):
            continue  # run_clip's own bookkeeping, not a stage
        if isinstance(record, dict) and record.get("status") == "ok":
            done.add(name)
    done |= Journal(context.run_dir).done()
    return {name for step in done for name in (step, SAME_AS.get(step, step))}


def tracked_people(context: RunContext) -> int:
    """How many people this run really found, once tracking has said.

    Before that the answer is the cap the run asked for, which is a guess; run_clip.py names
    a stage per person, so guessing sixteen would fill the list with steps that will never
    exist.
    """
    try:
        report = json.loads((context.run_dir / "tracks" / "tracks.json").read_text())
    except (OSError, ValueError):
        return 1
    return max(1, int(report.get("trackCount") or len(report.get("tracks") or ())))


def graph_shape(context: RunContext) -> tuple[int, bool]:
    """How many people this run has, and whether its graph names them separately."""
    options = context.options
    many = bool(options.get("all_people")) or int(options.get("people") or 1) > 1
    return tracked_people(context), many


def plan_for(context: RunContext) -> list[str]:
    """The steps this run has, named the way run_clip.py will name them.

    What is on the list is what the run was opened for: a run submitted with no world and no
    objects has no `marble_video` and no `objects`, because the agent is told to start
    everything that is ready and a catalogue entry on this list is work it will do. Listing
    the whole catalogue instead is how an operator who excluded a 1600-credit world gets one.

    Whether the per-person stages are `person_prep` or `person_prep_00` is decided by the
    shape of the graph the run asked for, not by how many people turned up: a run with
    --people 16 uses the multiperson graph even when tracking finds one. How many of them
    there are does come from tracking, so the list does not fill with fifteen stages that
    will never exist.
    """
    people, many = graph_shape(context)
    return people_steps(planned_steps(context.options), people, multiperson=many)


def waiting_on(step: str, done: set[str], people: int = 1, multiperson: bool = False) -> list[str]:
    """The prerequisites of a step that are not done."""
    return [need for need in needs_of(step, people, multiperson) if need not in done]


def in_flight(journal) -> dict[str, str]:
    """Steps started and not yet finished, by attempt."""
    running: dict[str, str] = {}
    for entry in journal.entries():
        step = entry.data.get("step")
        if entry.kind == "step.started" and step:
            running[step] = entry.data.get("attempt", "")
        elif entry.kind == "step.finished" and step:
            running.pop(step, None)
    return running


def command_ready(context: RunContext, args: argparse.Namespace) -> int:
    done = completed(context)
    journal = context.journal()
    people, many = graph_shape(context)
    running = in_flight(journal)
    ready, blocked = [], []
    for name in plan_for(context):
        if name in done:
            continue
        (blocked if waiting_on(name, done, people, many) else ready).append(name)
    if running:
        print("already running (you do not have to wait for these):")
        for name in running:
            print(f"  {name}")
        print()
    print("ready to run now:")
    ready = [name for name in ready if name not in running]
    if not ready:
        print("  (nothing — everything is either done or waiting on something)")
    for name in ready:
        step = STEPS[base_step(name)]
        last = journal.last_status(name)
        mark = "  PAID" if step.paid else ""
        again = f"  (last run: {last})" if last else ""
        print(f"  {name:<16}{mark:<7}{step.summary}{again}")
    if args.all and blocked:
        print("\nwaiting on something:")
        for name in blocked:
            print(f"  {name:<16} wants {', '.join(waiting_on(name, done, people, many))}")
    if done:
        print(f"\ndone: {', '.join(sorted(done & set(STEPS)))}")
    print(
        "\nThis is what the run directory says, and it is advice, not a gate: run something "
        "else if you have a reason to."
    )
    if ready:
        print(
            "Start all of these that are independent before you stop -- something already "
            "running is not a reason to wait."
        )
    return 0


def command_show(context: RunContext, args: argparse.Namespace) -> int:
    name = args.step
    step = STEPS.get(base_step(name))
    if step is None:
        print(f"no step called {name}; `wander ready` lists them", file=sys.stderr)
        return 1
    done = completed(context)
    journal = context.journal()
    people, many = graph_shape(context)
    print(f"{name} — {step.summary}")
    print(f"  writes     {step.writes}")
    if step.paid:
        print("  costs      money, every time it runs")
    if step.caution:
        print(f"  careful    {step.caution}")
    outstanding = waiting_on(name, done, people, many)
    print(f"  wants      {', '.join(needs_of(name, people, many)) or 'nothing'}")
    print(f"  waiting on {', '.join(outstanding) if outstanding else 'nothing — it can run now'}")
    properties = (step.parameters or {}).get("properties") or {}
    if properties:
        print("  flags      (it is already running at these)")
        for knob in sorted(properties):
            rule = properties[knob]
            default = repr(rule["default"]) if "default" in rule else "unset"
            print(
                f"    --{knob.replace('_', '-'):<16} = {default:<10} {rule.get('description', '')}"
            )
    attempts = [
        entry
        for entry in journal.entries()
        if entry.kind == "step.finished" and entry.data.get("step") == name
    ]
    if attempts:
        print("  attempts")
        for entry in attempts:
            data = entry.data
            detail = f" :: {data['error']}" if data.get("error") else ""
            print(
                f"    {entry.at[11:19]} {data.get('status')} in {data.get('seconds')}s"
                f"  log {data.get('log', '-')}{detail}"
            )
    return 0


def command_log(context: RunContext, args: argparse.Namespace) -> int:
    """The log from a step's last run, for reading what it actually said."""
    entries = [
        entry
        for entry in context.journal().entries()
        if entry.kind == "step.finished" and entry.data.get("step") == args.step
    ]
    if not entries or not entries[-1].data.get("log"):
        print(f"{args.step} has not run here yet", file=sys.stderr)
        return 1
    path = context.run_dir / entries[-1].data["log"]
    if not path.is_file():
        print(f"{path} is gone", file=sys.stderr)
        return 1
    lines = path.read_text(errors="replace").splitlines()
    for line in lines[-args.tail :]:
        print(line)
    return 0


def command_note(context: RunContext, args: argparse.Namespace) -> int:
    context.journal().append("note", text=args.text, step=args.step)
    return 0


def command_ask(context: RunContext, args: argparse.Namespace) -> int:
    context.journal().append("question", question=args.question, step=args.step)
    print(
        "asked, and the run is waiting. An operator answers with `wander answer`; "
        "stop working on this and finish your turn.",
        file=sys.stderr,
    )
    return 0


def command_answer(context: RunContext, args: argparse.Namespace) -> int:
    context.journal().append("answer", text=args.text, author=args.author, step=args.step)
    return 0


def command_finish(context: RunContext, args: argparse.Namespace) -> int:
    context.journal().append("run.finished", status=args.status, summary=args.summary)
    return 0


def command_status(context: RunContext, args: argparse.Namespace) -> int:
    journal = context.journal()
    done = journal.done()
    print(f"run {context.run_id} in {context.run_dir}")
    if journal.finished():
        print(f"  finished: {journal.finished()}")
    waiting = journal.unanswered()
    if waiting:
        print(f"  waiting on an answer to: {waiting}")
    print("\n  step              last        what it is for")
    for name in plan_for(context):
        described = STEPS[base_step(name)]
        last = journal.last_status(name) or "-"
        mark = "x" if name in done else " "
        print(f"  [{mark}] {name:<15} {last:<10}  {described.summary}")
    entries = journal.entries()[-int(args.tail) :] if args.tail else []
    if entries:
        print("\n  recently")
        for entry in entries:
            detail = entry.data.get("step") or entry.data.get("text") or entry.data.get("question")
            print(f"    {entry.at[11:19]} {entry.kind:<14} {str(detail)[:70]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wander", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    step = commands.add_parser("step", help="run a pipeline step and record it")
    step.add_argument("step", help="the step to run; `wander status` lists them")
    step.add_argument(
        "--wait",
        action="store_true",
        help="run it here and block until it finishes, instead of leaving it to run",
    )
    step.add_argument("--stream", action="store_true", help="with --wait, echo the log as it goes")
    step.add_argument("--number", type=int, help=argparse.SUPPRESS)
    step.add_argument("--attempt", help=argparse.SUPPRESS)
    step.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="extra flags passed straight to run_clip.py, such as --dilate 28",
    )
    step.set_defaults(handler=command_step)

    note = commands.add_parser("note", help="write an observation into the run's history")
    note.add_argument("text")
    note.add_argument("--step")
    note.set_defaults(handler=command_note)

    ask = commands.add_parser("ask", help="stop and ask an operator")
    ask.add_argument("question")
    ask.add_argument("--step")
    ask.set_defaults(handler=command_ask)

    answer = commands.add_parser("answer", help="answer a question the run is waiting on")
    answer.add_argument("text")
    answer.add_argument("--author", default="operator")
    answer.add_argument("--step")
    answer.set_defaults(handler=command_answer)

    finish = commands.add_parser("finish", help="declare the run over")
    finish.add_argument("status", choices=("succeeded", "failed", "blocked"))
    finish.add_argument("summary")
    finish.set_defaults(handler=command_finish)

    ready = commands.add_parser("ready", help="what can run right now")
    ready.add_argument("--all", action="store_true", help="also show what is waiting, and on what")
    ready.set_defaults(handler=command_ready)

    show = commands.add_parser("show", help="one step in detail: flags, prerequisites, attempts")
    show.add_argument("step")
    show.set_defaults(handler=command_show)

    log = commands.add_parser("log", help="the log from a step's last run")
    log.add_argument("step")
    log.add_argument("--tail", type=int, default=60)
    log.set_defaults(handler=command_log)

    status = commands.add_parser("status", help="what has been done and what is left")
    status.add_argument("--tail", type=int, default=8, help="recent journal lines to show")
    status.set_defaults(handler=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(RunContext.load(), args))


if __name__ == "__main__":
    raise SystemExit(main())
