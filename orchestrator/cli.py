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
from orchestrator.steps import STEPS, suggested_order

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


def command_step(context: RunContext, args: argparse.Namespace) -> int:
    step = args.step
    journal = context.journal()
    number = journal.attempt_number(step)
    attempt = f"{context.run_id}:{step}:{number}"
    command = run_clip_command(context, step, args.rest)
    described = STEPS.get(step)

    logs = context.run_dir / LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{step}.{number}.log"

    journal.append(
        "step.started",
        step=step,
        attempt=attempt,
        number=number,
        command=command,
        parameters=parse_flags(args.rest),
        paid=bool(described and described.paid),
    )
    print(f"$ {shlex.join(command)}", flush=True)
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
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
            code = process.wait()
        if code != 0:
            status, error = "failed", f"exited {code}"
    except KeyboardInterrupt:
        status, error = "blocked", "interrupted"
        raise
    except Exception as caught:  # noqa: BLE001 - the journal must say what happened
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
    print(
        f"{step} {status} in {time.monotonic() - started:.0f}s; log at {log.name}",
        file=sys.stderr,
        flush=True,
    )
    return 0 if status == "succeeded" else 1


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
    for name in suggested_order():
        described = STEPS[name]
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

    status = commands.add_parser("status", help="what has been done and what is left")
    status.add_argument("--tail", type=int, default=8, help="recent journal lines to show")
    status.set_defaults(handler=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(RunContext.load(), args))


if __name__ == "__main__":
    raise SystemExit(main())
