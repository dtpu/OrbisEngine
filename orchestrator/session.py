"""One agent, one run, for as long as the run takes.

The agent is given the run's directory, the catalogue of steps, what has happened so far, and
one command to do things with. It decides what to run, looks at what came out, and either goes
on, tries something else, asks, or calls the run done. Nothing here schedules anything: this
writes the brief, starts the session, and afterwards reads the journal to see what the agent
decided -- the journal being the run's own record rather than anything this process holds.

A session ends when the run is finished, when a question is waiting for an operator, or when
the agent stops without doing either, which is itself a thing to report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import dataclasses
import os
import sys

from orchestrator.agent.harness import HarnessAgent
from orchestrator.journal import Journal, RunProjection
from orchestrator.steps import STEPS, Step, suggested_order

SESSION_FILE = "session.id"
BRIEF_FILE = "BRIEF.md"
BIN_DIR = "bin"


def install_wander(run_dir: Path) -> Path:
    """Put `wander` on the agent's PATH, pointing at the interpreter that has the code.

    The brief tells the agent to type `wander step clean`, so that has to be a real command.
    Writing it into the run rather than installing it means the agent gets the interpreter
    this process is running under, which is the one with the dependencies -- a shell that
    finds some other python is how "no module named pydantic" becomes the agent's problem.
    """
    binaries = run_dir / BIN_DIR
    binaries.mkdir(parents=True, exist_ok=True)
    shim = binaries / "wander"
    # The agent's shell starts in the run directory, so the code has to be named absolutely:
    # the interpreter alone is not enough to find a package that was never installed.
    repository = os.environ.get("WANDER_REPOSITORY") or str(Path(__file__).resolve().parents[1])
    shim.write_text(
        "#!/bin/sh\n"
        f'PYTHONPATH="{repository}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'exec {sys.executable} -m orchestrator.cli "$@"\n'
    )
    shim.chmod(0o755)
    return binaries


def describe_step(step: Step, done: bool, last: str | None) -> str:
    mark = "x" if done else " "
    line = f"- [{mark}] `{step.name}` — {step.summary}"
    if last and not done:
        line += f" (last run: {last})"
    if step.paid:
        line += "  **costs money**"
    return line


def parameter_lines(step: Step) -> list[str]:
    properties = (step.parameters or {}).get("properties") or {}
    if not properties:
        return []
    lines = [f"  Defaults for `{step.name}`, which it is already running at:"]
    for name in sorted(properties):
        rule = properties[name]
        default = repr(rule["default"]) if "default" in rule else "unset"
        lines.append(f"    - `{name}` = {default} — {rule.get('description', '')}")
    return lines


def render_brief(run_id: str, run_dir: Path, name: str, journal: Journal, goal: str) -> str:
    """What the agent is told. Written to the run directory so it is auditable afterwards."""
    done = journal.done()
    entries = journal.entries()
    waiting = journal.unanswered()
    lines = [
        f"# Run `{name}`",
        "",
        goal,
        "",
        "## Where you are",
        "",
        f"Everything for this run is in `{run_dir}`, and that directory is the run. The stages",
        "write into it and read each other's output out of it; there is no artifact store to",
        "move things through and nothing hydrates or freezes between steps. Look at the files.",
        "",
        "## Doing things",
        "",
        "```sh",
        "wander status                 # what has been done, and what the steps are",
        "wander step clean             # run a step, streaming its log",
        "wander step clean --dilate 28 # same, with a flag passed to the stage",
        'wander note "what you saw"    # write an observation into the run history',
        'wander ask "..."              # stop and ask an operator, then end your turn',
        'wander finish succeeded "..."  # the run is done',
        "```",
        "",
        "Anything else you want to run, run it. Only `wander` commands go into the history, so",
        "use the shell freely to look at files, decode frames and measure things, and use",
        "`wander note` when you have learned something the next session should not have to",
        "learn again. Give every command a timeout; if one hangs, kill it and find the answer",
        "another way rather than starting it again the same way.",
        "",
        "## The steps",
        "",
        "This order is a suggestion, not a schedule. Run them in another order if you have a",
        "reason to; run one again if you think it produced the wrong thing. The legacy pipeline",
        "checks its own dependencies against what is really in the directory, so it will tell",
        "you when something genuinely cannot run yet.",
        "",
    ]
    for step_name in suggested_order():
        step = STEPS[step_name]
        lines.append(describe_step(step, step_name in done, journal.last_status(step_name)))
        if step.caution:
            lines.append(f"      {step.caution}")
        lines.extend(parameter_lines(step))
    lines += [
        "",
        "Move a default only when you can name the thing you saw and say why that value",
        "addresses it. A tolerance or a guard you want to relax is a question for an operator,",
        "not a flag to set.",
        "",
        "## Spending",
        "",
        "Steps marked **costs money** charge every time they run. A world generation is 1600",
        "credits and a quality failure does not authorise a second one. If a paid step has",
        "already been submitted, poll it rather than submitting again. When a paid step fails",
        "ambiguously -- you cannot tell whether it charged -- stop and ask rather than guessing.",
        "",
    ]
    if entries:
        lines += ["## What has happened", ""]
        for entry in entries[-25:]:
            data = entry.data
            if entry.kind == "step.finished":
                lines.append(
                    f"- {entry.at[11:19]} `{data.get('step')}` {data.get('status')}"
                    + (f": {data.get('error')}" if data.get("error") else "")
                )
            elif entry.kind == "note":
                lines.append(f"- {entry.at[11:19]} note: {data.get('text', '')}")
            elif entry.kind == "question":
                lines.append(f"- {entry.at[11:19]} you asked: {data.get('question', '')}")
            elif entry.kind == "answer":
                lines.append(f"- {entry.at[11:19]} answered: {data.get('text', '')}")
        lines.append("")
    if waiting:
        lines += [
            "## You are waiting on an answer",
            "",
            f"You asked: {waiting}",
            "",
            "It has not been answered. Do not run the step again hoping for a different result;",
            "end your turn and leave the run where an operator can see it.",
            "",
        ]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class SessionOutcome:
    """What a turn of the agent left behind, read back from the journal."""

    finished: str | None
    waiting: str | None
    steps_run: int
    error: str | None = None

    @property
    def wants_another_turn(self) -> bool:
        return self.finished is None and self.waiting is None and self.error is None


class RunSession:
    """The agent working on one run, one turn at a time."""

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        harness: HarnessAgent,
        *,
        repository=None,
        goal: str | None = None,
    ):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        # The agent reaches back into the run through `wander`, which needs to know which run
        # it is in. Without this the agent has a brief telling it to run steps and no way to.
        self.harness = HarnessAgent(
            dataclasses.replace(
                harness.policy,
                extra_environment={
                    **harness.policy.extra_environment,
                    "WANDER_RUN_ID": run_id,
                    "WANDER_RUN_DIR": str(Path(run_dir)),
                    "WANDER_REPOSITORY": os.environ.get("WANDER_REPOSITORY", str(Path.cwd())),
                    "PATH": os.pathsep.join(
                        (str(install_wander(Path(run_dir))), os.environ.get("PATH", ""))
                    ),
                    **{
                        name: os.environ[name]
                        for name in ("WANDER_DATABASE_URL", "HOME", "VIRTUAL_ENV")
                        if name in os.environ
                    },
                },
            )
        )
        self.projection = (
            RunProjection(repository, run_id, catalogue=STEPS) if repository is not None else None
        )
        self.goal = goal or (
            "Take this clip through the pipeline until there is a world with the people from "
            "the clip moving in it, or until you have a reason to stop that an operator needs "
            "to hear."
        )

    @property
    def described(self) -> dict:
        return json.loads((self.run_dir / "run.json").read_text())

    def journal(self) -> Journal:
        return Journal(self.run_dir, projection=self.projection)

    def read_session(self) -> str | None:
        try:
            return (self.run_dir / SESSION_FILE).read_text().strip() or None
        except OSError:
            return None

    def turn(self) -> SessionOutcome:
        """Give the agent one turn, then read the journal to see what it did."""
        journal = self.journal()
        before = len([e for e in journal.entries() if e.kind == "step.started"])
        brief = render_brief(self.run_id, self.run_dir, self.described["name"], journal, self.goal)
        (self.run_dir / BRIEF_FILE).write_text(brief)
        outcome = self.harness.run(
            None,
            self.run_dir,
            brief,
            session=self.read_session(),
        )
        if outcome.session:
            (self.run_dir / SESSION_FILE).write_text(outcome.session + "\n")
        elif outcome.result.status == "stalled":
            # A session that stopped answering is not the one to ask again; the journal is
            # this run's memory, not the conversation.
            (self.run_dir / SESSION_FILE).unlink(missing_ok=True)
        after = self.journal()
        error = None
        if outcome.result.status not in {"completed", "stalled"}:
            error = outcome.result.error
        elif outcome.result.status == "stalled":
            error = outcome.result.error
        return SessionOutcome(
            finished=after.finished(),
            waiting=after.unanswered(),
            steps_run=len([e for e in after.entries() if e.kind == "step.started"]) - before,
            error=error,
        )

    def work(self, turns: int = 12) -> SessionOutcome:
        """Keep giving the agent turns until the run is settled or the turns run out.

        A turn that ran nothing and decided nothing is where this stops: the agent has either
        lost the thread or has nothing left to do, and either way another identical turn is
        unlikely to differ.
        """
        outcome = SessionOutcome(finished=None, waiting=None, steps_run=0)
        for _ in range(turns):
            outcome = self.turn()
            if not outcome.wants_another_turn:
                return outcome
            if outcome.steps_run == 0:
                self.journal().append(
                    "note",
                    text="the agent took a turn without running a step or deciding anything",
                )
                return outcome
        return outcome
