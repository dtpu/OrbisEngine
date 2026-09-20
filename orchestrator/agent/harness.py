"""Run a coding harness (Codex CLI or Claude Code) as the reviewing agent for one attempt.

The harness brings its own file tools, so the agent can inspect and edit the attempt's outputs
in a scratch workspace. Its decision comes back as ``decision.json`` in that workspace, typed by
the same tool contracts the dispatcher enforces, so nothing the agent says is trusted until it
validates. The harness runs on the host under its own sandbox (it needs network for the model),
unlike the offline Docker sandbox in ``sandbox.py``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from orchestrator.agent.contracts import AgentResult, AgentTaskPacket
from orchestrator.agent.tools import CALL_ADAPTER, AgentToolCall
from orchestrator.workspace import atomic_json

DECISION_FILE = "decision.json"
INSTRUCTIONS_FILE = "TASK.md"
PACKET_FILE = "task.json"
PROMPT_FILE = "prompt.txt"
TRANSCRIPT_FILE = "transcript.jsonl"

# Codex announces the session it opened as the first event of a run.
THREAD_EVENT = "thread.started"

# How often the watchdog compares the transcript against its previous size.
POLL_SECONDS = 5

NUDGE = """You went quiet for {minutes:.0f} minutes, so the review was interrupted and resumed.
This is the same session: everything you had read is still yours.

Say in one line what you were doing and what you were waiting on, then finish. If a command
hung, do not start it again the same way -- run it with a timeout, on a smaller sample, or read
the file it was going to produce instead.

If you are genuinely stuck -- a tool you cannot run, an input that is not there, a question only
a person can answer -- then stop, and write {decision} with:

```json
{{"tool": "human.ask", "node_id": "{node_id}", "question": "what you need decided", "artifact_ids": []}}
```

That blocks this stage for an operator to look at and resume, which is a real outcome. Silence
is not: a review that writes nothing blocks the stage anyway, with nothing for them to read.
"""


@dataclass(frozen=True)
class HarnessPolicy:
    """Which harness to run and with what limits.

    ``kind`` is "codex", "claude", or "command" (an explicit argv, used by tests). The command
    always runs with the scratch workspace as its working directory; ``{workspace}`` in an
    explicit command is replaced with that path.
    """

    kind: str = "codex"
    model: str | None = None
    # Codex's own sandbox for the commands the agent runs. "workspace-write" confines writes to
    # the review workspace and is the right default. Some hosts cannot run Codex's bundled
    # bubblewrap at all (it fails setting up loopback or the uid map); running the agent there
    # means choosing "danger-full-access" deliberately, per host, never by silent fallback.
    sandbox: str = "workspace-write"
    timeout_seconds: int = 900
    # An agent that has written nothing for this long is interrupted and resumed with a nudge,
    # up to ``idle_nudges`` times. A hung command or a model waiting on nothing otherwise burns
    # the whole timeout in silence and the stage learns nothing from it. 0 disables the clock.
    idle_seconds: int = 300
    idle_nudges: int = 2
    command: tuple[str, ...] = ()
    # Only these host variables reach the harness. API credentials must be listed explicitly.
    passthrough: tuple[str, ...] = ("PATH", "HOME", "OPENAI_API_KEY")
    extra_environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"codex", "claude", "command"}:
            raise ValueError(f"unknown harness kind {self.kind!r}")
        if self.kind == "command" and not self.command:
            raise ValueError("a command harness needs an explicit command")
        if self.sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(f"unknown sandbox mode {self.sandbox!r}")
        if self.timeout_seconds < 1:
            raise ValueError("harness timeout must be positive")
        if self.idle_seconds < 0:
            raise ValueError("the idle clock cannot be negative")
        if self.idle_nudges < 0:
            raise ValueError("the nudge count cannot be negative")

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> HarnessPolicy:
        environ = os.environ if environ is None else environ
        kind = environ.get("WANDER_REVIEW_AGENT", "codex").strip().lower()
        command = tuple(environ.get("WANDER_REVIEW_COMMAND", "").split())
        if command:
            kind = "command"
        return cls(
            kind=kind,
            model=environ.get("WANDER_REVIEW_MODEL") or None,
            sandbox=environ.get("WANDER_REVIEW_SANDBOX", "workspace-write"),
            timeout_seconds=int(environ.get("WANDER_REVIEW_TIMEOUT", "900")),
            idle_seconds=int(environ.get("WANDER_REVIEW_IDLE", "300")),
            idle_nudges=int(environ.get("WANDER_REVIEW_NUDGES", "2")),
            command=command,
        )

    def argv(
        self, workspace: Path, instructions: Path, session: str | None = None
    ) -> tuple[str, ...]:
        if self.kind == "codex" and session:
            # Resuming keeps one session per run, so the agent remembers what it already
            # decided. `resume` takes neither --cd nor --sandbox, so the workspace comes from
            # the process directory and the sandbox from a config override.
            command = [
                "codex",
                "exec",
                "resume",
                session,
                "--json",
                "--skip-git-repo-check",
                "-c",
                f'sandbox_mode="{self.sandbox}"',
            ]
            if self.model:
                command += ["--model", self.model]
            return (*command, "-")
        if self.kind == "codex":
            # Prompt on stdin ("-"); edits are confined to the workspace by Codex's own sandbox.
            command = [
                "codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--sandbox",
                self.sandbox,
                "--cd",
                str(workspace),
            ]
            if self.model:
                command += ["--model", self.model]
            return (*command, "-")
        if self.kind == "claude":
            command = [
                "claude",
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions" if self.sandbox == "danger-full-access" else "acceptEdits",
                "--allowedTools",
                "Read",
                "Write",
                "Edit",
                "Glob",
                "Grep",
                "Bash",
                "--add-dir",
                str(workspace),
            ]
            if self.model:
                command += ["--model", self.model]
            return (*command, "-")
        # A custom harness that can continue a session says so by taking ``{session}``; that is
        # also what makes it worth nudging rather than starting over.
        return tuple(
            part.replace("{workspace}", str(workspace)).replace("{session}", session or "")
            for part in self.command
        )


def read_session(transcript: Path) -> str | None:
    """The session Codex opened, taken from the first event it emitted."""
    try:
        with transcript.open() as stream:
            for line in stream:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == THREAD_EVENT and event.get("thread_id"):
                    return str(event["thread_id"])
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class HarnessOutcome:
    result: AgentResult
    decision: AgentToolCall | None
    decision_error: str | None
    # The session this run used, so the next review of the same run can continue it.
    session: str | None = None


class Stalled(Exception):
    """The harness stopped writing for longer than the idle clock allows."""


class HarnessAgent:
    def __init__(self, policy: HarnessPolicy):
        self.policy = policy

    def run(
        self,
        packet: AgentTaskPacket,
        workspace: Path,
        instructions: str,
        session: str | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> HarnessOutcome:
        """Review one attempt, nudging the agent back to work if it goes silent.

        A review that says nothing for ``idle_seconds`` is interrupted and resumed in the same
        session with :data:`NUDGE`, which asks what it was waiting on and tells it that being
        stuck is an answer it may give. That turns the common silent failure -- a command that
        hangs -- from a lost timeout into either a finished judgement or a question an operator
        can act on. The transcript keeps every attempt, so the interruptions are visible.
        """
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        packet_path = workspace / PACKET_FILE
        instructions_path = workspace / INSTRUCTIONS_FILE
        transcript = workspace / TRANSCRIPT_FILE
        decision_path = workspace / DECISION_FILE
        prompt_path = workspace / PROMPT_FILE
        atomic_json(packet_path, packet.model_dump(mode="json"))
        instructions_path.write_text(instructions)
        decision_path.unlink(missing_ok=True)
        environment = {key: os.environ[key] for key in self.policy.passthrough if key in os.environ}
        environment.update(self.policy.extra_environment)

        prompt = instructions
        mode = "wb"
        nudges = 0
        budget = float(self.policy.timeout_seconds)
        while True:
            prompt_path.write_text(prompt)
            started = time.monotonic()
            try:
                code = self._spawn(
                    self.policy.argv(workspace, instructions_path, session),
                    workspace,
                    environment,
                    prompt_path,
                    transcript,
                    mode,
                    budget,
                    heartbeat,
                )
            except subprocess.TimeoutExpired:
                result = AgentResult(
                    status="timed_out",
                    transcript_path=str(transcript),
                    error=f"agent exceeded {self.policy.timeout_seconds}s",
                )
                return HarnessOutcome(
                    result, None, result.error, read_session(transcript) or session
                )
            except OSError as error:
                result = AgentResult(
                    status="failed",
                    transcript_path=str(transcript),
                    error=f"harness could not start: {error}",
                )
                return HarnessOutcome(result, None, result.error, session)
            except Stalled:
                budget -= time.monotonic() - started
                session = read_session(transcript) or session
                nudges += 1
                quiet = self.policy.idle_seconds / 60
                # A stall with a decision already written is a finished review whose process
                # did not exit. There is nothing to ask it, so take the decision.
                if decision_path.exists():
                    code = 0
                    break
                resumable = session is not None and (
                    self.policy.kind == "codex"
                    or any("{session}" in part for part in self.policy.command)
                )
                if resumable and nudges <= self.policy.idle_nudges and budget > POLL_SECONDS:
                    prompt = NUDGE.format(
                        minutes=quiet, decision=DECISION_FILE, node_id=packet.node_id
                    )
                    mode = "ab"
                    continue
                sent = nudges - 1
                if not resumable:
                    unanswered = "opened no session to resume"
                elif sent:
                    unanswered = f"did not answer {sent} nudge(s)"
                else:
                    unanswered = "was not nudged"
                result = AgentResult(
                    status="stalled",
                    transcript_path=str(transcript),
                    error=f"agent wrote nothing for {quiet:.0f} minutes and {unanswered}",
                )
                return HarnessOutcome(result, None, result.error, session)
            break

        status = "completed" if code == 0 else "failed"
        result = AgentResult(
            status=status,
            exit_code=code,
            transcript_path=str(transcript),
            response_path=str(decision_path) if decision_path.exists() else None,
            error=None if status == "completed" else f"agent exited {code}",
        )
        decision, decision_error = self._read_decision(decision_path, packet)
        return HarnessOutcome(result, decision, decision_error, read_session(transcript) or session)

    def _spawn(
        self,
        argv: tuple[str, ...],
        workspace: Path,
        environment: dict[str, str],
        prompt_path: Path,
        transcript: Path,
        mode: str,
        budget: float,
        heartbeat: Callable[[], None] | None = None,
    ) -> int:
        """Run one harness process, raising :class:`Stalled` if it stops writing.

        The prompt is a file rather than a pipe we write into, so a long brief cannot deadlock
        against a child that has not started reading yet. Silence is measured by the size of the
        transcript, which the child appends to as it works.
        """
        with prompt_path.open("rb") as prompt, transcript.open(mode) as output:
            process = subprocess.Popen(
                argv,
                cwd=workspace,
                env=environment,
                stdin=prompt,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started = time.monotonic()
            quiet_since = started
            written = transcript.stat().st_size
            while True:
                try:
                    return process.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                if heartbeat:
                    heartbeat()
                now = time.monotonic()
                size = transcript.stat().st_size
                if size != written:
                    written, quiet_since = size, now
                if now - started > budget:
                    _halt(process)
                    raise subprocess.TimeoutExpired(argv, budget)
                if self.policy.idle_seconds and now - quiet_since > self.policy.idle_seconds:
                    _halt(process)
                    raise Stalled()

    @staticmethod
    def _read_decision(
        path: Path, packet: AgentTaskPacket
    ) -> tuple[AgentToolCall | None, str | None]:
        if not path.is_file():
            return None, f"agent wrote no {DECISION_FILE}"
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            return None, f"{DECISION_FILE} is not valid JSON: {error}"
        try:
            call = CALL_ADAPTER.validate_python(raw)
        except Exception as error:
            return None, f"{DECISION_FILE} does not match a permitted decision: {error}"
        if call.tool not in packet.permitted_tools:
            return None, f"decision {call.tool!r} is not permitted for this review"
        if getattr(call, "node_id", packet.node_id) != packet.node_id:
            return None, "decision names a different stage than the one under review"
        return call, None


def _halt(process: subprocess.Popen) -> None:
    """Stop a harness and everything it started, and do not return until it is gone.

    The agent's own child processes are what usually hang, and they are not the process we
    spawned, so this signals the whole group the harness was given. Signalling a group is not
    atomic with waiting on one member of it: the direct child is reaped while a command it
    started is still dying, and a harness that returned then had not stopped -- its leftovers
    still held the workspace the next attempt was about to write into.
    """
    for stop in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, stop)
        except (ProcessLookupError, PermissionError):
            process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        if _group_gone(process.pid, timeout=5):
            return


def _group_gone(group: int, *, timeout: float) -> bool:
    """Whether every process in ``group`` has exited, waiting up to ``timeout`` for it."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(group, 0)
        except (ProcessLookupError, PermissionError):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
