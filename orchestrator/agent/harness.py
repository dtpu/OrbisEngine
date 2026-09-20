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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from orchestrator.agent.contracts import AgentResult, AgentTaskPacket
from orchestrator.agent.tools import CALL_ADAPTER, AgentToolCall
from orchestrator.workspace import atomic_json

DECISION_FILE = "decision.json"
INSTRUCTIONS_FILE = "TASK.md"
PACKET_FILE = "task.json"
TRANSCRIPT_FILE = "transcript.jsonl"


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
            command=command,
        )

    def argv(self, workspace: Path, instructions: Path) -> tuple[str, ...]:
        if self.kind == "codex":
            # Prompt on stdin ("-"); edits are confined to the workspace by Codex's own sandbox.
            command = [
                "codex",
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--ephemeral",
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
        return tuple(part.replace("{workspace}", str(workspace)) for part in self.command)


@dataclass(frozen=True)
class HarnessOutcome:
    result: AgentResult
    decision: AgentToolCall | None
    decision_error: str | None


class HarnessAgent:
    def __init__(self, policy: HarnessPolicy):
        self.policy = policy

    def run(self, packet: AgentTaskPacket, workspace: Path, instructions: str) -> HarnessOutcome:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        packet_path = workspace / PACKET_FILE
        instructions_path = workspace / INSTRUCTIONS_FILE
        transcript = workspace / TRANSCRIPT_FILE
        decision_path = workspace / DECISION_FILE
        atomic_json(packet_path, packet.model_dump(mode="json"))
        instructions_path.write_text(instructions)
        decision_path.unlink(missing_ok=True)
        environment = {key: os.environ[key] for key in self.policy.passthrough if key in os.environ}
        environment.update(self.policy.extra_environment)
        try:
            with transcript.open("wb") as output:
                completed = subprocess.run(
                    self.policy.argv(workspace, instructions_path),
                    cwd=workspace,
                    env=environment,
                    input=instructions.encode(),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=self.policy.timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            result = AgentResult(
                status="timed_out",
                transcript_path=str(transcript),
                error=f"agent exceeded {self.policy.timeout_seconds}s",
            )
            return HarnessOutcome(result, None, result.error)
        except OSError as error:
            result = AgentResult(
                status="failed",
                transcript_path=str(transcript),
                error=f"harness could not start: {error}",
            )
            return HarnessOutcome(result, None, result.error)
        status = "completed" if completed.returncode == 0 else "failed"
        result = AgentResult(
            status=status,
            exit_code=completed.returncode,
            transcript_path=str(transcript),
            response_path=str(decision_path) if decision_path.exists() else None,
            error=None if status == "completed" else f"agent exited {completed.returncode}",
        )
        decision, decision_error = self._read_decision(decision_path, packet)
        return HarnessOutcome(result, decision, decision_error)

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
