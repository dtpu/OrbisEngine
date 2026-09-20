"""Fail-closed Docker sandbox for Codex/Claude-style coding agents."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from orchestrator.agent.contracts import AgentResult, AgentTaskPacket
from orchestrator.workspace import atomic_json


@dataclass(frozen=True)
class SandboxPolicy:
    image: str
    agent_command: tuple[str, ...] = ("codex", "exec", "--json")
    timeout_seconds: int = 900
    memory: str = "2g"
    cpus: float = 2.0
    pids_limit: int = 256
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if "@sha256:" not in self.image:
            raise ValueError("agent sandbox image must be pinned by sha256 digest")
        if self.timeout_seconds < 1 or self.cpus <= 0 or self.pids_limit < 1:
            raise ValueError("sandbox resource limits must be positive")
        forbidden = {
            key
            for key in self.environment
            if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        }
        if forbidden:
            raise ValueError(
                f"secrets cannot be passed directly to the agent sandbox: {sorted(forbidden)}"
            )


class ContainerAgent:
    def __init__(
        self,
        repository: Path,
        policy: SandboxPolicy,
        *,
        docker: str = "docker",
    ):
        self.repository = Path(repository).resolve()
        self.policy = policy
        self.docker = docker

    def command(self, packet: Path, scratch: Path) -> tuple[str, ...]:
        packet = packet.resolve()
        scratch = scratch.resolve()
        if not packet.is_file():
            raise ValueError(f"agent packet is missing: {packet}")
        scratch.mkdir(parents=True, exist_ok=True)
        command = [
            self.docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.policy.pids_limit),
            "--memory",
            self.policy.memory,
            "--cpus",
            str(self.policy.cpus),
            "--mount",
            f"type=bind,src={self.repository},dst=/repo,readonly",
            "--mount",
            f"type=bind,src={packet},dst=/task/task.json,readonly",
            "--mount",
            f"type=bind,src={scratch},dst=/scratch",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=256m",
            "--workdir",
            "/scratch",
        ]
        for key, value in sorted(self.policy.environment.items()):
            command += ["--env", f"{key}={value}"]
        command += [self.policy.image, *self.policy.agent_command, "/task/task.json"]
        return tuple(command)

    def run(
        self,
        packet: AgentTaskPacket,
        workspace: Path,
    ) -> AgentResult:
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        task_path = workspace / "task.json"
        transcript = workspace / "transcript.jsonl"
        response = workspace / "response.json"
        atomic_json(task_path, packet.model_dump(mode="json"))
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", "/tmp"),
        }
        try:
            with transcript.open("wb") as output:
                completed = subprocess.run(
                    self.command(task_path, workspace / "scratch"),
                    cwd=self.repository,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=self.policy.timeout_seconds,
                    check=False,
                )
            status = "completed" if completed.returncode == 0 else "failed"
            error = None if completed.returncode == 0 else f"agent exited {completed.returncode}"
            return AgentResult(
                status=status,
                exit_code=completed.returncode,
                transcript_path=str(transcript),
                response_path=str(response) if response.exists() else None,
                error=error,
            )
        except subprocess.TimeoutExpired:
            return AgentResult(
                status="timed_out",
                transcript_path=str(transcript),
                error=f"agent exceeded {self.policy.timeout_seconds}s",
            )
