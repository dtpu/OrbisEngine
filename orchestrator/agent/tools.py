"""Typed, auditable tools exposed to the sandboxed coding agent."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from orchestrator.agent.contracts import AgentTaskPacket


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default_factory=lambda: f"tool:{uuid.uuid4().hex}")


class InspectGraph(ToolCall):
    tool: Literal["graph.inspect"]
    node_id: str | None = None


class PreviewArtifact(ToolCall):
    tool: Literal["artifact.preview"]
    artifact_id: str
    page: int = Field(default=0, ge=0)


class CompareArtifacts(ToolCall):
    tool: Literal["artifact.compare"]
    artifact_ids: tuple[str, ...] = Field(min_length=2, max_length=8)
    metric: Literal["visual", "metadata", "timing", "geometry"]


class ProposeParameters(ToolCall):
    tool: Literal["parameters.propose"]
    node_id: str
    parameters: dict[str, Any]
    hypothesis: str = Field(min_length=1)


class SubmitVerdict(ToolCall):
    tool: Literal["quality.verdict"]
    node_id: str
    attempt_id: str
    verdict: Literal["pass", "fail", "needs_human"]
    evidence_artifact_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class RequestRetry(ToolCall):
    tool: Literal["attempt.retry"]
    node_id: str
    attempt_id: str
    hypothesis: str = Field(min_length=1)
    parameters: dict[str, Any]


class AskHuman(ToolCall):
    tool: Literal["human.ask"]
    node_id: str
    question: str = Field(min_length=1)
    artifact_ids: tuple[str, ...] = ()


AgentToolCall = Annotated[
    InspectGraph
    | PreviewArtifact
    | CompareArtifacts
    | ProposeParameters
    | SubmitVerdict
    | RequestRetry
    | AskHuman,
    Field(discriminator="tool"),
]
CALL_ADAPTER = TypeAdapter(AgentToolCall)


class AgentToolBackend(Protocol):
    def execute(self, call: AgentToolCall) -> dict[str, Any]: ...


class AgentToolDispatcher:
    def __init__(
        self,
        packet: AgentTaskPacket,
        backend: AgentToolBackend,
        audit_log: Path,
    ):
        self.packet = packet
        self.backend = backend
        self.audit_log = Path(audit_log)

    def dispatch(self, raw: dict[str, Any]) -> dict[str, Any]:
        call = CALL_ADAPTER.validate_python(raw)
        if call.tool not in self.packet.permitted_tools:
            raise PermissionError(f"tool is not permitted for this task: {call.tool}")
        try:
            payload = self.backend.execute(call)
            result = {"id": call.id, "ok": True, "result": payload}
        except Exception as error:
            result = {
                "id": call.id,
                "ok": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        self._record(call, result)
        return result

    def _record(self, call: AgentToolCall, result: dict[str, Any]) -> None:
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runId": self.packet.run_id,
            "nodeId": self.packet.node_id,
            "call": call.model_dump(mode="json"),
            "result": result,
        }
        descriptor = os.open(
            self.audit_log,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, (json.dumps(record, separators=(",", ":")) + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
