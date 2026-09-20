"""Compact, versioned packets crossing the coding-agent sandbox boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.contracts import utc_now


class AgentArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: str
    media_type: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preview_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAttemptSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    number: int = Field(ge=1)
    status: str
    hypothesis: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    artifact_ids: tuple[str, ...] = ()


class AgentTaskPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["wander.agent-task/1"] = "wander.agent-task/1"
    run_id: str
    node_id: str
    objective: str = Field(min_length=1)
    stage_definition: dict[str, Any]
    dependency_state: dict[str, str]
    artifacts: tuple[AgentArtifact, ...] = ()
    attempts: tuple[AgentAttemptSummary, ...] = ()
    operator_messages: tuple[dict[str, Any], ...] = ()
    remaining_budget: dict[str, Any] = Field(default_factory=dict)
    permitted_tools: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["wander.agent-result/1"] = "wander.agent-result/1"
    status: Literal["completed", "failed", "timed_out"]
    exit_code: int | None = None
    transcript_path: str
    response_path: str | None = None
    error: str | None = None
