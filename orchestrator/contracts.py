"""Versioned contracts shared by workflows, workers, agents, and the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUN_SCHEMA = "wander.run/1"
GRAPH_SCHEMA = "wander.graph/1"
ATTEMPT_SCHEMA = "wander.stage-attempt/1"
ARTIFACT_SCHEMA = "wander.artifact/1"
EVENT_SCHEMA = "wander.run-event/1"
QUALITY_SCHEMA = "wander.quality-decision/1"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_HUMAN = "waiting_human"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"


class NodeStatus(StrEnum):
    QUEUED = "queued"
    READY = "ready"
    RUNNING = "running"
    WAITING_AGENT = "waiting_agent"
    WAITING_HUMAN = "waiting_human"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELED = "canceled"


class AttemptStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class StageKind(StrEnum):
    COMPUTE = "compute"
    AGENT = "agent"
    HUMAN = "human"
    JOIN = "join"
    EXPAND = "expand"


class QualityVerdict(StrEnum):
    PASS = "pass"
    RETRY = "retry"
    REQUEST_HUMAN = "request_human"
    BLOCK = "block"


class EventType(StrEnum):
    RUN = "run"
    NODE = "node"
    ATTEMPT = "attempt"
    LOG = "log"
    ARTIFACT = "artifact"
    QUALITY = "quality"
    AGENT = "agent"
    OPERATOR_MESSAGE = "operator_message"
    APPROVAL = "approval"
    COST = "cost"


class ArtifactContract(Contract):
    role: Identifier
    media_type: str = Field(min_length=1)
    contract: str | None = None
    required: bool = True
    multiple: bool = False


class ArtifactBinding(Contract):
    source: Literal["run_input", "stage_output"]
    role: Identifier
    stage_id: Identifier | None = None
    selected: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> ArtifactBinding:
        if (self.source == "stage_output") != (self.stage_id is not None):
            raise ValueError("stage_id is required only for a stage_output binding")
        return self


class ResourcePolicy(Contract):
    task_queue: Identifier = "local_cpu"
    concurrency_key: Identifier | None = None
    timeout_seconds: int = Field(default=3600, ge=1)
    cpu: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    network: bool = False


class RetryPolicy(Contract):
    automatic_attempts: int = Field(default=1, ge=1)
    paid: bool = False
    requires_changed_hypothesis: bool = False
    requires_changed_parameters: bool = False
    recover_before_retry: bool = False

    @model_validator(mode="after")
    def protect_paid_submission(self) -> RetryPolicy:
        if self.paid and self.automatic_attempts != 1:
            raise ValueError("paid stages cannot be submitted automatically more than once")
        return self


class QualityPolicy(Contract):
    automatic_checks: tuple[Identifier, ...] = ()
    agent_rubric: Identifier | None = None
    human_approval: bool = False


class StageDefinition(Contract):
    schema_version: Literal["wander.stage-definition/1"] = "wander.stage-definition/1"
    id: Identifier
    title: str = Field(min_length=1)
    kind: StageKind
    executor: Identifier | None = None
    inputs: dict[Identifier, ArtifactBinding] = Field(default_factory=dict)
    outputs: dict[Identifier, ArtifactContract] = Field(default_factory=dict)
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    resources: ResourcePolicy = Field(default_factory=ResourcePolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    quality: QualityPolicy = Field(default_factory=QualityPolicy)

    @model_validator(mode="after")
    def validate_executor(self) -> StageDefinition:
        requires_executor = self.kind in {StageKind.COMPUTE, StageKind.AGENT, StageKind.EXPAND}
        if requires_executor and self.executor is None:
            raise ValueError(f"{self.kind} stages require an executor")
        if self.kind == StageKind.HUMAN and self.executor is not None:
            raise ValueError("human stages do not have an executor")
        return self


class BudgetPolicy(Contract):
    currency: str = "USD"
    maximum_cost: float | None = Field(default=None, ge=0)
    maximum_attempts_by_stage: dict[Identifier, int] = Field(default_factory=dict)
    maximum_provider_operations: dict[Identifier, int] = Field(default_factory=dict)


class Run(Contract):
    schema_version: Literal["wander.run/1"] = RUN_SCHEMA
    id: Identifier
    graph_schema: Literal["wander.graph/1"] = GRAPH_SCHEMA
    graph_version: str = Field(min_length=1)
    code_revision: str = Field(min_length=7)
    container_digest: str | None = None
    source_sha256: Sha256
    source_artifact_id: Identifier
    configuration: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    status: RunStatus = RunStatus.QUEUED
    created_by: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    parent_run_id: Identifier | None = None
    branch_key: str | None = None


class Artifact(Contract):
    schema_version: Literal["wander.artifact/1"] = ARTIFACT_SCHEMA
    id: Identifier
    run_id: Identifier
    role: Identifier
    sha256: Sha256
    size: int = Field(ge=0)
    media_type: str = Field(min_length=1)
    storage_key: str = Field(min_length=1)
    producer_attempt_id: Identifier | None = None
    contract: str | None = None
    preview_artifact_id: Identifier | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CostRecord(Contract):
    provider: Identifier
    amount: float = Field(ge=0)
    currency: str = "USD"
    units: dict[str, float] = Field(default_factory=dict)
    estimated: bool = True


class StageAttempt(Contract):
    schema_version: Literal["wander.stage-attempt/1"] = ATTEMPT_SCHEMA
    id: Identifier
    run_id: Identifier
    node_id: Identifier
    number: int = Field(ge=1)
    retry_of: Identifier | None = None
    status: AttemptStatus = AttemptStatus.PENDING
    hypothesis: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    command: tuple[str, ...] = ()
    code_revision: str = Field(min_length=7)
    environment_fingerprint: Sha256
    input_artifact_ids: tuple[Identifier, ...] = ()
    output_artifact_ids: tuple[Identifier, ...] = ()
    provider_operation_id: str | None = None
    costs: tuple[CostRecord, ...] = ()
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> StageAttempt:
        terminal = self.status in {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELED,
        }
        if terminal and self.finished_at is None:
            raise ValueError("terminal attempts require finished_at")
        if self.finished_at is not None and self.started_at is None:
            raise ValueError("finished_at requires started_at")
        return self


class Evidence(Contract):
    artifact_id: Identifier
    observation: str = Field(min_length=1)


class QualityDecision(Contract):
    schema_version: Literal["wander.quality-decision/1"] = QUALITY_SCHEMA
    id: Identifier
    run_id: Identifier
    node_id: Identifier
    attempt_id: Identifier
    verdict: QualityVerdict
    rationale: str = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()
    proposed_parameters: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str | None = None
    decided_by: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_retry(self) -> QualityDecision:
        if self.verdict == QualityVerdict.RETRY and not self.hypothesis:
            raise ValueError("retry decisions require a hypothesis")
        return self


class Event(Contract):
    schema_version: Literal["wander.run-event/1"] = EVENT_SCHEMA
    id: Identifier
    run_id: Identifier
    sequence: int = Field(ge=1)
    type: EventType
    subject_id: Identifier | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
