"""Relational persistence for durable pipeline state and ordered run events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Json = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    graph_schema: Mapped[str] = mapped_column(String(64), nullable=False)
    graph_version: Mapped[str] = mapped_column(String(128), nullable=False)
    code_revision: Mapped[str] = mapped_column(String(128), nullable=False)
    container_digest: Mapped[str | None] = mapped_column(String(256))
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_artifact_id: Mapped[str] = mapped_column(String(128), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    budget: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    next_event_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    # Set on runs the workflow spawns for one admitted shot of a parent run's source.
    parent_run_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE")
    )
    branch_key: Mapped[str | None] = mapped_column(String(128))
    # Operator intent. It belongs beside the run, not inside a workflow's memory.
    paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    canceled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class NodeRecord(Base):
    __tablename__ = "pipeline_nodes"
    __table_args__ = (UniqueConstraint("run_id", "node_id"),)

    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), primary_key=True
    )
    node_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    stage_definition: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    dependencies: Mapped[list[dict[str, Any]]] = mapped_column(Json, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    selected_attempt_id: Mapped[str | None] = mapped_column(String(128))
    # The artifact IDs chosen per output role when this node's attempt was selected. Without it
    # a restart knows a node succeeded but not which artifacts downstream stages must consume.
    selected_artifacts: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    # Reviewing-agent state, kept here so a restart resumes the loop where it left off rather
    # than handing the agent a fresh budget.
    agent_retries: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    retry_parameters: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    parent_node_id: Mapped[str | None] = mapped_column(String(128))
    branch_key: Mapped[str | None] = mapped_column(String(128))
    blocked_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AttemptRecord(Base):
    __tablename__ = "pipeline_attempts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["run_id", "node_id"],
            ["pipeline_nodes.run_id", "pipeline_nodes.node_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("run_id", "node_id", "number"),
        Index("ix_pipeline_attempts_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    node_id: Mapped[str] = mapped_column(String(128), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    retry_of: Mapped[str | None] = mapped_column(String(128), ForeignKey("pipeline_attempts.id"))
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    parameters: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    command: Mapped[list[str]] = mapped_column(Json, nullable=False, default=list)
    code_revision: Mapped[str] = mapped_column(String(128), nullable=False)
    environment_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    input_artifact_ids: Mapped[list[str]] = mapped_column(Json, nullable=False, default=list)
    output_artifact_ids: Mapped[list[str]] = mapped_column(Json, nullable=False, default=list)
    provider_operation_id: Mapped[str | None] = mapped_column(String(256))
    costs: Mapped[list[dict[str, Any]]] = mapped_column(Json, nullable=False, default=list)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class ArtifactRecord(Base):
    __tablename__ = "pipeline_artifacts"
    # No uniqueness beyond the primary key: freeze_attempt derives the artifact ID from
    # run, attempt, relative path and digest, so one attempt may legitimately record the same
    # bytes at two paths (a staged input that the stage also republishes under outputs/).
    __table_args__ = (Index("ix_pipeline_artifacts_sha256", "sha256"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(128), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(String(256), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    producer_attempt_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("pipeline_attempts.id")
    )
    contract: Mapped[str | None] = mapped_column(String(128))
    preview_artifact_id: Mapped[str | None] = mapped_column(
        String(128), ForeignKey("pipeline_artifacts.id")
    )
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EventRecord(Base):
    __tablename__ = "pipeline_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence"),
        Index("ix_pipeline_events_resume", "run_id", "sequence"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str | None] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OperatorMessageRecord(Base):
    __tablename__ = "pipeline_operator_messages"
    __table_args__ = (Index("ix_pipeline_messages_scope", "run_id", "node_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str | None] = mapped_column(String(128))
    attempt_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("pipeline_attempts.id"))
    author: Mapped[str] = mapped_column(String(256), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    attachment_artifact_ids: Mapped[list[str]] = mapped_column(Json, nullable=False, default=list)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ApprovalRecord(Base):
    __tablename__ = "pipeline_approvals"
    __table_args__ = (
        UniqueConstraint("run_id", "node_id", "attempt_id", "kind"),
        Index("ix_pipeline_approvals_pending", "run_id", "decision"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_id: Mapped[str | None] = mapped_column(String(128), ForeignKey("pipeline_attempts.id"))
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    requested_by: Mapped[str] = mapped_column(String(256), nullable=False)
    decided_by: Mapped[str | None] = mapped_column(String(256))
    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class QualityDecisionRecord(Base):
    __tablename__ = "pipeline_quality_decisions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_attempts.id"), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(Json, nullable=False, default=list)
    proposed_parameters: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False, default=dict)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    decided_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProviderClaimRecord(Base):
    __tablename__ = "pipeline_provider_claims"
    __table_args__ = (
        UniqueConstraint("provider", "idempotency_key"),
        Index("ix_pipeline_claims_source_stage", "source_sha256", "logical_stage"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False
    )
    attempt_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("pipeline_attempts.id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    logical_stage: Mapped[str] = mapped_column(String(128), nullable=False)
    operation: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_operation_id: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    estimated_cost: Mapped[float | None] = mapped_column(Float)
