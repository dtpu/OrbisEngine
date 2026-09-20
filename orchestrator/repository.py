"""Transactional application repository for runs, graph state, events, and operator input."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from orchestrator.contracts import Event, EventType, Run
from orchestrator.database import (
    ApprovalRecord,
    ArtifactRecord,
    QualityDecisionRecord,
    AttemptRecord,
    EventRecord,
    NodeRecord,
    OperatorMessageRecord,
    ProviderClaimRecord,
    RunRecord,
)
from orchestrator.graph import RunGraph


TERMINAL_RUN_STATUSES = frozenset({"succeeded", "failed", "canceled"})
HALTED_NODE_STATUSES = frozenset({"succeeded", "skipped", "failed", "blocked", "canceled"})


def artifact_display_name(artifact: ArtifactRecord) -> str:
    """The human-facing name for an artifact: its recorded basename, else its ID."""
    relative_path = (artifact.artifact_metadata or {}).get("relative_path")
    return Path(relative_path).name if relative_path else artifact.id


def live_run_status(recorded: str, node_statuses: list[str]) -> str:
    """The status an operator should see now.

    The workflow writes the run row only when it finishes, so while it runs the row still says
    "queued". Until then the node states say what is really happening, and in particular
    whether a human decision is the only thing left.
    """
    if recorded in TERMINAL_RUN_STATUSES:
        return recorded
    statuses = set(node_statuses)
    if "waiting_human" in statuses:
        return "waiting_human"
    if statuses & {"running", "waiting_agent", "ready"}:
        return "running"
    # Nothing is running. A failed or blocked stage now needs the operator, whatever is still
    # queued behind it (those stages cannot start until the halted one is retried).
    if statuses & {"failed", "blocked"}:
        return "blocked"
    return recorded


def artifact_name(artifact: ArtifactRecord) -> str:
    """The file name an operator recognises: the path the stage wrote, else the artifact id."""
    metadata = artifact.artifact_metadata or {}
    relative = str(metadata.get("relative_path") or "").strip()
    return relative.rsplit("/", 1)[-1] if relative else artifact.id


class PipelineRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self.sessions = sessions

    def create_run(self, run: Run, graph: RunGraph) -> None:
        with self.sessions.begin() as session:
            if session.get(RunRecord, run.id):
                raise ValueError(f"run already exists: {run.id}")
            session.add(
                RunRecord(
                    id=run.id,
                    schema_version=run.schema_version,
                    graph_schema=run.graph_schema,
                    graph_version=run.graph_version,
                    code_revision=run.code_revision,
                    container_digest=run.container_digest,
                    source_sha256=run.source_sha256,
                    source_artifact_id=run.source_artifact_id,
                    configuration=run.configuration,
                    budget=run.budget.model_dump(mode="json"),
                    status=run.status,
                    created_by=run.created_by,
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                    next_event_sequence=1,
                    parent_run_id=run.parent_run_id,
                    branch_key=run.branch_key,
                )
            )
            for node in graph.nodes.values():
                session.add(
                    NodeRecord(
                        run_id=run.id,
                        node_id=node.id,
                        stage_type=node.stage_type,
                        stage_definition=node.definition.model_dump(mode="json"),
                        dependencies=list(node.dependencies),
                        status=node.status,
                        selected_attempt_id=node.selected_attempt_id,
                        parent_node_id=node.parent_node_id,
                        branch_key=node.branch_key,
                        blocked_reason=node.blocked_reason,
                        created_at=run.created_at,
                        updated_at=run.updated_at,
                    )
                )
        self.append_event(
            run.id,
            EventType.RUN,
            run.id,
            {"status": run.status, "graphFingerprint": graph.fingerprint},
        )

    def ping(self) -> None:
        with self.sessions() as session:
            session.execute(text("SELECT 1"))

    def add_source_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str,
        sha256: str,
        size: int,
        media_type: str,
        storage_key: str,
        filename: str,
    ) -> None:
        with self.sessions.begin() as session:
            if session.get(ArtifactRecord, artifact_id):
                raise ValueError(f"artifact already exists: {artifact_id}")
            session.add(
                ArtifactRecord(
                    id=artifact_id,
                    run_id=run_id,
                    role="source_video",
                    sha256=sha256,
                    size=size,
                    media_type=media_type,
                    storage_key=storage_key,
                    artifact_metadata={
                        "relative_path": filename,
                        "original_filename": filename,
                    },
                    created_at=datetime.now(timezone.utc),
                )
            )

    def append_event(
        self,
        run_id: str,
        event_type: EventType | str,
        subject_id: str | None,
        payload: dict[str, Any],
    ) -> Event:
        with self.sessions.begin() as session:
            run = session.get(RunRecord, run_id, with_for_update=True)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            sequence = run.next_event_sequence
            run.next_event_sequence += 1
            run.updated_at = datetime.now(timezone.utc)
            event = Event(
                id=f"event:{uuid.uuid4().hex}",
                run_id=run_id,
                sequence=sequence,
                type=event_type,
                subject_id=subject_id,
                payload=payload,
            )
            session.add(
                EventRecord(
                    id=event.id,
                    run_id=event.run_id,
                    sequence=event.sequence,
                    schema_version=event.schema_version,
                    type=event.type,
                    subject_id=event.subject_id,
                    payload=event.payload,
                    created_at=event.created_at,
                )
            )
            return event

    def events_after(self, run_id: str, sequence: int, limit: int = 500) -> list[Event]:
        with self.sessions() as session:
            records = session.scalars(
                select(EventRecord)
                .where(
                    EventRecord.run_id == run_id,
                    EventRecord.sequence > sequence,
                )
                .order_by(EventRecord.sequence)
                .limit(limit)
            ).all()
            return [
                Event(
                    id=record.id,
                    run_id=record.run_id,
                    sequence=record.sequence,
                    type=record.type,
                    subject_id=record.subject_id,
                    payload=record.payload,
                    created_at=record.created_at,
                )
                for record in records
            ]

    def run_summary(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            run = session.get(RunRecord, run_id)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            nodes = session.scalars(
                select(NodeRecord).where(NodeRecord.run_id == run_id).order_by(NodeRecord.node_id)
            ).all()
            attempts = session.scalars(
                select(AttemptRecord)
                .where(AttemptRecord.run_id == run_id)
                .order_by(AttemptRecord.node_id, AttemptRecord.number)
            ).all()
            artifacts = session.scalars(
                select(ArtifactRecord)
                .where(ArtifactRecord.run_id == run_id)
                .order_by(ArtifactRecord.created_at)
            ).all()
            provider_claims = session.scalars(
                select(ProviderClaimRecord)
                .where(ProviderClaimRecord.run_id == run_id)
                .order_by(ProviderClaimRecord.created_at)
            ).all()
            decisions = session.scalars(
                select(QualityDecisionRecord)
                .where(QualityDecisionRecord.run_id == run_id)
                .order_by(QualityDecisionRecord.created_at)
            ).all()
            return {
                "id": run.id,
                "status": live_run_status(run.status, [node.status for node in nodes]),
                "recordedStatus": run.status,
                "parentRunId": run.parent_run_id,
                "branchKey": run.branch_key,
                "reviews": [
                    {
                        "id": decision.id,
                        "nodeId": decision.node_id,
                        "attemptId": decision.attempt_id,
                        "verdict": decision.verdict,
                        "rationale": decision.rationale,
                        "hypothesis": decision.hypothesis,
                        "proposedParameters": decision.proposed_parameters,
                        "decidedBy": decision.decided_by,
                        "createdAt": decision.created_at.isoformat(),
                    }
                    for decision in decisions
                ],
                "graphVersion": run.graph_version,
                "sourceSha256": run.source_sha256,
                "configuration": run.configuration,
                "budget": run.budget,
                "createdAt": run.created_at.isoformat(),
                "updatedAt": run.updated_at.isoformat(),
                "nodes": [
                    {
                        "id": node.node_id,
                        "status": node.status,
                        "dependencies": node.dependencies,
                        "selectedAttemptId": node.selected_attempt_id,
                        "blockedReason": node.blocked_reason,
                        "definition": node.stage_definition,
                    }
                    for node in nodes
                ],
                "attempts": [
                    {
                        "id": attempt.id,
                        "nodeId": attempt.node_id,
                        "number": attempt.number,
                        "status": attempt.status,
                        "parameters": attempt.parameters,
                        "costs": attempt.costs,
                        "error": attempt.error,
                        "hypothesis": attempt.hypothesis,
                        "retryOf": attempt.retry_of,
                        "providerOperationId": attempt.provider_operation_id,
                        "inputArtifactIds": attempt.input_artifact_ids,
                        "outputArtifactIds": attempt.output_artifact_ids,
                        "startedAt": attempt.started_at.isoformat() if attempt.started_at else None,
                        "finishedAt": (
                            attempt.finished_at.isoformat() if attempt.finished_at else None
                        ),
                    }
                    for attempt in attempts
                ],
                "artifacts": [
                    {
                        "id": artifact.id,
                        "role": artifact.role,
                        "sha256": artifact.sha256,
                        "size": artifact.size,
                        "mediaType": artifact.media_type,
                        "attemptId": artifact.producer_attempt_id,
                        "previewArtifactId": artifact.preview_artifact_id,
                        "name": artifact_name(artifact),
                        "contract": artifact.contract,
                        "createdAt": artifact.created_at.isoformat(),
                    }
                    for artifact in artifacts
                ],
                "providerClaims": [
                    {
                        "id": claim.id,
                        "provider": claim.provider,
                        "operation": claim.operation,
                        "status": claim.status,
                        "providerOperationId": claim.provider_operation_id,
                        "estimatedCost": claim.estimated_cost,
                        "createdAt": claim.created_at.isoformat(),
                        "updatedAt": claim.updated_at.isoformat(),
                    }
                    for claim in provider_claims
                ],
            }

    def get_artifact(self, run_id: str, artifact_id: str) -> dict[str, Any]:
        """One artifact's serving details; a mismatched run is as unknown as a missing id."""
        with self.sessions() as session:
            artifact = session.get(ArtifactRecord, artifact_id)
            if artifact is None or artifact.run_id != run_id:
                raise KeyError(f"unknown artifact: {artifact_id}")
            return {
                "id": artifact.id,
                "runId": artifact.run_id,
                "role": artifact.role,
                "sha256": artifact.sha256,
                "size": artifact.size,
                "mediaType": artifact.media_type,
                "storageKey": artifact.storage_key,
                "attemptId": artifact.producer_attempt_id,
                "name": artifact_name(artifact),
            }

    def add_quality_decision(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt_id: str,
        verdict: str,
        rationale: str,
        decided_by: str,
        evidence: list[dict[str, Any]] | None = None,
        proposed_parameters: dict[str, Any] | None = None,
        hypothesis: str | None = None,
    ) -> str:
        identifier = f"decision:{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc)
        with self.sessions.begin() as session:
            session.add(
                QualityDecisionRecord(
                    id=identifier,
                    run_id=run_id,
                    node_id=node_id,
                    attempt_id=attempt_id,
                    schema_version="wander.quality-decision/1",
                    verdict=verdict,
                    rationale=rationale,
                    evidence=evidence or [],
                    proposed_parameters=proposed_parameters or {},
                    hypothesis=hypothesis,
                    decided_by=decided_by,
                    created_at=now,
                )
            )
        self.append_event(
            run_id,
            EventType.QUALITY,
            node_id,
            {
                "decision": identifier,
                "attemptId": attempt_id,
                "verdict": verdict,
                "decidedBy": decided_by,
                "rationale": rationale,
            },
        )
        return identifier

    def load_run_state(self, run_id: str) -> dict[str, Any] | None:
        """Everything a restart needs to continue this run, or None if it has no rows yet.

        This is the whole point of the design: a run's progress is rows, not a replay log, so a
        worker started on new code resumes exactly where the previous one stopped.
        """
        with self.sessions() as session:
            run = session.get(RunRecord, run_id)
            if run is None:
                return None
            nodes = session.scalars(select(NodeRecord).where(NodeRecord.run_id == run_id)).all()
            return {
                "paused": bool(run.paused),
                "canceled": bool(run.canceled),
                "nodes": {
                    node.node_id: {
                        "stage_type": node.stage_type or node.node_id,
                        "definition": node.stage_definition,
                        "dependencies": list(node.dependencies or ()),
                        "parent_node_id": node.parent_node_id,
                        "branch_key": node.branch_key,
                        "status": node.status,
                        "blocked_reason": node.blocked_reason,
                        "selected_attempt_id": node.selected_attempt_id,
                        "selected_artifacts": dict(node.selected_artifacts or {}),
                        "agent_retries": int(node.agent_retries or 0),
                        "retry_parameters": dict(node.retry_parameters or {}),
                    }
                    for node in nodes
                },
            }

    def save_run_state(self, run_id: str, state: dict[str, Any]) -> None:
        """Persist the scheduler's view of a run so it survives losing the scheduler."""
        with self.sessions.begin() as session:
            run = session.get(RunRecord, run_id, with_for_update=True)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            run.paused = bool(state.get("paused", False))
            run.canceled = bool(state.get("canceled", False))
            now = datetime.now(timezone.utc)
            run.updated_at = now
            for node_id, value in (state.get("nodes") or {}).items():
                node = session.get(
                    NodeRecord, {"run_id": run_id, "node_id": node_id}, with_for_update=True
                )
                if node is None:
                    # A node created by expansion: one person per track, one branch per object.
                    # It has no row until now, and without one a resumed run loses the work.
                    node = NodeRecord(
                        run_id=run_id,
                        node_id=node_id,
                        stage_type=value.get("stage_type") or node_id,
                        stage_definition=value.get("definition") or {},
                        dependencies=value.get("dependencies") or [],
                        status=value["status"],
                        parent_node_id=value.get("parent_node_id"),
                        branch_key=value.get("branch_key"),
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(node)
                node.status = value["status"]
                node.blocked_reason = value.get("blocked_reason")
                node.selected_attempt_id = value.get("selected_attempt_id")
                node.selected_artifacts = value.get("selected_artifacts") or {}
                node.agent_retries = int(value.get("agent_retries") or 0)
                node.retry_parameters = value.get("retry_parameters") or {}
                node.updated_at = now

    def set_node_status(
        self, run_id: str, node_id: str, status: str, blocked_reason: str | None = None
    ) -> None:
        """Record a node state the stage activity did not write, such as agent review."""
        with self.sessions.begin() as session:
            node = session.get(
                NodeRecord, {"run_id": run_id, "node_id": node_id}, with_for_update=True
            )
            if node is None:
                raise KeyError(f"unknown node: {run_id}/{node_id}")
            node.status = status
            node.blocked_reason = blocked_reason
            node.updated_at = datetime.now(timezone.utc)
        self.append_event(
            run_id, EventType.NODE, node_id, {"status": status, "blockedReason": blocked_reason}
        )

    def reopen_run(self, run_id: str) -> bool:
        """Mark a finished run running again, for a scheduler that has just taken it over.

        A run resumed after it ended -- because a stage produced the wrong thing, or because a
        person sent one back -- kept the status it finished with, so the dashboard read
        "succeeded" while the pipeline was working. Returns whether anything changed.
        """
        with self.sessions.begin() as session:
            run = session.get(RunRecord, run_id, with_for_update=True)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            if run.status not in TERMINAL_RUN_STATUSES:
                return False
            run.status = "running"
            run.updated_at = datetime.now(timezone.utc)
        self.append_event(run_id, EventType.RUN, run_id, {"status": "running"})
        return True

    def finish_run(self, run_id: str, status: str) -> None:
        if status not in TERMINAL_RUN_STATUSES:
            raise ValueError(f"finish_run needs a terminal status, got {status!r}")
        with self.sessions.begin() as session:
            run = session.get(RunRecord, run_id, with_for_update=True)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            run.status = status
            run.updated_at = datetime.now(timezone.utc)
        self.append_event(run_id, EventType.RUN, run_id, {"status": status})

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.sessions() as session:
            identifiers = session.scalars(
                select(RunRecord.id).order_by(RunRecord.created_at.desc()).limit(limit)
            ).all()
        return [self.run_summary(identifier) for identifier in identifiers]

    def add_message(
        self,
        *,
        run_id: str,
        author: str,
        message: str,
        node_id: str | None = None,
        attempt_id: str | None = None,
        attachment_artifact_ids: list[str] | None = None,
    ) -> str:
        identifier = f"message:{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc)
        with self.sessions.begin() as session:
            session.add(
                OperatorMessageRecord(
                    id=identifier,
                    run_id=run_id,
                    node_id=node_id,
                    attempt_id=attempt_id,
                    author=author,
                    message=message,
                    attachment_artifact_ids=attachment_artifact_ids or [],
                    resolved=False,
                    created_at=now,
                )
            )
        self.append_event(
            run_id,
            EventType.OPERATOR_MESSAGE,
            node_id or run_id,
            {"id": identifier, "author": author, "message": message},
        )
        return identifier

    def list_messages(self, run_id: str) -> list[dict[str, Any]]:
        with self.sessions() as session:
            records = session.scalars(
                select(OperatorMessageRecord)
                .where(OperatorMessageRecord.run_id == run_id)
                .order_by(OperatorMessageRecord.created_at)
            ).all()
            return [
                {
                    "id": record.id,
                    "nodeId": record.node_id,
                    "attemptId": record.attempt_id,
                    "author": record.author,
                    "message": record.message,
                    "attachmentArtifactIds": record.attachment_artifact_ids,
                    "resolved": record.resolved,
                    "createdAt": record.created_at.isoformat(),
                }
                for record in records
            ]

    def add_approval(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt_id: str | None,
        kind: str,
        decision: str,
        actor: str,
        rationale: str,
    ) -> str:
        identifier = f"approval:{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc)
        with self.sessions.begin() as session:
            # Approving the same attempt twice is an operator repeating themselves, not an
            # error: the first attempt may have been recorded without reaching the run. Update
            # the existing decision and keep its identifier so the caller can still signal.
            existing = session.scalars(
                select(ApprovalRecord).where(
                    ApprovalRecord.run_id == run_id,
                    ApprovalRecord.node_id == node_id,
                    ApprovalRecord.attempt_id == attempt_id,
                    ApprovalRecord.kind == kind,
                )
            ).one_or_none()
            if existing is not None:
                existing.decision = decision
                existing.decided_by = actor
                existing.rationale = rationale
                existing.decided_at = now
                return existing.id
            session.add(
                ApprovalRecord(
                    id=identifier,
                    run_id=run_id,
                    node_id=node_id,
                    attempt_id=attempt_id,
                    kind=kind,
                    decision=decision,
                    requested_by=actor,
                    decided_by=actor,
                    rationale=rationale,
                    created_at=now,
                    decided_at=now,
                )
            )
        self.append_event(
            run_id,
            EventType.APPROVAL,
            node_id,
            {
                "id": identifier,
                "attemptId": attempt_id,
                "kind": kind,
                "decision": decision,
                "actor": actor,
                "rationale": rationale,
            },
        )
        return identifier

    def reconcile_provider_claim(
        self,
        *,
        run_id: str,
        claim_id: str,
        status: str,
        provider_operation_id: str | None,
        actor: str,
        rationale: str,
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("provider reconciliation status must be completed or failed")
        with self.sessions.begin() as session:
            claim = session.get(ProviderClaimRecord, claim_id, with_for_update=True)
            if claim is None or claim.run_id != run_id:
                raise KeyError(f"unknown provider claim: {claim_id}")
            if claim.status not in {"pending", "unknown"}:
                raise ValueError("only pending/unknown provider claims can be reconciled")
            if (
                provider_operation_id
                and claim.provider_operation_id
                and claim.provider_operation_id != provider_operation_id
            ):
                raise ValueError("provider operation ID conflicts with durable evidence")
            if provider_operation_id:
                claim.provider_operation_id = provider_operation_id
            claim.status = status
            claim.updated_at = datetime.now(timezone.utc)
        self.append_event(
            run_id,
            EventType.COST,
            claim_id,
            {
                "status": status,
                "providerOperationId": provider_operation_id,
                "actor": actor,
                "rationale": rationale,
            },
        )
