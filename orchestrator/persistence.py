"""Database lifecycle hooks for Temporal stage activities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import func, select

from orchestrator.artifacts import AttemptManifest
from orchestrator.database import ArtifactRecord, AttemptRecord, NodeRecord
from orchestrator.workflows.run import StageActivityInput, StageActivityResult


class DatabaseAttemptLedger:
    def __init__(self, sessions, *, code_revision: str, environment: dict[str, str]):
        self.sessions = sessions
        self.code_revision = code_revision
        self.environment_fingerprint = hashlib.sha256(
            json.dumps(environment, sort_keys=True).encode()
        ).hexdigest()

    def started(self, request: StageActivityInput, attempt_id: str) -> None:
        with self.sessions.begin() as session:
            node = session.get(
                NodeRecord,
                {"run_id": request.run_id, "node_id": request.node_id},
                with_for_update=True,
            )
            if node is None:
                raise ValueError(f"unknown pipeline node: {request.node_id}")
            number = (
                session.scalar(
                    select(func.count())
                    .select_from(AttemptRecord)
                    .where(
                        AttemptRecord.run_id == request.run_id,
                        AttemptRecord.node_id == request.node_id,
                    )
                )
                + 1
            )
            session.add(
                AttemptRecord(
                    id=attempt_id,
                    run_id=request.run_id,
                    node_id=request.node_id,
                    number=number,
                    schema_version="wander.stage-attempt/1",
                    status="running",
                    hypothesis=request.parameters.get("hypothesis"),
                    parameters=request.parameters,
                    command=[],
                    code_revision=self.code_revision,
                    environment_fingerprint=self.environment_fingerprint,
                    input_artifact_ids=[
                        artifact
                        for artifacts in request.selected_inputs.values()
                        for artifact in artifacts
                    ],
                    output_artifact_ids=[],
                    costs=[],
                    started_at=datetime.now(timezone.utc),
                )
            )
            node.status = "running"
            node.updated_at = datetime.now(timezone.utc)

    def finished(
        self,
        request: StageActivityInput,
        result: StageActivityResult,
        manifest: AttemptManifest,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self.sessions.begin() as session:
            attempt = session.get(AttemptRecord, result.attempt_id, with_for_update=True)
            if attempt is None:
                raise ValueError(f"unknown pipeline attempt: {result.attempt_id}")
            artifact_ids = []
            for frozen in manifest.files:
                artifact_ids.append(frozen.artifact_id)
                if session.get(ArtifactRecord, frozen.artifact_id):
                    continue
                session.add(
                    ArtifactRecord(
                        id=frozen.artifact_id,
                        run_id=request.run_id,
                        role=frozen.role,
                        sha256=frozen.sha256,
                        size=frozen.size,
                        media_type=frozen.media_type,
                        storage_key=frozen.blob_key,
                        producer_attempt_id=result.attempt_id,
                        artifact_metadata={"relative_path": frozen.relative_path},
                        created_at=now,
                    )
                )
            attempt.status = result.status
            attempt.output_artifact_ids = artifact_ids
            attempt.finished_at = now
            attempt.error = result.error
            node = session.get(
                NodeRecord,
                {"run_id": request.run_id, "node_id": request.node_id},
                with_for_update=True,
            )
            # Only the newest attempt speaks for the node. A scheduler that was replaced can
            # still have work in flight, and when that work ends it must not overwrite the state
            # of the attempt that replaced it: a dead run's cancellation would otherwise mark a
            # healthy stage failed.
            newest = session.scalar(
                select(AttemptRecord.id)
                .where(
                    AttemptRecord.run_id == request.run_id,
                    AttemptRecord.node_id == request.node_id,
                )
                .order_by(AttemptRecord.started_at.desc())
                .limit(1)
            )
            if node and newest == result.attempt_id:
                node.status = result.status
                node.updated_at = now
