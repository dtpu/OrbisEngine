"""Run lifecycle activities: the persistence the workflow itself cannot do.

Workflow code is deterministic and has no database, so a child run the workflow spawns per
admitted shot has no rows until an activity creates them, and no run row learns the
workflow's outcome unless an activity writes it.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from temporalio import activity

from orchestrator.contracts import Run
from orchestrator.database import RunRecord
from orchestrator.graph import instantiate_graph
from orchestrator.artifacts import LocalCAS, UploadOutbox, freeze_attempt
from orchestrator.contracts import EventType
from orchestrator.repository import PipelineRepository
from orchestrator.workspace import RunWorkspace
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import (
    FinishRunInput,
    PublishRunInput,
    PublishRunResult,
    RecordApprovalInput,
    RegisterRunInput,
    RunState,
    RunStateInput,
    StageActivityInput,
    StageActivityResult,
)


class RunLifecycleActivities:
    def __init__(
        self,
        repository: PipelineRepository,
        *,
        graph_version: str,
        code_revision: str,
        container_digest: str | None,
        repository_root: Path | None = None,
        workspace_root: Path | None = None,
        store: LocalCAS | None = None,
        archive=None,
        attempt_ledger=None,
    ):
        self.repository = repository
        self.graph_version = graph_version
        self.code_revision = code_revision
        self.container_digest = container_digest
        self.repository_root = Path(repository_root) if repository_root else Path.cwd()
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.store = store
        self.archive = archive
        self.attempt_ledger = attempt_ledger

    @activity.defn(name="register_run")
    def register_run(self, request: RegisterRunInput) -> None:
        """Create the run and node rows for a child run. Safe to retry: a second call is a no-op."""
        with self.repository.sessions() as session:
            if session.get(RunRecord, request.run_id) is not None:
                return
            parent = session.get(RunRecord, request.parent_run_id)
            if parent is None:
                raise ValueError(f"unknown parent run: {request.parent_run_id}")
            budget = dict(parent.budget or {})
            created_by = parent.created_by
        options = GraphOptions.model_validate(request.options)
        run = Run(
            id=request.run_id,
            graph_version=self.graph_version,
            code_revision=self.code_revision,
            container_digest=self.container_digest,
            source_sha256=request.source_sha256,
            source_artifact_id=request.source_artifact_id,
            configuration={"options": options.model_dump(mode="json")},
            budget=budget,
            created_by=created_by,
            parent_run_id=request.parent_run_id,
            branch_key=request.branch_key,
        )
        self.repository.create_run(run, instantiate_graph(options))

    @activity.defn(name="finish_run")
    def finish_run(self, request: FinishRunInput) -> None:
        self.repository.finish_run(request.run_id, request.status)

    @activity.defn(name="load_run_state")
    def load_run_state(self, request: RunStateInput) -> RunState:
        """Read back a run's progress so a new scheduler can continue it.

        A scheduler starting is the run being worked on, so this is also where a run that had
        already finished stops saying so. Every workflow start passes through here, whichever
        way it was resumed.
        """
        try:
            self.repository.reopen_run(request.run_id)
        except KeyError:
            pass  # a child run's rows are written by register_run, which may not have run yet
        stored = self.repository.load_run_state(request.run_id)
        if stored is None:
            return RunState()
        return RunState(paused=stored["paused"], canceled=stored["canceled"], nodes=stored["nodes"])

    @activity.defn(name="save_run_state")
    def save_run_state(self, request: RunStateInput) -> None:
        self.repository.save_run_state(request.run_id, request.state or {})

    @activity.defn(name="record_approval")
    def record_approval(self, request: RecordApprovalInput) -> StageActivityResult:
        """Write a human stage's approval as an ordinary attempt.

        A human stage executes nothing, so without this it has no attempt and produces no
        artifact, and the stages that declare an approval as a required input can never be
        satisfied. The approval document names who allowed the run past and why, so the decision
        is as inspectable as any other output.
        """
        info = activity.info()
        attempt_id = f"{request.run_id}:{info.activity_id}:{info.attempt}:approval"
        workspace = RunWorkspace(self.workspace_root, request.run_id)
        workspace.initialize()
        attempt = workspace.create_attempt(
            request.node_id,
            attempt_id,
            {
                "runId": request.run_id,
                "nodeId": request.node_id,
                "stageType": request.stage_type,
                "definition": request.definition,
                "selectedInputs": {},
            },
        )
        document = {
            "schema": "wander.approval/1",
            "runId": request.run_id,
            "nodeId": request.node_id,
            "reviewedAttemptId": request.reviewed_attempt_id,
            "reviewedArtifacts": request.artifacts,
            "approvedBy": request.approved_by,
            "rationale": request.rationale,
            "approvedAt": datetime.now(timezone.utc).isoformat(),
        }
        (attempt.outputs / "approval.json").write_text(json.dumps(document, indent=2))
        stage_input = StageActivityInput(
            run_id=request.run_id,
            node_id=request.node_id,
            stage_type=request.stage_type,
            definition=request.definition,
            selected_inputs={},
        )
        if self.attempt_ledger:
            self.attempt_ledger.started(stage_input, attempt_id)
        attempt.finalize({"status": "succeeded", "error": None, "command": []})
        manifest = freeze_attempt(
            attempt,
            self.store,
            status="succeeded",
            roles={"outputs/approval.json": "approval"},
        )
        outbox = UploadOutbox(workspace.outbox, self.store, self.archive)
        outbox.enqueue(manifest)
        outbox.flush()
        artifacts: dict[str, list[str]] = {}
        for frozen in manifest.files:
            if frozen.role != "attempt_file":
                artifacts.setdefault(frozen.role, []).append(frozen.artifact_id)
        result = StageActivityResult(
            node_id=request.node_id,
            attempt_id=attempt_id,
            status="succeeded",
            artifacts=artifacts,
        )
        if self.attempt_ledger:
            self.attempt_ledger.finished(stage_input, result, manifest)
        self.repository.add_approval(
            run_id=request.run_id,
            node_id=request.node_id,
            attempt_id=attempt_id,
            kind="quality_gate",
            decision="approved",
            actor=request.approved_by,
            rationale=request.rationale,
        )
        return result

    @activity.defn(name="publish_run")
    def publish_run(self, request: PublishRunInput) -> PublishRunResult:
        """Copy a finished run's outputs to shared storage.

        Stages run with --no-publish so that no single stage decides to publish the run; this is
        the one place that does, once the whole run has succeeded. It shells out to the
        repository's own publisher rather than reimplementing the upload, so archived layout and
        credentials stay in one place.
        """
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            return PublishRunResult(
                published=False,
                detail="no AWS credentials in this environment; outputs stay local",
            )
        command = ["bun", "run", "runs:publish", "--archive-only"]
        evidence = os.environ.get("WANDER_EVIDENCE_DIR")
        if evidence:
            command += ["--evidence-dir", evidence]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                timeout=3300,
            )
        except (OSError, subprocess.SubprocessError) as error:
            detail = f"archive could not start: {type(error).__name__}: {error}"
            self.repository.append_event(
                request.run_id, EventType.RUN, request.run_id, {"archived": False, "detail": detail}
            )
            return PublishRunResult(published=False, detail=detail)
        tail = (completed.stdout or completed.stderr or "").strip().splitlines()
        detail = tail[-1] if tail else f"publisher exited {completed.returncode}"
        published = completed.returncode == 0
        self.repository.append_event(
            request.run_id,
            EventType.RUN,
            request.run_id,
            {"archived": published, "detail": detail[:500]},
        )
        return PublishRunResult(published=published, detail=detail[:500])
