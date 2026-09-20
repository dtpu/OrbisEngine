"""Run lifecycle activities: the persistence the workflow itself cannot do.

Workflow code is deterministic and has no database, so a child run the workflow spawns per
admitted shot has no rows until an activity creates them, and no run row learns the
workflow's outcome unless an activity writes it.
"""

from __future__ import annotations

from temporalio import activity

from orchestrator.contracts import Run
from orchestrator.database import RunRecord
from orchestrator.graph import instantiate_graph
from orchestrator.repository import PipelineRepository
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import (
    FinishRunInput,
    RegisterRunInput,
    RunState,
    RunStateInput,
)


class RunLifecycleActivities:
    def __init__(
        self,
        repository: PipelineRepository,
        *,
        graph_version: str,
        code_revision: str,
        container_digest: str | None,
    ):
        self.repository = repository
        self.graph_version = graph_version
        self.code_revision = code_revision
        self.container_digest = container_digest

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
        """Read back a run's progress so a new scheduler can continue it."""
        stored = self.repository.load_run_state(request.run_id)
        if stored is None:
            return RunState()
        return RunState(paused=stored["paused"], canceled=stored["canceled"], nodes=stored["nodes"])

    @activity.defn(name="save_run_state")
    def save_run_state(self, request: RunStateInput) -> None:
        self.repository.save_run_state(request.run_id, request.state or {})
