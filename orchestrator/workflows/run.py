"""Durable Temporal workflow for one admitted source or shot."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy as TemporalRetryPolicy

with workflow.unsafe.imports_passed_through():
    from orchestrator.contracts import NodeStatus, StageKind
    from orchestrator.graph import (
        BranchArtifact,
        GraphNode,
        RunGraph,
        ShotBranch,
        child_workflows_for_shots,
        instantiate_graph,
    )
    from orchestrator.stages import GraphOptions


@dataclass
class GenerationWorkflowInput:
    run_id: str
    source_sha256: str
    source_artifact_id: str
    options: dict[str, Any]
    run_inputs: dict[str, list[str]] = field(default_factory=dict)
    parent_run_id: str | None = None
    branch_key: str | None = None
    # Rubric'd stages are judged by the reviewing agent before the run moves on.
    agent_review: bool = True
    queue_limits: dict[str, int] = field(
        default_factory=lambda: {
            "local_cpu": 8,
            "modal_gpu": 3,
            "external_api": 2,
            "agent_qa": 2,
            "publisher": 2,
        }
    )


@dataclass
class StageActivityInput:
    run_id: str
    node_id: str
    stage_type: str
    definition: dict[str, Any]
    selected_inputs: dict[str, list[str]]
    parameters: dict[str, Any] = field(default_factory=dict)
    # The run's graph options. Some stages only exist in a particular shape of graph, so the
    # command that runs them has to ask for that shape too.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageActivityResult:
    node_id: str
    attempt_id: str
    status: str
    artifacts: dict[str, list[str]] = field(default_factory=dict)
    branches: list[dict[str, str]] = field(default_factory=list)
    shots: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class ApprovalSignal:
    node_id: str
    attempt_id: str
    artifacts: dict[str, list[str]]
    approved_by: str
    rationale: str


@dataclass
class OperatorMessage:
    id: str
    author: str
    message: str
    node_id: str | None = None
    attempt_id: str | None = None
    attachment_artifact_ids: list[str] = field(default_factory=list)


@dataclass
class DecisionSignal:
    node_id: str
    attempt_id: str | None
    rationale: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordApprovalInput:
    run_id: str
    node_id: str
    stage_type: str
    definition: dict[str, Any]
    reviewed_attempt_id: str | None
    artifacts: dict[str, list[str]]
    approved_by: str
    rationale: str


@dataclass
class RunStateInput:
    run_id: str
    state: dict[str, Any] | None = None


@dataclass
class RunState:
    """The scheduler's whole view of a run, as data."""

    paused: bool = False
    canceled: bool = False
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ReviewInput:
    run_id: str
    node_id: str
    stage_type: str
    definition: dict[str, Any]
    attempt_id: str
    artifacts: dict[str, list[str]]
    attempt_status: str = "succeeded"
    attempt_error: str | None = None
    operator_messages: list[dict[str, Any]] = field(default_factory=list)
    agent_retries: int = 0


@dataclass
class ReviewDecision:
    """The reviewing agent's judgement of one attempt.

    ``decision`` is "pass", "retry" or "needs_human". ``attempt_id``/``artifacts`` name the
    attempt to select on pass: the reviewed one, or an agent-revised copy of it.
    """

    node_id: str
    attempt_id: str
    decision: str
    rationale: str
    artifacts: dict[str, list[str]] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    hypothesis: str | None = None
    question: str | None = None
    revised: bool = False
    decided_by: str = "agent"


# How many times the agent may send one stage back on its own. After this it must ask, so the
# operator confirms before it tries again rather than the loop deciding for itself.
# A stage or review says it is alive every 20 seconds while it works. Without a heartbeat
# timeout, a worker that dies mid-stage is only noticed when the stage's own timeout expires,
# and a Marble stage's is four hours: the run sits still for all of them with nothing running.
# The allowance is far longer than the interval on purpose. Hashing and archiving a large
# attempt is a single step that cannot report progress, and a stage wrongly declared dead is
# started again -- which for a paid stage is a second charge. Ten minutes still turns four
# hours of nothing into ten.
HEARTBEAT_TIMEOUT = timedelta(seconds=600)
AGENT_RETRY_CAP = 3


@dataclass
class RegisterRunInput:
    run_id: str
    parent_run_id: str
    branch_key: str
    source_sha256: str
    source_artifact_id: str
    options: dict[str, Any]


@dataclass
class FinishRunInput:
    run_id: str
    status: str


@dataclass
class PublishRunInput:
    run_id: str


@dataclass
class PublishRunResult:
    """What the archive step did, so a run records whether its outputs left this machine."""

    published: bool
    detail: str


@dataclass
class WorkflowResult:
    run_id: str
    status: str
    graph: dict[str, Any]
    child_results: list[dict[str, Any]]


def activity_inputs(
    graph: RunGraph, node_id: str, run_artifacts: dict[str, list[str]]
) -> dict[str, list[str]]:
    node = graph.nodes[node_id]
    selected: dict[str, list[str]] = {}
    for name, binding in node.definition.inputs.items():
        if binding.source == "run_input":
            selected[name] = run_artifacts[binding.role]
        else:
            selected[name] = list(
                graph.nodes[binding.stage_id].selected_artifacts.get(binding.role, ())
            )
    return selected


def apply_activity_result(graph: RunGraph, result: StageActivityResult) -> None:
    if result.status == "succeeded":
        graph.select_attempt(
            result.node_id,
            result.attempt_id,
            {role: tuple(artifacts) for role, artifacts in result.artifacts.items()},
        )
        branches = tuple(BranchArtifact(**branch) for branch in result.branches)
        if result.node_id == "tracks":
            graph.expand_people(branches)
        elif result.node_id == "object_detect":
            graph.expand_objects(branches)
        return
    status = {
        "failed": NodeStatus.FAILED,
        "blocked": NodeStatus.BLOCKED,
        "canceled": NodeStatus.CANCELED,
    }.get(result.status)
    if status is None:
        raise ValueError(f"unknown stage activity status {result.status}")
    graph.set_status(result.node_id, status, result.error)


# Attempt outcomes the agent is asked to judge. A failure is where its judgement is worth the
# most: it can read the error, decide whether a different parameter would help, and say so,
# instead of the run simply stopping.
REVIEWABLE_STATUSES = frozenset({"succeeded", "failed", "blocked"})


def snapshot_state(
    graph: RunGraph,
    *,
    paused: bool,
    canceled: bool,
    agent_retries: dict[str, int],
    retry_parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Everything the scheduler knows, in a form the database can hold."""
    return {
        "paused": paused,
        "canceled": canceled,
        "nodes": {
            node.id: {
                "stage_type": node.stage_type,
                "definition": node.definition.model_dump(mode="json"),
                "dependencies": list(node.dependencies),
                "parent_node_id": node.parent_node_id,
                "branch_key": node.branch_key,
                "status": node.status.value,
                "blocked_reason": node.blocked_reason,
                "selected_attempt_id": node.selected_attempt_id,
                "selected_artifacts": {
                    role: list(ids) for role, ids in node.selected_artifacts.items()
                },
                "agent_retries": agent_retries.get(node.id, 0),
                "retry_parameters": retry_parameters.get(node.id, {}),
            }
            for node in graph.nodes.values()
        },
    }


def restore_state(
    graph: RunGraph,
    state: dict[str, Any],
    agent_retries: dict[str, int],
    retry_parameters: dict[str, dict[str, Any]],
) -> tuple[bool, bool]:
    """Put a stored state back onto a freshly instantiated graph.

    Nodes that were running when the previous scheduler stopped are returned to the queue: the
    work either finished and was recorded, or it did not and must be done again.
    """
    for node_id, value in (state.get("nodes") or {}).items():
        node = graph.nodes.get(node_id)
        if node is None:
            # A node the run created by expanding a stage. The base graph has no place for it,
            # so rebuild it from its row rather than dropping the work it represents.
            definition = value.get("definition")
            if not definition:
                continue
            if hasattr(definition, "model_dump"):
                # Temporal's workflow sandbox rebuilds the modules a workflow imports, so a
                # StageDefinition built outside it is not the class GraphNode validates
                # against, however identical it looks. Hand over the plain fields and let the
                # node build its own.
                definition = definition.model_dump(mode="json")
            node = GraphNode(
                id=node_id,
                stage_type=value.get("stage_type") or node_id,
                definition=definition,
                dependencies=tuple(value.get("dependencies") or ()),
                parent_node_id=value.get("parent_node_id"),
                branch_key=value.get("branch_key"),
            )
            graph.nodes[node_id] = node
        status = NodeStatus(value["status"])
        if status in {NodeStatus.RUNNING, NodeStatus.WAITING_AGENT}:
            status = NodeStatus.QUEUED
        node.status = status
        node.blocked_reason = value.get("blocked_reason")
        node.selected_attempt_id = value.get("selected_attempt_id")
        node.selected_artifacts = {
            role: tuple(ids) for role, ids in (value.get("selected_artifacts") or {}).items()
        }
        if value.get("agent_retries"):
            agent_retries[node_id] = int(value["agent_retries"])
        if value.get("retry_parameters"):
            retry_parameters[node_id] = dict(value["retry_parameters"])
    return bool(state.get("paused")), bool(state.get("canceled"))


def reviewable(definition: dict[str, Any]) -> bool:
    quality = definition.get("quality") or {}
    return bool(quality.get("agent_rubric")) and definition.get("kind") != "human"


def apply_review_decision(
    graph: RunGraph,
    result: StageActivityResult,
    decision: ReviewDecision,
    retry_parameters: dict[str, dict[str, Any]],
    agent_retries: dict[str, int],
    cap: int = AGENT_RETRY_CAP,
) -> str:
    """Apply the agent's judgement of ``result`` and return the outcome applied.

    A retry beyond ``cap`` becomes a human question rather than an endless loop, and a pass
    selects whichever attempt the agent named (its own revision, if it edited the outputs).
    """
    node_id = result.node_id
    if decision.decision == "pass" and result.status != "succeeded":
        graph.set_status(
            node_id,
            NodeStatus.WAITING_HUMAN,
            f"agent passed a {result.status} attempt; that needs a person: {decision.rationale}",
        )
        return "needs_human"
    if decision.decision == "pass":
        selected = StageActivityResult(
            node_id=node_id,
            attempt_id=decision.attempt_id,
            status="succeeded",
            artifacts=decision.artifacts or result.artifacts,
            branches=result.branches,
            shots=result.shots,
        )
        apply_activity_result(graph, selected)
        return "pass"
    if decision.decision == "retry" and agent_retries.get(node_id, 0) < cap:
        agent_retries[node_id] = agent_retries.get(node_id, 0) + 1
        parameters = dict(decision.parameters)
        if decision.hypothesis:
            parameters["hypothesis"] = decision.hypothesis
        retry_parameters[node_id] = parameters
        graph.retry_node(node_id)
        return "retry"
    reason = decision.question or decision.rationale
    if decision.decision == "retry":
        reason = (
            f"agent used its {cap} retries and wants another; confirm to let it try again: {reason}"
        )
    graph.set_status(node_id, NodeStatus.WAITING_HUMAN, reason)
    return "needs_human"


@workflow.defn(name="wander.generation")
class GenerationWorkflow:
    def __init__(self) -> None:
        self.graph: RunGraph | None = None
        self.paused = False
        self.canceled = False
        self.messages: list[OperatorMessage] = []
        # Monotonic count of changes to this run. Waiters compare it against the value they
        # last saw, and the persister against the value it last wrote; resetting it would make
        # a change look like no change and silently skip the write.
        # Operator-supplied stage parameters (e.g. a retry hypothesis) keyed by node ID.
        self.retry_parameters: dict[str, dict[str, Any]] = {}
        # Times the reviewing agent has sent each stage back, bounded by AGENT_RETRY_CAP.
        self.agent_retries: dict[str, int] = {}
        # Approvals waiting to be recorded as attempts, keyed by node.
        self.pending_approvals: dict[str, ApprovalSignal] = {}
        self.revision = 0

    @workflow.run
    async def run(self, request: GenerationWorkflowInput) -> WorkflowResult:
        self.graph = instantiate_graph(GraphOptions.model_validate(request.options))
        if request.parent_run_id is not None:
            await workflow.execute_activity(
                "register_run",
                RegisterRunInput(
                    run_id=request.run_id,
                    parent_run_id=request.parent_run_id,
                    branch_key=request.branch_key or "",
                    source_sha256=request.source_sha256,
                    source_artifact_id=request.source_artifact_id,
                    options=request.options,
                ),
                task_queue="local_cpu",
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=TemporalRetryPolicy(maximum_attempts=3),
            )
        run_artifacts = {
            "source_video": [request.source_artifact_id],
            **request.run_inputs,
        }
        run_inputs = set(run_artifacts)
        stored = await workflow.execute_activity(
            "load_run_state",
            RunStateInput(run_id=request.run_id),
            task_queue="local_cpu",
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=TemporalRetryPolicy(maximum_attempts=3),
            result_type=RunState,
        )
        if stored is not None and stored.nodes:
            self.paused, self.canceled = restore_state(
                self.graph,
                {"paused": stored.paused, "canceled": stored.canceled, "nodes": stored.nodes},
                self.agent_retries,
                self.retry_parameters,
            )
        running: dict[str, asyncio.Task[StageActivityResult]] = {}
        reviewing: dict[str, asyncio.Task[ReviewDecision]] = {}
        persisted_revision = -1
        # Results held while the agent judges them, keyed by node.
        pending: dict[str, StageActivityResult] = {}
        child_tasks: list[asyncio.Task[WorkflowResult]] = []
        child_results: list[dict[str, Any]] = []

        while True:
            if self.canceled:
                for node in self.graph.nodes.values():
                    if node.status not in {
                        NodeStatus.SUCCEEDED,
                        NodeStatus.SKIPPED,
                        NodeStatus.FAILED,
                        NodeStatus.BLOCKED,
                    }:
                        node.status = NodeStatus.CANCELED
                for task in running.values():
                    task.cancel()
                return await self._finish(request.run_id, "canceled", child_results)

            if self.paused:
                await workflow.wait_condition(lambda: not self.paused or self.canceled)
                continue

            if self.revision != persisted_revision:
                await self._persist(request.run_id)
                persisted_revision = self.revision
            for node_id, node in self.graph.nodes.items():
                if (
                    node.status == NodeStatus.WAITING_HUMAN
                    and node.definition.kind == StageKind.HUMAN
                    and node_id not in self.pending_approvals
                ):
                    # Gates advance on their own. The approval is still recorded as an attempt
                    # with an author, so who let the run past remains answerable.
                    self.pending_approvals[node_id] = ApprovalSignal(
                        node_id=node_id,
                        attempt_id="",
                        artifacts={},
                        approved_by="pipeline",
                        rationale="human gate advanced automatically",
                    )
            while self.pending_approvals:
                node_id, signal = self.pending_approvals.popitem()
                node = self.graph.nodes[node_id]
                result = await workflow.execute_activity(
                    "record_approval",
                    RecordApprovalInput(
                        run_id=request.run_id,
                        node_id=node_id,
                        stage_type=node.stage_type,
                        definition=node.definition.model_dump(mode="json"),
                        reviewed_attempt_id=signal.attempt_id or None,
                        artifacts=signal.artifacts,
                        approved_by=signal.approved_by,
                        rationale=signal.rationale,
                    ),
                    task_queue="local_cpu",
                    start_to_close_timeout=timedelta(seconds=120),
                    retry_policy=TemporalRetryPolicy(maximum_attempts=3),
                    result_type=StageActivityResult,
                )
                apply_activity_result(self.graph, result)
                self.revision += 1
            ready = self.graph.evaluate(run_inputs)
            queue_counts: dict[str, int] = {}
            for node_id in running:
                queue = self.graph.nodes[node_id].definition.resources.task_queue
                queue_counts[queue] = queue_counts.get(queue, 0) + 1
            for node_id in ready:
                if node_id in running:
                    continue
                node = self.graph.nodes[node_id]
                queue = node.definition.resources.task_queue
                limit = request.queue_limits.get(queue, 1)
                if queue_counts.get(queue, 0) >= limit:
                    continue
                node.status = NodeStatus.RUNNING
                queue_counts[queue] = queue_counts.get(queue, 0) + 1
                running[node_id] = asyncio.create_task(
                    workflow.execute_activity(
                        "run_stage",
                        StageActivityInput(
                            run_id=request.run_id,
                            node_id=node_id,
                            stage_type=node.stage_type,
                            definition=node.definition.model_dump(mode="json"),
                            selected_inputs=activity_inputs(self.graph, node_id, run_artifacts),
                            parameters=self.retry_parameters.get(node_id, {}),
                            options=request.options,
                        ),
                        task_queue=queue,
                        start_to_close_timeout=timedelta(
                            seconds=node.definition.resources.timeout_seconds
                        ),
                        heartbeat_timeout=HEARTBEAT_TIMEOUT,
                        retry_policy=TemporalRetryPolicy(
                            maximum_attempts=node.definition.retry.automatic_attempts
                        ),
                        result_type=StageActivityResult,
                    )
                )

            if running or reviewing:
                done, _ = await workflow.wait(
                    (*running.values(), *reviewing.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    if task in reviewing.values():
                        node_id = next(key for key, value in reviewing.items() if value is task)
                        del reviewing[node_id]
                        result = pending.pop(node_id)
                        try:
                            decision = task.result()
                        except Exception as error:
                            # The agent could not judge; that is a human's call, never a pass.
                            self.graph.set_status(
                                node_id, NodeStatus.WAITING_HUMAN, f"agent review failed: {error}"
                            )
                            self.revision += 1
                            continue
                        outcome = apply_review_decision(
                            self.graph, result, decision, self.retry_parameters, self.agent_retries
                        )
                        self.revision += 1
                        if outcome == "pass":
                            self._spawn_shot_children(request, result, child_tasks)
                        continue
                    node_id = next(key for key, value in running.items() if value is task)
                    del running[node_id]
                    try:
                        result = task.result()
                    except Exception as error:
                        self.graph.set_status(node_id, NodeStatus.FAILED, str(error))
                        continue
                    node = self.graph.nodes[node_id]
                    if (
                        request.agent_review
                        and result.status in REVIEWABLE_STATUSES
                        and reviewable(node.definition.model_dump(mode="json"))
                    ):
                        node.status = NodeStatus.WAITING_AGENT
                        pending[node_id] = result
                        reviewing[node_id] = asyncio.create_task(
                            workflow.execute_activity(
                                "review_attempt",
                                ReviewInput(
                                    run_id=request.run_id,
                                    node_id=node_id,
                                    stage_type=node.stage_type,
                                    definition=node.definition.model_dump(mode="json"),
                                    attempt_id=result.attempt_id,
                                    artifacts=result.artifacts,
                                    attempt_status=result.status,
                                    attempt_error=result.error,
                                    operator_messages=[
                                        message.__dict__ for message in self.messages
                                    ],
                                    agent_retries=self.agent_retries.get(node_id, 0),
                                ),
                                task_queue="external_api",
                                start_to_close_timeout=timedelta(seconds=1800),
                                heartbeat_timeout=HEARTBEAT_TIMEOUT,
                                retry_policy=TemporalRetryPolicy(maximum_attempts=1),
                                result_type=ReviewDecision,
                            )
                        )
                        self.revision += 1
                        continue
                    apply_activity_result(self.graph, result)
                    self.revision += 1
                    self._spawn_shot_children(request, result, child_tasks)
                continue

            unfinished = [
                node
                for node in self.graph.nodes.values()
                if node.status
                not in {
                    NodeStatus.SUCCEEDED,
                    NodeStatus.SKIPPED,
                    NodeStatus.FAILED,
                    NodeStatus.BLOCKED,
                    NodeStatus.CANCELED,
                }
            ]
            if unfinished:
                seen = self.revision
                await workflow.wait_condition(
                    lambda: self.canceled or self.paused or self.revision != seen
                )
                continue
            halted = [
                node
                for node in self.graph.nodes.values()
                if node.status in {NodeStatus.FAILED, NodeStatus.BLOCKED}
            ]
            if halted:
                # Nothing is running and at least one stage stopped on a failure or a policy
                # block. That is an operator's decision to make, so the run stays open for a
                # retry (after reconciling any provider claim) or a cancel; it does not end
                # itself, which would leave those signals with no workflow to reach.
                seen = self.revision
                await workflow.wait_condition(lambda: self.canceled or self.revision != seen)
                continue
            if child_tasks:
                child_results = [result.__dict__ for result in await asyncio.gather(*child_tasks)]
            failed = any(
                node.status in {NodeStatus.FAILED, NodeStatus.BLOCKED}
                for node in self.graph.nodes.values()
            )
            return await self._finish(
                request.run_id, "failed" if failed else "succeeded", child_results
            )

    @workflow.signal
    def approve(self, signal: ApprovalSignal) -> None:
        if self.graph is None:
            return
        node = self.graph.nodes[signal.node_id]
        if node.status != NodeStatus.WAITING_HUMAN:
            raise ValueError(f"{signal.node_id} is not waiting for human approval")
        if node.definition.kind == StageKind.HUMAN:
            # A human stage runs nothing, so it has no attempt and no artifact to select. Its
            # approval is its output: queue it and let the loop record it as an ordinary
            # attempt, so stages that require an approval have something real to consume.
            self.pending_approvals[signal.node_id] = signal
            self.revision += 1
            return
        self.graph.select_attempt(
            signal.node_id,
            signal.attempt_id,
            {role: tuple(values) for role, values in signal.artifacts.items()},
        )
        self.revision += 1

    @workflow.signal
    def operator_message(self, message: OperatorMessage) -> None:
        self.messages.append(message)
        self.revision += 1

    @workflow.signal
    def reject(self, signal: DecisionSignal) -> None:
        self.messages.append(
            OperatorMessage(
                id=f"rejection:{len(self.messages) + 1}",
                author="operator",
                message=signal.rationale,
                node_id=signal.node_id,
                attempt_id=signal.attempt_id,
            )
        )
        self.revision += 1

    @workflow.signal
    def retry(self, signal: DecisionSignal) -> None:
        if self.graph is None:
            return
        self.graph.retry_node(signal.node_id)
        if signal.parameters:
            self.retry_parameters[signal.node_id] = dict(signal.parameters)
        self.messages.append(
            OperatorMessage(
                id=f"retry:{len(self.messages) + 1}",
                author="operator",
                message=signal.rationale,
                node_id=signal.node_id,
                attempt_id=signal.attempt_id,
            )
        )
        self.revision += 1

    @workflow.signal
    def pause(self) -> None:  # noqa: D401 - revision bump persists the intent
        self.paused = True
        self.revision += 1

    @workflow.signal
    def resume(self) -> None:
        self.paused = False
        self.revision += 1

    @workflow.signal
    def cancel(self) -> None:
        self.canceled = True
        self.revision += 1

    @workflow.query
    def state(self) -> dict[str, Any]:
        return {
            "graph": self.graph.model_dump(mode="json") if self.graph else None,
            "paused": self.paused,
            "canceled": self.canceled,
            "messages": [message.__dict__ for message in self.messages],
        }

    async def _persist(self, run_id: str) -> None:
        """Write the scheduler's view to the database.

        A failure here must not fail the run: the stage and review activities have already
        recorded their own outcomes, so the rows stay usable and the next pass tries again.
        """
        assert self.graph is not None
        try:
            await workflow.execute_activity(
                "save_run_state",
                RunStateInput(
                    run_id=run_id,
                    state=snapshot_state(
                        self.graph,
                        paused=self.paused,
                        canceled=self.canceled,
                        agent_retries=self.agent_retries,
                        retry_parameters=self.retry_parameters,
                    ),
                ),
                task_queue="local_cpu",
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=TemporalRetryPolicy(maximum_attempts=3),
            )
        except Exception as error:
            workflow.logger.warning("run %s state not persisted: %s", run_id, error)

    def _spawn_shot_children(
        self,
        request: GenerationWorkflowInput,
        result: StageActivityResult,
        child_tasks: list[asyncio.Task[WorkflowResult]],
    ) -> None:
        if not result.shots or request.parent_run_id is not None:
            return
        specs = child_workflows_for_shots(
            request.run_id,
            request.source_sha256,
            tuple(ShotBranch(**shot) for shot in result.shots),
        )
        for spec in specs:
            child_tasks.append(
                asyncio.create_task(
                    workflow.execute_child_workflow(
                        GenerationWorkflow.run,
                        GenerationWorkflowInput(
                            run_id=spec.workflow_id,
                            source_sha256=spec.canonical_source_sha256,
                            source_artifact_id=spec.clip_artifact_id,
                            options=request.options,
                            run_inputs=request.run_inputs,
                            parent_run_id=request.run_id,
                            branch_key=spec.branch_key,
                            queue_limits=request.queue_limits,
                            agent_review=request.agent_review,
                        ),
                        id=spec.workflow_id,
                    )
                )
            )

    async def _finish(
        self, run_id: str, status: str, child_results: list[dict[str, Any]]
    ) -> WorkflowResult:
        """Record the outcome on the run row, then return it.

        The result is the workflow's authoritative answer; a failure to write it must not turn a
        finished run into a failed workflow, so the write is retried and then given up on. The
        API derives a live status from the node rows in the meantime.
        """
        if status == "succeeded":
            try:
                published = await workflow.execute_activity(
                    "publish_run",
                    PublishRunInput(run_id=run_id),
                    task_queue="local_cpu",
                    start_to_close_timeout=timedelta(seconds=3600),
                    retry_policy=TemporalRetryPolicy(maximum_attempts=2),
                    result_type=PublishRunResult,
                )
                workflow.logger.info("run %s archive: %s", run_id, published.detail)
            except Exception as error:
                # The outputs are already durable locally; failing to copy them off this machine
                # must not turn a finished run into a failed one.
                workflow.logger.warning("run %s was not archived: %s", run_id, error)
        try:
            await workflow.execute_activity(
                "finish_run",
                FinishRunInput(run_id=run_id, status=status),
                task_queue="local_cpu",
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=TemporalRetryPolicy(maximum_attempts=5),
            )
        except Exception as error:
            workflow.logger.warning(
                "run %s finished %s but the row was not updated: %s", run_id, status, error
            )
        return self._result(run_id, status, child_results)

    def _result(
        self, run_id: str, status: str, child_results: list[dict[str, Any]]
    ) -> WorkflowResult:
        assert self.graph is not None
        return WorkflowResult(
            run_id=run_id,
            status=status,
            graph=self.graph.model_dump(mode="json"),
            child_results=child_results,
        )
