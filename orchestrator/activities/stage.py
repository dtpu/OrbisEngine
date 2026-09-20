"""Typed subprocess activity runner with immutable success and failure artifacts."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from temporalio import activity

from orchestrator.artifacts import AttemptManifest, LocalCAS, UploadOutbox, freeze_attempt
from orchestrator.quality.validators import validate_stage_outputs
from orchestrator.workspace import AttemptWorkspace, RunWorkspace
from orchestrator.workflows.run import StageActivityInput, StageActivityResult


# How often a running stage tells Temporal it is still alive. Without this a worker that dies
# mid-stage is only noticed when the stage's whole timeout expires -- four hours for a Modal or
# Marble stage -- and the run sits still for all of them even though the work is long finished.
HEARTBEAT_SECONDS = 20


def beat() -> None:
    """Report liveness, when there is a Temporal to report it to.

    The runner is also called directly by tests and by the resume script, where there is no
    activity context and nothing to tell.
    """
    try:
        activity.heartbeat()
    except RuntimeError:
        pass


class ArtifactResolver(Protocol):
    def hydrate(self, artifact_id: str, destination_directory: Path) -> Path: ...


class PaidGuard(Protocol):
    def begin(self, request: StageActivityInput, attempt_id: str) -> str: ...

    def finish(self, claim_id: str, status: str, evidence: Path) -> None: ...


class AttemptLedger(Protocol):
    def started(self, request: StageActivityInput, attempt_id: str) -> None: ...

    def finished(
        self,
        request: StageActivityInput,
        result: StageActivityResult,
        manifest: AttemptManifest,
    ) -> None: ...


@dataclass(frozen=True)
class AdapterContext:
    request: StageActivityInput
    attempt: AttemptWorkspace
    inputs: dict[str, tuple[Path, ...]]
    repository: Path


def attempt_identifier(run_id: str, info, suffix: str = "") -> str:
    """A stable, unique ID for one attempt.

    Temporal numbers activity IDs per workflow *execution*, so they restart at 1 both for a new
    run and for a resumed one. The run ID separates runs; the workflow execution ID separates a
    resumed scheduler from the one it replaced, which would otherwise reuse the same numbers.
    """
    execution = (getattr(info, "workflow_run_id", "") or "").replace("-", "")[:8]
    return f"{run_id}:{execution}:{info.activity_id}:{info.attempt}{suffix}"


@dataclass(frozen=True)
class StageExecution:
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str] = field(default_factory=dict)
    output_roles: dict[str, str] = field(default_factory=dict)
    unknown_on_failure: bool = False
    # Names of credentials this stage needs, forwarded from the worker's own environment when
    # set. Stage subprocesses otherwise get no secrets: a stage receives a key only by asking
    # for it, so a provider credential never reaches a stage with no business holding it.
    credentials: tuple[str, ...] = ()
    # Exit codes the stage treats as a completed run. Tools that report a finding through their
    # exit status (rather than a crash) declare the finding code here.
    success_exit_codes: frozenset[int] = frozenset({0})


class StageAdapter(Protocol):
    def build(self, context: AdapterContext) -> StageExecution: ...

    def finalize(self, context: AdapterContext) -> None: ...

    def branches(self, context: AdapterContext) -> list[dict[str, str]]: ...

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]: ...


class CommandAdapter:
    """Small adapter useful for local scripts with explicit output paths."""

    def __init__(self, builder):
        self.builder = builder

    def build(self, context: AdapterContext) -> StageExecution:
        return self.builder(context)

    def branches(self, context: AdapterContext) -> list[dict[str, str]]:
        return []

    def shots(self, context: AdapterContext) -> list[dict[str, Any]]:
        return []

    def finalize(self, context: AdapterContext) -> None:
        return None


class StageActivityRunner:
    def __init__(
        self,
        *,
        repository: Path,
        workspace_root: Path,
        store: LocalCAS,
        outbox_archive,
        resolver: ArtifactResolver,
        adapters: dict[str, StageAdapter],
        base_environment: dict[str, str] | None = None,
        paid_guard: PaidGuard | None = None,
        attempt_ledger: AttemptLedger | None = None,
    ):
        self.repository = Path(repository).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.store = store
        self.archive = outbox_archive
        self.resolver = resolver
        self.adapters = adapters
        self.base_environment = dict(base_environment or {})
        self.paid_guard = paid_guard
        self.attempt_ledger = attempt_ledger

    @activity.defn(name="run_stage")
    def run_stage(self, request: StageActivityInput) -> StageActivityResult:
        attempt_id = attempt_identifier(request.run_id, activity.info())
        return self.execute(request, attempt_id)

    def execute(self, request: StageActivityInput, attempt_id: str) -> StageActivityResult:
        adapter = self.adapters.get(request.definition.get("executor"))
        if adapter is None:
            return StageActivityResult(
                node_id=request.node_id,
                attempt_id=attempt_id,
                status="blocked",
                error=f"no activity adapter for executor {request.definition.get('executor')!r}",
            )
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
                "selectedInputs": request.selected_inputs,
            },
        )
        if self.attempt_ledger:
            self.attempt_ledger.started(request, attempt_id)
        inputs = self._hydrate_inputs(attempt, request.selected_inputs)
        context = AdapterContext(request, attempt, inputs, self.repository)
        execution = adapter.build(context)
        absent = [name for name in execution.credentials if name not in os.environ]
        if absent:
            # A stage names the credentials it needs, and a worker without one cannot do the
            # work. Running the command anyway gets an exit code and a message from whatever
            # script happened to look first -- a marble poll of an already-generated world
            # failed with "set WLT_API_KEY" and read as a failed stage. Blocked is what this
            # is: nothing is wrong with the attempt, the worker is missing configuration, and
            # the stage runs as it stands once that is fixed.
            error = "worker is missing " + ", ".join(absent)
            attempt.finalize(
                {"status": "blocked", "error": error, "command": list(execution.command)}
            )
            manifest = freeze_attempt(attempt, self.store, status="blocked")
            self._publish(workspace, manifest)
            result = StageActivityResult(
                node_id=request.node_id,
                attempt_id=attempt_id,
                status="blocked",
                error=error,
            )
            if self.attempt_ledger:
                self.attempt_ledger.finished(request, result, manifest)
            return result
        paid = bool(request.definition.get("retry", {}).get("paid"))
        try:
            claim_id = (
                self.paid_guard.begin(request, attempt_id)
                if paid and self.paid_guard is not None
                else None
            )
        except (ValueError, PermissionError) as caught:
            error = f"paid operation blocked: {caught}"
            attempt.finalize(
                {
                    "status": "blocked",
                    "error": error,
                    "command": list(execution.command),
                }
            )
            manifest = freeze_attempt(attempt, self.store, status="blocked")
            self._publish(workspace, manifest)
            result = StageActivityResult(
                node_id=request.node_id,
                attempt_id=attempt_id,
                status="blocked",
                error=error,
            )
            if self.attempt_ledger:
                self.attempt_ledger.finished(request, result, manifest)
            return result
        status, error = "succeeded", None
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            **{name: os.environ[name] for name in execution.credentials if name in os.environ},
            **self.base_environment,
            **execution.environment,
        }
        with attempt.stdout.open("wb") as stdout, attempt.stderr.open("wb") as stderr:
            process = None
            try:
                process = subprocess.Popen(
                    execution.command,
                    cwd=execution.cwd,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                )
                while True:
                    try:
                        code = process.wait(timeout=HEARTBEAT_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        beat()
                if code not in execution.success_exit_codes:
                    status = "unknown" if execution.unknown_on_failure else "failed"
                    error = f"command exited {code}"
            except BaseException as caught:
                if process and process.poll() is None:
                    # Nothing will read this stage's output now, and a paid job left running
                    # would report to a worker that is not listening.
                    process.kill()
                    process.wait()
                status = "unknown" if execution.unknown_on_failure else "failed"
                error = f"{type(caught).__name__}: {caught}"
        provider_status = "completed" if status == "succeeded" else status
        if status == "succeeded":
            try:
                finalize = getattr(adapter, "finalize", None)
                if finalize:
                    finalize(context)
            except BaseException as caught:
                status = "failed"
                error = f"adapter finalization {type(caught).__name__}: {caught}"
        if status == "succeeded":
            issues = validate_stage_outputs(
                request.definition,
                attempt.root,
                execution.output_roles,
            )
            if issues:
                status = "failed"
                error = "automatic QA failed: " + "; ".join(issues)
        attempt.finalize(
            {
                "status": status,
                "error": error,
                "command": list(execution.command),
            }
        )
        beat()
        manifest = freeze_attempt(
            attempt,
            self.store,
            status=status,
            roles=execution.output_roles,
        )
        beat()
        # Hashing and archiving a stage's outputs takes as long as the outputs are large. Say so
        # before and after, so a slow freeze is not read as a dead worker.
        self._publish(workspace, manifest)
        beat()
        if claim_id:
            self.paid_guard.finish(
                claim_id,
                provider_status,
                attempt.result,
            )
        artifacts: dict[str, list[str]] = {}
        if status == "succeeded":
            for frozen in manifest.files:
                if frozen.role == "attempt_file":
                    continue
                artifacts.setdefault(frozen.role, []).append(frozen.artifact_id)
        branches = adapter.branches(context) if status == "succeeded" else []
        shots = adapter.shots(context) if status == "succeeded" else []
        by_path = {item.relative_path: item.artifact_id for item in manifest.files}
        for descriptor in branches:
            relative = descriptor.pop("relative_path", None)
            if relative:
                descriptor["artifact_id"] = by_path[relative]
        for descriptor in shots:
            relative = descriptor.pop("relative_path", None)
            if relative:
                descriptor["clip_artifact_id"] = by_path[relative]
        result = StageActivityResult(
            node_id=request.node_id,
            attempt_id=attempt_id,
            status="blocked" if status == "unknown" else status,
            artifacts=artifacts,
            branches=branches,
            shots=shots,
            error=error,
        )
        if self.attempt_ledger:
            self.attempt_ledger.finished(request, result, manifest)
        return result

    def _publish(self, workspace: RunWorkspace, manifest: AttemptManifest) -> None:
        """Queue the attempt's files for the archive and push the queue now.

        Downstream stages, including the child runs spawned per shot, hydrate their inputs from
        the archive, so an attempt's outputs must be there before its result is returned. The
        outbox entry stays on disk until a flush succeeds; a flush that fails here is retried by
        the next attempt of this run, and the artifacts stay readable in the local store.
        """
        outbox = UploadOutbox(workspace.outbox, self.store, self.archive)
        outbox.enqueue(manifest)
        try:
            outbox.flush()
        except Exception as error:
            activity.logger.warning(
                "outbox flush deferred for %s %s: %s", manifest.run_id, manifest.attempt_id, error
            )

    def _hydrate_inputs(
        self, attempt: AttemptWorkspace, selected: dict[str, list[str]]
    ) -> dict[str, tuple[Path, ...]]:
        hydrated: dict[str, tuple[Path, ...]] = {}
        root = attempt.root / "inputs"
        for name, artifact_ids in selected.items():
            paths = []
            destination = root / name
            destination.mkdir(parents=True, exist_ok=True)
            for artifact_id in artifact_ids:
                path = self.resolver.hydrate(artifact_id, destination)
                if (
                    not path.resolve().is_relative_to(destination.resolve())
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    raise ValueError(
                        f"artifact resolver escaped its input directory for {artifact_id}"
                    )
                paths.append(path)
            hydrated[name] = tuple(paths)
        return hydrated
