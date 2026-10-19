"""FastAPI control plane with durable commands and resumable SSE."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from orchestrator.api.artifacts import ArtifactReader, register_artifact_routes, safe_filename
from orchestrator.api.progress import (
    StageSnapshot,
    find_run_directory,
    register_progress_routes,
    stage_snapshot,
)
from orchestrator.api.reviews import register_review_routes
from orchestrator.api.questions import apply_question_hold, restore_legacy_question_states
from orchestrator.contracts import BudgetPolicy, Run
from orchestrator.journal import Journal
from orchestrator.repository import PipelineRepository, TERMINAL_RUN_STATUSES, live_run_status
from orchestrator.steps import STEPS, RunOptions, base_step

GRAPH_VERSION = "wander.agent-run/1"


def stage_definition(name: str, existing: dict | None = None) -> dict:
    described = STEPS[base_step(name)]
    human_gate = name == "review"
    return {
        "id": name,
        "title": described.summary,
        "kind": "human" if human_gate else "compute",
        "executor": None if human_gate else "wander",
        "writes": described.writes,
        "paid": described.paid,
        "parameter_schema": described.parameters or {},
        **(existing or {}),
    }


def with_stage_snapshot(summary: dict, snapshot: StageSnapshot) -> dict:
    """Overlay the directory's current graph without fabricating database history."""
    existing = {node["id"]: node for node in summary["nodes"]}
    attempted = {attempt["nodeId"] for attempt in summary["attempts"]}
    reviewed = {review["nodeId"] for review in summary["reviews"]}
    nodes = []
    for stage in snapshot.stages:
        name = stage["name"]
        old = existing.get(name, {})
        old_status = old.get("status")
        stage_status = stage["status"]
        if old_status == "waiting_human" or (
            old_status in {"waiting_agent", "running"} and stage_status in {"pending", "queued"}
        ):
            stage_status = old_status
        node = {
            **old,
            "id": name,
            "status": stage_status,
            "dependencies": snapshot.required_dependencies[name],
            "selectedAttemptId": old.get("selectedAttemptId"),
            "blockedReason": (
                old.get("blockedReason")
                if old.get("status") == "waiting_human"
                else (stage["reason"] or old.get("blockedReason"))
                if stage["status"] in {"blocked", "failed"}
                else None
            ),
            "definition": stage_definition(name, old.get("definition")),
        }
        for field in ("execution", "freshInference", "sourceRun"):
            if field in stage:
                node[field] = stage[field]
        nodes.append(node)
    planned = {stage["name"] for stage in snapshot.stages}
    pending = snapshot.journal.pending_question()
    question_step = pending.data.get("step") if pending is not None else None
    # A seeded, never-started node from the original one-person graph is stale once
    # tracking exposes more people. Actual attempts and operator decisions remain visible.
    for node in summary["nodes"]:
        if node["id"] in planned or not (
            node["id"] in attempted
            or node["id"] in reviewed
            or node["id"] == question_step
            or node.get("selectedAttemptId")
            or node["status"] not in {"queued", "pending"}
        ):
            continue
        if base_step(node["id"]) in STEPS:
            node = {**node, "definition": stage_definition(node["id"], node.get("definition"))}
        nodes.append(node)
    summary["nodes"] = nodes
    summary["status"] = live_run_status(
        summary.get("recordedStatus", summary["status"]), [node["status"] for node in nodes]
    )
    return summary


def source_filename(name: str) -> str:
    """Keep the source's extension even when its untrusted basename needs truncation."""
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(basename).suffix
    if suffix and len(suffix) <= 20 and suffix[1:].isalnum():
        stem = safe_filename(Path(basename).stem)[: 120 - len(suffix)]
        return stem + suffix
    return safe_filename(basename)


class Runs(Protocol):
    """Where runs live on disk. The supervisor picks up whatever appears here."""

    def open(
        self,
        *,
        run_id: str,
        name: str,
        source: Path,
        options: dict[str, Any],
        source_artifact_id: str | None = None,
    ) -> Path: ...

    def journal(self, run_id: str) -> Journal: ...

    def pause(self, run_id: str, paused: bool) -> None: ...


class SourceStore(Protocol):
    def put(self, source: Path, storage_key: str) -> None: ...


class ApiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bearer_token: str = Field(min_length=24)
    code_revision: str = Field(min_length=7)
    container_digest: str | None = None
    maximum_upload_bytes: int = Field(default=2_000_000_000, ge=1)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_id: str = Field(min_length=1, max_length=128)
    options: RunOptions = Field(default_factory=RunOptions)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)


class SavedInputsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=20_000)
    node_id: str | None = None
    attempt_id: str | None = None
    attachment_artifact_ids: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    attempt_id: str
    artifacts: dict[str, list[str]]
    rationale: str = Field(min_length=1)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    attempt_id: str | None = None
    rationale: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ReconcileProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern=r"^(completed|failed)$")
    provider_operation_id: str | None = None
    rationale: str = Field(min_length=1)


def create_app(
    repository: PipelineRepository,
    runs: Runs,
    settings: ApiSettings,
    source_store: SourceStore | None = None,
    artifact_reader: ArtifactReader | None = None,
    review_workspace_root: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Wander Pipeline", version="1")

    @app.get("/healthz")
    async def health():
        repository.ping()
        return {"status": "ok"}

    async def actor(authorization: str | None = Header(default=None)) -> str:
        expected = f"Bearer {settings.bearer_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return "operator"

    @app.get("/api/pipeline/options")
    async def pipeline_options(user: str = Depends(actor)):
        del user
        return RunOptions.model_json_schema()

    @app.post("/api/pipeline/doctor")
    def doctor(options: RunOptions, user: str = Depends(actor)):
        del user
        from orchestrator.doctor import inspect_setup

        return inspect_setup(options.model_dump(mode="json"))

    def require_supported_budget(budget: BudgetPolicy) -> None:
        if budget.maximum_attempts_by_stage or budget.maximum_provider_operations:
            raise HTTPException(
                422,
                "Custom run budgets are not enforced by the directory runner.",
            )
        if budget.maximum_cost is not None:
            raise HTTPException(
                422,
                "maximum_cost is not enforced by the directory runner. Use provider-side spending limits; monetary-capped submissions are disabled until all provider calls are metered.",
            )

    def require_new_run(run_id: str) -> None:
        # Check before archiving bytes or downloading a replay. The unique database key
        # still arbitrates concurrent submissions below.
        try:
            repository.run_summary(run_id)
        except KeyError:
            return
        raise HTTPException(status_code=409, detail=f"run already exists: {run_id}")

    def submit(run: Run, source: Path, filename: str, source_metadata: dict | None = None):
        created = False
        try:
            repository.create_run_row(run)
            created = True
            if source_metadata is not None:
                repository.add_source_artifact(**source_metadata)
            runs.open(
                run_id=run.id,
                name=Path(filename).stem or run.id,
                source=source,
                options=run.configuration["options"],
                source_artifact_id=run.source_artifact_id,
            )
        except Exception as error:
            if created:
                repository.delete_unstarted_run(run.id)
            if isinstance(error, (ValueError, FileExistsError, IntegrityError)):
                raise HTTPException(status_code=409, detail=str(error)) from error
            raise

    @app.post("/api/pipeline/runs", status_code=201)
    async def create_run(request: CreateRunRequest, user: str = Depends(actor)):
        require_supported_budget(request.budget)
        run_id = request.id or f"run-{uuid.uuid4().hex}"
        require_new_run(run_id)
        if artifact_reader is None:
            raise HTTPException(status_code=503, detail="source artifact reads are not configured")
        try:
            artifact = repository.get_source_artifact(request.source_artifact_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if artifact["sha256"] != request.source_sha256:
            raise HTTPException(status_code=422, detail="source digest does not match artifact")
        if artifact["size"] > settings.maximum_upload_bytes:
            raise HTTPException(status_code=413, detail="source upload is too large")
        filename = source_filename(artifact["name"])
        run = Run(
            id=run_id,
            graph_version=GRAPH_VERSION,
            code_revision=settings.code_revision,
            container_digest=settings.container_digest,
            source_sha256=request.source_sha256,
            source_artifact_id=request.source_artifact_id,
            configuration={"options": request.options.model_dump(mode="json")},
            budget=request.budget,
            created_by=user,
        )
        # Artifact IDs are durable references, not runnable inputs. Hydrate and verify a
        # local copy before publishing anything the supervisor can discover.
        with tempfile.TemporaryDirectory(prefix="wander-source-") as directory:
            local = Path(directory) / filename
            payload = artifact_reader.read(artifact["storageKey"])
            if payload.path is not None:
                shutil.copyfile(payload.path, local)
            elif payload.body is not None and payload.status_code == 200:
                local.write_bytes(payload.body)
            else:
                raise HTTPException(status_code=422, detail="source artifact has no complete bytes")
            digest = hashlib.sha256()
            with local.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if (
                local.stat().st_size != artifact["size"]
                or digest.hexdigest() != request.source_sha256
            ):
                raise HTTPException(
                    status_code=422, detail="source artifact bytes failed verification"
                )
            if not local.stat().st_size:
                raise HTTPException(status_code=422, detail="source upload is empty")
            submit(run, local, filename)
        return {"id": run.id, "status": run.status}

    @app.post("/api/pipeline/runs/upload", status_code=201)
    async def upload_run(
        source: UploadFile = File(),
        options: str = Form(default="{}"),
        budget: str = Form(default="{}"),
        requested_id: str | None = Form(
            default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
        ),
        user: str = Depends(actor),
    ):
        if source_store is None:
            raise HTTPException(status_code=503, detail="source upload is not configured")
        try:
            graph_options = RunOptions.model_validate_json(options)
            budget_policy = BudgetPolicy.model_validate_json(budget)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        require_supported_budget(budget_policy)
        run_id = requested_id or f"run-{uuid.uuid4().hex}"
        require_new_run(run_id)
        filename = source_filename(source.filename or "source.bin")
        digest = hashlib.sha256()
        size = 0
        # The filename preserves the video extension through the local run copy, and the
        # context manager removes uploads on every error path, including oversized bodies.
        with tempfile.TemporaryDirectory(prefix="wander-source-") as directory:
            stored_path = Path(directory) / filename
            with stored_path.open("wb") as temporary:
                while chunk := await source.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.maximum_upload_bytes:
                        raise HTTPException(status_code=413, detail="source upload is too large")
                    digest.update(chunk)
                    temporary.write(chunk)
            if not size:
                raise HTTPException(status_code=422, detail="source upload is empty")
            source_sha256 = digest.hexdigest()
            artifact_id = f"artifact:{run_id}:source"
            if len(artifact_id) > 128:
                artifact_id = f"artifact:{hashlib.sha256(run_id.encode()).hexdigest()}:source"
            storage_key = f"sources/{source_sha256[:2]}/{source_sha256}"
            source_store.put(stored_path, storage_key)
            run = Run(
                id=run_id,
                graph_version=GRAPH_VERSION,
                code_revision=settings.code_revision,
                container_digest=settings.container_digest,
                source_sha256=source_sha256,
                source_artifact_id=artifact_id,
                configuration={"options": graph_options.model_dump(mode="json")},
                budget=budget_policy,
                created_by=user,
            )
            submit(
                run,
                stored_path,
                filename,
                {
                    "artifact_id": artifact_id,
                    "run_id": run_id,
                    "sha256": source_sha256,
                    "size": size,
                    "media_type": source.content_type or "application/octet-stream",
                    "storage_key": storage_key,
                    "filename": filename,
                },
            )
        return {
            "id": run_id,
            "status": run.status,
            "sourceArtifactId": artifact_id,
            "sourceSha256": source_sha256,
        }

    @app.get("/api/pipeline")
    async def list_runs(user: str = Depends(actor)):
        del user
        return {"runs": [with_live_state(summary) for summary in repository.list_runs()]}

    def with_live_state(summary: dict) -> dict:
        # Read the same durable holds as the supervisor, even when the database mirror
        # missed a journal write or a question has no associated stage.
        try:
            directory = (
                find_run_directory(review_workspace_root, summary["id"])
                if review_workspace_root
                else None
            )
            journal = Journal(directory) if directory else runs.journal(summary["id"])
            paused = (journal.run_dir / "paused").exists()
        except (KeyError, OSError, ValueError, TypeError, AttributeError):
            summary["live"] = {"source": "journal", "available": False}
            return summary
        restore_legacy_question_states(summary, journal, repository)
        snapshot = stage_snapshot(journal.run_dir)
        if snapshot is not None and snapshot.run_id == summary["id"]:
            with_stage_snapshot(summary, snapshot)
        summary["live"] = {"source": "journal", "paused": paused}
        # The terminal journal decision survives a failed database projection and late
        # detached-stage completions. Do not turn it back into a request for UI input.
        finished = journal.finished()
        if finished in TERMINAL_RUN_STATUSES:
            summary["status"] = finished
        if summary["status"] not in TERMINAL_RUN_STATUSES:
            apply_question_hold(summary, journal)
            summary["status"] = live_run_status(
                summary.get("recordedStatus", summary["status"]),
                [node["status"] for node in summary["nodes"]],
            )
            if paused:
                summary["status"] = "paused"
            elif journal.unanswered():
                summary["status"] = "waiting_human"
        return summary

    @app.get("/api/pipeline/runs/{run_id}")
    async def get_run(run_id: str, user: str = Depends(actor)):
        del user
        try:
            summary = repository.run_summary(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return with_live_state(summary)

    @app.get("/api/pipeline/runs/{run_id}/saved-inputs")
    def saved_inputs(run_id: str, user: str = Depends(actor)):
        try:
            repository.run_summary(run_id)
            inspect = getattr(runs, "saved_inputs", None)
            return (
                inspect(run_id)
                if inspect
                else {
                    "available": False,
                    "sourceRunId": run_id,
                    "reason": "Saved inputs are unavailable",
                }
            )
        except KeyError as error:
            raise HTTPException(404, "Unknown run") from error

    @app.post("/api/pipeline/runs/{run_id}/saved-inputs", status_code=201)
    def prepare_saved_inputs(run_id: str, request: SavedInputsRequest, user: str = Depends(actor)):
        new_id = request.id or f"run-{uuid.uuid4().hex}"
        require_new_run(new_id)
        prepare = getattr(runs, "prepare_saved_inputs", None)
        if prepare is None:
            raise HTTPException(409, "Saved inputs are unavailable")
        try:
            repository.run_summary(run_id)
            prepare(run_id, new_id, settings=settings, user=user)
        except KeyError as error:
            raise HTTPException(404, "Unknown run or missing saved inputs") from error
        except (ValueError, OSError, IntegrityError) as error:
            raise HTTPException(409, str(error)) from error
        return {"id": new_id, "status": "paused"}

    @app.post("/api/pipeline/runs/{run_id}/messages", status_code=202)
    async def add_message(run_id: str, request: MessageRequest, user: str = Depends(actor)):
        identifier = repository.add_message(
            run_id=run_id,
            author=user,
            message=request.message,
            node_id=request.node_id,
            attempt_id=request.attempt_id,
            attachment_artifact_ids=request.attachment_artifact_ids,
        )
        runs.journal(run_id).append(
            "answer", text=request.message, author=user, step=request.node_id
        )
        return {"id": identifier}

    @app.post(
        "/api/pipeline/runs/{run_id}/provider-claims/{claim_id}/reconcile",
        status_code=202,
    )
    async def reconcile_provider(
        run_id: str,
        claim_id: str,
        request: ReconcileProviderRequest,
        user: str = Depends(actor),
    ):
        try:
            repository.reconcile_provider_claim(
                run_id=run_id,
                claim_id=claim_id,
                status=request.status,
                provider_operation_id=request.provider_operation_id,
                actor=user,
                rationale=request.rationale,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"accepted": True}

    @app.post("/api/pipeline/runs/{run_id}/approve", status_code=202)
    async def approve(run_id: str, request: ApprovalRequest, user: str = Depends(actor)):
        identifier = repository.add_approval(
            run_id=run_id,
            node_id=request.node_id,
            attempt_id=request.attempt_id,
            kind="quality_gate",
            decision="approved",
            actor=user,
            rationale=request.rationale,
        )
        runs.journal(run_id).append(
            "answer",
            text=f"approved: {request.rationale}",
            author=user,
            step=request.node_id,
        )
        return {"id": identifier}

    @app.post("/api/pipeline/runs/{run_id}/{command}", status_code=202)
    def command(
        run_id: str,
        command: str,
        request: DecisionRequest,
        user: str = Depends(actor),
    ):
        if command not in {"reject", "retry", "pause", "resume", "cancel"}:
            raise HTTPException(status_code=404)
        # Validate before recording a command. A stale operator tab must receive a
        # useful 404 without appending an event or creating a stray run directory.
        try:
            summary = repository.run_summary(run_id)
            if command == "retry" and summary["configuration"].get("retainedDelivery") is not None:
                raise HTTPException(409, "Stage retries are disabled for saved-input delivery runs")
            journal = runs.journal(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        # The local executor holds this lock only until its subprocess starts. A control
        # accepted after that point affects the next stage; one accepted before it stops
        # the launch. This endpoint runs in FastAPI's threadpool so waiting on flock does
        # not block unrelated async requests.
        with journal.admission_lock():
            repository.append_event(
                run_id,
                "approval" if command == "reject" else "run",
                request.node_id,
                {
                    "command": command,
                    "actor": user,
                    "rationale": request.rationale,
                    "parameters": request.parameters,
                },
            )
            if command in {"retry", "resume"}:
                journal.append("run.resumed", reason=request.rationale)
            if command in {"reject", "retry"}:
                asked = "run again" if command == "retry" else "do not accept"
                journal.append(
                    "answer",
                    text=f"{asked} `{request.node_id}`: {request.rationale}",
                    author=user,
                    step=request.node_id,
                )
            elif command == "cancel":
                journal.append("run.finished", status="canceled", summary=request.rationale)
                runs.pause(run_id, False)
            else:
                runs.pause(run_id, command == "pause")
        return {"accepted": True}

    @app.get("/api/pipeline/runs/{run_id}/events")
    async def events(
        run_id: str,
        http_request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        user: str = Depends(actor),
    ):
        del user
        try:
            cursor = int(last_event_id or http_request.query_params.get("after", "0"))
        except ValueError as error:
            raise HTTPException(status_code=400, detail="invalid event cursor") from error

        async def stream():
            nonlocal cursor
            while not await http_request.is_disconnected():
                batch = repository.events_after(run_id, cursor)
                if not batch:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1)
                    continue
                for event in batch:
                    cursor = event.sequence
                    payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
                    yield f"id: {event.sequence}\nevent: {event.type}\ndata: {payload}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    register_artifact_routes(app, repository, actor, artifact_reader)
    register_review_routes(app, repository, actor, review_workspace_root)
    register_progress_routes(app, repository, actor, review_workspace_root)
    return app
