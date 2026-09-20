"""FastAPI control plane with durable commands and resumable SSE."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
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

from orchestrator.api.artifacts import ArtifactReader, register_artifact_routes
from orchestrator.api.reviews import register_review_routes
from orchestrator.contracts import BudgetPolicy, Run
from orchestrator.graph import GRAPH_VERSION, instantiate_graph
from orchestrator.repository import PipelineRepository
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import (
    ApprovalSignal,
    DecisionSignal,
    GenerationWorkflowInput,
    OperatorMessage,
)


class WorkflowControl(Protocol):
    async def start(self, request: GenerationWorkflowInput) -> None: ...

    async def signal(self, run_id: str, name: str, value: Any = None) -> None: ...

    async def state(self, run_id: str) -> dict[str, Any]: ...


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

    id: str | None = None
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifact_id: str
    options: GraphOptions = Field(default_factory=GraphOptions)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)


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
    workflows: WorkflowControl,
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

    @app.post("/api/pipeline/runs", status_code=201)
    async def create_run(request: CreateRunRequest, user: str = Depends(actor)):
        run_id = request.id or f"run-{uuid.uuid4().hex}"
        graph = instantiate_graph(request.options)
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
        try:
            repository.create_run(run, graph)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        workflow_request = GenerationWorkflowInput(
            run_id=run.id,
            source_sha256=run.source_sha256,
            source_artifact_id=run.source_artifact_id,
            options=request.options.model_dump(mode="json"),
        )
        try:
            await workflows.start(workflow_request)
        except Exception as error:
            repository.append_event(
                run.id,
                "run",
                run.id,
                {"status": "blocked", "error": type(error).__name__},
            )
            raise HTTPException(status_code=503, detail="workflow could not start") from error
        return {"id": run.id, "status": run.status, "graphFingerprint": graph.fingerprint}

    @app.post("/api/pipeline/runs/upload", status_code=201)
    async def upload_run(
        source: UploadFile = File(),
        options: str = Form(default="{}"),
        budget: str = Form(default="{}"),
        requested_id: str | None = Form(default=None),
        user: str = Depends(actor),
    ):
        if source_store is None:
            raise HTTPException(status_code=503, detail="source upload is not configured")
        try:
            graph_options = GraphOptions.model_validate_json(options)
            budget_policy = BudgetPolicy.model_validate_json(budget)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        run_id = requested_id or f"run-{uuid.uuid4().hex}"
        filename = Path(source.filename or "source.bin").name
        digest = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(prefix="wander-source-", delete=True) as temporary:
            while chunk := await source.read(1024 * 1024):
                size += len(chunk)
                if size > settings.maximum_upload_bytes:
                    raise HTTPException(status_code=413, detail="source upload is too large")
                digest.update(chunk)
                temporary.write(chunk)
            temporary.flush()
            source_sha256 = digest.hexdigest()
            artifact_id = f"artifact:{run_id}:source"
            storage_key = f"sources/{source_sha256[:2]}/{source_sha256}"
            source_store.put(Path(temporary.name), storage_key)
        graph = instantiate_graph(graph_options)
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
        try:
            repository.create_run(run, graph)
            repository.add_source_artifact(
                artifact_id=artifact_id,
                run_id=run_id,
                sha256=source_sha256,
                size=size,
                media_type=source.content_type or "application/octet-stream",
                storage_key=storage_key,
                filename=filename,
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        try:
            await workflows.start(
                GenerationWorkflowInput(
                    run_id=run_id,
                    source_sha256=source_sha256,
                    source_artifact_id=artifact_id,
                    options=graph_options.model_dump(mode="json"),
                )
            )
        except Exception as error:
            repository.append_event(
                run_id,
                "run",
                run_id,
                {"status": "blocked", "error": type(error).__name__},
            )
            raise HTTPException(status_code=503, detail="workflow could not start") from error
        return {
            "id": run_id,
            "status": run.status,
            "sourceArtifactId": artifact_id,
            "sourceSha256": source_sha256,
            "graphFingerprint": graph.fingerprint,
        }

    @app.get("/api/pipeline")
    async def list_runs(user: str = Depends(actor)):
        del user
        return {"runs": repository.list_runs()}

    @app.get("/api/pipeline/runs/{run_id}")
    async def get_run(run_id: str, user: str = Depends(actor)):
        del user
        try:
            summary = repository.run_summary(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        try:
            summary["live"] = await workflows.state(run_id)
        except Exception as error:
            # A finished workflow can no longer answer queries once the workflow code has
            # changed (Temporal replays its history under the new code and finds it
            # nondeterministic). The recorded rows still describe the run; say why live is empty.
            summary["live"] = None
            summary["liveError"] = type(error).__name__
        return summary

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
        message = OperatorMessage(
            id=identifier,
            author=user,
            message=request.message,
            node_id=request.node_id,
            attempt_id=request.attempt_id,
            attachment_artifact_ids=request.attachment_artifact_ids,
        )
        await workflows.signal(run_id, "operator_message", message)
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
        await workflows.signal(
            run_id,
            "approve",
            ApprovalSignal(
                node_id=request.node_id,
                attempt_id=request.attempt_id,
                artifacts=request.artifacts,
                approved_by=user,
                rationale=request.rationale,
            ),
        )
        return {"id": identifier}

    @app.post("/api/pipeline/runs/{run_id}/{command}", status_code=202)
    async def command(
        run_id: str,
        command: str,
        request: DecisionRequest,
        user: str = Depends(actor),
    ):
        if command not in {"reject", "retry", "pause", "resume", "cancel"}:
            raise HTTPException(status_code=404)
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
        value = (
            DecisionSignal(
                node_id=request.node_id,
                attempt_id=request.attempt_id,
                rationale=request.rationale,
                parameters=request.parameters,
            )
            if command in {"reject", "retry"}
            else None
        )
        await workflows.signal(run_id, command, value)
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
    return app
