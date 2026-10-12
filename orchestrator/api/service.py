"""Production filesystem adapter and explicit environment-configured ASGI factory."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from orchestrator.api.app import ApiSettings, SourceStore, create_app
from orchestrator.api.artifacts import ArtifactReader, safe_filename
from orchestrator.api.storage import FilesystemArtifactReader, FilesystemSourceStore
from orchestrator.config import resolve_runs_root
from orchestrator.journal import Journal, RunProjection
from orchestrator.repository import PipelineRepository
from orchestrator.steps import STEPS
from orchestrator.supervisor import name_for, open_run


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


class FilesystemRuns:
    """Publish complete run directories for the supervisor to discover."""

    def __init__(self, root: Path, repository: PipelineRepository):
        self.root = resolve_runs_root(root)
        self.repository = repository

    def _dir(self, run_id: str) -> Path:
        for described in self.root.glob("*/run.json"):
            if described.parent.is_symlink():
                continue
            if json.loads(described.read_text()).get("runId") == run_id:
                return described.parent
        raise KeyError(f"no run directory for {run_id}")

    def open(
        self,
        *,
        run_id: str,
        name: str,
        source: Path,
        options: dict,
        source_artifact_id: str | None = None,
    ) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise ValueError("invalid run id")
        if source is None or not Path(source).is_file():
            raise ValueError("a run requires a local source file")
        try:
            self._dir(run_id)
        except KeyError:
            pass
        else:
            raise FileExistsError(f"run directory already exists: {run_id}")
        name = safe_filename(name)
        self.root.mkdir(parents=True, exist_ok=True)
        staging_root = self.root / ".submissions"
        staging_root.mkdir(exist_ok=True)
        # A nested temporary directory has no root-level run.json, so workers cannot see a
        # partially copied source or a plan whose database rows are still being seeded.
        with tempfile.TemporaryDirectory(prefix="run-", dir=staging_root) as staging:
            run_dir = open_run(
                Path(staging),
                run_id=run_id,
                name=name,
                source=source,
                options=options,
                repository=self.repository,
                source_artifact_id=source_artifact_id,
            )
            destination = self.root / name_for(self.root, name, run_id)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"run directory already exists: {destination.name}")
            described = json.loads((run_dir / "run.json").read_text())
            described["name"] = destination.name
            described["source"] = str(destination / Path(described["source"]).name)
            (run_dir / "run.json").write_text(json.dumps(described, indent=2))
            run_dir.rename(destination)
            return destination

    def journal(self, run_id: str) -> Journal:
        run_dir = self._dir(run_id)
        return Journal(run_dir, projection=RunProjection(self.repository, run_id, catalogue=STEPS))

    def pause(self, run_id: str, paused: bool) -> None:
        marker = self._dir(run_id) / "paused"
        if paused:
            marker.write_text("paused by an operator\n")
        else:
            marker.unlink(missing_ok=True)


def create_service(
    *,
    repository: PipelineRepository | None = None,
    runs_root: str | Path | None = None,
    settings: ApiSettings | None = None,
    artifact_root: str | Path | None = None,
    source_store: SourceStore | None = None,
    artifact_reader: ArtifactReader | None = None,
):
    """Build the real API explicitly; importing its adapter requires no credentials or I/O."""
    if repository is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        database_url = required("WANDER_DATABASE_URL").replace(
            "postgresql+asyncpg://", "postgresql+psycopg://"
        )
        engine = create_engine(database_url, pool_pre_ping=True)
        repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
    if settings is None:
        settings = ApiSettings(
            bearer_token=required("WANDER_API_TOKEN"),
            code_revision=required("WANDER_CODE_REVISION"),
            container_digest=os.environ.get("WANDER_CONTAINER_DIGEST"),
        )
    if source_store is None and artifact_reader is None:
        artifact_root = artifact_root or os.environ.get("WANDER_ARTIFACT_ROOT")
        if artifact_root:
            source_store = FilesystemSourceStore(Path(artifact_root))
            artifact_reader = FilesystemArtifactReader(Path(artifact_root))
        else:
            import boto3

            from orchestrator.api.storage import S3ArtifactReader, S3SourceStore

            s3 = boto3.client(
                "s3",
                endpoint_url=required("WANDER_S3_ENDPOINT"),
                aws_access_key_id=required("WANDER_S3_ACCESS_KEY"),
                aws_secret_access_key=required("WANDER_S3_SECRET_KEY"),
                region_name=os.environ.get("WANDER_S3_REGION", "us-east-1"),
            )
            bucket = os.environ.get("WANDER_S3_BUCKET", "wander-artifacts")
            source_store = S3SourceStore(s3, bucket)
            artifact_reader = S3ArtifactReader(s3, bucket)
    root = resolve_runs_root(runs_root)
    runs = FilesystemRuns(root, repository)
    app = create_app(
        repository,
        runs,
        settings,
        source_store,
        artifact_reader=artifact_reader,
        review_workspace_root=root,
    )
    app.state.repository = repository
    app.state.runs = runs
    return app


def __getattr__(name: str):
    # Preserve `uvicorn orchestrator.api.service:app` without constructing clients when a
    # test or command merely imports FilesystemRuns/create_service.
    if name == "app":
        app = create_service()
        globals()[name] = app
        return app
    raise AttributeError(name)
