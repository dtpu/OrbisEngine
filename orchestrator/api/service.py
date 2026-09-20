"""Environment-configured ASGI service used by deployment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.api.app import ApiSettings, create_app
from orchestrator.api.storage import (
    FilesystemArtifactReader,
    FilesystemSourceStore,
    S3ArtifactReader,
    S3SourceStore,
)
from orchestrator.journal import Journal, RunProjection
from orchestrator.repository import PipelineRepository
from orchestrator.steps import STEPS
from orchestrator.supervisor import open_run


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


class FilesystemRuns:
    """Runs are directories. Opening one is making its directory; the supervisor finds it.

    Nothing is scheduled here and nothing is signalled. An operator's answer is a line in the
    run's journal, which the agent reads at the start of its next turn.
    """

    def __init__(self, root: Path, repository: PipelineRepository):
        self.root = Path(root)
        self.repository = repository

    def _dir(self, run_id: str) -> Path:
        for described in self.root.glob("*/run.json"):
            if json.loads(described.read_text()).get("runId") == run_id:
                return described.parent
        raise KeyError(f"no run directory for {run_id}")

    def open(self, *, run_id: str, name: str, source: Path, options: dict) -> Path:
        return open_run(
            self.root,
            run_id=run_id,
            name=name,
            source=source,
            options=options,
            repository=self.repository,
        )

    def journal(self, run_id: str) -> Journal:
        run_dir = self._dir(run_id)
        return Journal(run_dir, projection=RunProjection(self.repository, run_id, catalogue=STEPS))

    def pause(self, run_id: str, paused: bool) -> None:
        marker = self._dir(run_id) / "paused"
        if paused:
            marker.write_text("paused by an operator\n")
        else:
            marker.unlink(missing_ok=True)


database_url = required("WANDER_DATABASE_URL").replace(
    "postgresql+asyncpg://", "postgresql+psycopg://"
)
engine = create_engine(database_url, pool_pre_ping=True)
sessions = sessionmaker(engine, expire_on_commit=False)
artifact_root = os.environ.get("WANDER_ARTIFACT_ROOT")
if artifact_root:
    source_store = FilesystemSourceStore(artifact_root)
    artifact_reader = FilesystemArtifactReader(artifact_root)
else:
    s3 = boto3.client(
        "s3",
        endpoint_url=required("WANDER_S3_ENDPOINT"),
        aws_access_key_id=required("WANDER_S3_ACCESS_KEY"),
        aws_secret_access_key=required("WANDER_S3_SECRET_KEY"),
        region_name=os.environ.get("WANDER_S3_REGION", "us-east-1"),
    )
    bucket = os.environ.get("WANDER_S3_BUCKET", "wander-artifacts")
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
    source_store = S3SourceStore(s3, bucket)
    artifact_reader = S3ArtifactReader(s3, bucket)
repository = PipelineRepository(sessions)
runs_root = Path(os.environ.get("WANDER_RUNS_ROOT", "/var/lib/wander/runs"))
app = create_app(
    repository,
    FilesystemRuns(runs_root, repository),
    ApiSettings(
        bearer_token=required("WANDER_API_TOKEN"),
        code_revision=required("WANDER_CODE_REVISION"),
        container_digest=os.environ.get("WANDER_CONTAINER_DIGEST"),
    ),
    source_store,
    artifact_reader=artifact_reader,
    # Same default as the worker; the review transcripts the admin tails live here.
    review_workspace_root=runs_root,
)
