"""Environment-configured ASGI service used by deployment."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from temporalio.client import Client

from orchestrator.api.app import ApiSettings, create_app
from orchestrator.api.storage import (
    FilesystemArtifactReader,
    FilesystemSourceStore,
    S3ArtifactReader,
    S3SourceStore,
)
from orchestrator.repository import PipelineRepository
from orchestrator.temporal import TemporalWorkflowControl


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


class LazyTemporalControl:
    def __init__(self, address: str):
        self.address = address
        self._control: TemporalWorkflowControl | None = None
        self._lock = asyncio.Lock()

    async def control(self) -> TemporalWorkflowControl:
        async with self._lock:
            if self._control is None:
                self._control = TemporalWorkflowControl(await Client.connect(self.address))
            return self._control

    async def start(self, request) -> None:
        await (await self.control()).start(request)

    async def signal(self, run_id: str, name: str, value: Any = None) -> None:
        await (await self.control()).signal(run_id, name, value)

    async def state(self, run_id: str) -> dict[str, Any]:
        return await (await self.control()).state(run_id)


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
app = create_app(
    PipelineRepository(sessions),
    LazyTemporalControl(required("WANDER_TEMPORAL_ADDRESS")),
    ApiSettings(
        bearer_token=required("WANDER_API_TOKEN"),
        code_revision=required("WANDER_CODE_REVISION"),
        container_digest=os.environ.get("WANDER_CONTAINER_DIGEST"),
    ),
    source_store,
    artifact_reader=artifact_reader,
    # Same default as the worker; the review transcripts the admin tails live here.
    review_workspace_root=Path(os.environ.get("WANDER_WORKSPACE_ROOT", "/var/lib/wander/runs")),
)
