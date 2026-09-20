"""Temporal worker entrypoint for control and resource-specific activity queues."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from pathlib import Path

import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from temporalio.client import Client
from temporalio.worker import Worker

from orchestrator.activities.adapters import default_adapters
from orchestrator.activities.lifecycle import RunLifecycleActivities
from orchestrator.activities.review import ReviewActivities
from orchestrator.agent.harness import HarnessAgent, HarnessPolicy
from orchestrator.activities.stage import StageActivityRunner
from orchestrator.artifacts import (
    DatabaseArtifactResolver,
    DatabaseFilesystemArtifactResolver,
    LocalCAS,
    S3Archive,
    UploadOutbox,
)
from orchestrator.graph import GRAPH_VERSION
from orchestrator.paid import DatabasePaidGuard
from orchestrator.persistence import DatabaseAttemptLedger
from orchestrator.repository import PipelineRepository
from orchestrator.workflows.run import GenerationWorkflow

ACTIVITY_QUEUES = ("local_cpu", "modal_gpu", "external_api", "ssh_gpu")


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


async def heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        path.touch()
        await asyncio.sleep(5)


def flush_pending_outboxes(workspace_root: Path, store: LocalCAS, archive) -> int:
    """Push attempt manifests that were queued but never reached the archive.

    An attempt enqueues its files and flushes them immediately, but a flush can fail — a full
    disk is the usual reason — and the attempt still finishes, because its outputs are already
    safe in the local store. Nothing retried those entries afterwards, so the database recorded
    artifacts whose blobs the archive never received, and the next stage to want one failed with
    "artifact storage object is unavailable". Recovering at start-up costs nothing when there is
    nothing to do.
    """
    recovered = 0
    for outbox in sorted(Path(workspace_root).glob("*/outbox")):
        try:
            recovered += UploadOutbox(outbox, store, archive).flush()
        except Exception as error:  # noqa: BLE001 - one bad run must not stop the worker
            print(f"could not flush {outbox}: {error}", flush=True)
    return recovered


async def main() -> None:
    database_url = required("WANDER_DATABASE_URL").replace(
        "postgresql+asyncpg://", "postgresql+psycopg://"
    )
    engine = create_engine(database_url, pool_pre_ping=True)
    sessions = sessionmaker(engine, expire_on_commit=False)
    artifact_root = os.environ.get("WANDER_ARTIFACT_ROOT")
    if artifact_root:
        archive = LocalCAS(Path(artifact_root))
        archive.initialize()
        resolver = DatabaseFilesystemArtifactResolver(sessions, artifact_root)
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
        archive = S3Archive(s3, bucket, prefix="pipeline")
        resolver = DatabaseArtifactResolver(sessions, s3, bucket)
    code_revision = required("WANDER_CODE_REVISION")
    runner = StageActivityRunner(
        repository=Path(os.environ.get("WANDER_REPOSITORY", "/app")),
        workspace_root=Path(os.environ.get("WANDER_WORKSPACE_ROOT", "/var/lib/wander/runs")),
        store=LocalCAS(Path(os.environ.get("WANDER_CAS_ROOT", "/var/lib/wander/cas"))),
        outbox_archive=archive,
        resolver=resolver,
        adapters=default_adapters(),
        base_environment={"MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "dtpu")},
        paid_guard=DatabasePaidGuard(
            sessions,
            code_version={
                "repository": code_revision,
                "container": os.environ.get("WANDER_CONTAINER_DIGEST", "development"),
            },
        ),
        attempt_ledger=DatabaseAttemptLedger(
            sessions,
            code_revision=code_revision,
            environment={"container": os.environ.get("WANDER_CONTAINER_DIGEST", "development")},
        ),
    )
    review = ReviewActivities(
        PipelineRepository(sessions),
        workspace_root=Path(os.environ.get("WANDER_WORKSPACE_ROOT", "/var/lib/wander/runs")),
        store=LocalCAS(Path(os.environ.get("WANDER_CAS_ROOT", "/var/lib/wander/cas"))),
        outbox_archive=archive,
        resolver=resolver,
        attempt_ledger=DatabaseAttemptLedger(
            sessions,
            code_revision=code_revision,
            environment={"container": os.environ.get("WANDER_CONTAINER_DIGEST", "development")},
        ),
        harness=HarnessAgent(HarnessPolicy.from_environment()),
    )
    lifecycle = RunLifecycleActivities(
        PipelineRepository(sessions),
        graph_version=GRAPH_VERSION,
        code_revision=code_revision,
        container_digest=os.environ.get("WANDER_CONTAINER_DIGEST"),
        repository_root=Path(os.environ.get("WANDER_REPOSITORY", "/app")),
        workspace_root=Path(os.environ.get("WANDER_WORKSPACE_ROOT", "/var/lib/wander/runs")),
        store=LocalCAS(Path(os.environ.get("WANDER_CAS_ROOT", "/var/lib/wander/cas"))),
        archive=archive,
        attempt_ledger=DatabaseAttemptLedger(
            sessions,
            code_revision=code_revision,
            environment={"container": os.environ.get("WANDER_CONTAINER_DIGEST", "development")},
        ),
    )
    flushed = flush_pending_outboxes(
        Path(os.environ.get("WANDER_WORKSPACE_ROOT", "/var/lib/wander/runs")),
        LocalCAS(Path(os.environ.get("WANDER_CAS_ROOT", "/var/lib/wander/cas"))),
        archive,
    )
    if flushed:
        print(f"recovered {flushed} unflushed attempt manifests", flush=True)
    temporal = await Client.connect(required("WANDER_TEMPORAL_ADDRESS"))
    async with AsyncExitStack() as stack:
        activity_executor = ThreadPoolExecutor(
            max_workers=int(os.environ.get("WANDER_ACTIVITY_THREADS", "8"))
        )
        stack.callback(activity_executor.shutdown)
        heartbeat_task = asyncio.create_task(
            heartbeat(Path(os.environ.get("WANDER_HEARTBEAT", "/tmp/wander-worker")))
        )
        stack.callback(heartbeat_task.cancel)
        await stack.enter_async_context(
            Worker(
                temporal,
                task_queue=os.environ.get("WANDER_CONTROL_QUEUE", "pipeline-control"),
                workflows=[GenerationWorkflow],
            )
        )
        for queue in ACTIVITY_QUEUES:
            await stack.enter_async_context(
                Worker(
                    temporal,
                    task_queue=queue,
                    activities=[
                        runner.run_stage,
                        review.review_attempt,
                        lifecycle.register_run,
                        lifecycle.finish_run,
                        lifecycle.load_run_state,
                        lifecycle.save_run_state,
                        lifecycle.record_approval,
                        lifecycle.publish_run,
                    ],
                    activity_executor=activity_executor,
                )
            )
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
