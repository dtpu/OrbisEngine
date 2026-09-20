"""Representative deterministic run and artifact fixtures."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.contracts import (
    Artifact,
    AttemptStatus,
    BudgetPolicy,
    CostRecord,
    Run,
    RunStatus,
    StageAttempt,
)

FIXTURE_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXTURE_CODE_REVISION = "0123456789abcdef0123456789abcdef01234567"
FIXTURE_ENVIRONMENT_SHA256 = hashlib.sha256(b"wander-fixture-environment-v1").hexdigest()


@dataclass(frozen=True)
class ArtifactFixture:
    artifact: Artifact
    content: bytes
    filename: str

    def write(self, directory: Path) -> Path:
        destination = Path(directory) / self.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != self.content:
            raise FileExistsError(f"fixture destination has different bytes: {destination}")
        destination.write_bytes(self.content)
        return destination


@dataclass(frozen=True)
class RepresentativeRunFixture:
    run: Run
    artifacts: tuple[ArtifactFixture, ...]
    attempt: StageAttempt

    def artifact_for_role(self, role: str) -> ArtifactFixture:
        matches = [fixture for fixture in self.artifacts if fixture.artifact.role == role]
        if len(matches) != 1:
            raise KeyError(f"expected one fixture artifact for role {role!r}")
        return matches[0]

    def write_artifacts(self, directory: Path) -> dict[str, Path]:
        return {fixture.artifact.id: fixture.write(directory) for fixture in self.artifacts}


def artifact_fixture(
    role: str,
    content: bytes | str,
    *,
    run_id: str = "run-fixture-001",
    media_type: str = "application/json",
    filename: str | None = None,
    producer_attempt_id: str | None = None,
    contract: str | None = None,
) -> ArtifactFixture:
    value = content.encode() if isinstance(content, str) else content
    digest = hashlib.sha256(value).hexdigest()
    name = filename or f"{role}.json"
    identity = hashlib.sha256(f"{run_id}\0{role}\0{digest}".encode()).hexdigest()[:24]
    artifact = Artifact(
        id=f"artifact:{identity}",
        run_id=run_id,
        role=role,
        sha256=digest,
        size=len(value),
        media_type=media_type,
        storage_key=f"fixtures/blobs/{digest[:2]}/{digest}",
        producer_attempt_id=producer_attempt_id,
        contract=contract,
        metadata={"fixture": True, "relative_path": name},
        created_at=FIXTURE_TIME,
    )
    return ArtifactFixture(artifact=artifact, content=value, filename=name)


def representative_run_fixture() -> RepresentativeRunFixture:
    """A small successful generation run with paid-stage evidence and binary outputs."""

    run_id = "run-fixture-001"
    attempt_id = "attempt:marble-video-submit:1"
    poll_attempt_id = "attempt:marble-video-poll:1"
    source = artifact_fixture(
        "source_video",
        b"deterministic fake mp4 bytes\n",
        run_id=run_id,
        media_type="video/mp4",
        filename="source.mp4",
    )
    provider_operation = artifact_fixture(
        "provider_operation",
        (
            '{"operation_id":"marble:fixture-operation","schema":'
            '"wander.fake-provider-operation/1","status":"completed"}\n'
        ),
        run_id=run_id,
        filename="marble-generation.json",
        producer_attempt_id=attempt_id,
    )
    world = artifact_fixture(
        "world_splat",
        b"deterministic fake SPZ world\n",
        run_id=run_id,
        media_type="application/octet-stream",
        filename="world.spz",
        producer_attempt_id=poll_attempt_id,
    )
    world_receipt = artifact_fixture(
        "world_receipt",
        '{"operation_id":"marble:fixture-operation","schema":"wander.fake-marble-world/1"}\n',
        run_id=run_id,
        filename="world-receipt.json",
        producer_attempt_id=poll_attempt_id,
    )
    object_shape = artifact_fixture(
        "object_shape",
        b"ply\nformat ascii 1.0\nend_header\n",
        run_id=run_id,
        media_type="application/octet-stream",
        filename="object.ply",
        producer_attempt_id="attempt:object-shape:1",
    )
    finetuned_world = artifact_fixture(
        "finetuned_world",
        b"deterministic fake fine-tuned SPZ world\n",
        run_id=run_id,
        media_type="application/octet-stream",
        filename="finetuned.spz",
        producer_attempt_id="attempt:finetune:1",
    )
    artifacts = (
        source,
        provider_operation,
        world,
        world_receipt,
        object_shape,
        finetuned_world,
    )
    run = Run(
        id=run_id,
        graph_version="wander.generation-graph/1",
        code_revision=FIXTURE_CODE_REVISION,
        container_digest="sha256:" + hashlib.sha256(b"fixture-container").hexdigest(),
        source_sha256=source.artifact.sha256,
        source_artifact_id=source.artifact.id,
        configuration={
            "marble": "video",
            "objects": True,
            "finetune": True,
            "fixture": True,
        },
        budget=BudgetPolicy(
            maximum_cost=10.0,
            maximum_attempts_by_stage={"marble_video_submit": 1},
            maximum_provider_operations={"marble": 1},
        ),
        status=RunStatus.SUCCEEDED,
        created_by="fixture",
        created_at=FIXTURE_TIME,
        updated_at=FIXTURE_TIME,
    )
    attempt = StageAttempt(
        id=attempt_id,
        run_id=run_id,
        node_id="marble_video_submit",
        number=1,
        status=AttemptStatus.SUCCEEDED,
        parameters={"mode": "video", "seed": 7},
        command=("fake-marble", "submit", "--mode", "video"),
        code_revision=FIXTURE_CODE_REVISION,
        environment_fingerprint=FIXTURE_ENVIRONMENT_SHA256,
        input_artifact_ids=(source.artifact.id,),
        output_artifact_ids=(provider_operation.artifact.id,),
        provider_operation_id="marble:fixture-operation",
        costs=(
            CostRecord(
                provider="marble",
                amount=2.5,
                units={"generations": 1.0},
                estimated=False,
            ),
        ),
        started_at=FIXTURE_TIME,
        finished_at=FIXTURE_TIME,
    )
    return RepresentativeRunFixture(run=run, artifacts=artifacts, attempt=attempt)
