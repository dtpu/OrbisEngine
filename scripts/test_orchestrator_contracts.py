"""Contract tests for durable run, graph, attempt, artifact, event, and QA records."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contracts import (
    Artifact,
    ArtifactBinding,
    ArtifactContract,
    AttemptStatus,
    Event,
    EventType,
    QualityDecision,
    QualityVerdict,
    RetryPolicy,
    Run,
    StageAttempt,
    StageDefinition,
    StageKind,
)

HASH = "a" * 64
NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)


class ContractTests(unittest.TestCase):
    def test_run_and_artifact_round_trip_to_versioned_json(self):
        run = Run(
            id="run-1",
            graph_version="single-video-v1",
            code_revision="1234567",
            source_sha256=HASH,
            source_artifact_id="source-1",
            created_by="test",
        )
        artifact = Artifact(
            id="artifact-1",
            run_id=run.id,
            role="source_video",
            sha256=HASH,
            size=42,
            media_type="video/mp4",
            storage_key=f"blobs/{HASH}",
        )

        self.assertEqual(Run.model_validate_json(run.model_dump_json()), run)
        self.assertEqual(Artifact.model_validate_json(artifact.model_dump_json()), artifact)
        self.assertEqual(run.schema_version, "wander.run/1")
        self.assertEqual(artifact.schema_version, "wander.artifact/1")

    def test_stage_requires_typed_executor_and_artifact_binding(self):
        stage = StageDefinition(
            id="motion",
            title="Animate person",
            kind=StageKind.COMPUTE,
            executor="lhm_motion",
            inputs={
                "cameras": ArtifactBinding(source="stage_output", stage_id="pi3x", role="cameras")
            },
            outputs={
                "sequence": ArtifactContract(
                    role="person_sequence",
                    media_type="application/json",
                    contract="wander.person-sequence/1",
                )
            },
        )

        self.assertEqual(stage.inputs["cameras"].stage_id, "pi3x")
        with self.assertRaises(ValidationError):
            ArtifactBinding(source="stage_output", role="cameras")
        with self.assertRaises(ValidationError):
            StageDefinition(id="bad", title="Bad", kind=StageKind.COMPUTE)
        with self.assertRaises(ValidationError):
            StageDefinition(
                id="bad-human",
                title="Bad human",
                kind=StageKind.HUMAN,
                executor="must-not-run",
            )

    def test_paid_stages_cannot_have_automatic_resubmission(self):
        with self.assertRaises(ValidationError):
            RetryPolicy(paid=True, automatic_attempts=2)

    def test_attempt_terminal_state_requires_timing_evidence(self):
        common = dict(
            id="attempt-1",
            run_id="run-1",
            node_id="pi3x",
            number=1,
            code_revision="1234567",
            environment_fingerprint=HASH,
        )
        with self.assertRaises(ValidationError):
            StageAttempt(**common, status=AttemptStatus.SUCCEEDED)
        attempt = StageAttempt(
            **common,
            status=AttemptStatus.SUCCEEDED,
            started_at=NOW,
            finished_at=NOW,
        )
        self.assertEqual(attempt.status, "succeeded")

    def test_retry_quality_decision_requires_hypothesis(self):
        common = dict(
            id="decision-1",
            run_id="run-1",
            node_id="clean",
            attempt_id="attempt-1",
            verdict=QualityVerdict.RETRY,
            rationale="Residual person pixels remain",
            decided_by="agent:codex",
        )
        with self.assertRaises(ValidationError):
            QualityDecision(**common)
        decision = QualityDecision(
            **common,
            hypothesis="Increase mask dilation to cover the residual silhouette",
            proposed_parameters={"dilate": 70},
        )
        self.assertEqual(decision.proposed_parameters["dilate"], 70)

    def test_events_are_ordered_and_reject_unknown_fields(self):
        event = Event(
            id="event-1",
            run_id="run-1",
            sequence=1,
            type=EventType.NODE,
            subject_id="clean",
            payload={"status": "running"},
        )
        self.assertEqual(event.sequence, 1)
        with self.assertRaises(ValidationError):
            Event(
                id="event-2",
                run_id="run-1",
                sequence=0,
                type=EventType.RUN,
                payload={},
            )
        with self.assertRaises(ValidationError):
            Run(
                id="run-2",
                graph_version="v1",
                code_revision="1234567",
                source_sha256=HASH,
                source_artifact_id="source-2",
                created_by="test",
                surprise=True,
            )


if __name__ == "__main__":
    unittest.main()
