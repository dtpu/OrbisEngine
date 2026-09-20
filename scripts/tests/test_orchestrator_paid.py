"""Provider-agnostic paid-operation accounting tests."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from orchestrator.database import (
    AttemptRecord,
    Base,
    NodeRecord,
    ProviderClaimRecord,
    RunRecord,
)
from orchestrator.paid import PaidOperationPolicy

NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)
SOURCE = "a" * 64
ENVIRONMENT = "b" * 64


class PaidPolicyTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.addCleanup(self.engine.dispose)
        self.addCleanup(self.session.close)
        self.session.add(
            RunRecord(
                id="run-1",
                schema_version="wander.run/1",
                graph_schema="wander.graph/1",
                graph_version="v1",
                code_revision="1234567",
                source_sha256=SOURCE,
                source_artifact_id="source",
                configuration={},
                budget={},
                status="running",
                created_by="test",
                created_at=NOW,
                updated_at=NOW,
                next_event_sequence=1,
            )
        )
        self.session.add(
            NodeRecord(
                run_id="run-1",
                node_id="clean",
                stage_definition={},
                dependencies=[],
                status="running",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        self.session.add(
            AttemptRecord(
                id="attempt-1",
                run_id="run-1",
                node_id="clean",
                number=1,
                schema_version="wander.stage-attempt/1",
                status="running",
                parameters={},
                command=[],
                code_revision="1234567",
                environment_fingerprint=ENVIRONMENT,
                input_artifact_ids=[],
                output_artifact_ids=[],
                costs=[],
            )
        )
        self.session.flush()
        self.policy = PaidOperationPolicy(maximum_attempts=3)

    def begin(self, **changes):
        values = {
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "provider": "modal",
            "idempotency_key": "clean:1",
            "source_sha256": SOURCE,
            "logical_stage": "inpainting",
            "operation": "clean",
            "parameters": {"dilate": 20},
            "code_version": {"worker": "one"},
            "hypothesis": None,
        }
        values.update(changes)
        return self.policy.begin(self.session, **values)

    def test_repeating_a_paid_stage_is_recorded_and_never_refused(self):
        """Paid stages are not special-cased: the ledger records, it does not gate.

        Earlier revisions refused a repeat without a new hypothesis, refused unchanged
        parameters, and refused any run while a previous claim was unresolved. Those gates were
        removed deliberately: whether a step costs money is not a reason to change control flow.
        """
        first = self.begin()
        self.policy.finish(self.session, first.id, status="completed", estimated_cost=1.5)
        # Same parameters, no hypothesis: allowed.
        second = self.begin(idempotency_key="clean:2")
        self.assertEqual(second.number, 2)
        self.policy.finish(self.session, second.id, status="unknown")
        # A previous claim left unresolved does not block the next operation.
        third = self.begin(idempotency_key="clean:3", parameters={"dilate": 40})
        self.assertEqual(third.number, 3)
        # Reconciliation still exists for the record, and still refuses a resolved claim.
        self.policy.reconcile(
            self.session, second.id, status="failed", provider_operation_id="operation-2"
        )
        with self.assertRaisesRegex(ValueError, "only pending/unknown"):
            self.policy.reconcile(self.session, second.id, status="completed")

    def test_attempt_numbers_count_within_the_run(self):
        self.begin()
        self.assertEqual(self.begin(idempotency_key="clean:2").number, 2)

    def test_provider_operation_id_is_immutable(self):
        claim = self.begin()
        self.policy.record_operation_id(self.session, claim.id, "operation-1")
        self.policy.record_operation_id(self.session, claim.id, "operation-1")
        with self.assertRaisesRegex(ValueError, "cannot be replaced"):
            self.policy.record_operation_id(self.session, claim.id, "operation-other")

    def test_run_budget_reserves_estimates_and_caps_provider_operations(self):
        run = self.session.get(RunRecord, "run-1")
        run.budget = {
            "maximum_cost": 2.0,
            "maximum_attempts_by_stage": {},
            "maximum_provider_operations": {"modal": 1},
        }
        self.session.flush()
        with self.assertRaisesRegex(ValueError, "estimated cost"):
            self.begin()
        with self.assertRaisesRegex(ValueError, "cost budget"):
            self.begin(estimated_cost=3.0)
        claim = self.begin(estimated_cost=1.0)
        self.policy.finish(self.session, claim.id, status="completed")
        with self.assertRaisesRegex(ValueError, "provider operation allowance"):
            self.begin(idempotency_key="clean:2", parameters={"dilate": 30}, estimated_cost=0.5)


class RunScopedRetryRuleTests(unittest.TestCase):
    """A fresh run on the same clip is never treated as a retry of an earlier run."""

    def setUp(self):
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.policy = PaidOperationPolicy()
        with self.sessions.begin() as session:
            for run_id in ("run-a", "run-b"):
                session.add(
                    RunRecord(
                        id=run_id,
                        schema_version="wander.run/1",
                        graph_schema="wander.graph/1",
                        graph_version="wander.generation-graph/1",
                        code_revision="rev",
                        source_sha256=SOURCE,
                        source_artifact_id=f"artifact:{run_id}:source",
                        configuration={},
                        budget={},
                        status="queued",
                        created_by="tests",
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        next_event_sequence=1,
                    )
                )

    def begin(self, run_id, attempt_id, key, *, hypothesis=None, parameters=None):
        with self.sessions.begin() as session:
            return self.policy.begin(
                session,
                run_id=run_id,
                attempt_id=attempt_id,
                provider="modal",
                idempotency_key=key,
                source_sha256=SOURCE,
                logical_stage="clean",
                operation="clean",
                parameters=parameters or {},
                code_version={"repository": "rev"},
                hypothesis=hypothesis,
            )

    def resolve_all(self, status="completed"):
        with self.sessions.begin() as session:
            for record in session.scalars(select(ProviderClaimRecord)).all():
                record.status = status

    def test_a_new_run_on_the_same_clip_starts_at_attempt_one(self):
        first = self.begin("run-a", "run-a:1:1", "key-a")
        self.assertEqual(first.number, 1)
        # Same clip and stage, different run, identical parameters, no hypothesis, and the
        # earlier claim still pending: none of that makes this a retry.
        fresh = self.begin("run-b", "run-b:1:1", "key-b")
        self.assertEqual(fresh.number, 1)

    def test_attempts_accumulate_per_run(self):
        self.begin("run-a", "run-a:1:1", "key-a")
        self.assertEqual(self.begin("run-a", "run-a:1:2", "key-a2").number, 2)
        self.assertEqual(self.begin("run-b", "run-b:1:1", "key-b").number, 1)


if __name__ == "__main__":
    unittest.main()
