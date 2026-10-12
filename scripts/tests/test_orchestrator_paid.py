"""Provider-agnostic paid-operation accounting tests."""

import sys
import tempfile
import threading
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
        if "attempt_id" not in changes and values["idempotency_key"] != "clean:1":
            values["attempt_id"] = f"attempt:{values['idempotency_key']}"
        if self.session.get(AttemptRecord, values["attempt_id"]) is None:
            self.session.add(
                AttemptRecord(
                    id=values["attempt_id"],
                    run_id="run-1",
                    node_id="clean",
                    number=self.session.query(AttemptRecord).count() + 1,
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
        return self.policy.begin(self.session, **values)

    def test_retries_require_new_hypothesis_and_changed_identity(self):
        first = self.begin()
        self.policy.finish(self.session, first.id, status="completed", estimated_cost=1.5)
        with self.assertRaisesRegex(ValueError, "explicit new hypothesis"):
            self.begin(idempotency_key="clean:2", parameters={"dilate": 30})
        with self.assertRaisesRegex(ValueError, "parameters or code"):
            self.begin(idempotency_key="clean:2", hypothesis="Cover silhouette")
        second = self.begin(
            idempotency_key="clean:2", hypothesis="Cover silhouette", parameters={"dilate": 30}
        )
        self.assertEqual(second.number, 2)
        self.policy.finish(self.session, second.id, status="completed")
        with self.assertRaisesRegex(ValueError, "new hypothesis"):
            self.begin(
                idempotency_key="clean:3",
                hypothesis="  COVER   SILHOUETTE ",
                parameters={"dilate": 40},
            )

    def test_pending_and_unknown_require_reconciliation(self):
        first = self.begin()
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(idempotency_key="clean:2", operation="clean_first")
        self.policy.finish(self.session, first.id, status="unknown")
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin(idempotency_key="clean:2", operation="clean_first")
        self.policy.reconcile(
            self.session, first.id, status="failed", provider_operation_id="operation-1"
        )
        second = self.begin(idempotency_key="clean:2", operation="clean_first")
        self.policy.finish(self.session, second.id, status="unknown")
        self.policy.reconcile(
            self.session, second.id, status="failed", provider_operation_id="operation-2"
        )
        with self.assertRaisesRegex(ValueError, "only pending/unknown"):
            self.policy.reconcile(self.session, second.id, status="completed")

    def test_distinct_operations_share_three_execution_allowance(self):
        for number, operation in enumerate(("clean", "clean_first", "clean_multi"), start=1):
            claim = self.begin(idempotency_key=f"clean:{number}", operation=operation)
            self.assertEqual(claim.number, number)
            self.policy.finish(self.session, claim.id, status="completed")
        with self.assertRaisesRegex(ValueError, "execution allowance exhausted"):
            self.begin(
                idempotency_key="clean:4", hypothesis="Widen mask", parameters={"dilate": 40}
            )

    def test_explicit_stage_cap_can_stop_execution_sooner(self):
        self.session.get(RunRecord, "run-1").budget = {
            "maximum_attempts_by_stage": {"inpainting": 1}
        }
        claim = self.begin()
        self.policy.finish(self.session, claim.id, status="completed")
        with self.assertRaisesRegex(ValueError, "execution allowance exhausted"):
            self.begin(idempotency_key="clean:2", operation="clean_first")

    def test_changed_worker_code_allows_justified_retry(self):
        first = self.begin()
        self.policy.finish(self.session, first.id, status="failed")
        claim = self.begin(
            idempotency_key="clean:2",
            hypothesis="Repair mask association",
            code_version={"worker": "two"},
        )
        self.assertEqual(claim.number, 2)

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

    def test_cost_budget_reserves_all_providers(self):
        self.session.get(RunRecord, "run-1").budget = {"maximum_cost": 2.0}
        claim = self.begin(estimated_cost=1.5)
        self.policy.finish(self.session, claim.id, status="completed")
        with self.assertRaisesRegex(ValueError, "cost budget"):
            self.begin(
                provider="openai",
                logical_stage="review",
                operation="review",
                idempotency_key="review:1",
                estimated_cost=0.6,
            )
        self.begin(
            provider="openai",
            logical_stage="review",
            operation="review",
            idempotency_key="review:1",
            estimated_cost=0.5,
        )

    def test_cost_estimates_must_be_finite_and_nonnegative(self):
        for cost in (-1, float("nan"), float("inf")):
            with (
                self.subTest(cost=cost),
                self.assertRaisesRegex(ValueError, "finite and nonnegative"),
            ):
                self.begin(estimated_cost=cost)
        claim = self.begin()
        for cost in (-1, float("nan"), float("inf")):
            with (
                self.subTest(cost=cost),
                self.assertRaisesRegex(ValueError, "finite and nonnegative"),
            ):
                self.policy.finish(self.session, claim.id, status="completed", estimated_cost=cost)

    def test_existing_unpriced_operation_blocks_capped_budget(self):
        claim = self.begin()
        self.policy.finish(self.session, claim.id, status="completed")
        self.session.get(RunRecord, "run-1").budget = {"maximum_cost": 2.0}
        with self.assertRaisesRegex(ValueError, "existing paid operations need estimated costs"):
            self.begin(idempotency_key="clean:2", operation="clean_first", estimated_cost=0.5)


class RunScopedRetryRuleTests(unittest.TestCase):
    """A run/candidate rename preserves the source's execution history."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = create_engine(f"sqlite:///{Path(temporary.name) / 'paid.db'}")
        self.addCleanup(engine.dispose)
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
                session.add(
                    NodeRecord(
                        run_id=run_id,
                        node_id="clean",
                        stage_definition={},
                        dependencies=[],
                        status="running",
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )

    def ensure_attempt(self, session, run_id, attempt_id):
        if session.get(AttemptRecord, attempt_id) is not None:
            return
        records = session.scalars(select(AttemptRecord).where(AttemptRecord.run_id == run_id)).all()
        session.add(
            AttemptRecord(
                id=attempt_id,
                run_id=run_id,
                node_id="clean",
                number=len(records) + 1,
                schema_version="wander.stage-attempt/1",
                status="running",
                parameters={},
                command=[],
                code_revision="rev",
                environment_fingerprint=ENVIRONMENT,
                input_artifact_ids=[],
                output_artifact_ids=[],
                costs=[],
            )
        )
        session.flush()

    def begin(self, run_id, attempt_id, key, *, hypothesis=None, parameters=None):
        with self.sessions.begin() as session:
            self.ensure_attempt(session, run_id, attempt_id)
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

    def test_a_new_run_cannot_bypass_unresolved_execution(self):
        first = self.begin("run-a", "run-a:1:1", "key-a")
        self.assertEqual(first.number, 1)
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin("run-b", "run-b:1:1", "key-b")
        self.resolve_all("unknown")
        with self.assertRaisesRegex(ValueError, "evidence reconciliation"):
            self.begin("run-b", "run-b:1:1", "key-b")

    def test_attempts_accumulate_across_runs(self):
        self.begin("run-a", "run-a:1:1", "key-a")
        self.resolve_all()
        with self.assertRaisesRegex(ValueError, "explicit new hypothesis"):
            self.begin("run-b", "run-b:1:1", "key-b", parameters={"dilate": 30})
        second = self.begin(
            "run-b", "run-b:1:1", "key-b", hypothesis="Widen mask", parameters={"dilate": 30}
        )
        self.assertEqual(second.number, 2)
        self.resolve_all()
        third = self.begin(
            "run-a", "run-a:1:2", "key-a2", hypothesis="Adjust overlap", parameters={"overlap": 8}
        )
        self.assertEqual(third.number, 3)
        self.resolve_all()
        with self.assertRaisesRegex(ValueError, "execution allowance exhausted"):
            self.begin(
                "run-b", "run-b:1:2", "key-b2", hypothesis="Change model", parameters={"model": 2}
            )

    def test_concurrent_runs_allow_one_pending_claim(self):
        with self.sessions.begin() as session:
            self.ensure_attempt(session, "run-a", "attempt-a")
            self.ensure_attempt(session, "run-b", "attempt-b")
        barrier = threading.Barrier(2)
        claims, errors = [], []

        def launch(run_id, attempt_id):
            barrier.wait(timeout=5)
            try:
                claims.append(self.begin(run_id, attempt_id, f"key:{run_id}"))
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=launch, args=("run-a", "attempt-a")),
            threading.Thread(target=launch, args=("run-b", "attempt-b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(claims), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertIn("evidence reconciliation", str(errors[0]))
        with self.sessions() as session:
            records = session.scalars(select(ProviderClaimRecord)).all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].status, "pending")


if __name__ == "__main__":
    unittest.main()
