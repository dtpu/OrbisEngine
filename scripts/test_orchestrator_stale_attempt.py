"""A superseded attempt must not overwrite the node state of the attempt that replaced it.

A scheduler that has been replaced can still have work in flight. When that work ends — often
because it was cancelled — it reports through the same rows as the live attempt. Letting it
write would mark a healthy stage failed on the strength of a dead run's cancellation.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.artifacts import AttemptManifest
from orchestrator.contracts import Run
from orchestrator.database import Base, NodeRecord
from orchestrator.graph import instantiate_graph
from orchestrator.persistence import DatabaseAttemptLedger
from orchestrator.repository import PipelineRepository
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import StageActivityInput, StageActivityResult

RUN_ID = "run-stale"


class StaleAttemptTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-stale-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        engine = create_engine(f"sqlite:///{self.root / 'meta.db'}")
        Base.metadata.create_all(engine)
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        graph = instantiate_graph(GraphOptions(all_people=True, people=4))
        PipelineRepository(self.sessions).create_run(
            Run(
                id=RUN_ID,
                graph_version="wander.generation-graph/1",
                code_revision="revision-1",
                source_sha256="d" * 64,
                source_artifact_id=f"artifact:{RUN_ID}:source",
                created_by="tests",
            ),
            graph,
        )
        self.definition = graph.nodes["tracks"].definition.model_dump(mode="json")
        self.ledger = DatabaseAttemptLedger(
            self.sessions, code_revision="revision-1", environment={"container": "test"}
        )

    def request(self):
        return StageActivityInput(
            run_id=RUN_ID,
            node_id="tracks",
            stage_type="tracks",
            definition=self.definition,
            selected_inputs={},
        )

    def manifest(self, attempt_id):
        return AttemptManifest(
            run_id=RUN_ID, node_id="tracks", attempt_id=attempt_id, status="failed", files=()
        )

    def finish(self, attempt_id, status, error=None):
        self.ledger.finished(
            self.request(),
            StageActivityResult(
                node_id="tracks", attempt_id=attempt_id, status=status, error=error
            ),
            self.manifest(attempt_id),
        )

    def node_status(self):
        with self.sessions() as session:
            return session.get(NodeRecord, {"run_id": RUN_ID, "node_id": "tracks"}).status

    def test_the_newest_attempt_sets_the_node(self):
        self.ledger.started(self.request(), "old:1:1")
        self.ledger.started(self.request(), "new:1:1")
        self.finish("new:1:1", "succeeded")
        self.assertEqual(self.node_status(), "succeeded")

    def test_a_superseded_attempt_cannot_overwrite_it(self):
        """The real case: a killed orphan reported 'command exited -15' over a live attempt."""
        self.ledger.started(self.request(), "old:1:1")
        self.ledger.started(self.request(), "new:1:1")
        self.finish("new:1:1", "succeeded")
        self.finish("old:1:1", "blocked", error="command exited -15")
        self.assertEqual(self.node_status(), "succeeded")

    def test_the_superseded_attempt_still_records_its_own_outcome(self):
        self.ledger.started(self.request(), "old:1:1")
        self.ledger.started(self.request(), "new:1:1")
        self.finish("old:1:1", "blocked", error="command exited -15")
        with self.sessions() as session:
            from orchestrator.database import AttemptRecord

            old = session.get(AttemptRecord, "old:1:1")
            self.assertEqual(old.status, "blocked")
            self.assertEqual(old.error, "command exited -15")

    def test_a_lone_attempt_still_sets_the_node(self):
        self.ledger.started(self.request(), "only:1:1")
        self.finish("only:1:1", "failed", error="command exited 1")
        self.assertEqual(self.node_status(), "failed")


if __name__ == "__main__":
    unittest.main()
