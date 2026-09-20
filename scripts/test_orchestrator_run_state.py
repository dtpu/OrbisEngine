"""A run's progress is rows, not a replay log, so it survives losing its scheduler."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.contracts import NodeStatus, Run
from orchestrator.database import Base
from orchestrator.graph import instantiate_graph
from orchestrator.repository import PipelineRepository
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import restore_state, snapshot_state

RUN_ID = "run-state"
SHA = "c" * 64


class RunStateRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-state-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        engine = create_engine(f"sqlite:///{self.root / 'meta.db'}")
        Base.metadata.create_all(engine)
        self.repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
        self.graph = instantiate_graph(GraphOptions())
        self.repository.create_run(
            Run(
                id=RUN_ID,
                graph_version="wander.generation-graph/1",
                code_revision="revision-1",
                source_sha256=SHA,
                source_artifact_id=f"artifact:{RUN_ID}:source",
                created_by="tests",
            ),
            self.graph,
        )

    def advance(self):
        """Put the graph into a state a real run reaches part-way through."""
        self.graph.select_attempt(
            "admission",
            f"{RUN_ID}:1:1",
            {"source_metadata": ("artifact:meta",), "shots": ("artifact:shots",)},
        )
        self.graph.set_status("pi3x", NodeStatus.FAILED, "SVD did not converge")
        self.graph.set_status("clean", NodeStatus.WAITING_HUMAN, "agent asks: is the mat right?")
        self.graph.nodes["world_prompt"].status = NodeStatus.RUNNING
        return (
            {"clean": 2},
            {"clean": {"dilate": 40, "hypothesis": "mask too tight"}},
        )

    def test_a_new_scheduler_resumes_from_rows_alone(self):
        agent_retries, retry_parameters = self.advance()
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph,
                paused=True,
                canceled=False,
                agent_retries=agent_retries,
                retry_parameters=retry_parameters,
            ),
        )

        # A brand new scheduler: fresh graph, no memory of anything.
        revived = instantiate_graph(GraphOptions())
        restored_retries: dict = {}
        restored_parameters: dict = {}
        stored = self.repository.load_run_state(RUN_ID)
        paused, canceled = restore_state(revived, stored, restored_retries, restored_parameters)

        self.assertTrue(paused)
        self.assertFalse(canceled)
        self.assertEqual(revived.nodes["admission"].status, NodeStatus.SUCCEEDED)
        self.assertEqual(revived.nodes["admission"].selected_attempt_id, f"{RUN_ID}:1:1")
        self.assertEqual(
            revived.nodes["admission"].selected_artifacts["shots"], ("artifact:shots",)
        )
        self.assertEqual(revived.nodes["pi3x"].status, NodeStatus.FAILED)
        self.assertEqual(revived.nodes["pi3x"].blocked_reason, "SVD did not converge")
        self.assertEqual(revived.nodes["clean"].status, NodeStatus.WAITING_HUMAN)
        self.assertEqual(restored_retries["clean"], 2)
        self.assertEqual(restored_parameters["clean"]["dilate"], 40)

    def test_work_that_was_in_flight_is_requeued_not_assumed_done(self):
        """A stage running when the scheduler died must be redone, never silently skipped."""
        self.advance()
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
            ),
        )
        revived = instantiate_graph(GraphOptions())
        restore_state(revived, self.repository.load_run_state(RUN_ID), {}, {})
        self.assertEqual(revived.nodes["world_prompt"].status, NodeStatus.QUEUED)

    def test_a_node_under_agent_review_is_requeued_too(self):
        self.graph.nodes["clean"].status = NodeStatus.WAITING_AGENT
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
            ),
        )
        revived = instantiate_graph(GraphOptions())
        restore_state(revived, self.repository.load_run_state(RUN_ID), {}, {})
        self.assertEqual(revived.nodes["clean"].status, NodeStatus.QUEUED)

    def test_cancel_intent_outlives_the_scheduler(self):
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph, paused=False, canceled=True, agent_retries={}, retry_parameters={}
            ),
        )
        _, canceled = restore_state(
            instantiate_graph(GraphOptions()), self.repository.load_run_state(RUN_ID), {}, {}
        )
        self.assertTrue(canceled)

    def test_an_unknown_run_has_no_state(self):
        self.assertIsNone(self.repository.load_run_state("no-such-run"))

    def test_saving_state_for_an_unknown_run_is_refused(self):
        with self.assertRaises(KeyError):
            self.repository.save_run_state("no-such-run", {"nodes": {}})


if __name__ == "__main__":
    unittest.main()
