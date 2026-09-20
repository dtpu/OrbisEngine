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


class ExpandedNodeTests(unittest.TestCase):
    """Nodes a run creates by expanding a stage must survive losing the scheduler."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-expand-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        engine = create_engine(f"sqlite:///{self.root / 'meta.db'}")
        Base.metadata.create_all(engine)
        self.repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
        self.options = GraphOptions(all_people=True, people=4)
        self.graph = instantiate_graph(self.options)
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

    def expand(self):
        """Expand one person, the way a finished tracks stage does."""
        from orchestrator.graph import BranchArtifact

        # Expansion requires tracks to have selected a successful attempt first.
        self.graph.select_attempt(
            "tracks",
            f"{RUN_ID}:tracks:1",
            {
                contract.role: (f"artifact:tracks:{contract.role}",)
                for contract in self.graph.nodes["tracks"].definition.outputs.values()
            },
        )
        self.graph.expand_people((BranchArtifact(key="00", artifact_id="artifact:track00"),))
        return [n for n in self.graph.nodes if n not in instantiate_graph(self.options).nodes]

    def test_expanded_nodes_are_written_and_rebuilt(self):
        created = self.expand()
        self.assertTrue(created, "precondition: expansion created nodes")
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
            ),
        )
        revived = instantiate_graph(self.options)
        for node_id in created:
            self.assertNotIn(node_id, revived.nodes, "precondition: base graph lacks them")
        restore_state(revived, self.repository.load_run_state(RUN_ID), {}, {})
        for node_id in created:
            self.assertIn(node_id, revived.nodes)
            rebuilt, original = revived.nodes[node_id], self.graph.nodes[node_id]
            self.assertEqual(rebuilt.stage_type, original.stage_type)
            self.assertEqual(rebuilt.dependencies, original.dependencies)
            self.assertEqual(rebuilt.branch_key, original.branch_key)
            self.assertEqual(rebuilt.definition.id, original.definition.id)

    def test_a_rebuilt_node_keeps_its_progress(self):
        created = self.expand()
        target = created[0]
        self.graph.set_status(target, NodeStatus.FAILED, "modal ran out of memory")
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph,
                paused=False,
                canceled=False,
                agent_retries={target: 2},
                retry_parameters={target: {"hypothesis": "smaller batch"}},
            ),
        )
        revived = instantiate_graph(self.options)
        retries, parameters = {}, {}
        restore_state(revived, self.repository.load_run_state(RUN_ID), retries, parameters)
        self.assertEqual(revived.nodes[target].status, NodeStatus.FAILED)
        self.assertEqual(revived.nodes[target].blocked_reason, "modal ran out of memory")
        self.assertEqual(retries[target], 2)
        self.assertEqual(parameters[target]["hypothesis"], "smaller batch")

    def test_saving_twice_does_not_duplicate_a_node(self):
        created = self.expand()
        state = snapshot_state(
            self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
        )
        self.repository.save_run_state(RUN_ID, state)
        self.repository.save_run_state(RUN_ID, state)
        stored = self.repository.load_run_state(RUN_ID)["nodes"]
        self.assertEqual(len(stored), len(self.graph.nodes))
        self.assertTrue(set(created) <= set(stored))


if __name__ == "__main__":
    unittest.main()
