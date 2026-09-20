"""A run's progress is rows, not a replay log, so it survives losing its scheduler."""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.contracts import NodeStatus, Run, StageDefinition
from orchestrator.database import Base
from orchestrator.graph import instantiate_graph, rebuild_expanded_node
from orchestrator.repository import PipelineRepository
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import restore_state, snapshot_state

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resume_run import requeue

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

    def test_a_resumed_run_stops_saying_it_finished(self):
        """The dashboard read "succeeded" while the pipeline was working on the run again."""
        self.repository.finish_run(RUN_ID, "succeeded")
        self.assertTrue(self.repository.reopen_run(RUN_ID))
        self.assertEqual(self.repository.run_summary(RUN_ID)["status"], "running")
        self.assertFalse(self.repository.reopen_run(RUN_ID), "a running run is already open")

    def test_an_unknown_run_has_no_state(self):
        self.assertIsNone(self.repository.load_run_state("no-such-run"))

    def test_saving_state_for_an_unknown_run_is_refused(self):
        with self.assertRaises(KeyError):
            self.repository.save_run_state("no-such-run", {"nodes": {}})


def alien(definition: dict):
    """The same StageDefinition, built by a separately loaded copy of its module.

    This is what Temporal's workflow sandbox hands back: a class with the same name and the
    same fields that is not the one anything else validates against.
    """
    specification = importlib.util.find_spec("orchestrator.contracts")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    assert module.StageDefinition is not StageDefinition, "the copy is the original"
    return module.StageDefinition.model_validate(definition)


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

    def test_requeueing_a_stage_clears_what_it_produced(self):
        """Resuming exists for a run whose scheduler was lost; --rerun for one whose stage was.

        A stage that succeeded on the code of the day but produced the wrong thing has to run
        again, and everything downstream has to follow from the new result, so its selected
        attempt goes with it rather than being left for a dependent to read.
        """
        self.graph.select_attempt(
            "tracks",
            "attempt:tracks",
            {
                contract.role: (f"artifact:tracks:{contract.role}",)
                for contract in self.graph.nodes["tracks"].definition.outputs.values()
            },
        )
        self.repository.save_run_state(
            RUN_ID,
            snapshot_state(
                self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
            ),
        )
        self.assertEqual(requeue(self.repository, RUN_ID, ["tracks", "nonesuch"]), ["tracks"])
        stored = self.repository.load_run_state(RUN_ID)["nodes"]["tracks"]
        self.assertEqual(stored["status"], "queued")
        self.assertIsNone(stored["selected_attempt_id"])
        self.assertEqual(stored["selected_artifacts"], {})

        revived = instantiate_graph(self.options)
        restore_state(revived, self.repository.load_run_state(RUN_ID), {}, {})
        self.assertEqual(revived.nodes["tracks"].status, NodeStatus.QUEUED)

    def test_a_definition_that_arrives_as_a_model_is_rebuilt_too(self):
        """Temporal's workflow sandbox rebuilds the modules a workflow imports.

        A StageDefinition constructed outside the sandbox is then a different class from the
        one GraphNode validates against, however identical it looks, and passing it through
        raised "Input should be a valid dictionary or instance of StageDefinition" for an
        instance of exactly that. The whole run crash-looped on restore. Whatever the state
        carries, it has to come back as a node.
        """
        created = self.expand()
        target = created[0]
        stored = snapshot_state(
            self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
        )
        stored["nodes"][target]["definition"] = alien(stored["nodes"][target]["definition"])
        revived = instantiate_graph(self.options)
        restore_state(revived, stored, {}, {})
        self.assertIn(target, revived.nodes)
        self.assertEqual(
            revived.nodes[target].definition.id, self.graph.nodes[target].definition.id
        )

    def test_a_rebuilt_node_takes_todays_stage_definition(self):
        """A stage definition fixed after a run expanded has to reach that run.

        lhm_frozen never declared the clip every legacy command opens. Rebuilding expanded
        nodes from the copy stored at expansion time froze the omission into every run that
        had got that far, so the fix could not be deployed into them -- the stage went on
        failing to build its command until the clip was started again from nothing.
        """
        created = self.expand()
        frozen = next(node for node in created if node.startswith("lhm_frozen"))
        stored = snapshot_state(
            self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
        )["nodes"][frozen]
        # The run as it was: no run inputs declared, and the wiring it discovered for itself.
        stale = dict(stored["definition"])
        stale["inputs"] = {
            name: binding
            for name, binding in stale["inputs"].items()
            if binding["source"] == "stage_output"
        }
        self.assertNotIn("source", stale["inputs"], "precondition: the stale copy lacks the clip")

        node = rebuild_expanded_node(
            frozen,
            "lhm_frozen",
            stale,
            dependencies=tuple(stored["dependencies"]),
            branch_key=stored["branch_key"],
        )
        self.assertEqual(node.definition.inputs["source"].source, "run_input")
        self.assertEqual(node.definition.inputs["source"].role, "source_video")
        # The run's own wiring survives: no registry knows which person this node reads.
        prepared = node.definition.inputs["prepared_person"]
        self.assertEqual(prepared.source, "stage_output")
        self.assertTrue(prepared.stage_id.startswith("person_prep:"))

    def test_a_rebuilt_join_keeps_the_producers_it_found(self):
        """package_people is wired to one motion node per person, which no registry knows."""
        self.expand()
        stored = snapshot_state(
            self.graph, paused=False, canceled=False, agent_retries={}, retry_parameters={}
        )["nodes"]["package_people"]
        node = rebuild_expanded_node(
            "package_people",
            "package_people",
            stored["definition"],
            dependencies=tuple(stored["dependencies"]),
        )
        wired = {
            name: binding.stage_id
            for name, binding in node.definition.inputs.items()
            if binding.source == "stage_output"
        }
        self.assertTrue(any(v.startswith("lhm_motion:") for v in wired.values()), wired)
        self.assertNotIn(
            "lhm_motion",
            set(wired.values()),
            "the base graph's unexpanded producer must not come back",
        )

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
