"""Deterministic graph and artifact-readiness tests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contracts import NodeStatus
from orchestrator.graph import (
    BranchArtifact,
    ShotBranch,
    child_workflows_for_shots,
    instantiate_graph,
)
from orchestrator.stages import GraphOptions


def artifacts_for(graph, node_id):
    return {
        contract.role: (f"artifact:{node_id}:{contract.role}",)
        for contract in graph.nodes[node_id].definition.outputs.values()
        if contract.required
    }


def succeed(graph, node_id):
    graph.select_attempt(node_id, f"attempt:{node_id}", artifacts_for(graph, node_id))


class GraphTests(unittest.TestCase):
    def test_video_graph_matches_artifact_dependencies(self):
        graph = instantiate_graph(GraphOptions(marble="video", objects=True))
        self.assertEqual(
            set(graph.nodes),
            {
                "admission",
                "pi3x",
                "frame_align",
                "clean",
                "world_prompt",
                "clean_review",
                "marble_video_submit",
                "marble_video",
                "person_prep",
                "lhm_frozen",
                "lhm_motion",
                "package_people",
                "scale_fit",
                "place_fit",
                "anchors",
                "verify",
            },
        )
        self.assertIn("marble_video", graph.nodes["scale_fit"].dependencies)
        self.assertIn("package_people", graph.nodes["scale_fit"].dependencies)
        self.assertIn("pi3x", graph.nodes["lhm_motion"].dependencies)

    def test_admission_gates_all_parallel_roots(self):
        graph = instantiate_graph(GraphOptions(marble="video"))
        self.assertEqual(graph.evaluate({"source_video"}), ("admission",))
        succeed(graph, "admission")
        ready = set(graph.evaluate({"source_video"}))
        self.assertEqual(ready, {"clean", "person_prep", "pi3x", "world_prompt"})

    def test_human_gate_waits_for_selected_clean_artifacts(self):
        graph = instantiate_graph(GraphOptions(marble="video"))
        run_inputs = {"source_video"}
        succeed(graph, "admission")
        graph.evaluate(run_inputs)
        succeed(graph, "clean")
        succeed(graph, "world_prompt")
        graph.evaluate(run_inputs)
        self.assertEqual(graph.nodes["clean_review"].status, NodeStatus.WAITING_HUMAN)
        self.assertEqual(graph.nodes["marble_video"].status, NodeStatus.QUEUED)
        succeed(graph, "clean_review")
        self.assertIn("marble_video_submit", graph.evaluate(run_inputs))

    def test_failed_dependency_blocks_only_descendants(self):
        graph = instantiate_graph(GraphOptions(marble="none"))
        succeed(graph, "admission")
        graph.evaluate({"source_video"})
        graph.set_status("pi3x", NodeStatus.FAILED, "provider failed")
        ready = set(graph.evaluate({"source_video"}))
        self.assertEqual(graph.nodes["frame_align"].status, NodeStatus.BLOCKED)
        self.assertIn("clean", ready)
        self.assertIn("person_prep", ready)

    def test_attempt_selection_requires_declared_outputs(self):
        graph = instantiate_graph(GraphOptions(marble="none"))
        with self.assertRaisesRegex(ValueError, "lacks required artifact roles"):
            graph.select_attempt("admission", "attempt:bad", {})
        with self.assertRaisesRegex(ValueError, "undeclared artifact roles"):
            graph.select_attempt(
                "admission",
                "attempt:bad",
                {**artifacts_for(graph, "admission"), "secret": ("artifact:secret",)},
            )

    def test_modes_rebind_world_and_review_inputs(self):
        image = instantiate_graph(GraphOptions(marble="image"))
        self.assertEqual(
            image.nodes["clean_review"].definition.inputs["clean_frame"].stage_id,
            "clean_first",
        )
        self.assertEqual(
            image.nodes["scale_fit"].definition.inputs["world"].stage_id,
            "marble_image",
        )
        multi = instantiate_graph(GraphOptions(marble="multi"))
        self.assertIn("world_mode", multi.nodes["clean_multi"].dependencies)
        self.assertEqual(
            multi.nodes["scale_fit"].definition.inputs["world"].stage_id,
            "marble_multi",
        )

    def test_fingerprint_is_deterministic_and_configuration_sensitive(self):
        first = instantiate_graph(GraphOptions(marble="video"))
        second = instantiate_graph(GraphOptions(marble="video"))
        image = instantiate_graph(GraphOptions(marble="image"))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, image.fingerprint)

    def test_multiperson_graph_pauses_at_track_expansion(self):
        graph = instantiate_graph(GraphOptions(marble="none", all_people=True, people=4))
        self.assertIn("tracks", graph.nodes)
        self.assertNotIn("package_people", graph.nodes)
        self.assertFalse(any(node.startswith("person_prep:") for node in graph.nodes))

    def test_person_expansion_creates_isolated_branches_and_join(self):
        graph = instantiate_graph(GraphOptions(marble="none", all_people=True, people=4))
        succeed(graph, "admission")
        succeed(graph, "pi3x")
        succeed(graph, "tracks")
        created = graph.expand_people(
            (
                BranchArtifact(key="00", artifact_id="track-artifact-0"),
                BranchArtifact(key="01", artifact_id="track-artifact-1"),
            )
        )
        self.assertIn("person_prep:00", created)
        self.assertIn("lhm_motion:01", created)
        self.assertIn("package_people", created)
        self.assertEqual(graph.nodes["person_prep:00"].branch_key, "00")
        self.assertEqual(
            set(graph.nodes["package_people"].dependencies),
            {"admission", "frame_align", "lhm_motion:00", "lhm_motion:01"},
        )

    def test_person_expansion_enforces_cap_and_unique_keys(self):
        graph = instantiate_graph(GraphOptions(marble="none", all_people=True, people=1))
        succeed(graph, "admission")
        succeed(graph, "pi3x")
        succeed(graph, "tracks")
        with self.assertRaisesRegex(ValueError, "configured cap"):
            graph.expand_people(
                (
                    BranchArtifact(key="00", artifact_id="track-0"),
                    BranchArtifact(key="01", artifact_id="track-1"),
                )
            )

    def test_object_expansion_keeps_each_attempt_branch_visible(self):
        graph = instantiate_graph(
            GraphOptions(marble="none", all_people=True, people=2, objects=True)
        )
        succeed(graph, "admission")
        succeed(graph, "pi3x")
        succeed(graph, "tracks")
        graph.expand_people((BranchArtifact(key="00", artifact_id="track-artifact-0"),))
        succeed(graph, "object_detect")
        created = graph.expand_objects(
            (
                BranchArtifact(key="ball", artifact_id="object-ball"),
                BranchArtifact(key="bat", artifact_id="object-bat"),
            )
        )
        self.assertIn("object_lift:ball", created)
        self.assertIn("object_shape:bat", created)
        self.assertEqual(
            graph.nodes["object_shape:ball"].parent_node_id,
            "object_detect",
        )
        self.assertIn("object_package", graph.nodes)

    def test_shot_children_share_canonical_budget_identity(self):
        source_sha = "a" * 64
        children = child_workflows_for_shots(
            "run-1",
            source_sha,
            (
                ShotBranch(
                    key="00",
                    clip_artifact_id="shot-0",
                    start_seconds=0,
                    end_seconds=2,
                ),
                ShotBranch(
                    key="01",
                    clip_artifact_id="shot-1",
                    start_seconds=2,
                    end_seconds=4,
                ),
            ),
        )
        self.assertEqual({child.budget_key for child in children}, {source_sha})
        self.assertEqual(
            {child.workflow_id for child in children},
            {"run-1:shot:00", "run-1:shot:01"},
        )


if __name__ == "__main__":
    unittest.main()
