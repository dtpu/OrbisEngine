"""Workflow state-transition tests that do not start providers or a Temporal server."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contracts import NodeStatus
from orchestrator.graph import instantiate_graph
from orchestrator.stages import GraphOptions
from orchestrator.workflows.run import (
    ApprovalSignal,
    GenerationWorkflow,
    StageActivityResult,
    activity_inputs,
    apply_activity_result,
)


def outputs(graph, node_id):
    return {
        contract.role: [f"artifact:{node_id}:{contract.role}"]
        for contract in graph.nodes[node_id].definition.outputs.values()
    }


class WorkflowTests(unittest.TestCase):
    def test_activity_input_resolves_artifact_ids_not_logical_names(self):
        graph = instantiate_graph(GraphOptions(marble="video"))
        graph.select_attempt("admission", "attempt:admission", outputs(graph, "admission"))
        selected = activity_inputs(
            graph,
            "clean",
            {"source_video": ["artifact:source"]},
        )
        self.assertEqual(selected["source"], ["artifact:source"])

    def test_activity_result_selects_attempt_and_expands_people(self):
        graph = instantiate_graph(GraphOptions(marble="none", all_people=True, people=2))
        graph.select_attempt("admission", "attempt:admission", outputs(graph, "admission"))
        graph.select_attempt("pi3x", "attempt:pi3x", outputs(graph, "pi3x"))
        result = StageActivityResult(
            node_id="tracks",
            attempt_id="attempt:tracks",
            status="succeeded",
            artifacts=outputs(graph, "tracks"),
            branches=[
                {"key": "00", "artifact_id": "artifact:track:00"},
                {"key": "01", "artifact_id": "artifact:track:01"},
            ],
        )
        apply_activity_result(graph, result)
        self.assertEqual(graph.nodes["tracks"].selected_attempt_id, "attempt:tracks")
        self.assertIn("person_prep:00", graph.nodes)
        self.assertIn("package_people", graph.nodes)

    def test_activity_failure_blocks_dependents_without_losing_error(self):
        graph = instantiate_graph(GraphOptions(marble="none"))
        apply_activity_result(
            graph,
            StageActivityResult(
                node_id="pi3x",
                attempt_id="attempt:pi3x",
                status="failed",
                error="provider unavailable",
            ),
        )
        self.assertEqual(graph.nodes["pi3x"].status, NodeStatus.FAILED)
        self.assertEqual(graph.nodes["pi3x"].blocked_reason, "provider unavailable")

    def test_approval_signal_promotes_waiting_human_node(self):
        workflow = GenerationWorkflow()
        workflow.graph = instantiate_graph(GraphOptions(marble="video"))
        node = workflow.graph.nodes["clean_review"]
        node.status = NodeStatus.WAITING_HUMAN
        workflow.approve(
            ApprovalSignal(
                node_id="clean_review",
                attempt_id="approval:1",
                artifacts=outputs(workflow.graph, "clean_review"),
                approved_by="operator",
                rationale="Reviewed every sampled frame",
            )
        )
        self.assertEqual(node.status, NodeStatus.SUCCEEDED)
        self.assertEqual(node.selected_attempt_id, "approval:1")


if __name__ == "__main__":
    unittest.main()
