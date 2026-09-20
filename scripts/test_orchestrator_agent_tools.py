"""Typed coding-agent tool authorization and audit tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from orchestrator.agent.contracts import AgentTaskPacket
from orchestrator.agent.tools import AgentToolDispatcher


class Backend:
    def __init__(self):
        self.calls = []

    def execute(self, call):
        self.calls.append(call)
        return {"accepted": True, "tool": call.tool}


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "tools.jsonl"
        self.backend = Backend()
        self.dispatcher = AgentToolDispatcher(
            AgentTaskPacket(
                run_id="run-1",
                node_id="clean",
                objective="Review clean attempt",
                stage_definition={},
                dependency_state={},
                permitted_tools=("artifact.preview", "quality.verdict"),
            ),
            self.backend,
            self.log,
        )

    def test_permitted_calls_are_strictly_typed_and_audited(self):
        result = self.dispatcher.dispatch(
            {
                "tool": "artifact.preview",
                "artifact_id": "artifact-1",
                "page": 0,
            }
        )
        self.assertTrue(result["ok"])
        record = json.loads(self.log.read_text())
        self.assertEqual(record["call"]["tool"], "artifact.preview")
        self.assertEqual(record["runId"], "run-1")

    def test_unpermitted_and_malformed_calls_never_reach_backend(self):
        with self.assertRaises(PermissionError):
            self.dispatcher.dispatch(
                {
                    "tool": "attempt.retry",
                    "node_id": "clean",
                    "attempt_id": "attempt-1",
                    "hypothesis": "Wider mask",
                    "parameters": {"dilate": 40},
                }
            )
        with self.assertRaises(ValidationError):
            self.dispatcher.dispatch(
                {
                    "tool": "quality.verdict",
                    "node_id": "clean",
                    "attempt_id": "attempt-1",
                    "verdict": "ship_it",
                    "rationale": "No",
                }
            )
        self.assertEqual(self.backend.calls, [])
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
