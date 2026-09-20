"""Fail-closed coding-agent sandbox contract tests."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.agent import AgentTaskPacket, ContainerAgent, SandboxPolicy


class AgentSandboxTests(unittest.TestCase):
    def test_image_must_be_immutable_and_secrets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            SandboxPolicy(image="wander-agent:latest")
        with self.assertRaisesRegex(ValueError, "secrets"):
            SandboxPolicy(
                image="wander-agent@sha256:" + "a" * 64,
                environment={"OPENAI_API_KEY": "forbidden"},
            )

    def test_container_has_no_network_or_host_write_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet = root / "task.json"
            packet.write_text("{}")
            repository = root / "repository"
            repository.mkdir()
            sandbox = ContainerAgent(
                repository,
                SandboxPolicy(image="wander-agent@sha256:" + "a" * 64),
            )
            command = sandbox.command(packet, root / "scratch")
            rendered = " ".join(command)
            self.assertIn("--network none", rendered)
            self.assertIn("--read-only", command)
            self.assertIn("dst=/repo,readonly", rendered)
            self.assertNotIn("/var/run/docker.sock", rendered)

    def test_task_packet_rejects_undeclared_context(self):
        packet = AgentTaskPacket(
            run_id="run-1",
            node_id="clean",
            objective="Review retained mask evidence",
            stage_definition={},
            dependency_state={"admission": "succeeded"},
            permitted_tools=("artifact.preview", "quality.verdict"),
        )
        self.assertEqual(packet.schema_version, "wander.agent-task/1")
        with self.assertRaises(ValueError):
            AgentTaskPacket(
                **packet.model_dump(),
                hidden_host_path="/home/operator",
            )


if __name__ == "__main__":
    unittest.main()
