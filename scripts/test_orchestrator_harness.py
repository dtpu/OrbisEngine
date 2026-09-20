"""The reviewing harness's idle clock. No model is called here.

An agent that stops writing used to burn its whole timeout in silence and hand the stage back
with nothing to read. These check the clock that interrupts it, that working slowly is not
mistaken for silence, and that a hung grandchild goes with it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.agent import harness as harness_module
from orchestrator.agent.contracts import AgentTaskPacket
from orchestrator.agent.harness import DECISION_FILE, HarnessAgent, HarnessPolicy

PACKET = AgentTaskPacket(
    run_id="run-1",
    node_id="clean",
    objective="Judge the cleaned clip",
    stage_definition={},
    dependency_state={},
    permitted_tools=("quality.verdict",),
)
VERDICT = {
    "tool": "quality.verdict",
    "node_id": "clean",
    "attempt_id": "attempt-1",
    "verdict": "pass",
    "evidence_artifact_ids": [],
    "rationale": "no residual person in the sampled frames",
}


class IdleClockTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-harness-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        polled = harness_module.POLL_SECONDS
        harness_module.POLL_SECONDS = 0.2
        self.addCleanup(setattr, harness_module, "POLL_SECONDS", polled)

    def agent(self, body: str, **policy) -> HarnessAgent:
        script = self.root / "agent.py"
        script.write_text(textwrap.dedent(body))
        return HarnessAgent(
            HarnessPolicy(
                kind="command",
                command=(sys.executable, "-u", str(script)),
                timeout_seconds=60,
                **policy,
            )
        )

    def test_a_silent_agent_is_stopped_and_reported_as_stalled(self):
        agent = self.agent(
            """
            import time
            time.sleep(60)
            """,
            idle_seconds=1,
        )
        started = time.monotonic()
        outcome = agent.run(PACKET, self.root / "scratch", "review this")
        self.assertLess(time.monotonic() - started, 45, "the idle clock did not fire")
        self.assertEqual(outcome.result.status, "stalled")
        self.assertIn("wrote nothing", outcome.result.error)
        self.assertIsNone(outcome.decision)

    def test_an_agent_that_keeps_writing_is_left_alone(self):
        agent = self.agent(
            f"""
            import json, pathlib, time
            for _ in range(10):
                print("still reading frames")
                time.sleep(0.2)
            pathlib.Path("{DECISION_FILE}").write_text(json.dumps({VERDICT!r}))
            """,
            idle_seconds=5,
        )
        outcome = agent.run(PACKET, self.root / "scratch", "review this")
        self.assertEqual(outcome.result.status, "completed")
        self.assertIsNotNone(outcome.decision)
        self.assertEqual(outcome.decision.verdict, "pass")

    def test_a_decision_written_before_the_stall_is_still_taken(self):
        """The judgement is what the stage needs; the process failing to exit is not its problem."""
        agent = self.agent(
            f"""
            import json, pathlib, time
            pathlib.Path("{DECISION_FILE}").write_text(json.dumps({VERDICT!r}))
            time.sleep(60)
            """,
            idle_seconds=1,
        )
        outcome = agent.run(PACKET, self.root / "scratch", "review this")
        self.assertEqual(outcome.result.status, "completed")
        self.assertIsNotNone(outcome.decision)

    def test_the_hung_command_the_agent_started_is_killed_with_it(self):
        """The agent's own child is usually what hangs, and it is not the process we spawned."""
        agent = self.agent(
            """
            import pathlib, subprocess, time
            child = subprocess.Popen(["sleep", "300"])
            pathlib.Path("child.pid").write_text(str(child.pid))
            time.sleep(60)
            """,
            idle_seconds=1,
        )
        scratch = self.root / "scratch"
        agent.run(PACKET, scratch, "review this")
        pid = int((scratch / "child.pid").read_text())
        self.assertFalse(
            Path(f"/proc/{pid}").exists()
            and "sleep" in Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf8", "replace"),
            "the hung grandchild outlived the harness",
        )

    def test_a_resumable_agent_is_nudged_and_finishes(self):
        """The point of the clock: the silence becomes a question, and the question an answer.

        A harness that can continue a session takes ``{session}`` in its command. On the first
        run it is empty, and this one goes quiet; the nudge resumes it with the session it
        announced, and it decides.
        """
        script = self.root / "resumable.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import json, pathlib, sys, time
                session = sys.argv[1]
                if not session:
                    print(json.dumps({{"type": "thread.started", "thread_id": "t1"}}))
                    time.sleep(60)
                pathlib.Path("nudge.txt").write_text(sys.stdin.read())
                pathlib.Path("{DECISION_FILE}").write_text(json.dumps({VERDICT!r}))
                """
            )
        )
        agent = HarnessAgent(
            HarnessPolicy(
                kind="command",
                command=(sys.executable, "-u", str(script), "{session}"),
                timeout_seconds=60,
                idle_seconds=1,
            )
        )
        scratch = self.root / "scratch"
        outcome = agent.run(PACKET, scratch, "review this")
        self.assertEqual(outcome.result.status, "completed")
        self.assertIsNotNone(outcome.decision)
        self.assertEqual(outcome.session, "t1")
        self.assertIn("went quiet", (scratch / "nudge.txt").read_text())

    def test_a_silent_agent_gives_up_after_its_nudges(self):
        script = self.root / "mute.py"
        script.write_text(
            textwrap.dedent(
                """
                import json, sys, time
                if not sys.argv[1]:
                    print(json.dumps({"type": "thread.started", "thread_id": "t1"}))
                time.sleep(60)
                """
            )
        )
        agent = HarnessAgent(
            HarnessPolicy(
                kind="command",
                command=(sys.executable, "-u", str(script), "{session}"),
                timeout_seconds=60,
                idle_seconds=1,
                idle_nudges=1,
            )
        )
        outcome = agent.run(PACKET, self.root / "scratch", "review this")
        self.assertEqual(outcome.result.status, "stalled")
        self.assertIn("did not answer 1 nudge(s)", outcome.result.error)

    def test_the_clock_can_be_switched_off(self):
        agent = self.agent(
            f"""
            import json, pathlib, time
            time.sleep(1.5)
            pathlib.Path("{DECISION_FILE}").write_text(json.dumps({VERDICT!r}))
            """,
            idle_seconds=0,
        )
        outcome = agent.run(PACKET, self.root / "scratch", "review this")
        self.assertEqual(outcome.result.status, "completed")

    def test_the_prompt_reaches_the_agent_as_stdin(self):
        agent = self.agent(
            f"""
            import json, pathlib, sys
            pathlib.Path("seen.txt").write_text(sys.stdin.read())
            pathlib.Path("{DECISION_FILE}").write_text(json.dumps({VERDICT!r}))
            """,
            idle_seconds=5,
        )
        scratch = self.root / "scratch"
        agent.run(PACKET, scratch, "look at frame 12")
        self.assertEqual((scratch / "seen.txt").read_text(), "look at frame 12")


class PolicyTests(unittest.TestCase):
    def test_the_clock_is_configured_from_the_environment(self):
        policy = HarnessPolicy.from_environment(
            {"WANDER_REVIEW_IDLE": "120", "WANDER_REVIEW_NUDGES": "1"}
        )
        self.assertEqual(policy.idle_seconds, 120)
        self.assertEqual(policy.idle_nudges, 1)

    def test_the_default_clock_is_five_minutes(self):
        policy = HarnessPolicy.from_environment({})
        self.assertEqual(policy.idle_seconds, 300)

    def test_a_negative_clock_is_refused(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            HarnessPolicy(idle_seconds=-1)

    def test_a_resumed_session_keeps_the_sandbox_and_drops_the_directory(self):
        """`codex exec resume` accepts neither --cd nor --sandbox; the config override is how."""
        policy = HarnessPolicy(kind="codex", sandbox="danger-full-access", model="gpt-6-astra")
        fresh = policy.argv(Path("/ws"), Path("/ws/TASK.md"))
        resumed = policy.argv(Path("/ws"), Path("/ws/TASK.md"), session="abc-123")
        self.assertIn("--cd", fresh)
        self.assertNotIn("--cd", resumed)
        self.assertIn("resume", resumed)
        self.assertIn("abc-123", resumed)
        self.assertIn('sandbox_mode="danger-full-access"', resumed)

    def test_the_nudge_names_the_stage_and_the_way_out(self):
        nudge = harness_module.NUDGE.format(minutes=5, decision=DECISION_FILE, node_id="clean")
        self.assertIn("human.ask", nudge)
        self.assertIn('"node_id": "clean"', nudge)
        json.loads(nudge[nudge.index("{") : nudge.index("}") + 1])


if __name__ == "__main__":
    unittest.main()
