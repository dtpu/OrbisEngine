"""The agent drives the run: brief in, journal out.

No model is called here. The harness runs a stand-in agent that does what a real one would --
reads its brief, runs a step through `wander`, writes down what it saw and stops -- so what is
under test is the wiring: that the brief says the true thing, that the agent can actually reach
back into its run, and that the supervisor reads the outcome out of the journal rather than
holding it in memory.
"""

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.agent.harness import HarnessAgent, HarnessPolicy
from orchestrator.journal import Journal
from orchestrator.session import RunSession, install_wander, render_brief
from orchestrator.steps import STEPS
from orchestrator.supervisor import open_run, runs_needing_work, work_on

AGENT = """
import os, pathlib, subprocess
brief = pathlib.Path(os.environ["WANDER_RUN_DIR"]) / "BRIEF.md"
text = brief.read_text()
assert "wander step" in text, "the brief does not say how to run a step"
{body}
"""


class BriefTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-brief-"))
        self.addCleanup(shutil.rmtree, self.root, True)

    def brief(self, journal: Journal) -> str:
        return render_brief("run-1", self.root, "demo", journal, "Get this clip through.")

    def test_the_brief_is_the_situation_and_points_at_the_tools(self):
        """What a step is for belongs in the standing instructions, not in every turn."""
        text = self.brief(Journal(self.root))
        self.assertIn("wander ready", text)
        self.assertIn("end your turn", text)
        self.assertIn("Not done yet:", text)

    def test_the_standing_instructions_carry_the_job(self):
        from orchestrator.session import SKILL_SOURCE

        skill = SKILL_SOURCE.read_text()
        self.assertIn("1600", skill, "the price of a world is in the standing instructions")
        self.assertIn("exit code of zero means the command ran", skill)
        self.assertIn("wander ask", skill)

    def test_what_a_step_costs_is_a_command_away(self):
        """The brief no longer lists nineteen steps; `wander show` has the detail."""
        from orchestrator.steps import STEPS

        self.assertTrue(STEPS["marble_video"].paid)
        self.assertEqual(STEPS["clean"].parameters["properties"]["dilate"]["default"], 20)

    def test_what_has_happened_is_carried_into_the_next_turn(self):
        journal = Journal(self.root)
        journal.append("step.started", step="clean", attempt="a", number=1)
        journal.append(
            "step.finished", step="clean", attempt="a", status="failed", error="exited 1"
        )
        journal.append("note", text="the mask eats the railing")
        text = self.brief(journal)
        self.assertIn("`clean` failed: exited 1", text)
        self.assertIn("the mask eats the railing", text)

    def test_a_run_waiting_on_an_answer_is_told_to_stop(self):
        journal = Journal(self.root)
        journal.append("question", question="is this clip truncated?", step="clean")
        text = self.brief(journal)
        self.assertIn("You are waiting on an answer", text)
        self.assertIn("end your turn", text)

    def test_wander_is_a_real_command_in_the_run(self):
        """The brief tells the agent to type it, so it has to exist and work."""
        binaries = install_wander(self.root)
        shim = binaries / "wander"
        self.assertTrue(shim.is_file() and shim.stat().st_mode & 0o111)
        self.assertIn(str(ROOT), shim.read_text(), "the shim must name the code absolutely")


class RunThroughTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-session-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"clip")

    def agent(self, body: str) -> HarnessPolicy:
        script = self.root / "agent.py"
        script.write_text(AGENT.format(body=body))
        return HarnessPolicy(
            kind="command",
            command=(sys.executable, str(script)),
            timeout_seconds=120,
            idle_seconds=0,
        )

    def open(self, **options) -> Path:
        return open_run(
            self.runs,
            run_id="run-1",
            name="demo",
            source=self.clip,
            options=options or {"marble": "none"},
        )

    def test_the_agent_runs_a_step_and_the_run_records_it(self):
        run_dir = self.open()
        session = RunSession(
            "run-1",
            run_dir,
            HarnessAgent(
                self.agent(
                    'subprocess.run(["wander", "step", "clean"])\n'
                    'subprocess.run(["wander", "finish", "blocked", "source is not a video"])'
                )
            ),
        )
        outcome = session.work(turns=2)
        self.assertEqual(outcome.finished, "blocked")
        journal = Journal(run_dir)
        # The step runs on its own, so its outcome lands after the agent's turn is over.
        for _ in range(200):
            if journal.last_status("clean"):
                break
            time.sleep(0.1)
        # The clip is four bytes, so the real stage really does fail.
        self.assertEqual(journal.last_status("clean"), "failed")
        finished = [e for e in journal.entries() if e.kind == "step.finished"][0]
        self.assertTrue((run_dir / finished.data["log"]).is_file(), "the step's log is kept")

    def test_a_question_ends_the_turn_and_holds_the_run(self):
        run_dir = self.open()
        session = RunSession(
            "run-1",
            run_dir,
            HarnessAgent(
                self.agent('subprocess.run(["wander", "ask", "is this clip truncated?"])')
            ),
        )
        outcome = session.work(turns=3)
        self.assertEqual(outcome.waiting, "is this clip truncated?")
        self.assertIsNone(outcome.finished)
        self.assertEqual(runs_needing_work(self.runs), [], "a waiting run is not picked up again")

    def test_a_turn_that_decides_nothing_stops_rather_than_repeating(self):
        run_dir = self.open()
        session = RunSession("run-1", run_dir, HarnessAgent(self.agent("pass")))
        outcome = session.work(turns=5)
        self.assertEqual(outcome.steps_run, 0)
        self.assertIn(
            "without running a step",
            [e.data.get("text", "") for e in Journal(run_dir).entries()][-1],
        )

    def test_the_supervisor_reads_the_outcome_out_of_the_journal(self):
        self.open()
        policy = self.agent('subprocess.run(["wander", "finish", "succeeded", "done"])')
        self.assertIn("finished succeeded", work_on(self.runs / "demo", None, policy, 2))
        self.assertEqual(runs_needing_work(self.runs), [], "a finished run is not picked up again")

    def test_a_run_holds_its_own_clip_and_its_own_plan(self):
        run_dir = self.open(marble="none", people=1)
        described = json.loads((run_dir / "run.json").read_text())
        self.assertTrue(Path(described["source"]).is_file())
        self.assertNotEqual(described["source"], str(self.clip), "the run keeps its own copy")
        self.assertEqual(described["options"]["marble"], "none")

    def test_a_step_runs_on_its_own_and_the_turn_ends(self):
        """An agent watching a ten-minute GPU step is a session held open to do nothing."""
        run_dir = self.open()
        session = RunSession(
            "run-1",
            run_dir,
            HarnessAgent(
                self.agent(
                    'out = subprocess.run(["wander", "step", "clean"], capture_output=True, text=True)\n'
                    'assert "end your turn" in out.stdout, out.stdout\n'
                )
            ),
        )
        outcome = session.turn()
        journal = Journal(run_dir)
        started = [e for e in journal.entries() if e.kind == "step.started"]
        self.assertEqual(len(started), 1, "the step was started")
        self.assertEqual(outcome.steps_run, 1)
        # The child journals its own finish, after the agent's turn is over.
        for _ in range(200):
            if journal.last_status("clean"):
                break
            time.sleep(0.1)
        self.assertEqual(journal.last_status("clean"), "failed", "the four-byte clip really fails")

    def test_the_standing_instructions_arrive_where_the_agent_will_read_them(self):
        run_dir = self.open()
        RunSession("run-1", run_dir, HarnessAgent(self.agent("pass"))).turn()
        skill = (run_dir / "AGENTS.md").read_text()
        self.assertIn("wander ready", skill)
        self.assertIn("Ending your turn is the normal thing to do", skill)
        self.assertIn("never wrap a command in a timeout", skill, "no babysitting")

    def test_every_planned_step_is_one_the_catalogue_describes(self):
        from orchestrator.steps import planned_steps

        for name in planned_steps({"marble": "video", "people": 4, "finetune": True}):
            self.assertIn(name, STEPS)


if __name__ == "__main__":
    unittest.main()
