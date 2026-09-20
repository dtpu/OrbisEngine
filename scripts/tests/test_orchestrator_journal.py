"""A run is its directory and its journal; everything else is derived from them.

These cover the record the agent writes as it works, and the database mirror the dashboard
reads. The mirror is a projection: losing it, or building a run on a machine that never had a
database, must not lose the run.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.cli import RunContext, main, parse_flags, run_clip_command, step_environment
from orchestrator.database import Base
from orchestrator.journal import Journal, RunProjection
from orchestrator.repository import PipelineRepository
from orchestrator.steps import STEPS, suggested_order


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-journal-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.journal = Journal(self.root)

    def run_step(self, step: str, status: str) -> None:
        number = self.journal.attempt_number(step)
        self.journal.append("step.started", step=step, attempt=f"a:{step}:{number}", number=number)
        self.journal.append("step.finished", step=step, attempt=f"a:{step}:{number}", status=status)

    def test_a_step_that_failed_and_then_passed_counts_as_done(self):
        self.run_step("clean", "failed")
        self.run_step("clean", "succeeded")
        self.assertEqual(self.journal.attempt_number("clean"), 3)
        self.assertEqual(self.journal.last_status("clean"), "succeeded")
        self.assertEqual(self.journal.done(), {"clean"})

    def test_a_step_that_passed_and_then_failed_is_not_done(self):
        """Rerunning something that worked and breaking it undoes it."""
        self.run_step("pi3x", "succeeded")
        self.run_step("pi3x", "failed")
        self.assertEqual(self.journal.done(), set())

    def test_a_question_holds_the_run_until_it_is_answered(self):
        self.journal.append("question", question="is this clip truncated?", step="clean")
        self.assertEqual(self.journal.unanswered(), "is this clip truncated?")
        self.journal.append("answer", text="yes, use the trimmed copy")
        self.assertIsNone(self.journal.unanswered())

    def test_the_history_survives_a_line_that_is_not_one(self):
        """A crash mid-write, or something else appending, must not lose the rest."""
        self.run_step("clean", "succeeded")
        with self.journal.path.open("a") as stream:
            stream.write("{ truncated\n\n")
        self.run_step("pi3x", "succeeded")
        self.assertEqual(self.journal.done(), {"clean", "pi3x"})

    def test_an_unknown_kind_is_refused_rather_than_written(self):
        with self.assertRaisesRegex(ValueError, "not a kind of thing"):
            self.journal.append("step.exploded", step="clean")

    def test_two_writers_interleave_whole_lines(self):
        """An operator answering while the agent is working writes to the same file."""
        for index in range(40):
            Journal(self.root).append("note", text=f"note {index}")
        entries = self.journal.entries()
        self.assertEqual(len(entries), 40)
        self.assertEqual([e.data["text"] for e in entries][-1], "note 39")


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-projection-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        engine = create_engine(f"sqlite:///{self.root / 'meta.db'}")
        Base.metadata.create_all(engine)
        self.repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
        from orchestrator.contracts import Run

        self.repository.create_run_row(
            Run(
                id="run-1",
                graph_version="wander.agent-run/1",
                code_revision="test-revision",
                source_sha256="a" * 64,
                source_artifact_id="artifact:source",
                created_by="tests",
            )
        )
        self.journal = Journal(
            self.root, projection=RunProjection(self.repository, "run-1", catalogue=STEPS)
        )

    def test_a_step_the_run_never_planned_still_gets_a_row(self):
        """The plan is a suggestion, so the dashboard has to show what was really done."""
        self.journal.append(
            "step.started", step="pi3x", attempt="run-1:pi3x:1", number=1, command=["run_clip"]
        )
        self.journal.append(
            "step.finished", step="pi3x", attempt="run-1:pi3x:1", status="succeeded"
        )
        summary = self.repository.run_summary("run-1")
        node = next(n for n in summary["nodes"] if n["id"] == "pi3x")
        self.assertEqual(node["status"], "succeeded")
        attempt = next(a for a in summary["attempts"] if a["id"] == "run-1:pi3x:1")
        self.assertEqual(attempt["status"], "succeeded")

    def test_the_rows_can_be_rebuilt_from_the_journal_alone(self):
        """A database that was lost, or never existed while the run was working."""
        plain = Journal(self.root)
        for step in ("clean", "pi3x"):
            plain.append("step.started", step=step, attempt=f"run-1:{step}:1", number=1)
            plain.append("step.finished", step=step, attempt=f"run-1:{step}:1", status="succeeded")
        projection = RunProjection(self.repository, "run-1", catalogue=STEPS)
        self.assertEqual(projection.replay(plain), 4)
        statuses = {n["id"]: n["status"] for n in self.repository.run_summary("run-1")["nodes"]}
        self.assertEqual(statuses.get("clean"), "succeeded")
        self.assertEqual(statuses.get("pi3x"), "succeeded")

    def test_a_broken_mirror_does_not_stop_the_run(self):
        """The journal line is already on disk; the database is the derived copy."""

        class Broken:
            def __getattr__(self, name):
                def explode(*args, **kwargs):
                    raise RuntimeError("database is gone")

                return explode

        journal = Journal(self.root, projection=RunProjection(Broken(), "run-1", catalogue=STEPS))
        journal.append("step.started", step="clean", attempt="run-1:clean:1", number=1)
        self.assertEqual(len(journal.entries()), 1)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-cli-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.run_dir = self.root / "demo"
        self.run_dir.mkdir()
        (self.run_dir / "source.mp4").write_bytes(b"clip")
        (self.run_dir / "run.json").write_text(
            json.dumps(
                {
                    "runId": "run-demo",
                    "name": "demo",
                    "source": str(self.run_dir / "source.mp4"),
                    "repository": str(ROOT),
                    "options": {"marble": "none", "people": 4},
                }
            )
        )
        for name, value in (("WANDER_RUN_DIR", str(self.run_dir)), ("WANDER_RUN_ID", "run-demo")):
            previous = os.environ.get(name)
            os.environ[name] = value
            self.addCleanup(
                lambda n=name, p=previous: (
                    os.environ.__setitem__(n, p) if p is not None else os.environ.pop(n, None)
                )
            )
        os.environ.pop("WANDER_DATABASE_URL", None)
        self.context = RunContext.load()

    def test_a_step_runs_the_legacy_stage_in_this_run_directory(self):
        command = run_clip_command(self.context, "clean", ["--dilate", "28"])
        self.assertIn("--only", command)
        self.assertEqual(command[command.index("--only") + 1], "clean")
        self.assertEqual(command[command.index("--name") + 1], "demo")
        self.assertIn("--dilate", command)
        self.assertEqual(command[command.index("--people") + 1], "4")
        self.assertIn("--stage-ledger", command, "the legacy spend ledger must stay in play")

    def test_the_legacy_pipeline_writes_into_this_run_and_nowhere_else(self):
        environment = step_environment(self.context)
        self.assertEqual(environment["WANDER_RUNS_DIR"], str(self.run_dir.parent))
        self.assertTrue(environment["WANDER_PUBLIC_DIR"].startswith(str(self.run_dir)))
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")

    def test_flags_are_recorded_as_what_was_asked_for(self):
        self.assertEqual(
            parse_flags(["--dilate", "28", "--moved-mask", "--lama-px", "1280"]),
            {"dilate": "28", "moved_mask": True, "lama_px": "1280"},
        )

    def test_starting_a_step_records_it_and_returns(self):
        """Starting work is not waiting for it: the agent ends its turn and is woken."""
        self.assertEqual(main(["step", "clean"]), 0)
        journal = Journal(self.run_dir)
        started = [e for e in journal.entries() if e.kind == "step.started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0].data["step"], "clean")

    def test_a_step_run_here_reports_what_became_of_it(self):
        """The clip here is four bytes, so the stage really does fail."""
        code = main(["step", "--wait", "clean"])
        self.assertEqual(code, 1)
        journal = Journal(self.run_dir)
        self.assertEqual(journal.last_status("clean"), "failed")
        finished = [e for e in journal.entries() if e.kind == "step.finished"][-1]
        self.assertTrue((self.run_dir / finished.data["log"]).is_file())

    def test_our_own_flags_read_the_same_on_either_side_of_the_step(self):
        """`rest` is a REMAINDER, so a flag after the name was forwarded to run_clip.py.

        The agent wrote `wander step review --wait`, which is what the standing instructions
        show, and run_clip.py rejected an argument it had never heard of.
        """
        from orchestrator.cli import build_parser, take_our_flags

        after = build_parser().parse_args(["step", "clean", "--wait", "--dilate", "28"])
        take_our_flags(after)
        self.assertTrue(after.wait)
        self.assertEqual(after.rest, ["--dilate", "28"])

        before = build_parser().parse_args(["step", "--wait", "clean", "--dilate", "28"])
        take_our_flags(before)
        self.assertTrue(before.wait)
        self.assertEqual(before.rest, ["--dilate", "28"])

    def test_a_step_that_needs_a_key_this_run_lacks_says_which(self):
        """Running it anyway spends a turn to be told by the stage, and a paid one, worse."""
        from orchestrator.cli import missing_credentials

        previous = os.environ.pop("WLT_API_KEY", None)
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("WLT_API_KEY", previous) if previous is not None else None
            )
        )
        self.assertEqual(missing_credentials("marble_video"), ["WLT_API_KEY"])
        self.assertEqual(missing_credentials("pi3x"), [], "Modal reads ~/.modal.toml")

    def test_asking_leaves_the_run_waiting_and_finishing_ends_it(self):
        main(["ask", "is this clip truncated?", "--step", "clean"])
        journal = Journal(self.run_dir)
        self.assertEqual(journal.unanswered(), "is this clip truncated?")
        main(["finish", "blocked", "source will not decode"])
        self.assertEqual(journal.finished(), "blocked")

    def test_the_catalogue_offers_an_order_without_imposing_one(self):
        order = suggested_order()
        self.assertLess(order.index("pi3x"), order.index("frame_align"))
        self.assertLess(order.index("lhm_frozen"), order.index("lhm_motion"))
        self.assertLess(order.index("scale_fit"), order.index("place_fit"))
        self.assertEqual(set(order), set(STEPS))


if __name__ == "__main__":
    unittest.main()
