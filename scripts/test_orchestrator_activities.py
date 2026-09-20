"""Typed activity execution and immutable failure-capture tests."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

from orchestrator.activities import stage as stage_module
from orchestrator.activities.adapters import default_adapters
from orchestrator.activities.stage import CommandAdapter, StageActivityRunner, StageExecution
from orchestrator.artifacts import LocalCAS
from orchestrator.workflows.run import StageActivityInput


class Resolver:
    def __init__(self, files):
        self.files = files

    def hydrate(self, artifact_id, destination_directory):
        destination = destination_directory / self.files[artifact_id].name
        shutil.copyfile(self.files[artifact_id], destination)
        return destination


class DenyPaidGuard:
    def begin(self, request, attempt_id):
        raise ValueError("budget exhausted")

    def finish(self, claim_id, status, evidence):
        raise AssertionError("a denied claim must not be finished")


class ActivityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.txt"
        self.source.write_text("source")
        self.store = LocalCAS(self.root / "cas")
        self.store.initialize()
        self.archive = LocalCAS(self.root / "archive")
        self.archive.initialize()

    def request(self, executor="test"):
        return StageActivityInput(
            run_id="run-1",
            node_id="example",
            stage_type="example",
            definition={"executor": executor},
            selected_inputs={"source": ["source-artifact"]},
        )

    def runner(self, adapter, **options):
        return StageActivityRunner(
            repository=Path(__file__).resolve().parents[1],
            workspace_root=self.root / "runs",
            store=self.store,
            outbox_archive=self.archive,
            resolver=Resolver({"source-artifact": self.source}),
            adapters={"test": adapter},
            **options,
        )

    def test_success_returns_only_declared_output_artifacts(self):
        def build(context):
            output = context.attempt.outputs / "result.json"
            code = f"from pathlib import Path; Path({str(output)!r}).write_text('{{\"ok\":true}}')"
            return StageExecution(
                command=(sys.executable, "-c", code),
                cwd=context.repository,
                output_roles={"outputs/result.json": "report"},
            )

        result = self.runner(CommandAdapter(build)).execute(self.request(), "attempt-1")
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(set(result.artifacts), {"report"})
        self.assertEqual(len(result.artifacts["report"]), 1)
        attempt = self.root / "runs/run-1/attempts/example/attempt-1"
        self.assertEqual(json.loads((attempt / "result.json").read_text())["status"], "succeeded")

    def test_failure_retains_stdout_stderr_and_partial_output(self):
        def build(context):
            partial = context.attempt.outputs / "partial.bin"
            code = (
                "import sys; from pathlib import Path; "
                f"Path({str(partial)!r}).write_bytes(b'partial'); "
                "print('started'); print('failed', file=sys.stderr); raise SystemExit(7)"
            )
            return StageExecution(
                command=(sys.executable, "-c", code),
                cwd=context.repository,
                output_roles={"outputs/partial.bin": "partial_result"},
            )

        result = self.runner(CommandAdapter(build)).execute(self.request(), "attempt-1")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.artifacts, {})
        attempt = self.root / "runs/run-1/attempts/example/attempt-1"
        self.assertIn("started", (attempt / "stdout.log").read_text())
        self.assertIn("failed", (attempt / "stderr.log").read_text())
        self.assertEqual((attempt / "outputs/partial.bin").read_bytes(), b"partial")
        entry = json.loads((self.root / "runs/run-1/outbox/attempt-1.json").read_text())
        paths = {item["relative_path"] for item in entry["manifest"]["files"]}
        self.assertIn("outputs/partial.bin", paths)

    def test_a_running_stage_reports_that_it_is_alive(self):
        """Without this a dead worker is only noticed when the stage's own timeout expires.

        A Marble stage's is four hours, so a worker killed mid-stage left the run sitting still
        for all of them with nothing running and the work already done on disk.
        """
        beats = []
        interval = stage_module.HEARTBEAT_SECONDS
        stage_module.HEARTBEAT_SECONDS = 0.2
        self.addCleanup(setattr, stage_module, "HEARTBEAT_SECONDS", interval)
        original = stage_module.beat
        stage_module.beat = lambda: beats.append(1)
        self.addCleanup(setattr, stage_module, "beat", original)

        adapter = CommandAdapter(
            lambda context: StageExecution(
                command=(sys.executable, "-c", "import time; time.sleep(1.2)"),
                cwd=context.repository,
            )
        )
        result = self.runner(adapter).execute(self.request(), "attempt-1")
        self.assertEqual(result.status, "succeeded")
        self.assertGreaterEqual(len(beats), 3, "the stage went quiet while it was working")

    def test_reporting_liveness_outside_an_activity_is_harmless(self):
        """The runner is also called by the tests and the resume script, with no Temporal."""
        stage_module.beat()

    def test_tracking_reports_one_branch_per_person_it_kept(self):
        """The graph only grows a person chain for a branch this list names.

        While it was empty nothing downstream of tracking existed: no person was prepared,
        reconstructed, animated or packaged, and the run finished "succeeded" holding a
        generated room with nobody in it.
        """
        adapter = default_adapters()["track_people"]
        request = self.request("track_people")
        name = adapter.legacy_name(
            SimpleNamespace(request=SimpleNamespace(run_id=request.run_id, node_id=request.node_id))
        )
        attempt = self.root / "attempt"
        tracks = attempt / "outputs" / "runs" / name / "tracks"
        (tracks / "track_00").mkdir(parents=True)
        (tracks / "track_02").mkdir(parents=True)
        (tracks / "track_00" / "motion.json").write_text("{}")
        (tracks / "tracks.json").write_text(
            json.dumps({"trackCount": 3, "tracks": [{"track": 0}, {"track": 1}, {"track": 2}]})
        )
        context = SimpleNamespace(
            request=request, attempt=SimpleNamespace(outputs=attempt / "outputs")
        )
        self.assertEqual(
            adapter.branches(context),
            [{"key": "00", "relative_path": f"outputs/runs/{name}/tracks/track_00/motion.json"}],
            "a track without a motion file must not be named; only track 00 wrote one",
        )

    def test_a_stage_writes_its_progress_as_it_goes(self):
        """Python buffers stdout to a file, so a stage printed nothing until it exited.

        A stage working hard and a stage hung on a dead socket then looked identical in the
        log, which is the one thing the log is read for.
        """
        adapter = CommandAdapter(
            lambda context: StageExecution(
                command=(
                    sys.executable,
                    "-c",
                    "import time; print('scoring frame 1'); time.sleep(0.6)",
                ),
                cwd=context.repository,
            )
        )
        runner = self.runner(adapter)
        seen = []
        original = stage_module.beat
        attempt = self.root / "runs/run-1/attempts/example/attempt-1/stdout.log"
        stage_module.beat = lambda: seen.append(attempt.read_text() if attempt.exists() else "")
        self.addCleanup(setattr, stage_module, "beat", original)
        interval = stage_module.HEARTBEAT_SECONDS
        stage_module.HEARTBEAT_SECONDS = 0.2
        self.addCleanup(setattr, stage_module, "HEARTBEAT_SECONDS", interval)

        self.assertEqual(runner.execute(self.request(), "attempt-1").status, "succeeded")
        self.assertTrue(
            any("scoring frame 1" in text for text in seen),
            "the stage's output only arrived after it exited",
        )

    def test_a_stage_missing_its_credential_is_blocked_not_run(self):
        """Running it anyway gets whatever message the script happens to print.

        A Marble poll of an already-generated world failed with "set WLT_API_KEY" and exit 1,
        and read as a failed stage rather than a worker that is missing configuration.
        """
        marker = self.root / "ran"
        adapter = CommandAdapter(
            lambda context: StageExecution(
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
                cwd=context.repository,
                credentials=("WANDER_KEY_THAT_IS_NOT_SET",),
            )
        )
        result = self.runner(adapter).execute(self.request(), "attempt-1")
        self.assertEqual(result.status, "blocked")
        self.assertIn("WANDER_KEY_THAT_IS_NOT_SET", result.error)
        self.assertFalse(marker.exists(), "the stage ran without the credential it declared")

    def test_a_command_that_cannot_be_built_blocks_with_the_reason(self):
        """Raising here left the ledger holding an attempt that said running for ever.

        lhm_frozen asked for a clip it had never declared, so building its command raised
        before anything ran: no command, no logs, no reason, and a node the dashboard showed
        as working.
        """

        def build(context):
            raise ValueError("requires exactly one source artifact")

        result = self.runner(CommandAdapter(build)).execute(self.request(), "attempt-1")
        self.assertEqual(result.status, "blocked")
        self.assertIn("could not be built", result.error)
        self.assertIn("source artifact", result.error)
        attempt = self.root / "runs/run-1/attempts/example/attempt-1"
        self.assertEqual(json.loads((attempt / "result.json").read_text())["status"], "blocked")

    def test_missing_adapter_blocks_without_starting_attempt(self):
        result = self.runner(CommandAdapter(lambda context: None)).execute(
            self.request("missing"), "attempt-1"
        )
        self.assertEqual(result.status, "blocked")
        self.assertIn("no activity adapter", result.error)
        self.assertFalse((self.root / "runs/run-1/attempts").exists())

    def test_paid_policy_denial_is_persisted_without_launching_provider(self):
        marker = self.root / "provider-launched"
        adapter = CommandAdapter(
            lambda context: StageExecution(
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
                cwd=context.repository,
            )
        )
        request = self.request()
        request.definition["retry"] = {"paid": True}
        result = self.runner(adapter, paid_guard=DenyPaidGuard()).execute(request, "attempt-1")
        self.assertEqual(result.status, "blocked")
        self.assertIn("budget exhausted", result.error)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
