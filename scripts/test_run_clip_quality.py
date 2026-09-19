"""Offline runner acceptance, resume and no-submission regressions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_clip
import world_quality_review


class RunnerQualityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = object.__new__(run_clip.Pipeline)
        p = self.pipeline
        p.ctx = self.root / "run"
        p.ctx.mkdir()
        p.name = "fixture"
        p.clip = self.root / "clip.mp4"
        p.clip.write_bytes(b"source")
        p.state = run_clip.State(p.ctx / "state.json")
        p.state.record("_scale", scale0=1.0)
        p.a = SimpleNamespace(
            only=None,
            force=None,
            skip_finetune=False,
            reuse_world=None,
            quality_plan=None,
            quality_result=None,
            quality_evidence_root=None,
        )
        p.world_spz = Mock(return_value=self.root / "world.spz")
        p.cameras = Mock(return_value=self.root / "cameras.json")
        p.stages = ["verify"]
        p.deps = {"verify": []}
        p.marble = "video"
        p.multi = False
        p.summary = Mock()
        p.vacant = Mock(return_value=False)
        self.root_patch = patch.object(run_clip, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def review_args(self):
        p = self.pipeline
        p.a.quality_plan = str(self.root / "plan.json")
        p.a.quality_result = str(self.root / "result.json")
        p.a.quality_evidence_root = str(self.root / "evidence")

    def test_default_diagnostics_cannot_pass_and_never_retry(self):
        p = self.pipeline
        with (
            patch.object(run_clip, "run") as invoke,
            self.assertRaises(run_clip.QualityStop) as error,
        ):
            p.verify()
        self.assertEqual(error.exception.status, "blocked")
        command = invoke.call_args.args[0]
        self.assertIn("--no-vlm", command)
        self.assertEqual(invoke.call_args.kwargs["attempts"], 1)
        self.assertEqual(command[1], "scripts/verify_world.py")
        self.assertNotEqual(
            invoke.call_args.args[1].parent, Path(command[command.index("--out") + 1])
        )
        self.assertFalse(p.state.data["stages"]["_verify"]["passed"])

    def test_new_diagnostics_preserve_previous_report(self):
        p = self.pipeline
        old = p.ctx / "verify" / "report.json"
        old.parent.mkdir()
        old.write_text('{"retained": true}')
        destinations = []
        for _ in range(2):
            with patch.object(run_clip, "run") as invoke, self.assertRaises(run_clip.QualityStop):
                p.verify()
            command = invoke.call_args.args[0]
            destinations.append(command[command.index("--out") + 1])
        self.assertNotEqual(*destinations)
        self.assertEqual(json.loads(old.read_text()), {"retained": True})

    def test_partial_review_arguments_block_without_renderer(self):
        self.pipeline.a.quality_plan = "plan.json"
        with patch.object(run_clip, "run") as invoke, self.assertRaises(run_clip.QualityStop):
            self.pipeline.verify()
        invoke.assert_not_called()

    def test_import_pass_fail_and_missing_verdict_propagate_without_render_or_provider(self):
        self.review_args()
        for status in ("passed", "failed", "blocked"):
            with self.subTest(status=status), patch.object(run_clip, "run") as invoke:
                with patch.object(
                    world_quality_review,
                    "assess",
                    return_value={
                        "status": status,
                        "errors": [] if status == "passed" else ["rejected evidence"],
                    },
                ) as assess:
                    if status == "passed":
                        self.pipeline.verify()
                    else:
                        with self.assertRaises(run_clip.QualityStop) as error:
                            self.pipeline.verify()
                        self.assertEqual(error.exception.status, status)
                invoke.assert_not_called()
                self.assertEqual(assess.call_args.kwargs["clip"], self.pipeline.clip)
                self.assertEqual(
                    self.pipeline.state.data["stages"]["_verify"]["passed"], status == "passed"
                )

    def test_invalid_registration_blocks_even_prior_success(self):
        for scale in (None, True, -1, 0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                self.pipeline.state.record("_scale", scale0=scale)
                self.pipeline.state.record("_verify", passed=True)
                with (
                    patch.object(run_clip, "run") as invoke,
                    self.assertRaises(run_clip.QualityStop),
                ):
                    self.pipeline.verify()
                invoke.assert_not_called()
                self.assertFalse(self.pipeline.state.data["stages"]["_verify"]["passed"])

    def test_cached_success_always_reimports_and_new_failure_blocks(self):
        p = self.pipeline
        self.review_args()
        for status, code in (("passed", 0), ("blocked", 1), ("failed", 1)):
            p.state.record("verify", status="ok")
            with (
                patch.object(
                    world_quality_review,
                    "assess",
                    return_value={
                        "status": status,
                        "errors": ["evidence changed"] if code else [],
                    },
                ) as assess,
                patch.object(run_clip, "run") as invoke,
            ):
                self.assertEqual(p.go(), code)
            assess.assert_called_once()
            invoke.assert_not_called()
            self.assertEqual(
                p.state.data["stages"]["verify"]["status"], "ok" if not code else status
            )

    def test_real_importer_rejects_mutated_world_on_cached_resume(self):
        from test_world_quality_review import create_fixture

        kwargs, _ = create_fixture(self.root / "bound-review")
        p = self.pipeline
        p.clip = kwargs["clip"]
        p.world_spz.return_value = kwargs["world"]
        p.cameras.return_value = kwargs["cameras"]
        p.a.quality_plan = str(kwargs["plan_path"])
        p.a.quality_result = str(kwargs["result_path"])
        p.a.quality_evidence_root = str(kwargs["evidence_root"])
        with (
            patch.object(world_quality_review, "source_timestamps", return_value=[0, 1, 2]),
            patch.object(
                world_quality_review,
                "source_frame_png",
                side_effect=lambda clip, index: f"source-{index}".encode(),
            ),
            patch.object(run_clip, "run") as invoke,
        ):
            self.assertEqual(p.go(), 0)
            kwargs["world"].write_bytes(b"different world")
            self.assertEqual(p.go(), 1)
        invoke.assert_not_called()
        self.assertFalse(p.state.done("verify"))
        self.assertFalse(p.state.data["stages"]["_verify"]["passed"])

    def test_review_cannot_bypass_verify_via_only(self):
        p = self.pipeline
        self.review_args()
        p.stages.append("package")
        p.deps["package"] = []
        p.a.only = "package"
        p.state.record("verify", status="ok")
        with patch.object(p, "stage_fn") as stage:
            self.assertEqual(p.go(), 1)
        stage.assert_not_called()
        self.assertFalse(p.state.data["stages"]["_verify"]["passed"])
        self.assertFalse(p.state.done("verify"))

    def test_skipped_review_invalidates_cached_subset_and_no_world_graph(self):
        for no_world in (False, True):
            with self.subTest(no_world=no_world):
                p = self.pipeline
                p.stages = ["package"] if no_world else ["package", "verify"]
                p.deps = {stage: [] for stage in p.stages}
                p.a.only = "package"
                p.state.record("verify", status="ok")
                p.state.record("_verify", passed=True)
                p.state.record("package", status="ok")
                with patch.object(p, "stage_fn") as stage:
                    self.assertEqual(p.go(), 0)
                stage.assert_not_called()
                self.assertFalse(p.state.done("verify"))
                self.assertFalse(p.state.data["stages"]["_verify"]["passed"])

    def test_forced_upstream_invalidates_prior_acceptance_without_paid_retry(self):
        p = self.pipeline
        p.stages.append("package")
        p.deps["package"] = []
        p.a.only = p.a.force = "package"
        p.state.record("verify", status="ok")
        p.state.record("_verify", passed=True)
        work = Mock()
        with (
            patch.object(p, "stage_fn", return_value=work),
            patch.object(run_clip, "run") as invoke,
        ):
            self.assertEqual(p.go(), 0)
        work.assert_called_once()
        invoke.assert_not_called()
        self.assertFalse(p.state.data["stages"]["_verify"]["passed"])
        self.assertFalse(p.state.done("verify"))


if __name__ == "__main__":
    unittest.main()
