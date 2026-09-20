"""Missing-pose repair preserves submission and registration safeguards, without a GPU."""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_multiperson_recovery as existing_tests

from stages import lhm_execution, lhm_recovery
from worker import modal_multiperson
from stage_attempts import command_identity
import run_clip


class MissingPoseRecoveryTests(unittest.TestCase):
    def setUp(self):
        existing_tests.MultipersonRecoveryTests.setUp(self)

    def test_recovery_requires_measured_scale_before_creating_receipt(self):
        with patch.object(modal_multiperson, "animate_track") as function:
            with self.assertRaisesRegex(ValueError, "measured fixed world scale"):
                modal_multiperson.animate(**self.arguments, recover_missing_poses=True)
        self.assertFalse(self.destination.exists())
        function.spawn.assert_not_called()

    def test_recovery_records_intent_and_never_repeats_an_interrupted_submission(self):
        function = Mock()
        function.spawn.return_value.object_id = "fc-missing-pose"
        function.spawn.return_value.get.side_effect = KeyboardInterrupt
        with patch.object(modal_multiperson, "animate_track", function), patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                modal_multiperson.animate(
                    **self.arguments, recover_missing_poses=True, fixed_world_scale=0.2
                )
            with self.assertRaises(FileExistsError):
                modal_multiperson.animate(
                    **self.arguments, recover_missing_poses=True, fixed_world_scale=0.2
                )
        function.spawn.assert_called_once()
        self.assertTrue(function.spawn.call_args.kwargs["recover_missing_poses"])
        receipt = lhm_recovery.load_receipt(self.destination / lhm_recovery.RECEIPT_NAME)
        self.assertTrue(receipt["execution"]["recoverMissingPoses"])
        self.assertEqual(receipt["status"], "wait_interrupted")
        self.assertEqual(receipt["execution"]["flags"][-2:], ["--fixed-world-scale", "0.2"])

    def test_recovery_changes_paid_identity(self):
        root = Path(__file__).resolve().parents[2]
        base = [
            "modal",
            "run",
            "worker/modal_multiperson.py::animate",
            "--fixed-world-scale",
            "0.2",
        ]
        normal, _, _ = command_identity(base, root)
        repair, _, _ = command_identity(base + ["--recover-missing-poses"], root)
        self.assertNotEqual(normal, repair)

    def test_runner_passes_recovery_and_scale_to_saved_track_worker(self):
        pipeline = object.__new__(run_clip.Pipeline)
        pipeline.a = types.SimpleNamespace(recover_missing_poses=True, fixed_world_scale=0.2)
        pipeline.ctx = self.root
        pipeline.clip = self.inputs / "source.mp4"
        pipeline.track_or_skip = Mock(return_value={"quality": {"firstSample": 1}})
        pipeline.recover_lhm = Mock(return_value=False)
        pipeline.paid_run = Mock()
        (self.root / "pi3x").mkdir()
        (self.root / "pi3x/frame_001.ply").write_bytes(b"measured depth")
        track = self.root / "tracks/track_00"
        track.mkdir(parents=True)
        (track / "motion.json").write_text(
            json.dumps({"frames": [{"sample": 1, "maskBox": [10, 20, 30, 40]}]})
        )
        pipeline.lhm_motion(0)
        command = pipeline.paid_run.call_args.args[1]
        self.assertIn("--recover-missing-poses", command)
        self.assertEqual(command[-2:], ["--fixed-world-scale", "0.2"])

    def test_worker_only_enables_estimation_with_explicit_recovery(self):
        for enabled in (False, True):
            with self.subTest(recover_missing_poses=enabled):
                checkpoint = self.root / f"checkpoint-{enabled}"
                output = checkpoint / "output"
                output.mkdir(parents=True)
                inputs = {
                    name: b"fixture"
                    for name in (
                        "source.mp4",
                        "canonical-state.pt",
                        "reference.json",
                        "cameras.json",
                        "depth-reference.ply",
                        "seed-poses.pt",
                        "seed-motion.json",
                    )
                }

                def inspect_command(command, log, **kwargs):
                    self.assertEqual("--track-only" in command, not enabled)
                    self.assertIn("--seed-poses", command)
                    self.assertIn("--seed-motion", command)
                    self.assertEqual(command[-2:], ["--fixed-world-scale", "0.2"])
                    log.write_text("Stopped before inference in offline test")
                    raise RuntimeError("offline test stop")

                with (
                    patch.dict(
                        sys.modules,
                        {
                            "huggingface_hub": types.SimpleNamespace(
                                snapshot_download=Mock(return_value="/fixture/model")
                            )
                        },
                    ),
                    patch.object(
                        lhm_recovery, "start_checkpoint", return_value=(checkpoint, output)
                    ),
                    patch.object(lhm_execution, "link_model_caches", return_value=[]),
                    patch.object(
                        lhm_execution, "run_logged_inference", side_effect=inspect_command
                    ),
                    patch.object(modal_multiperson, "cache", types.SimpleNamespace(commit=Mock())),
                    patch("builtins.print"),
                    self.assertWarnsRegex(UserWarning, "executing locally"),
                ):
                    result = modal_multiperson.animate_track.local(
                        inputs,
                        ["--fixed-world-scale", "0.2"],
                        recovery_id="a" * 31 + str(int(enabled)),
                        recover_missing_poses=enabled,
                    )
                self.assertEqual(result["status"], "failed")
                report = json.loads((output / "modal-run.json").read_text())
                self.assertEqual(report["recoverMissingPoses"], enabled)


if __name__ == "__main__":
    unittest.main()
