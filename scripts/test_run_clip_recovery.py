"""No-spend runner recovery and non-resubmitting paid-submission contracts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_clip

from worker.stages import lhm_recovery


class Volume:
    def __init__(self, root):
        self.root = root

    def read_file(self, name):
        yield (self.root / name).read_bytes()


class RunnerRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = object.__new__(run_clip.Pipeline)
        self.pipeline.ctx = self.root / "run"
        self.pipeline.ctx.mkdir()
        self.pipeline.clip = self.root / "source.mov"
        self.pipeline.clip.write_bytes(b"private source")
        (self.pipeline.ctx / "pi3x").mkdir()
        (self.pipeline.ctx / "pi3x/frame_000.ply").write_bytes(b"supported person depth")
        self.pipeline.a = SimpleNamespace(force="lhm_frozen,lhm_motion")
        self.pipeline.state = run_clip.State(self.pipeline.ctx / "state.json")
        self.pipeline.track_or_skip = Mock(return_value={"quality": {"firstSample": 0}})
        self.volume = Volume(self.root / "volume")

    def destination(self, mode, idx=None):
        suffix = "" if idx is None else f"-{idx:02d}"
        return self.pipeline.ctx / f"lhm-{mode}{suffix}"

    def checkpoint(self, mode, *, status="complete"):
        dest = self.destination(mode)
        receipt_path, receipt = lhm_recovery.new_receipt(dest, mode)
        directory, output = lhm_recovery.start_checkpoint(
            self.volume.root, receipt["recoveryId"], mode
        )
        (output / "modal-run.json").write_text(
            json.dumps({"error": "inference failed" if status == "failed" else None})
        )
        (output / "inference.log").write_text("retained generated evidence")
        if mode == "frozen":
            for name in ("person-posed.ply", "canonical-state.pt"):
                (output / name).write_bytes(b"generated output")
            (output / "result.json").write_text("{}")
        else:
            (output / "frame_000.ply").write_bytes(b"generated person")
            (output / "source-poses.pt").write_bytes(b"generated poses")
            missing = [{"sample": 1}] if status == "partial" else []
            (output / "missing-poses.json").write_text(json.dumps(missing))
            (output / "motion.json").write_text(json.dumps({"frames": [{}], "missing": missing}))
            (output / "sequence.json").write_text(
                json.dumps(
                    {
                        "frames": ["frame_000.ply"],
                        "count": 1,
                        "frame_sha256": [lhm_recovery.file_hash(output / "frame_000.ply")],
                        "timestamps": [0],
                        "sourceIndices": [0],
                        "requestedSamples": 2 if missing else 1,
                        "missingPoseSamples": missing,
                        "allRequestedSamplesReconstructed": not missing,
                    }
                )
            )
        lhm_recovery.finish_checkpoint(
            directory, mode=mode, error="failed" if status == "failed" else None
        )
        receipt.update(status="wait_interrupted", functionCallId="fc-expired")
        lhm_recovery.atomic_json(receipt_path, receipt)
        return dest

    def recover_command(self, cmd, log, **kwargs):
        self.assertEqual(cmd[0], run_clip.PY)
        self.assertTrue(cmd[1].endswith("worker/stages/lhm_recovery.py"))
        self.assertNotIn(run_clip.MODAL, cmd)
        self.assertEqual(kwargs["attempts"], 1)
        receipt = Path(cmd[cmd.index("--receipt") + 1])
        destination = Path(cmd[cmd.index("--out") + 1])
        manifest = lhm_recovery.recover_outputs(receipt, self.volume, destination)
        if manifest["status"] != "complete":
            raise RuntimeError("partial/failed outputs recovered")

    def test_complete_interrupted_frozen_and_motion_recover_without_gpu(self):
        for mode in ("frozen", "motion"):
            with self.subTest(mode=mode):
                dest = self.checkpoint(mode)
                with patch.object(run_clip, "run", side_effect=self.recover_command) as invoke:
                    getattr(self.pipeline, "lhm_" + mode)()
                invoke.assert_called_once()
                receipt = json.loads((dest / "recovery-receipt.json").read_text())
                self.assertEqual(receipt["status"], "recovered")
                self.assertEqual(receipt["functionCallId"], "fc-expired")
                evidence = self.pipeline.state.data["stages"]["_lhm_" + mode + "_recovery"]
                self.assertEqual(evidence["status"], "complete")

    def test_partial_motion_and_failed_frozen_remain_blocked_or_failed(self):
        for mode, status, expected in (
            ("motion", "partial", "blocked"),
            ("frozen", "failed", "failed"),
        ):
            with self.subTest(status=status):
                dest = self.checkpoint(mode, status=status)
                with (
                    patch.object(run_clip, "run", side_effect=self.recover_command) as invoke,
                    self.assertRaises(run_clip.QualityStop) as caught,
                ):
                    getattr(self.pipeline, "lhm_" + mode)()
                self.assertEqual(caught.exception.status, expected)
                self.assertEqual(
                    json.loads((dest / "recovery-manifest.json").read_text())["status"], status
                )
                self.assertTrue((dest / "inference.log").is_file())
                invoke.assert_called_once()

    def test_force_preserves_all_legacy_single_and_multiperson_destinations(self):
        for mode in ("frozen", "motion"):
            for idx in (None, 0):
                with self.subTest(mode=mode, idx=idx):
                    dest = self.destination(mode, idx)
                    dest.mkdir()
                    retained = dest / "precious-output.ply"
                    retained.write_bytes(b"already paid")
                    with (
                        patch.object(run_clip, "run") as invoke,
                        self.assertRaisesRegex(run_clip.QualityStop, "explicit inspection"),
                    ):
                        getattr(self.pipeline, "lhm_" + mode)(idx)
                    invoke.assert_not_called()
                    self.assertEqual(retained.read_bytes(), b"already paid")

    def test_unreadable_unknown_and_running_receipts_never_launch_gpu(self):
        dest = self.destination("frozen")
        dest.mkdir()
        receipt = dest / "recovery-receipt.json"
        for content in ("broken json", "[]", '{"mode":"motion"}'):
            receipt.write_text(content)
            with patch.object(run_clip, "run") as invoke, self.assertRaises(run_clip.QualityStop):
                self.pipeline.lhm_frozen()
            invoke.assert_not_called()
            self.assertEqual(receipt.read_text(), content)
        for status in ("waiting", "submission_uncertain", "unknown"):
            content = json.dumps({"mode": "frozen", "status": status})
            receipt.write_text(content)

            def unavailable(cmd, *args, **kwargs):
                self.assertEqual(cmd[0], run_clip.PY)
                raise RuntimeError("manifest unavailable; no resubmission")

            with (
                patch.object(run_clip, "run", side_effect=unavailable) as invoke,
                self.assertRaises(run_clip.QualityStop) as caught,
            ):
                self.pipeline.lhm_frozen()
            self.assertEqual(caught.exception.status, "blocked")
            invoke.assert_called_once()
            self.assertEqual(receipt.read_text(), content)

    def test_symlink_destinations_and_receipts_are_never_followed(self):
        dest = self.destination("frozen")
        dest.symlink_to(self.root, target_is_directory=True)
        with patch.object(run_clip, "run") as invoke, self.assertRaises(run_clip.QualityStop):
            self.pipeline.lhm_frozen()
        invoke.assert_not_called()
        dest.unlink()
        dest.mkdir()
        (dest / "recovery-receipt.json").symlink_to(self.pipeline.clip)
        with patch.object(run_clip, "run") as invoke, self.assertRaises(run_clip.QualityStop):
            self.pipeline.lhm_frozen()
        invoke.assert_not_called()

    def test_zero_recovery_exit_without_complete_evidence_still_blocks(self):
        dest = self.destination("frozen")
        dest.mkdir()
        (dest / "recovery-receipt.json").write_text('{"mode":"frozen","status":"waiting"}')
        with (
            patch.object(run_clip, "run") as invoke,
            self.assertRaisesRegex(run_clip.QualityStop, "complete verified evidence"),
        ):
            self.pipeline.lhm_frozen()
        invoke.assert_called_once()
        self.assertEqual(invoke.call_args.args[0][0], run_clip.PY)

    def test_new_single_person_stages_submit_once_with_existing_arguments(self):
        for mode in ("frozen", "motion"):
            with self.subTest(mode=mode), patch.object(run_clip, "run") as invoke:
                getattr(self.pipeline, "lhm_" + mode)()
                invoke.assert_called_once()
                command = invoke.call_args.args[0]
                self.assertEqual(command[:3], [run_clip.MODAL, "run", "worker/modal_lhm.py"])
                self.assertEqual(command[command.index("--out") + 1], str(self.destination(mode)))
                if mode == "frozen":
                    self.assertIn("--prepared", command)
                else:
                    self.assertIn("--canonical", command)
                    self.assertIn("--cameras", command)
                    self.assertEqual(command[command.index("--video") + 1], str(self.pipeline.clip))

    def test_camera_only_success_cannot_submit_person_registration(self):
        (self.pipeline.ctx / "pi3x/frame_000.ply").unlink()
        with (
            patch.object(run_clip, "run") as invoke,
            self.assertRaisesRegex(run_clip.QualityStop, "camera-only success"),
        ):
            self.pipeline.lhm_motion()
        invoke.assert_not_called()
        self.assertFalse(self.destination("motion").exists())

    def test_forced_scheduler_cannot_erase_legacy_outputs_or_advance_dependents(self):
        dest = self.destination("frozen")
        dest.mkdir()
        (dest / "retained.ply").write_bytes(b"paid artifact")
        self.pipeline.a = SimpleNamespace(
            only=None, force="lhm_frozen", skip_finetune=False, reuse_world=None
        )
        self.pipeline.marble = "none"
        self.pipeline.multi = False
        self.pipeline.stages = ["lhm_frozen", "lhm_motion"]
        self.pipeline.deps = {"lhm_frozen": [], "lhm_motion": ["lhm_frozen"]}
        self.pipeline.state.record("lhm_frozen", status="ok")
        self.pipeline.summary = Mock()
        with patch.object(run_clip, "run") as invoke:
            result = self.pipeline.go()
        self.assertEqual(result, 1)
        self.assertEqual(self.pipeline.state.data["stages"]["lhm_frozen"]["status"], "blocked")
        self.assertEqual(self.pipeline.state.data["stages"]["lhm_motion"]["status"], "failed")
        self.assertEqual((dest / "retained.ply").read_bytes(), b"paid artifact")
        invoke.assert_not_called()


class PaidRunTests(unittest.TestCase):
    def test_paid_modal_run_never_retries_rate_limit_after_completed_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            for command in (
                [run_clip.MODAL, "run", "worker/modal_lhm.py"],
                ["/alternate/bin/modal", "--profile", "dtpu", "run", "worker/modal_motion.py"],
                [run_clip.PY, "-m", "modal", "run", "worker/modal_lhm.py"],
            ):

                def failed(_cmd, **kwargs):
                    kwargs["stdout"].write(
                        "GPU work completed\nrate limit while transferring outputs\n"
                    )
                    return SimpleNamespace(returncode=143)

                with (
                    self.subTest(command=command),
                    patch.object(run_clip.subprocess, "run", side_effect=failed) as process,
                    patch.object(run_clip, "_space_modal_launch"),
                    patch.object(run_clip.time, "sleep") as sleep,
                    self.assertRaises(RuntimeError),
                ):
                    run_clip.run(command, Path(temporary) / "paid.log", attempts=4)
                process.assert_called_once()
                sleep.assert_not_called()

    def test_read_only_recovery_can_use_explicit_bounded_transport_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            count = 0

            def transient(_cmd, **kwargs):
                nonlocal count
                count += 1
                kwargs["stdout"].write("rate limit on volume read\n")
                return SimpleNamespace(returncode=1 if count == 1 else 0)

            with (
                patch.object(run_clip.subprocess, "run", side_effect=transient) as process,
                patch.object(run_clip.time, "sleep"),
            ):
                run_clip.run(
                    [run_clip.PY, "worker/stages/lhm_recovery.py", "--receipt", "receipt.json"],
                    Path(temporary) / "recovery.log",
                    attempts=2,
                )
            self.assertEqual(process.call_count, 2)


if __name__ == "__main__":
    unittest.main()
