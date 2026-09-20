"""Offline saved-actor execution tests; no model downloads or provider calls."""

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from stage_attempts import command_identity
from stages import lhm_execution, lhm_recovery

from worker import modal_multiperson


class DiskVolume:
    def __init__(self, root):
        self.root = root

    def read_file(self, name):
        yield (self.root / name).read_bytes()


class MultipersonRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        for name in (
            "source.mp4",
            "canonical-state.pt",
            "result.json",
            "cameras.json",
            "frame_000.ply",
            "source-poses.pt",
            "motion.json",
        ):
            (self.inputs / name).write_bytes(b"{}" if name.endswith(".json") else name.encode())
        self.destination = self.root / "output"
        self.arguments = {
            "video": str(self.inputs / "source.mp4"),
            "canonical": str(self.inputs),
            "cameras": str(self.inputs / "cameras.json"),
            "track_dir": str(self.inputs),
            "out": str(self.destination),
        }

    def test_invalid_scale_timeout_fail_before_destination_or_submission(self):
        for value in (0, -1, 10, float("inf"), float("nan"), True):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                modal_multiperson.animate(**self.arguments, fixed_world_scale=value)
        for value in (-1, 3601, 1.5, True):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                modal_multiperson.animate(**self.arguments, execution_timeout=value)
        self.assertFalse(self.destination.exists())

    def test_fixed_scale_and_timeout_have_paid_command_identity(self):
        root = Path(__file__).resolve().parents[1]
        base = ["modal", "run", "worker/modal_multiperson.py::animate"]
        first, code, _ = command_identity(
            base + ["--fixed-world-scale", "1.9", "--execution-timeout", "600"], root
        )
        changed, _, _ = command_identity(
            base + ["--fixed-world-scale", "1.8", "--execution-timeout", "600"], root
        )
        self.assertNotEqual(first, changed)
        self.assertEqual(first["options"]["--execution-timeout"], 600)
        self.assertIn("worker/stages/lhm_recovery.py", code)
        self.assertIn("worker/stages/lhm_execution.py", code)
        for flag, value in (
            ("--fixed-world-scale", "nan"),
            ("--fixed-world-scale", "10"),
            ("--execution-timeout", "600.1"),
        ):
            with self.assertRaises(ValueError):
                command_identity(base + [flag, value], root)

    def test_interrupted_wait_records_identity_and_repeat_cli_never_resubmits(self):
        function = Mock()
        bounded = function.with_options.return_value
        call = bounded.spawn.return_value
        call.object_id = "fc-retained-track"

        def interrupt():
            receipt = lhm_recovery.load_receipt(self.destination / lhm_recovery.RECEIPT_NAME)
            self.assertEqual(receipt["status"], "waiting")
            self.assertEqual(receipt["functionCallId"], call.object_id)
            self.assertEqual(
                receipt["execution"]["inputSha256"]["source.mp4"],
                hashlib.sha256(b"source.mp4").hexdigest(),
            )
            raise KeyboardInterrupt

        call.get.side_effect = interrupt
        with patch.object(modal_multiperson, "animate_track", function), patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                modal_multiperson.animate(
                    **self.arguments, fixed_world_scale=1.9, execution_timeout=600
                )
            with self.assertRaises(FileExistsError):
                modal_multiperson.animate(
                    **self.arguments, fixed_world_scale=1.9, execution_timeout=600
                )
        function.with_options.assert_called_once_with(
            timeout=600, cpu=(4, 4), memory=(65536, 65536), retries=0, max_containers=1
        )
        bounded.spawn.assert_called_once()
        self.assertEqual(
            bounded.spawn.call_args.kwargs["flags"],
            ["--track-id", "0", "--fixed-world-scale", "1.9"],
        )
        self.assertEqual(bounded.spawn.call_args.kwargs["execution_timeout"], 600)
        self.assertEqual(
            lhm_recovery.load_receipt(self.destination / lhm_recovery.RECEIPT_NAME)["status"],
            "wait_interrupted",
        )

    def test_uncertain_submission_is_not_retried_and_default_has_no_resource_override(self):
        function = Mock()
        function.spawn.side_effect = OSError("connection lost after dispatch")
        with patch.object(modal_multiperson, "animate_track", function), patch("builtins.print"):
            with self.assertRaises(OSError):
                modal_multiperson.animate(**self.arguments)
            with self.assertRaises(FileExistsError):
                modal_multiperson.animate(**self.arguments)
        function.with_options.assert_not_called()
        function.spawn.assert_called_once()
        self.assertEqual(function.spawn.call_args.kwargs["flags"], ["--track-id", "0"])
        receipt = lhm_recovery.load_receipt(self.destination / lhm_recovery.RECEIPT_NAME)
        self.assertEqual(receipt["status"], "submission_uncertain")
        self.assertEqual(receipt["execution"]["timeoutSeconds"], 3600)

    def test_subprocess_timeout_kills_writer_and_preserves_its_generated_output(self):
        target = self.root / "generated.ply"
        log = self.root / "inference.log"
        code = "from pathlib import Path; import time; p=Path(__import__('sys').argv[1]); p.write_bytes(b'partial'); print('started', flush=True); time.sleep(5); p.write_bytes(b'late')"
        with self.assertRaises(subprocess.TimeoutExpired):
            lhm_execution.run_logged_inference(
                [sys.executable, "-c", code, str(target)],
                log,
                started=time.monotonic(),
                execution_timeout=1,
            )
        self.assertEqual(target.read_bytes(), b"partial")
        self.assertIn("started", log.read_text())
        with self.assertRaises(TimeoutError):
            lhm_execution.run_logged_inference(
                [sys.executable, "-c", "raise Exception"],
                log,
                started=time.monotonic() - 2,
                execution_timeout=1,
            )

    def test_partial_cli_recovers_and_archives_once_then_refuses_resubmission(self):
        volume = DiskVolume(self.root / "volume")
        function = Mock()

        def spawn(inputs, **kwargs):
            directory, output = lhm_recovery.start_checkpoint(
                volume.root, kwargs["recovery_id"], "motion"
            )
            (output / "frame_000.ply").write_bytes(b"saved partial frame")
            (output / "modal-run.json").write_text('{"error":null}')
            (output / "inference.log").write_text("one saved pose completed")
            manifest = lhm_recovery.finish_checkpoint(directory, mode="motion")
            self.assertEqual(manifest["status"], "partial")
            result = {
                "recoveryId": kwargs["recovery_id"],
                "remotePath": lhm_recovery.remote_root(kwargs["recovery_id"]),
                "manifestSha256": lhm_recovery.file_hash(directory / "manifest.json"),
                "report": {"error": None},
            }
            return types.SimpleNamespace(object_id="fc-partial", get=lambda: result)

        function.spawn.side_effect = spawn
        with (
            patch.object(modal_multiperson, "animate_track", function),
            patch.object(modal_multiperson, "cache", volume),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "partial/failed"):
                modal_multiperson.animate(**self.arguments)
            with self.assertRaises(FileExistsError):
                modal_multiperson.animate(**self.arguments)
        function.spawn.assert_called_once()
        self.assertEqual((self.destination / "frame_000.ply").read_bytes(), b"saved partial frame")
        self.assertTrue((self.destination / "artifacts.tar.gz").is_file())
        receipt = lhm_recovery.load_receipt(self.destination / lhm_recovery.RECEIPT_NAME)
        self.assertEqual(receipt["status"], "recovered")
        self.assertEqual(receipt["resultStatus"], "partial")
        (self.destination / "frame_000.ply").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            lhm_recovery.recover_outputs(self.destination / lhm_recovery.RECEIPT_NAME, volume)
        function.spawn.assert_called_once()

    def test_cache_links_use_only_staged_trees_and_reject_conflicts(self):
        repo, cache, package = self.root / "repo", self.root / "cache", self.root / "gfpgan"
        repo.mkdir()
        package.mkdir()
        for name in ("data/pretrained_models", "data/gfpgan", "gfpgan/weights"):
            (cache / name).mkdir(parents=True)
        (cache / "gfpgan/weights/GFPGANv1.3.pth").write_bytes(b"staged model")
        links = lhm_execution.link_model_caches(repo, cache, package)
        self.assertEqual(len(links), 3)
        self.assertEqual((repo / "gfpgan").resolve(), (cache / "data/gfpgan").resolve())
        self.assertEqual((package / "weights/GFPGANv1.3.pth").read_bytes(), b"staged model")
        (repo / "gfpgan").unlink()
        (repo / "gfpgan").mkdir()
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            lhm_execution.link_model_caches(repo, cache, package)

    def test_worker_timeout_commits_partial_generated_outputs_and_hash_verified_recovery(self):
        volume = DiskVolume(self.root / "volume")
        receipt_path, receipt = lhm_recovery.new_receipt(self.destination, "motion")
        real_start = lhm_recovery.start_checkpoint
        commits = []

        def start(_cache, recovery_id, mode):
            return real_start(volume.root, recovery_id, mode)

        def commit():
            commits.append(
                json.loads((volume.root / receipt["remotePath"] / "manifest.json").read_text())
            )

        def timeout(command, log, **kwargs):
            self.assertEqual(kwargs["execution_timeout"], 600)
            self.assertEqual(kwargs["cwd"], "/opt/lhm")
            self.assertIn("--track-only", command)
            self.assertEqual(command[-2:], ["--fixed-world-scale", "1.9"])
            (log.parent / "frame_000.ply").write_bytes(b"generated before timeout")
            log.write_text("started saved-pose animation\n")
            raise subprocess.TimeoutExpired(command, 540)

        inputs = {
            name: name.encode()
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
        hub = types.SimpleNamespace(snapshot_download=Mock(return_value="/cache/model"))
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub}),
            patch.object(lhm_recovery, "start_checkpoint", side_effect=start),
            patch.object(lhm_execution, "link_model_caches", return_value=[]),
            patch.object(lhm_execution, "run_logged_inference", side_effect=timeout),
            patch.object(modal_multiperson, "cache", types.SimpleNamespace(commit=commit)),
            patch("builtins.print"),
            self.assertWarnsRegex(UserWarning, "executing locally"),
        ):
            result = modal_multiperson.animate_track.local(
                inputs,
                ["--fixed-world-scale", "1.9"],
                recovery_id=receipt["recoveryId"],
                execution_timeout=600,
            )
        self.assertEqual([m["status"] for m in commits], ["running", "failed"])
        self.assertNotIn("archive", result)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(hub.snapshot_download.call_args.kwargs["local_files_only"])
        manifest = lhm_recovery.recover_outputs(receipt_path, volume)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(
            (self.destination / "frame_000.ply").read_bytes(), b"generated before timeout"
        )
        self.assertFalse((self.destination / "source.mp4").exists())
        self.assertEqual(
            manifest["files"]["frame_000.ply"]["sha256"],
            lhm_recovery.file_hash(self.destination / "frame_000.ply"),
        )
        self.assertEqual(
            result["manifestSha256"],
            lhm_recovery.file_hash(volume.root / receipt["remotePath"] / "manifest.json"),
        )


if __name__ == "__main__":
    unittest.main()
