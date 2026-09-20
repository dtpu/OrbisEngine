"""Offline durable-output contracts. These tests do not run models or establish visual quality."""

import hashlib
import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
from stages import lhm_recovery as recovery


class DiskVolume:
    """Exercise the installed SDK's read_file streaming interface without provider calls."""

    def __init__(self, root):
        self.root = root
        self.reads = []

    def read_file(self, path):
        self.reads.append(path)
        with (self.root / path).open("rb") as stream:
            while data := stream.read(7):
                yield data


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.volume = DiskVolume(self.root / "volume")
        self.receipt_path, self.receipt = recovery.new_receipt(self.root / "local", "frozen")
        self.checkpoint, self.output = recovery.start_checkpoint(
            self.volume.root, self.receipt["recoveryId"], "frozen"
        )

    def write_frozen(self):
        for name, content in {
            "person-posed.ply": b"generated-ply",
            "canonical-state.pt": b"generated-state",
            "result.json": b'{"prepared":{"time":1.5}}',
            "modal-run.json": b'{"error":null}',
            "inference.log": b"exported person\n",
            "source-pose.json": b"{}",
            "head-input.png": b"derived-face",
        }.items():
            (self.output / name).write_bytes(content)

    def finish(self, **kwargs):
        return recovery.finish_checkpoint(self.checkpoint, mode="frozen", **kwargs)

    def update_manifest(self, mutate):
        path = self.checkpoint / "manifest.json"
        value = json.loads(path.read_text())
        mutate(value)
        recovery.atomic_json(path, value)

    def motion(self, *, missing=False):
        receipt = dict(self.receipt, mode="motion")
        recovery.atomic_json(self.receipt_path, receipt)
        for name in ("source-poses.pt", "inference.log"):
            (self.output / name).write_text("generated")
        (self.output / "modal-run.json").write_text('{"error":null}')
        (self.output / "frame_000.ply").write_bytes(b"generated-frame")
        missing_samples = [{"sample": 1}] if missing else []
        sequence = {
            "frames": ["frame_000.ply"],
            "frame_sha256": [recovery.file_hash(self.output / "frame_000.ply")],
            "count": 1,
            "timestamps": [0],
            "sourceIndices": [0],
            "requestedSamples": 2 if missing else 1,
            "allRequestedSamplesReconstructed": not missing,
            "missingPoseSamples": missing_samples,
        }
        (self.output / "sequence.json").write_text(json.dumps(sequence))
        (self.output / "motion.json").write_text(
            json.dumps({"frames": [{}], "missing": missing_samples})
        )
        (self.output / "missing-poses.json").write_text(json.dumps(missing_samples))
        return sequence

    def test_successful_recovery_is_repeatable_and_hash_verified(self):
        self.write_frozen()
        manifest = self.finish()
        self.assertEqual(manifest["status"], "complete")
        for _ in range(2):
            result = recovery.recover_outputs(self.receipt_path, self.volume)
            self.assertEqual(result["status"], "complete")
        receipt = recovery.load_receipt(self.receipt_path)
        self.assertEqual(receipt["resultStatus"], "complete")
        self.assertIsNone(receipt["functionCallId"])
        for name, item in manifest["files"].items():
            path = self.root / "local" / name
            self.assertEqual(recovery.file_hash(path), item["sha256"])
        self.assertFalse((self.root / "local" / "source.mp4").exists())

    def test_call_identity_is_saved_before_interrupted_wait_and_never_resubmits(self):
        self.write_frozen()
        self.finish()
        call = Mock(object_id="fc-test")

        def interrupt():
            receipt = recovery.load_receipt(self.receipt_path)
            self.assertEqual(receipt["status"], "waiting")
            self.assertEqual(receipt["functionCallId"], "fc-test")
            raise KeyboardInterrupt

        call.get.side_effect = interrupt
        function = Mock()
        function.spawn.return_value = call
        with self.assertRaises(KeyboardInterrupt):
            recovery.submit_once(function, self.receipt_path, {"source.mp4": b"private"})
        self.assertEqual(recovery.load_receipt(self.receipt_path)["status"], "wait_interrupted")
        with self.assertRaisesRegex(ValueError, "cannot resubmit"):
            recovery.submit_once(function, self.receipt_path, {})
        recovered = recovery.recover_outputs(self.receipt_path, self.volume)
        self.assertEqual(recovered["status"], "complete")
        function.spawn.assert_called_once()
        call.get.assert_called_once()

    def test_uncertain_spawn_and_unreadable_receipt_never_retry(self):
        function = Mock()
        function.spawn.side_effect = OSError("connection lost")
        with self.assertRaises(OSError):
            recovery.submit_once(function, self.receipt_path, {})
        self.assertEqual(recovery.load_receipt(self.receipt_path)["status"], "submission_uncertain")
        with self.assertRaises(ValueError):
            recovery.submit_once(function, self.receipt_path, {})
        self.receipt_path.write_text("broken")
        with self.assertRaises(ValueError):
            recovery.submit_once(function, self.receipt_path, {})
        function.spawn.assert_called_once()

    def test_concurrent_submit_has_one_owner_before_spawn(self):
        entered, release = threading.Event(), threading.Event()
        function = Mock()

        def spawn(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("test release timed out")
            raise OSError("provider outcome unknown")

        function.spawn.side_effect = spawn
        with ThreadPoolExecutor(max_workers=2) as pool:
            owner = pool.submit(recovery.submit_once, function, self.receipt_path, {})
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(ValueError, "busy"):
                    recovery.submit_once(function, self.receipt_path, {})
                with self.assertRaisesRegex(ValueError, "busy"):
                    recovery.recover_outputs(self.receipt_path, self.volume)
            finally:
                release.set()
            with self.assertRaises(OSError):
                owner.result(timeout=5)
        function.spawn.assert_called_once()
        self.assertEqual(recovery.load_receipt(self.receipt_path)["status"], "submission_uncertain")
        with self.assertRaisesRegex(ValueError, "cannot resubmit"):
            recovery.submit_once(function, self.receipt_path, {})

    def test_concurrent_recovery_cannot_touch_the_active_download(self):
        self.write_frozen()
        self.finish()
        entered, release = threading.Event(), threading.Event()
        original = self.volume.read_file

        def delayed(path):
            if path.endswith("person-posed.ply"):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("test release timed out")
            yield from original(path)

        with patch.object(self.volume, "read_file", side_effect=delayed):
            with ThreadPoolExecutor(max_workers=2) as pool:
                owner = pool.submit(recovery.recover_outputs, self.receipt_path, self.volume)
                try:
                    self.assertTrue(entered.wait(5))
                    active = list((self.root / "local").glob("*.part"))
                    self.assertEqual(len(active), 1)
                    with self.assertRaisesRegex(ValueError, "busy"):
                        recovery.recover_outputs(self.receipt_path, self.volume)
                    self.assertTrue(active[0].exists())
                finally:
                    release.set()
                self.assertEqual(owner.result(timeout=5)["status"], "complete")
        self.assertFalse(list((self.root / "local").glob("*.part")))
        self.assertEqual(
            recovery.recover_outputs(self.receipt_path, self.volume)["status"], "complete"
        )

    def test_return_receipt_pins_manifest_and_excludes_large_payload(self):
        self.write_frozen()
        self.finish()
        function = Mock()
        function.spawn.return_value.object_id = "fc-test"
        function.spawn.return_value.get.return_value = {
            "recoveryId": self.receipt["recoveryId"],
            "remotePath": self.receipt["remotePath"],
            "manifestSha256": recovery.file_hash(self.checkpoint / "manifest.json"),
            "totalWorkerSeconds": 12,
        }
        recovery.submit_once(function, self.receipt_path, {})
        self.update_manifest(lambda value: value.update(finishedAtEpoch=1))
        with self.assertRaisesRegex(ValueError, "differs from completed"):
            recovery.recover_outputs(self.receipt_path, self.volume)

    def test_running_or_absent_checkpoint_is_not_relabelled_complete(self):
        with self.assertRaisesRegex(ValueError, "running or incomplete"):
            recovery.recover_outputs(self.receipt_path, self.volume)
        (self.checkpoint / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "do not resubmit"):
            recovery.recover_outputs(self.receipt_path, self.volume)

    def test_failed_and_partial_outputs_are_recovered_with_explicit_status(self):
        (self.output / "inference.log").write_text("model failed after partial export")
        (self.output / "modal-run.json").write_text('{"error":"model failed"}')
        self.finish(error="model failed")
        result = recovery.recover_outputs(self.receipt_path, self.volume)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["inferenceError"])
        self.assertTrue(result["problems"])
        self.assertEqual(recovery.load_receipt(self.receipt_path)["resultStatus"], "failed")

    def test_motion_missing_samples_are_partial_even_with_successful_subprocess(self):
        self.motion(missing=True)
        manifest = recovery.finish_checkpoint(self.checkpoint, mode="motion")
        self.assertEqual(manifest["status"], "partial")
        result = recovery.recover_outputs(self.receipt_path, self.volume)
        self.assertEqual(result["status"], "partial")
        self.assertTrue((self.root / "local" / "frame_000.ply").exists())

    def test_complete_motion_covers_all_declared_frame_hashes(self):
        self.motion()
        self.assertEqual(
            recovery.finish_checkpoint(self.checkpoint, mode="motion")["status"], "complete"
        )
        (self.output / "frame_001.ply").write_bytes(b"unlisted")
        self.assertEqual(
            recovery.finish_checkpoint(self.checkpoint, mode="motion")["status"], "partial"
        )
        (self.output / "frame_001.ply").unlink()
        (self.output / "frame_000.ply").write_bytes(b"changed")
        self.assertEqual(
            recovery.finish_checkpoint(self.checkpoint, mode="motion")["status"], "partial"
        )

    def test_missing_registration_and_malformed_metadata_are_partial(self):
        self.motion()
        manifest = recovery.finish_checkpoint(
            self.checkpoint, mode="motion", require_registration=True
        )
        self.assertEqual(manifest["status"], "partial")
        (self.output / "registration.json").write_text("{}")
        (self.output / "sequence.json").write_text("[]")
        self.assertEqual(
            recovery.finish_checkpoint(self.checkpoint, mode="motion")["status"], "partial"
        )

    def test_original_inputs_and_path_traversal_are_rejected(self):
        for name in (
            "../escape",
            "/absolute",
            "x/../escape",
            "x//y",
            "source.mp4",
            "clip.mov",
            "source.png",
            "recovery-receipt.json",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                recovery.safe_relative(name)
        self.write_frozen()
        (self.output / "source.mp4").write_bytes(b"private footage")
        with self.assertRaisesRegex(ValueError, "original-input"):
            self.finish()
        self.assertFalse(json.loads((self.checkpoint / "manifest.json").read_text())["finalized"])

    def test_tampered_remote_bytes_and_local_conflicts_are_not_overwritten(self):
        self.write_frozen()
        self.finish()
        (self.output / "person-posed.ply").write_bytes(b"tampered-ply")
        with self.assertRaisesRegex(ValueError, "size/hash"):
            recovery.recover_outputs(self.receipt_path, self.volume)
        self.assertFalse((self.root / "local" / "person-posed.ply").exists())
        (self.output / "person-posed.ply").write_bytes(b"generated-ply")
        (self.root / "local" / "person-posed.ply").write_bytes(b"unrelated local output")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            recovery.recover_outputs(self.receipt_path, self.volume)
        self.assertEqual(
            (self.root / "local" / "person-posed.ply").read_bytes(), b"unrelated local output"
        )

    def test_interrupted_download_resumes_without_a_function_call(self):
        self.write_frozen()
        self.finish()
        original = self.volume.read_file

        def disconnect(path):
            if path.endswith("person-posed.ply"):
                yield b"gener"
                raise OSError("interrupted transfer")
            yield from original(path)

        with (
            patch.object(self.volume, "read_file", side_effect=disconnect),
            self.assertRaises(OSError),
        ):
            recovery.recover_outputs(self.receipt_path, self.volume)
        manifest = recovery.recover_outputs(self.receipt_path, self.volume)
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(list((self.root / "local").glob("*.part")))

    def test_manifest_paths_identity_and_completion_cannot_be_forged(self):
        self.write_frozen()
        original = self.finish()
        for change in (
            {"recoveryId": "f" * 32},
            {"status": "complete", "inferenceError": True},
            {"status": "complete", "problems": ["missing frame"]},
            {"files": {"../escape": {"bytes": 1, "sha256": "a" * 64}}},
        ):
            with self.subTest(change=change):
                recovery.atomic_json(self.checkpoint / "manifest.json", dict(original, **change))
                with self.assertRaises(ValueError):
                    recovery.recover_outputs(self.receipt_path, self.volume)
        forged = dict(original)
        forged["files"] = {
            key: value for key, value in original["files"].items() if key != "person-posed.ply"
        }
        recovery.atomic_json(self.checkpoint / "manifest.json", forged)
        with self.assertRaisesRegex(ValueError, "claimed complete"):
            recovery.recover_outputs(self.receipt_path, self.volume)

    def test_symlinks_and_duplicate_checkpoint_or_destination_fail_closed(self):
        self.write_frozen()
        (self.output / "linked.ply").symlink_to(self.output / "person-posed.ply")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.finish()
        (self.output / "linked.ply").unlink()
        self.finish()
        (self.root / "local" / "person-posed.ply").symlink_to(self.output / "person-posed.ply")
        with self.assertRaisesRegex(ValueError, "symlink"):
            recovery.recover_outputs(self.receipt_path, self.volume)
        with self.assertRaises(FileExistsError):
            recovery.new_receipt(self.root / "local", "frozen")
        with self.assertRaises(FileExistsError):
            recovery.start_checkpoint(self.volume.root, self.receipt["recoveryId"], "frozen")

    def test_atomic_receipt_update_survives_an_unrelated_stale_temporary(self):
        self.receipt_path.with_name(self.receipt_path.name + ".tmp").write_text("interrupted")
        recovery.atomic_json(self.receipt_path, dict(self.receipt, status="waiting"))
        self.assertEqual(recovery.load_receipt(self.receipt_path)["status"], "waiting")

    def test_normal_cli_keeps_prepared_fixed_inputs_and_local_archive(self):
        import tarfile

        from worker import modal_lhm

        prepared = self.root / "prepared"
        fixed = self.root / "fixed"
        prepared.mkdir()
        fixed.mkdir()
        for name in ("source.png", "mask.png", "prepared.json"):
            (prepared / name).write_bytes(b"private prepared input")
        for name in ("source-pose.pt", "source-pose.json", "head-input.png"):
            (fixed / name).write_bytes(b"derived fixed input")
        function = Mock()

        def spawn(inputs, **kwargs):
            self.assertEqual(inputs["source.png"], b"private prepared input")
            self.assertEqual(inputs["source-pose.pt"], b"derived fixed input")
            self.assertTrue(kwargs["larger_model"])
            self.assertFalse(kwargs["animate"])
            directory, output = recovery.start_checkpoint(
                self.volume.root, kwargs["recovery_id"], "frozen"
            )
            for name, data in {
                "person-posed.ply": b"generated person",
                "canonical-state.pt": b"generated state",
                "result.json": b"{}",
                "modal-run.json": b'{"error":null}',
                "inference.log": b"exported person",
            }.items():
                (output / name).write_bytes(data)
            recovery.finish_checkpoint(directory, mode="frozen")
            result = {
                "recoveryId": kwargs["recovery_id"],
                "remotePath": recovery.remote_root(kwargs["recovery_id"]),
                "manifestSha256": recovery.file_hash(directory / "manifest.json"),
                "report": {"error": None},
            }
            return types.SimpleNamespace(object_id="fc-test", get=lambda: result)

        function.spawn.side_effect = spawn
        destination = self.root / "cli-output"
        with (
            patch.object(modal_lhm, "frozen", function),
            patch.object(modal_lhm, "cache", self.volume),
            patch("builtins.print"),
        ):
            modal_lhm.main(
                prepared=str(prepared),
                out=str(destination),
                fixed_inputs=str(fixed),
                larger_model=True,
            )
        with tarfile.open(destination / "artifacts.tar.gz") as archive:
            self.assertIn("person-posed.ply", archive.getnames())
            self.assertNotIn("source.png", archive.getnames())
            self.assertNotIn("prepared.json", archive.getnames())
        self.assertTrue((destination / "recovery-receipt.json").is_file())
        function.spawn.assert_called_once()

    def test_canonical_cli_retains_camera_seed_inputs_and_interruption_receipt(self):
        from worker import modal_lhm

        folder = self.root / "canonical"
        folder.mkdir()
        for name in (
            "canonical-state.pt",
            "result.json",
            "cameras.json",
            "frame_000.ply",
            "source-poses.pt",
            "motion.json",
        ):
            (folder / name).write_bytes(name.encode())
        video = self.root / "private.mov"
        video.write_bytes(b"original source")
        function = Mock()
        function.spawn.return_value.object_id = "fc-canonical"
        function.spawn.return_value.get.side_effect = KeyboardInterrupt
        destination = self.root / "motion-output"
        with (
            patch.object(modal_lhm, "frozen", function),
            patch("builtins.print"),
            self.assertRaises(KeyboardInterrupt),
        ):
            modal_lhm.main(
                canonical=str(folder),
                video=str(video),
                cameras=str(folder / "cameras.json"),
                seed=str(folder),
                out=str(destination),
            )
        inputs = function.spawn.call_args.args[0]
        self.assertEqual(inputs["source.mp4"], b"original source")
        self.assertEqual(inputs["depth-reference.ply"], b"frame_000.ply")
        self.assertEqual(inputs["seed-poses.pt"], b"source-poses.pt")
        self.assertTrue(function.spawn.call_args.kwargs["animate"])
        receipt = recovery.load_receipt(destination / recovery.RECEIPT_NAME)
        self.assertEqual(receipt["functionCallId"], "fc-canonical")
        self.assertEqual(receipt["status"], "wait_interrupted")
        self.assertEqual(
            sorted(path.name for path in destination.iterdir()),
            [recovery.RECEIPT_NAME, recovery.RECEIPT_NAME + ".lock"],
        )

    def test_worker_failure_commits_generated_evidence_before_small_return(self):
        from worker import modal_lhm

        commits = []
        real_start = recovery.start_checkpoint
        checkpoint_root = self.root / "worker-volume"

        def start(_cache, recovery_id, mode):
            return real_start(checkpoint_root, recovery_id, mode)

        def commit():
            manifest = checkpoint_root / self.receipt["remotePath"] / "manifest.json"
            commits.append(json.loads(manifest.read_text()))

        cache = types.SimpleNamespace(commit=commit)
        hub = types.SimpleNamespace(snapshot_download=Mock())
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub}),
            patch.object(recovery, "start_checkpoint", side_effect=start),
            patch.object(modal_lhm, "cache", cache),
            patch.object(
                modal_lhm, "link_model_caches", side_effect=RuntimeError("missing model cache")
            ),
            self.assertWarnsRegex(UserWarning, "executing locally"),
            patch("builtins.print"),
        ):
            result = modal_lhm.frozen.local(
                {"source.mp4": b"private source"},
                animate=True,
                recovery_id=self.receipt["recoveryId"],
            )
        self.assertEqual([item["status"] for item in commits], ["running", "failed"])
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("archive", result)
        self.assertLess(len(json.dumps(result)), 4096)
        durable = checkpoint_root / self.receipt["remotePath"] / "output"
        self.assertTrue((durable / "inference.log").is_file())
        self.assertTrue((durable / "modal-run.json").is_file())
        self.assertFalse((durable / "source.mp4").exists())
        hub.snapshot_download.assert_not_called()
        manifest_bytes = (durable.parent / "manifest.json").read_bytes()
        self.assertEqual(result["manifestSha256"], hashlib.sha256(manifest_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
