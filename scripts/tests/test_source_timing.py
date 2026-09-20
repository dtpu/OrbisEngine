"""CPU-only pixel identity checks for cleaning source provenance."""

import sys
from pathlib import Path

# The modules under test are this directory's parent; importing them by name is what
# running from scripts/ used to give for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "scripts"))
from wander_worker import source_timing
from wander_worker.source_timing import resample_source, select_provenance


class SourceTimingTests(unittest.TestCase):
    def make_clip(self, root, rate, count, expression=None):
        clip = root / "source.mkv"
        # Every decoded frame has a unique, exact pixel identity, independent
        # of timestamp labels. No media persists outside this temporary fixture.
        raw = b"".join(bytes([index, 255 - index, index // 2]) * 16 * 16 for index in range(count))
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "16x16",
            "-r",
            rate,
            "-i",
            "-",
        ]
        if expression:
            command += ["-vf", expression]
        command += ["-fps_mode", "passthrough", "-c:v", "ffv1", str(clip)]
        subprocess.run(command, input=raw, capture_output=True, check=True, timeout=30)
        return clip

    def check_pixels(self, clip, fps):
        sampled, provenance = resample_source(clip, fps, width=16, height=16)
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        legacy = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip),
                "-vf",
                f"fps={fps},scale=16:16",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        self.assertEqual(sampled, legacy)
        frame_size = 16 * 16 * 3
        for frame in provenance["frames"]:
            index = frame["frameIndex"]
            source = frame["sourceIndex"]
            self.assertEqual(
                sampled[index * frame_size : (index + 1) * frame_size],
                decoded[source * frame_size : (source + 1) * frame_size],
            )
        _, mapping_only = resample_source(clip, fps)
        self.assertEqual(mapping_only["frames"], provenance["frames"])
        self.assertEqual(provenance["sourceSha256"], hashlib.sha256(clip.read_bytes()).hexdigest())
        return provenance

    def test_cfr_and_ntsc_downsampling_is_not_rounded_average_fps(self):
        for rate in ("60", "60000/1001"):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as root:
                clip = self.make_clip(Path(root), rate, 120)
                provenance = self.check_pixels(clip, 12.0)
                indices = [frame["sourceIndex"] for frame in provenance["frames"]]
                self.assertEqual(indices[:3], [2, 7, 12])
                self.assertNotEqual(indices[:3], [0, 5, 10])

    def test_vfr_with_nonzero_source_pts_drops_and_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            clip = self.make_clip(
                Path(root),
                "60",
                30,
                "setpts=5/TB+if(lt(N\\,12)\\,N\\,12+(N-12)*5)/(60*TB)",
            )
            provenance = self.check_pixels(clip, 12.0)
            self.assertGreater(provenance["sourceFirstPts"], 0)
            frames = provenance["frames"]
            self.assertGreater(frames[0]["sourceTimeSeconds"], 4.9)
            self.assertLess(frames[0]["sourceRelativeTimeSeconds"], 0.1)
            self.assertLess(len({frame["sourceIndex"] for frame in frames}), 30)

    def test_upsample_duplicates_keep_same_source_binding_and_only_order(self):
        with tempfile.TemporaryDirectory() as root:
            clip = self.make_clip(Path(root), "3", 6)
            provenance = self.check_pixels(clip, 12.0)
            indices = [frame["sourceIndex"] for frame in provenance["frames"]]
            self.assertGreater(len(indices), len(set(indices)))
            selected = select_provenance(provenance, [8, 0, 3])
            self.assertEqual([frame["frameIndex"] for frame in selected], [8, 0, 3])
            self.assertEqual(selected[0], provenance["frames"][8])
            for invalid in ([], [-1], [len(indices)], [1, 1]):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    select_provenance(provenance, invalid)

    def test_worker_partial_report_masks_and_returned_images_keep_bindings(self):
        import modal_clean_video

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = self.make_clip(root, "60", 30)
            _, provenance = resample_source(clip, 12.0)
            count = len(provenance["frames"])
            masks = io.BytesIO()
            np.savez_compressed(masks, masks=np.packbits(np.zeros((count, 16, 16), bool), axis=-1))
            worker_root = root / "worker"
            worker_root.mkdir()
            modules = {
                "cv2": SimpleNamespace(),
                "torch": SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: True), device=str
                ),
                "simple_lama_inpainting": SimpleNamespace(SimpleLama=lambda device: None),
                "wander_worker.masks": SimpleNamespace(moved_content_masks=None, people_masks=None),
            }
            with (
                patch.dict(sys.modules, modules),
                patch.object(modal_clean_video.cache, "commit"),
                patch("tempfile.mkdtemp", return_value=str(worker_root)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = modal_clean_video.clean.local(
                    clip.read_bytes(),
                    fps=12.0,
                    width=16,
                    height=16,
                    only="3,0",
                    return_frames=True,
                    encode=False,
                    masks_npz=masks.getvalue(),
                )
            report = result["report"]
            self.assertIsNone(report["error"])
            self.assertEqual(report["keptIndices"], [3, 0])
            for observed, expected in zip(
                report["keptFrameProvenance"], select_provenance(provenance, [3, 0]), strict=True
            ):
                self.assertEqual({key: observed[key] for key in expected}, expected)
            self.assertEqual(report["sourceProvenance"]["frames"], provenance["frames"])
            with np.load(io.BytesIO(result["masks"])) as saved:
                self.assertEqual(str(saved["source_sha256"]), report["sourceSha256"])
                self.assertEqual(
                    saved["source_pts"].tolist(), [f["sourcePts"] for f in provenance["frames"]]
                )
            with tarfile.open(fileobj=io.BytesIO(result["frames"])) as archive:
                self.assertEqual(archive.getnames(), ["f_0003.png", "f_0000.png"])
                for frame in report["keptFrameProvenance"]:
                    with archive.extractfile(f"f_{frame['frameIndex']:04d}.png") as stream:
                        pixel = Image.open(stream).getpixel((0, 0))
                    self.assertEqual(pixel[0], frame["sourceIndex"])

    def test_nonmonotonic_source_pts_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            clip = self.make_clip(Path(root), "60", 6, "setpts=floor(N/2)/(60*TB)")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                resample_source(clip, 12.0)

    def test_missing_ffmpeg_selection_diagnostics_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            clip = self.make_clip(Path(root), "60", 12)
            run = source_timing._run

            def damaged_log(command):
                result = run(command)
                if command[0] == "ffmpeg" and command[-1] == "-":
                    result.stderr = b"\n".join(
                        line for line in result.stderr.splitlines() if b"Writing frame" not in line
                    )
                return result

            with (
                patch.object(source_timing, "_run", side_effect=damaged_log),
                self.assertRaisesRegex(ValueError, "Incomplete FPS"),
            ):
                resample_source(clip, 12.0)

    def test_runner_exact_selection_blocks_mismatches_before_spend(self):
        import run_clip

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = self.make_clip(root, "60", 30)
            _, provenance = resample_source(clip, 12.0)
            selected = select_provenance(provenance, [3, 0])
            pipeline = object.__new__(run_clip.Pipeline)
            pipeline.ctx = root / "run"
            camera_root = pipeline.ctx / "pi3x"
            camera_root.mkdir(parents=True)
            pipeline.clip = clip
            pipeline.a = SimpleNamespace(fps=12.0, reuse_world=None)
            pipeline.clean_flags = Mock(return_value=[])
            pipeline.paid_run = Mock()
            pipeline.marble_world = Mock()
            pipeline.marble_saved_operation = Mock(return_value=None)
            decision = {
                "mode": "multi-image",
                "frames": [frame["sourceIndex"] for frame in selected],
                "azimuth": [90, 0],
            }
            pipeline.world_mode = Mock(return_value=decision)
            pipeline.mode_json.write_text(json.dumps(decision))
            camera_doc = {
                "sourceSha256": provenance["sourceSha256"],
                "cameras": [
                    {
                        "sourceIndex": frame["sourceIndex"],
                        "time": frame["sourceRelativeTimeSeconds"],
                    }
                    for frame in selected
                ],
            }
            camera_path = camera_root / "cameras.json"
            camera_path.write_text(json.dumps(camera_doc))
            legacy_report = pipeline.ctx / "clean.json"
            legacy_report.write_text('{"indices": [0, 5, 10]}')
            pipeline.clean_multi()
            command = pipeline.paid_run.call_args.args[1]
            self.assertEqual(command[command.index("--only") + 1], "3,0")
            self.assertEqual(legacy_report.read_text(), '{"indices": [0, 5, 10]}')
            outputs = pipeline.ctx / "clean-multi"
            outputs.mkdir()
            for frame in selected:
                path = outputs / f"f_{frame['frameIndex']:04d}.png"
                path.write_bytes(str(frame["sourceIndex"]).encode())
                frame["cleanedImageSha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report = {
                "sourceSha256": provenance["sourceSha256"],
                "sourceProvenance": provenance,
                "keptFrameProvenance": selected,
            }
            (pipeline.ctx / "clean-multi.json").write_text(json.dumps(report))
            pipeline.marble_multi()
            images = pipeline.marble_world.call_args.kwargs["extra"][1:3]
            self.assertTrue(images[0].endswith("f_0003.png:90"))
            self.assertTrue(images[1].endswith("f_0000.png:0"))
            pipeline.marble_world.reset_mock()
            (outputs / "f_0003.png").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "image bytes"):
                pipeline.marble_multi()
            pipeline.marble_world.assert_not_called()
            pipeline.paid_run.reset_mock()
            camera_doc["cameras"][0]["time"] += 0.001
            camera_path.write_text(json.dumps(camera_doc))
            with self.assertRaisesRegex(ValueError, "actual source PTS"):
                pipeline.clean_multi()
            pipeline.paid_run.assert_not_called()
            camera_doc["cameras"][0]["time"] -= 0.001
            camera_doc["sourceSha256"] = "0" * 64
            camera_path.write_text(json.dumps(camera_doc))
            with self.assertRaisesRegex(RuntimeError, "source SHA256"):
                pipeline.clean_multi()
            pipeline.paid_run.assert_not_called()

    def test_runner_existing_world_and_operation_recovery_need_no_clean_inputs(self):
        import run_clip

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = object.__new__(run_clip.Pipeline)
            pipeline.ctx = root
            pipeline.name = "fixture"
            pipeline.key = "synthetic-key-not-used"
            pipeline.state = Mock()
            pipeline.multi_source_selection = Mock(side_effect=AssertionError("must not resample"))
            pipeline.marble_reuse = Mock()
            pipeline.marble_poll = Mock()
            pipeline.marble_saved_operation = Mock(
                side_effect=AssertionError("explicit reuse does not inspect submissions")
            )
            pipeline.a = SimpleNamespace(reuse_world="existing-world")
            with (
                patch.object(run_clip, "MARBLE_DIR", root),
                patch.object(run_clip, "run") as invoke,
            ):
                pipeline.marble_multi()
                pipeline.marble_reuse.assert_called_once_with("multi", "multi")
                pipeline.marble_poll.assert_not_called()
                pipeline.multi_source_selection.assert_not_called()
                invoke.assert_not_called()
                pipeline.a.reuse_world = None
                pipeline.marble_saved_operation = Mock(return_value="known-operation")
                pipeline.marble_multi()
                pipeline.marble_poll.assert_called_once_with("multi", "known-operation", "multi")
                pipeline.multi_source_selection.assert_not_called()
                invoke.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
