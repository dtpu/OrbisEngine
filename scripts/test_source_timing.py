"""CPU-only pixel identity checks for cleaning source provenance."""

import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from wander_worker import source_timing
from wander_worker.source_timing import resample_source, select_provenance

# The fps filter states which frame it read and which one it then wrote. Each
# of those is a single finished log call, so, unlike a composed showinfo report,
# it survives a busy debug log and can referee the module's replayed rule.
_FPS_READ = re.compile(r"\[fps@sample @ [^\]]+\] Read frame with in pts (-?\d+), out pts (-?\d+)$")
_FPS_WRITE = re.compile(r"\[fps@sample @ [^\]]+\] Writing frame with pts (-?\d+) to pts (-?\d+)$")


class SourceTimingTests(unittest.TestCase):
    def make_clip(self, root, rate, count, expression=None, static=False):
        clip = root / "source.mkv"
        # Every decoded frame has a unique, exact pixel identity, independent
        # of timestamp labels. No media persists outside this temporary fixture.
        # A static clip instead repeats one frame, so pixels identify nothing.
        raw = b"".join(
            bytes([0, 0, 0] if static else [index, 255 - index, index // 2]) * 16 * 16
            for index in range(count)
        )
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

    def make_long_vfr_clip(self, root, count):
        """A long, irregularly timed H.264 clip: the shape that broke real runs.

        H.264 decodes on several worker threads, each free to log while the
        showinfo filter is midway through composing one of its own log lines.
        Timestamps sit on a 1/1200 grid with a repeating skew, so they are
        neither constant-rate nor a rounded average frame rate.
        """
        clip = root / "long.mp4"
        raw = b"".join(
            bytes([index % 256, 255 - index % 256, index // 256]) * 16 * 16
            for index in range(count)
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "16x16",
                # A 1/1200 input timebase keeps the skew on exact, whole ticks.
                "-r",
                "1200",
                "-i",
                "-",
                "-vf",
                "setpts=N*20+mod(N\\,7)*2+mod(N\\,3)",
                "-fps_mode",
                "passthrough",
                "-video_track_timescale",
                "1200",
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                str(clip),
            ],
            input=raw,
            capture_output=True,
            check=True,
            timeout=300,
        )
        return clip

    def filter_log_selection(self, clip, fps):
        """What the fps filter itself says it retained, as (source ordinal, out pts).

        This is the reference the module used to parse at runtime. Reading it
        here keeps the replayed rounding rule answerable to the filter even
        where identical pixels leave the module's checksums unable to tell two
        neighbouring source frames apart.
        """
        log = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-nostdin",
                "-loglevel",
                "repeat+debug",
                "-i",
                str(clip),
                "-map",
                "0:v:0",
                "-vf",
                f"fps@sample={fps}",
                "-fps_mode",
                "passthrough",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=300,
        ).stderr.decode("utf-8", errors="replace")
        latest, reads, writes = {}, 0, []
        for line in log.splitlines():
            read, write = _FPS_READ.search(line), _FPS_WRITE.search(line)
            if read:
                latest[int(read.group(2))] = reads
                reads += 1
            elif write:
                writes.append((latest[int(write.group(1))], int(write.group(2))))
        self.assertTrue(writes, "the fps filter reported no retained frames")
        return writes

    def check_selection(self, clip, fps, provenance):
        observed = [(frame["sourceIndex"], frame["resampledPts"]) for frame in provenance["frames"]]
        self.assertEqual(observed, self.filter_log_selection(clip, fps))

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
            timeout=300,
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
            timeout=300,
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
        self.check_selection(clip, fps, provenance)
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
            clip = self.make_clip(Path(root), "60", 60)
            run = source_timing._run

            def damaged_report(command):
                result = run(command)
                for argument in map(str, command):
                    if not argument.endswith("sampled.framecrc"):
                        continue
                    report = Path(argument)
                    rows = report.read_text().splitlines()
                    frames = [number for number, row in enumerate(rows) if not row.startswith("#")]
                    dropped = frames[len(frames) // 2]
                    report.write_text("\n".join(rows[:dropped] + rows[dropped + 1 :]) + "\n")
                return result

            with (
                patch.object(source_timing, "_run", side_effect=damaged_report),
                self.assertRaisesRegex(ValueError, "Incomplete FPS"),
            ):
                resample_source(clip, 12.0)

    def test_selection_does_not_read_the_ffmpeg_debug_log(self):
        """Regression: showinfo log lines are not atomic, so they cannot be parsed.

        FFmpeg composes one showinfo line from several unterminated av_log
        calls. A decoder worker thread logging at -loglevel debug splices its
        message into the middle of that line and moves the frame's checksum
        field onto a line of its own, which is how long clips lost frames.
        """
        with tempfile.TemporaryDirectory() as root:
            clip = self.make_clip(Path(root), "60", 30)
            _, expected = resample_source(clip, 12.0)
            run = source_timing._run
            spliced = (
                b"[showinfo@source @ 0x0] n:   0 pts: 0 pts_time:0 iskey:1 type:B "
                b"nal_unit_type: 1(Coded slice of a non-IDR picture), nal_ref_idc: 2\n"
            )

            def mangled_log(command):
                result = run(command)
                result.stderr = spliced
                return result

            with patch.object(source_timing, "_run", side_effect=mangled_log):
                _, observed = resample_source(clip, 12.0)
            self.assertEqual(observed, expected)

    def test_identical_frames_still_bind_to_the_frame_the_filter_kept(self):
        """Repeated pixels make every checksum equal, so only the rule decides.

        The module verifies its replayed selection against per-frame CRCs, and
        those agree with any neighbour here. The filter's own account still has
        to match, on both a constant and an irregular timeline.
        """
        for expression in (None, "setpts=(3*N+mod(N\\,3))/(60*TB)"):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as root:
                clip = self.make_clip(Path(root), "60", 90, expression, static=True)
                _, provenance = resample_source(clip, 12.0)
                self.assertGreater(len({frame["sourcePts"] for frame in provenance["frames"]}), 1)
                self.check_selection(clip, 12.0, provenance)

    def test_long_vfr_clip_binds_every_retained_frame(self):
        count = 1600
        with tempfile.TemporaryDirectory() as root:
            clip = self.make_long_vfr_clip(Path(root), count)
            provenance = self.check_pixels(clip, 12.0)
            self.assertEqual(provenance["sourceFrameCount"], count)
            indices = [frame["sourceIndex"] for frame in provenance["frames"]]
            self.assertEqual(indices, sorted(indices))
            # The filter stops within one output period of the end, so the last
            # binding sits inside the final group of source frames, not before.
            self.assertGreaterEqual(indices[-1], count - 6)
            self.assertGreater(len(indices), 300)
            retained = [frame["sourcePts"] for frame in provenance["frames"]]
            steps = {b - a for a, b in pairwise(retained) if b != a}
            self.assertGreater(len(steps), 1, "the fixture must be variable frame rate")
            _, repeated = resample_source(clip, 12.0)
            self.assertEqual(repeated["frames"], provenance["frames"])

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
