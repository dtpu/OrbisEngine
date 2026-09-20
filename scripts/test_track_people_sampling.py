"""Which source frames the tracking stage samples, and how it gets them off disk.

The tracking stage used to derive its own sample indices from the container's average frame
rate (`round(time * average_fps)`) and then reach each one with `CAP_PROP_POS_FRAMES`. Both
halves fail on a variable-rate source: the camera solver (worker/stages/dense_pi3x.py) selects
its samples from decoded presentation timestamps the way ffmpeg's `fps` filter does, so the two
lists disagree and the stage aborts, and seeking by index cannot reach the end of such a file.

Everything here is CPU-only and imports no torch: the index decisions are pure functions, and
the decode test builds a tiny variable-rate clip with ffmpeg whose frames encode their own
ordinal as a flat grey, then asks for irregular indices including the very last frame. It skips
itself when ffmpeg is missing.

  uv run --locked python scripts/test_track_people_sampling.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker" / "stages"))
import track_people as tp

# The shape the 59.49 fps phone clip's solve really has: 4- and 5-frame steps that start at 2,
# which no `round(time * average_fps)` schedule can produce.
VFR_CAMERA_INDICES = [2, 7, 12, 17, 22, 27, 31, 36, 41, 46, 51, 56, 60]
VFR_CAMERA_TIMES = [i / 59.49 for i in VFR_CAMERA_INDICES]

GREY_STEP = 11
GREY_BASE = 12
CLIP_FRAMES = 20
# Quantised by the concat demuxer, but deliberately uneven: the gaps below are 1, 2 and 3
# frame periods, so no single rate describes the clip.
CLIP_DURATIONS = [0.04, 0.08, 0.04, 0.12, 0.04, 0.08, 0.12, 0.04]


def cameras(indices, times):
    """cameras.json records carrying only the two fields this stage reads."""
    return [dict(sourceIndex=int(index), time=float(time)) for index, time in zip(indices, times)]


def build_vfr_clip(folder):
    """A short variable-rate H.264 clip whose every frame is a grey that names its ordinal."""
    lines = []
    for i in range(CLIP_FRAMES):
        frame = np.full((48, 64, 3), GREY_BASE + GREY_STEP * i, np.uint8)
        cv2.imwrite(str(folder / f"f{i:03d}.png"), frame)
        lines.append(f"file 'f{i:03d}.png'")
        lines.append(f"duration {CLIP_DURATIONS[i % len(CLIP_DURATIONS)]}")
    # The concat demuxer gives the last entry no duration unless it is named twice, so the
    # encoded clip holds one extra frame: a repeat of the final image.
    lines.append(f"file 'f{CLIP_FRAMES - 1:03d}.png'")
    (folder / "list.txt").write_text("\n".join(lines) + "\n")
    clip = folder / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "list.txt",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        cwd=folder,
        check=True,
    )
    return clip


def grey_ordinal(frame):
    """The ordinal a decoded frame's flat grey stands for."""
    return int(round((float(frame.mean()) - GREY_BASE) / GREY_STEP))


class SampleSelection(unittest.TestCase):
    def test_cameras_are_adopted_exactly(self):
        given = cameras(VFR_CAMERA_INDICES, VFR_CAMERA_TIMES)
        indices, times, authority = tp.plan_samples(given, 12, 59.49, 477)
        self.assertEqual(authority, tp.SAMPLE_AUTHORITY_CAMERAS)
        self.assertEqual([int(i) for i in indices], VFR_CAMERA_INDICES)
        np.testing.assert_allclose(times, VFR_CAMERA_TIMES)

    def test_the_old_rule_disagreed_with_those_cameras(self):
        """The regression itself: the fps rule cannot reproduce a variable-rate solve's plan."""
        legacy, _ = tp.fps_rule_indices(12, 59.49, 477)
        self.assertNotEqual([int(i) for i in legacy[: len(VFR_CAMERA_INDICES)]], VFR_CAMERA_INDICES)
        self.assertEqual(int(legacy[0]), 0)

    def test_camera_times_are_used_not_recomputed(self):
        """A camera whose time is not `index / average_fps` keeps its own measured time."""
        given = cameras([0, 5, 9], [0.0, 0.0836, 0.1502])
        _, times, _ = tp.plan_samples(given, 12, 59.49, 477)
        np.testing.assert_allclose(times, [0.0, 0.0836, 0.1502])

    def test_malformed_camera_indices_are_refused(self):
        for bad in ([3, 3, 9], [3, 9, 8]):
            with self.assertRaises(ValueError):
                tp.plan_samples(cameras(bad, [0.0, 0.1, 0.2]), 12, 24.0, 240)
        with self.assertRaises(ValueError):
            tp.plan_samples(cameras([], []), 12, 24.0, 240)

    def test_no_cameras_keeps_the_fps_rule(self):
        indices, times, authority = tp.plan_samples(None, 12, 24.0, 240)
        self.assertEqual(authority, tp.SAMPLE_AUTHORITY_FPS_RULE)
        self.assertEqual([int(i) for i in indices], list(range(0, 240, 2)))
        np.testing.assert_allclose(times, np.arange(0, 240, 2) / 24.0)

    def test_fps_rule_stays_inside_the_clip(self):
        indices, _, _ = tp.plan_samples(None, 12, 24.0, 61)
        self.assertEqual(int(indices[-1]), 60)
        self.assertTrue(all(0 <= int(i) < 61 for i in indices))

    def test_fps_rule_refuses_to_repeat_a_frame(self):
        with self.assertRaises(ValueError):
            tp.plan_samples(None, 30, 10.0, 100)


class DecodeShortfall(unittest.TestCase):
    def test_silence_when_every_frame_arrived(self):
        self.assertIsNone(tp.decode_shortfall([2, 7, 12], {2, 7, 12, 99}))

    def test_names_the_first_missing_frame_and_the_count(self):
        message = tp.decode_shortfall([2, 7, 12, 17], {2, 7})
        self.assertIn("Decoded 2 of 4 planned source frames", message)
        self.assertIn("source frame 12 never decoded", message)
        self.assertIn("1 more up to 17", message)

    def test_a_single_missing_frame_reads_plainly(self):
        message = tp.decode_shortfall([2, 7], {2})
        self.assertIn("source frame 7 never decoded.", message)
        self.assertNotIn("more up to", message)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is needed to build the test clip")
class SequentialDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory(prefix="track-sampling-")
        cls.folder = Path(cls._temp.name)
        cls.clip = build_vfr_clip(cls.folder)
        cls.fps, cls.count, cls.size = tp.source_metadata(cls.clip)
        cls.last = cls.count - 1
        # Irregular, and the last frame is in the set on purpose: index seeking cannot reach
        # the tail of a variable-rate file, which is what the sequential walk is here to fix.
        cls.wanted = [0, 3, 7, cls.last - 1, cls.last]

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def test_the_clip_really_is_variable_rate(self):
        stamps = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pts_time",
                "-of",
                "csv=p=0",
                str(self.clip),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        times = [float(s.rstrip(",")) for s in stamps if s.strip(",")]
        gaps = {round(b - a, 3) for a, b in zip(times, times[1:])}
        self.assertGreater(len(gaps), 1, f"expected uneven frame spacing, got {gaps}")

    def test_opencv_backend_is_chosen_for_this_clip(self):
        backend, reason = tp.choose_decode_backend(*tp.opencv_first_frame(cv2, self.clip))
        self.assertEqual(backend, tp.DECODE_OPENCV)
        self.assertTrue(reason)

    def test_sequential_decode_returns_exactly_the_wanted_frames(self):
        got = list(tp.opencv_frames(self.clip, set(self.wanted)))
        self.assertEqual([index for index, _ in got], self.wanted)
        for index, frame in got:
            self.assertEqual(frame.shape, (self.size[1], self.size[0], 3))
            # The last encoded frame repeats the last image, so both name the same ordinal.
            self.assertEqual(grey_ordinal(frame), min(index, CLIP_FRAMES - 1))

    def test_sample_stream_follows_the_chosen_backend(self):
        wanted = {index: sample for sample, index in enumerate(self.wanted)}
        got = list(tp.sample_stream(self.clip, tp.DECODE_OPENCV, self.size, set(wanted)))
        self.assertEqual([index for index, _ in got], self.wanted)

    def test_the_ffmpeg_fallback_sees_the_same_frames(self):
        got = list(tp.ffmpeg_frames(self.clip, self.size, set(self.wanted)))
        self.assertEqual([index for index, _ in got], self.wanted)
        for index, frame in got:
            self.assertEqual(grey_ordinal(frame), min(index, CLIP_FRAMES - 1))

    def test_adopted_camera_indices_decode_without_shortfall(self):
        """The end-to-end shape of the fix: plan from cameras, decode, guard reports nothing."""
        times = [index / self.fps for index in self.wanted]
        indices, _, authority = tp.plan_samples(
            cameras(self.wanted, times), 12, self.fps, self.count
        )
        self.assertEqual(authority, tp.SAMPLE_AUTHORITY_CAMERAS)
        decoded = {index for index, _ in tp.opencv_frames(self.clip, set(int(i) for i in indices))}
        self.assertIsNone(tp.decode_shortfall(indices, decoded))

    def test_a_frame_past_the_end_is_reported_precisely(self):
        beyond = self.count + 5
        decoded = {index for index, _ in tp.opencv_frames(self.clip, {0, beyond})}
        message = tp.decode_shortfall([0, beyond], decoded)
        self.assertIn(f"source frame {beyond} never decoded", message)


if __name__ == "__main__":
    unittest.main()
