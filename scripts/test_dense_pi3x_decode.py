#!/usr/bin/env python3
"""Decoding, sample selection and intrinsic fallbacks for the Pi3X dense solve.

Three real clips failed inside `worker/stages/dense_pi3x.py` for reasons that had nothing to
do with the reconstruction: index seeking could not reach the tail of a variable-rate file,
OpenCV's bundled libavcodec could not decode AV1 at all, and one frame with too few valid
rays aborted a whole solve. These checks cover the decisions behind those three fixes. The
decode cases build their own tiny clip with ffmpeg; everything else is pure and needs no media.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

from dense_pi3x import (  # noqa: E402
    DECODE_FFMPEG,
    DECODE_OPENCV,
    MIN_INTRINSIC_FIT_FRACTION,
    MIN_OUTPUT_SAMPLES,
    SELECT_CONTAINER_INDEX,
    SELECT_PTS_SLOTS,
    TIMES_AVERAGE_FPS,
    TIMES_POS_MSEC,
    TIMES_PTS,
    choose_decode_backend,
    clamp_plan,
    container_samples,
    decode_ffmpeg,
    decode_opencv,
    decoded_timeline,
    fit_ray_intrinsics,
    median_intrinsics,
    opencv_first_frame,
    output_slots,
    probe_source,
    require_intrinsic_fits,
    sample_plan,
    usable_timestamps,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# The synthetic clip: 24 source frames at 25 fps with four of them dropped, so its timestamps
# are genuinely uneven, and every frame is a flat grey whose level is its own source index.
GREY_BASE, GREY_STEP, DROPPED = 6, 10, range(5, 9)
CLIP_FRAMES = 24


def encoded_index(frame):
    """The source index the frame carries, recovered from its grey level."""
    return int(round((float(np.asarray(frame).mean()) - GREY_BASE) / GREY_STEP))


def build_clip(directory):
    """A tiny variable-rate H.264 clip whose every frame states which source frame it is."""
    import cv2

    for index in range(CLIP_FRAMES):
        level = GREY_BASE + GREY_STEP * index
        cv2.imwrite(str(directory / f"g{index:03d}.png"), np.full((48, 64, 3), level, np.uint8))
    constant = directory / "constant.mp4"
    variable = directory / "variable.mp4"
    common = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p"]
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-framerate", "25", "-i", str(directory / "g%03d.png")]
        + common
        + [str(constant)],
        check=True,
    )
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(constant),
            "-vf",
            f"select='not(between(n,{DROPPED.start},{DROPPED.stop - 1}))'",
            "-fps_mode",
            "passthrough",
        ]
        + common
        + [str(variable)],
        check=True,
    )
    return variable


def probe_frame_count(video):
    """What ffprobe counts by decoding, which is the number a container may contradict."""
    done = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(json.loads(done.stdout)["streams"][0]["nb_read_frames"])


class FakeCapture:
    """Just enough of cv2.VideoCapture to drive the decode decisions without media."""

    def __init__(self, opened=True, frames=0, retrieve_fails_at=None, msec=None):
        self.opened, self.frames = opened, frames
        self.retrieve_fails_at, self.msec = retrieve_fails_at, msec
        self.position, self.released, self.seeks = 0, False, []

    def isOpened(self):  # noqa: N802 - the OpenCV spelling
        return self.opened

    def grab(self):
        if not self.opened or self.position >= self.frames:
            return False
        self.position += 1
        return True

    def retrieve(self):
        index = self.position - 1
        if self.retrieve_fails_at is not None and index >= self.retrieve_fails_at:
            return False, None
        return True, np.full((4, 4, 3), index, np.uint8)

    def read(self):
        return (True, np.zeros((4, 4, 3), np.uint8)) if self.grab() else (False, None)

    def get(self, prop):
        if self.msec is not None:
            return self.msec[self.position - 1] * 1000.0
        return (self.position - 1) * 40.0

    def set(self, prop, value):
        self.seeks.append(value)
        return True

    def release(self):
        self.released = True


class FakeCv2:
    CAP_PROP_POS_MSEC = 0
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, capture):
        self.capture = capture

    def VideoCapture(self, path):  # noqa: N802 - the OpenCV spelling
        return self.capture


class BackendChoice(unittest.TestCase):
    def test_opencv_is_kept_when_it_decodes(self):
        backend, reason = choose_decode_backend(True, True)
        self.assertEqual(backend, DECODE_OPENCV)
        self.assertIn("decoded", reason)

    def test_a_file_that_opens_but_never_decodes_falls_back(self):
        # The AV1 case: frame count, size and rate all read back, every read fails.
        backend, reason = choose_decode_backend(True, False)
        self.assertEqual(backend, DECODE_FFMPEG)
        self.assertIn("could not decode", reason)

    def test_a_file_that_will_not_open_falls_back(self):
        backend, reason = choose_decode_backend(False, False)
        self.assertEqual(backend, DECODE_FFMPEG)
        self.assertIn("could not open", reason)

    def test_probe_reports_an_opened_capture_that_cannot_read(self):
        capture = FakeCapture(opened=True, frames=0)
        opened, first = opencv_first_frame(FakeCv2(capture), "clip.mp4")
        self.assertEqual((opened, first), (True, False))
        self.assertTrue(capture.released)
        self.assertEqual(choose_decode_backend(opened, first)[0], DECODE_FFMPEG)

    def test_probe_reports_a_capture_that_reads(self):
        capture = FakeCapture(opened=True, frames=3)
        self.assertEqual(opencv_first_frame(FakeCv2(capture), "clip.mp4"), (True, True))

    def test_probe_releases_a_capture_that_never_opened(self):
        capture = FakeCapture(opened=False)
        self.assertEqual(opencv_first_frame(FakeCv2(capture), "clip.mp4"), (False, False))
        self.assertTrue(capture.released)


class Timestamps(unittest.TestCase):
    def test_decoder_timestamps_are_preferred_and_labelled(self):
        pts = [0.0, 0.04, 0.09, 0.12]
        times, label = decoded_timeline(4, pts_seconds=pts, average_fps=25.0)
        self.assertEqual(label, TIMES_PTS)
        self.assertEqual(times, pts)

    def test_timestamps_are_relative_to_the_first_decoded_frame(self):
        times, label = decoded_timeline(3, pts_seconds=[10.0, 10.04, 10.08])
        self.assertEqual(label, TIMES_PTS)
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[2], 0.08)

    def test_opencv_positions_are_used_when_ffprobe_has_none(self):
        times, label = decoded_timeline(3, pts_seconds=[], pos_msec_seconds=[0.0, 0.04, 0.1])
        self.assertEqual(label, TIMES_POS_MSEC)
        self.assertEqual(times, [0.0, 0.04, 0.1])

    def test_ffprobe_wins_over_the_opencv_first_frame_position_quirk(self):
        # Real soccer clip: OpenCV reports -50.278 ms for frame 0 where the stream's own PTS
        # is 0, so its first interval is four frames long. Ordered, but not the stream's clock.
        quirk = [-0.050278, 0.016667, 0.033333, 0.05]
        pts = [0.0, 0.016667, 0.033333, 0.05]
        times, label = decoded_timeline(4, pts_seconds=pts, pos_msec_seconds=quirk)
        self.assertEqual(label, TIMES_PTS)
        self.assertEqual(times, pts)

    def test_positions_that_go_backwards_fall_through_to_the_average_rate(self):
        times, label = decoded_timeline(
            4, pos_msec_seconds=[0.0, 0.016, 0.010, 0.05], average_fps=60.0
        )
        self.assertEqual(label, TIMES_AVERAGE_FPS)
        self.assertAlmostEqual(times[1], 1 / 60.0)

    def test_average_rate_is_the_last_resort_and_says_so(self):
        times, label = decoded_timeline(3, average_fps=25.0)
        self.assertEqual(label, TIMES_AVERAGE_FPS)
        self.assertEqual(times, [0.0, 0.04, 0.08])

    def test_a_short_or_unusable_timestamp_list_is_ignored(self):
        self.assertFalse(usable_timestamps([0.0]))
        self.assertFalse(usable_timestamps([0.0, float("nan")]))
        self.assertFalse(usable_timestamps([0.0, 0.04, 0.04]))
        self.assertFalse(usable_timestamps([0.0, None]))
        _, label = decoded_timeline(4, pts_seconds=[0.0, 0.04], average_fps=25.0)
        self.assertEqual(label, TIMES_AVERAGE_FPS)

    def test_nothing_usable_at_all_is_an_error(self):
        with self.assertRaises(ValueError):
            decoded_timeline(4, average_fps=0)


class SlotSelection(unittest.TestCase):
    def test_one_frame_per_output_slot_at_half_the_source_rate(self):
        times = [index / 24.0 for index in range(24)]
        samples, empty = output_slots(times, 12.0)
        # Every other frame. Frame 23 rounds up into the slot at the stream's end, which the fps
        # filter does not write: `ffmpeg -vf fps=12` on one second of 24 fps gives 12 frames.
        self.assertEqual(samples, list(range(0, 23, 2)))
        self.assertEqual(empty, 0)
        self.assertEqual(len(samples), len(set(samples)))

    def test_the_last_frame_in_a_slot_wins_as_the_fps_filter_does(self):
        # Two frames land in slot 1; ffmpeg keeps the later one.
        samples, empty = output_slots([0.0, 0.07, 0.085, 0.17], 12.0)
        self.assertEqual(samples, [0, 2, 3])
        self.assertEqual(empty, 0)

    def test_a_hole_in_the_source_is_counted_not_duplicated(self):
        # Slots 2 and 3 have no frame behind them; the fps filter would repeat a neighbour.
        times = [0.0, 1 / 24, 2 / 24, 8 / 24, 9 / 24]
        samples, empty = output_slots(times, 12.0)
        # the last frame rounds into the end slot, which is never written
        self.assertEqual(samples, [0, 2, 3])
        self.assertEqual(empty, 2)
        self.assertEqual(len(samples), len(set(samples)))

    def test_no_frames_select_nothing(self):
        self.assertEqual(output_slots([], 12.0), ([], 0))

    def test_a_last_frame_rounding_into_the_end_slot_is_not_sampled(self):
        # 2 s at 30 fps: frame 59 (1.9667 s) rounds up into slot 24, which starts at the stream's
        # end and which FFmpeg's fps filter never writes; it writes 24 frames, ending on frame 58
        samples, empty = output_slots([i / 30 for i in range(60)], 12.0)
        self.assertEqual((len(samples), samples[-1], empty), (24, 58, 0))


class ContainerSelection(unittest.TestCase):
    def test_indices_follow_the_declared_average_rate(self):
        self.assertEqual(container_samples(48, 24.0, 12.0), list(range(0, 48, 2)))

    def test_a_source_slower_than_the_request_is_refused(self):
        with self.assertRaisesRegex(ValueError, "below requested independent output density"):
            container_samples(40, 8.0, 12.0)


class PlanSelection(unittest.TestCase):
    def test_decoded_timestamps_select_slots_and_metadata_says_so(self):
        plan = sample_plan(12.0, 24, 24.0, pts_seconds=[index / 24.0 for index in range(24)])
        self.assertEqual(plan["sampleSelection"], SELECT_PTS_SLOTS)
        self.assertEqual(plan["timestampSource"], TIMES_PTS)
        self.assertEqual(plan["samples"], list(range(0, 23, 2)))
        self.assertEqual(plan["times"][1], 1 / 12.0)
        self.assertEqual(plan["plannedSamples"], len(plan["samples"]))

    def test_container_metadata_selection_is_named_when_no_timestamps_survive(self):
        plan = sample_plan(12.0, 24, 24.0)
        self.assertEqual(plan["sampleSelection"], SELECT_CONTAINER_INDEX)
        self.assertEqual(plan["timestampSource"], TIMES_AVERAGE_FPS)
        self.assertEqual(plan["samples"], list(range(0, 24, 2)))


class ClampToDecodableFrames(unittest.TestCase):
    def plan_for(self, samples, fps=12.0):
        return dict(
            samples=list(samples),
            times=[index / fps for index in range(len(samples))],
            timestampSource=TIMES_PTS,
            sampleSelection=SELECT_PTS_SLOTS,
            emptyOutputSlots=0,
            plannedSamples=len(samples),
        )

    def test_an_over_reporting_container_loses_only_its_tail(self):
        # The soccer clip: 712 declared frames, the last few unreachable.
        plan = self.plan_for(range(0, 712, 5))
        clamped = clamp_plan(plan, decodable=700, container_frames=712, fps=12.0)
        self.assertEqual(clamped["droppedTailSamples"], 3)
        self.assertEqual(clamped["containerFrames"], 712)
        self.assertEqual(clamped["decodableFrames"], 700)
        self.assertEqual(clamped["samples"][-1], 695)
        self.assertEqual(len(clamped["samples"]), len(clamped["times"]))
        self.assertTrue(all(index < 700 for index in clamped["samples"]))

    def test_an_honest_container_drops_nothing(self):
        clamped = clamp_plan(self.plan_for(range(0, 240, 2)), 240, 240, 12.0)
        self.assertEqual(clamped["droppedTailSamples"], 0)
        self.assertEqual(clamped["emptyOutputSlots"], 0)
        self.assertEqual(clamped["outputSlotFill"], 1.0)

    def test_too_few_samples_survive_is_an_error_naming_the_counts(self):
        plan = self.plan_for(range(0, 600, 50))
        with self.assertRaisesRegex(RuntimeError, f"at least {MIN_OUTPUT_SAMPLES}"):
            clamp_plan(plan, decodable=100, container_frames=100, fps=12.0)

    def test_a_truncated_file_is_refused_rather_than_half_solved(self):
        plan = self.plan_for(range(0, 1000, 2))
        with self.assertRaisesRegex(RuntimeError, "truncated or corrupt"):
            clamp_plan(plan, decodable=200, container_frames=1000, fps=12.0)

    def test_an_unreadable_source_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "unreadable"):
            clamp_plan(self.plan_for([0]), decodable=1, container_frames=1, fps=12.0)

    def test_a_source_that_cannot_fill_the_output_rate_is_refused(self):
        # Samples an eighth of a second apart cannot be a 12 fps independent sequence.
        plan = dict(
            samples=list(range(40)),
            times=[index / 4.0 for index in range(40)],
            timestampSource=TIMES_PTS,
            sampleSelection=SELECT_PTS_SLOTS,
            emptyOutputSlots=0,
            plannedSamples=40,
        )
        with self.assertRaisesRegex(RuntimeError, "requested independent frame rate"):
            clamp_plan(plan, decodable=200, container_frames=200, fps=12.0)

    def test_the_original_plan_is_not_mutated(self):
        plan = self.plan_for(range(0, 712, 5))
        before = list(plan["samples"])
        clamp_plan(plan, 700, 712, 12.0)
        self.assertEqual(plan["samples"], before)


def fitted_entry(fx, fy, cx, cy, rmse=0.4):
    """An entry shaped exactly like `fit_ray_intrinsics` returns."""
    return dict(
        intrinsics=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        source_intrinsics=[[fx * 2, 0.0, cx * 2], [0.0, fy * 2, cy * 2], [0.0, 0.0, 1.0]],
        image_size=[672, 378],
        source_image_size=[1344, 756],
        intrinsic_fit_pixel_rmse=rmse,
        intrinsics_method="fit",
    )


class IntrinsicFallback(unittest.TestCase):
    def test_a_real_fit_still_works_and_is_flagged_by_the_caller(self):
        rays = np.zeros((32, 48, 3), np.float32)
        y, x = np.mgrid[:32, :48]
        rays[..., 0] = (x + 0.5 - 24.0) / 300.0
        rays[..., 1] = (y + 0.5 - 16.0) / 300.0
        rays[..., 2] = 1.0
        fit = fit_ray_intrinsics(rays, np.ones((32, 48), bool), (96, 64))
        self.assertAlmostEqual(fit["intrinsics"][0][0], 300.0, places=3)
        self.assertAlmostEqual(fit["intrinsics"][0][2], 24.0, places=3)
        self.assertEqual(fit["source_image_size"], [96, 64])

    def test_a_frame_with_no_valid_rays_still_raises_for_the_caller_to_catch(self):
        rays = np.zeros((32, 48, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "Insufficient valid rays"):
            fit_ray_intrinsics(rays, np.zeros((32, 48), bool), (96, 64))

    def test_the_median_is_per_parameter_and_ignores_one_wild_fit(self):
        entries = [
            fitted_entry(300.0, 302.0, 336.0, 189.0),
            fitted_entry(304.0, 306.0, 338.0, 191.0),
            fitted_entry(9000.0, 9000.0, 10.0, 10.0),
        ]
        median = median_intrinsics(entries)
        self.assertEqual(median["intrinsics"][0][0], 304.0)
        self.assertEqual(median["intrinsics"][1][1], 306.0)
        self.assertEqual(median["intrinsics"][0][2], 336.0)
        self.assertEqual(median["intrinsics"][1][2], 189.0)

    def test_the_median_keeps_the_shape_downstream_readers_require(self):
        median = median_intrinsics([fitted_entry(300.0, 300.0, 336.0, 189.0)])
        self.assertEqual(median["image_size"], [672, 378])
        self.assertEqual(median["source_image_size"], [1344, 756])
        self.assertEqual(median["source_intrinsics"][2], [0, 0, 1])
        self.assertEqual(median["source_intrinsics"][0][0], 600.0)
        self.assertEqual(median["source_intrinsics"][1][2], 378.0)
        self.assertEqual(json.loads(json.dumps(median))["intrinsicsMedianFrames"], 1)

    def test_the_median_carries_no_invented_residual(self):
        median = median_intrinsics([fitted_entry(300.0, 300.0, 336.0, 189.0, rmse=0.7)])
        self.assertIsNone(median["intrinsic_fit_pixel_rmse"])
        self.assertEqual(median["intrinsicsMedianPixelRmse"], 0.7)
        self.assertIn("did not", median["intrinsics_method"])

    def test_mixed_image_sizes_are_refused(self):
        other = fitted_entry(300.0, 300.0, 336.0, 189.0)
        other["image_size"] = [512, 288]
        with self.assertRaisesRegex(ValueError, "image size"):
            median_intrinsics([fitted_entry(300.0, 300.0, 336.0, 189.0), other])

    def test_no_fits_at_all_is_refused(self):
        with self.assertRaises(ValueError):
            median_intrinsics([])

    def test_one_bad_frame_in_a_clip_does_not_abort(self):
        self.assertAlmostEqual(require_intrinsic_fits(349, 350), 349 / 350)

    def test_a_clip_whose_intrinsics_are_mostly_borrowed_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a measurement of this camera"):
            require_intrinsic_fits(100, 350)

    def test_the_minimum_fraction_is_the_named_constant(self):
        self.assertEqual(MIN_INTRINSIC_FIT_FRACTION, 0.5)
        total = 200
        require_intrinsic_fits(int(total * MIN_INTRINSIC_FIT_FRACTION), total)
        with self.assertRaises(ValueError):
            require_intrinsic_fits(int(total * MIN_INTRINSIC_FIT_FRACTION) - 1, total)

    def test_nothing_fitted_is_refused(self):
        with self.assertRaises(ValueError):
            require_intrinsic_fits(0, 0)
        with self.assertRaises(ValueError):
            require_intrinsic_fits(0, 10)


class SequentialDecodeWithoutMedia(unittest.TestCase):
    def test_frames_are_decoded_in_order_and_never_seeked(self):
        capture = FakeCapture(opened=True, frames=10)
        kept = []
        count, stamps = decode_opencv(
            FakeCv2(capture), "clip.mp4", {0, 4, 9}, lambda i, f: kept.append((i, int(f[0, 0, 0])))
        )
        self.assertEqual(count, 10)
        self.assertEqual(kept, [(0, 0), (4, 4), (9, 9)])
        self.assertEqual(capture.seeks, [])
        self.assertEqual(len(stamps), 10)
        self.assertTrue(capture.released)

    def test_a_frame_that_stops_decoding_bounds_the_count_instead_of_raising(self):
        capture = FakeCapture(opened=True, frames=10, retrieve_fails_at=7)
        kept = []
        count, stamps = decode_opencv(
            FakeCv2(capture), "clip.mp4", {0, 7}, lambda i, f: kept.append(i)
        )
        self.assertEqual(count, 7)
        self.assertEqual(kept, [0])
        self.assertEqual(len(stamps), 7)

    def test_positions_are_returned_in_seconds(self):
        capture = FakeCapture(opened=True, frames=3, msec=[0.0, 0.04, 0.1])
        _, stamps = decode_opencv(FakeCv2(capture), "clip.mp4", set(), lambda *_: None)
        self.assertEqual(stamps, [0.0, 0.04, 0.1])


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required to build a test clip")
class SequentialDecodeOfAVariableRateClip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="dense-pi3x-decode-")
        cls.clip = build_clip(Path(cls.temp.name))
        cls.decodable = probe_frame_count(cls.clip)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_the_fixture_really_is_variable_rate(self):
        probe = probe_source(self.clip)
        gaps = {round(b - a, 4) for a, b in zip(probe["pts"], probe["pts"][1:])}
        self.assertGreater(len(gaps), 1, "the fixture must not be constant rate")
        self.assertEqual(len(probe["pts"]), self.decodable)
        self.assertEqual((probe["width"], probe["height"]), (64, 48))
        self.assertEqual(probe["codec"], "h264")

    def test_every_wanted_frame_arrives_with_its_own_content(self):
        import cv2

        wanted = {0, 5, 11, self.decodable - 1}
        kept = {}
        count, stamps = decode_opencv(cv2, self.clip, wanted, lambda i, f: kept.__setitem__(i, f))
        self.assertEqual(count, self.decodable)
        self.assertEqual(set(kept), wanted)
        self.assertEqual(len(stamps), self.decodable)
        # Frames 5 onward are the source frames after the hole, so the grey level proves the
        # decoder handed back the frame at that ordinal and not its neighbour.
        survivors = [i for i in range(CLIP_FRAMES) if i not in DROPPED]
        for ordinal, frame in kept.items():
            self.assertEqual(encoded_index(frame), survivors[ordinal], f"ordinal {ordinal}")

    def test_the_tail_frame_decodes_in_order_where_a_seek_need_not(self):
        import cv2

        last = self.decodable - 1
        kept = {}
        decode_opencv(cv2, self.clip, {last}, lambda i, f: kept.__setitem__(i, f))
        self.assertIn(last, kept)
        self.assertEqual(encoded_index(kept[last]), CLIP_FRAMES - 1)

    def test_the_ffmpeg_backend_returns_the_same_frames(self):
        import cv2

        wanted = {0, 6, self.decodable - 1}
        opencv_kept, ffmpeg_kept = {}, {}
        opencv_count, _ = decode_opencv(cv2, self.clip, wanted, opencv_kept.__setitem__)
        ffmpeg_count, _ = decode_ffmpeg(self.clip, 64, 48, wanted, ffmpeg_kept.__setitem__)
        self.assertEqual(ffmpeg_count, opencv_count)
        self.assertEqual(set(ffmpeg_kept), wanted)
        for ordinal in wanted:
            self.assertEqual(ffmpeg_kept[ordinal].shape, (48, 64, 3))
            self.assertEqual(
                encoded_index(ffmpeg_kept[ordinal]), encoded_index(opencv_kept[ordinal])
            )

    def test_the_ffmpeg_backend_names_a_codec_it_cannot_decode(self):
        undecodable = Path(self.temp.name) / "not-a-video.mp4"
        undecodable.write_bytes(b"this is not a video stream")
        with self.assertRaisesRegex(RuntimeError, "needs a decoder for this codec"):
            decode_ffmpeg(undecodable, 64, 48, {0}, lambda *_: None)

    def test_the_plan_binds_samples_to_decoded_timestamps(self):
        import cv2

        probe = probe_source(self.clip)
        plan = sample_plan(12.0, len(probe["pts"]), probe["averageFps"], pts_seconds=probe["pts"])
        self.assertEqual(plan["timestampSource"], TIMES_PTS)
        self.assertEqual(plan["sampleSelection"], SELECT_PTS_SLOTS)
        kept = {}
        count, _ = decode_opencv(cv2, self.clip, set(plan["samples"]), kept.__setitem__)
        clamped = clamp_plan(plan, count, probe["containerFrames"], 12.0, minimum=2, slot_fill=0.5)
        self.assertEqual(clamped["decodableFrames"], self.decodable)
        self.assertEqual(clamped["droppedTailSamples"], 0)
        self.assertEqual(len(clamped["samples"]), len(clamped["times"]))
        self.assertTrue(all(b > a for a, b in zip(clamped["times"], clamped["times"][1:])))
        self.assertGreaterEqual(clamped["times"][0], 0.0)
        for sample, seconds in zip(clamped["samples"], clamped["times"]):
            self.assertAlmostEqual(seconds, probe["pts"][sample] - probe["pts"][0], places=6)
            self.assertIn(sample, kept)

    def test_every_sample_is_a_frame_the_cleaner_would_also_keep(self):
        """The property `source_timing.match_camera_selection` enforces on this stage.

        The cleaner resamples with FFmpeg's own fps filter and then refuses any camera whose
        source frame the filter did not retain. Selecting from the same decoded timestamps
        gives exactly the filter's retained set, minus the duplicates it emits to fill a hole
        in the source -- a duplicate is not an independent reconstruction.
        """
        sys.path.insert(0, str(ROOT / "worker"))
        from wander_worker.source_timing import resample_source

        try:
            _, provenance = resample_source(self.clip, 12.0)
        except ValueError as error:  # the filter diagnostics this ffmpeg build emits
            raise unittest.SkipTest(f"ffmpeg fps diagnostics unavailable: {error}") from error
        retained = [frame["sourceIndex"] for frame in provenance["frames"]]
        probe = probe_source(self.clip)
        plan = sample_plan(12.0, len(probe["pts"]), probe["averageFps"], pts_seconds=probe["pts"])
        self.assertEqual(plan["samples"], sorted(set(retained)))
        self.assertLess(len(plan["samples"]), len(retained), "the fixture must contain a hole")
        self.assertEqual(plan["emptyOutputSlots"], len(retained) - len(set(retained)))
        relative = {
            frame["sourceIndex"]: frame["sourceRelativeTimeSeconds"]
            for frame in provenance["frames"]
        }
        for sample, seconds in zip(plan["samples"], plan["times"]):
            self.assertAlmostEqual(seconds, relative[sample], places=6)

    def test_a_display_rotation_is_decoded_the_same_way_by_both_backends(self):
        import cv2

        rotated = Path(self.temp.name) / "rotated.mp4"
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-display_rotation", "90"]
            + ["-i", str(self.clip), "-c", "copy", str(rotated)],
            check=True,
        )
        probe = probe_source(rotated)
        self.assertEqual(probe["rotation"], 90)
        self.assertEqual((probe["width"], probe["height"]), (64, 48))
        opencv_kept, ffmpeg_kept = {}, {}
        decode_opencv(cv2, rotated, {1}, opencv_kept.__setitem__)
        decode_ffmpeg(rotated, probe["width"], probe["height"], {1}, ffmpeg_kept.__setitem__)
        self.assertEqual(opencv_kept[1].shape, (48, 64, 3))
        self.assertEqual(ffmpeg_kept[1].shape, opencv_kept[1].shape)
        self.assertEqual(encoded_index(ffmpeg_kept[1]), encoded_index(opencv_kept[1]))

    def test_a_missing_ffprobe_still_plans_from_opencv_positions(self):
        import cv2

        count, stamps = decode_opencv(cv2, self.clip, set(), lambda *_: None)
        plan = sample_plan(12.0, count, 25.0, pts_seconds=[], pos_msec_seconds=stamps)
        self.assertIn(plan["timestampSource"], (TIMES_POS_MSEC, TIMES_AVERAGE_FPS))
        self.assertTrue(all(index < count for index in plan["samples"]))


if __name__ == "__main__":
    unittest.main()
