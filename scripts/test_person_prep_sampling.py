"""Which frame the person stages prepare an avatar from, and how they get it off disk.

Two independent failures live here, and both are checked without a GPU or torch.

The frame CHOICE: `prepare_track_person.py` only down-weighted samples whose person runs past a
frame edge, so on hp-fly-s63 it picked frame 0 -- a boy on a broom with head and feet cropped,
heightFraction 1.79 -- whose torso-only mask is wider than tall, and `prepare_lhm_person.py`
refused it after Mask R-CNN had already run. A cropped sample is now excluded whenever a whole
one exists, the fallback records why it was needed, and a track that is never more than a face
in frame is refused by name instead of as a CalledProcessError.

The frame DECODE: the index comes from the solve's `cameras.json`/the tracker, where it names a
position in the decoder's own sequential order. Reaching it with `CAP_PROP_POS_FRAMES` cannot
work on a variable-rate file -- seeks miss near the end and can land on a neighbour elsewhere --
so the decode test builds a tiny variable-rate clip whose frames encode their own ordinal as a
flat grey and asks for irregular indices including the very last. It skips itself without ffmpeg.

  uv run --locked python scripts/test_person_prep_sampling.py
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "worker" / "stages"))
import lhm_animate as la
import prepare_lhm_person as prep
import prepare_track_person as ptp
import track_people as tp

# Front and back views: prepare_track_person projects [0,0,1] through the root rotation, so a
# half turn about Y points the body at the lens (forward.z = -1) and no rotation faces away.
FACING_CAMERA = [0.0, math.pi, 0.0]
FACING_AWAY = [0.0, 0.0, 0.0]

FRAME_W, FRAME_H = 1920, 1080

GREY_STEP = 11
GREY_BASE = 12
CLIP_FRAMES = 20
# Quantised by the concat demuxer, but deliberately uneven: gaps of 1, 2 and 3 frame periods,
# so no single rate describes the clip.
CLIP_DURATIONS = [0.04, 0.08, 0.04, 0.12, 0.04, 0.08, 0.12, 0.04]


def sample_record(
    sample,
    source_index,
    height_fraction,
    mask_box,
    fully_in_frame,
    score=0.6,
    facing=FACING_CAMERA,
    occluded=0.0,
):
    """One `track_NN/motion.json` frame, holding the fields the ranking reads."""
    x0, y0, x1, y1 = mask_box
    joints = np.zeros((22, 2), float)
    joints[:, 1] = np.linspace(y0, y1, 22)
    joints[:, 0] = (x0 + x1) / 2
    # 16/17 are the SMPL-X shoulders; their separation is the shoulder-spread term.
    joints[16, 0] = (x0 + x1) / 2 - (x1 - x0) * 0.2
    joints[17, 0] = (x0 + x1) / 2 + (x1 - x0) * 0.2
    return dict(
        sample=sample,
        sourceIndex=source_index,
        time=source_index / 24.0,
        score=score,
        projectedBodyJoints=joints.tolist(),
        jointProjectionInImage=[bool(fully_in_frame)] * 22,
        rootRotationVector=list(facing),
        source_intrinsics=[[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]],
        occludedFraction=occluded,
        heightFraction=height_fraction,
        box=[x0, y0 - 400, x1, y1 + 400],
        maskBox=[x0, y0, x1, y1],
    )


def whole_person(sample=5, source_index=10, height_fraction=0.8, **extra):
    """A person standing entirely inside the frame: tall mask box, nothing at an edge."""
    return sample_record(
        sample, source_index, height_fraction, [800.0, 120.0, 1180.0, 980.0], True, **extra
    )


def cropped_person(sample=0, source_index=0, height_fraction=1.79, **extra):
    """hp-fly-s63's frame 0: bigger on screen, but head and feet are outside the frame."""
    return sample_record(
        sample, source_index, height_fraction, [1002.0, 142.0, 1909.0, 940.0], False, **extra
    )


def close_up(sample=0, source_index=0, height_fraction=6.5, **extra):
    """movie-s17: a face filling the frame, a body six frame-heights tall."""
    return sample_record(
        sample, source_index, height_fraction, [300.0, 0.0, 1400.0, 1079.0], False, **extra
    )


def build_vfr_clip(folder):
    """A short variable-rate H.264 clip whose every frame is a grey that names its ordinal."""
    lines = []
    for i in range(CLIP_FRAMES):
        cv2.imwrite(
            str(folder / f"f{i:03d}.png"), np.full((48, 64, 3), GREY_BASE + GREY_STEP * i, np.uint8)
        )
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


class SampleTerms(unittest.TestCase):
    def test_visible_body_fraction_is_the_share_inside_the_frame(self):
        self.assertAlmostEqual(ptp.visible_body_fraction(2.0), 0.5)
        self.assertAlmostEqual(ptp.visible_body_fraction(6.5), 1 / 6.5)
        # A person smaller than the frame is wholly visible, not "more than whole".
        self.assertAlmostEqual(ptp.visible_body_fraction(0.4), 1.0)

    def test_portrait_mask_is_taller_than_wide(self):
        self.assertTrue(ptp.portrait_mask([0, 0, 100, 200]))
        self.assertFalse(ptp.portrait_mask([1002, 142, 1909, 940]))

    def test_size_no_longer_rewards_a_closer_crop(self):
        """The ranking bug in one term: a body 3 frames tall scored 3x a whole body."""
        near = ptp.rank([cropped_person(height_fraction=3.0)])[0]
        far = ptp.rank([cropped_person(height_fraction=1.0)])[0]
        self.assertAlmostEqual(near["fit"], far["fit"])


class Selection(unittest.TestCase):
    def test_a_whole_person_wins_over_a_bigger_cropped_one(self):
        records = [cropped_person(), whole_person()]
        chosen = ptp.select(records)
        self.assertEqual(chosen["basis"], "fully-in-frame")
        self.assertEqual(chosen["chosen"]["sample"], 5)
        self.assertEqual(chosen["excluded"], 1)
        self.assertIn("excluded before ranking", chosen["reason"])

    def test_the_old_ranking_preferred_the_cropped_sample(self):
        """The regression itself, on hp-fly-s63's shape: fit alone put frame 0 on top."""
        records = [cropped_person(score=0.53), whole_person(score=0.5)]
        raw = sorted(records, key=lambda r: -r["heightFraction"] * r["score"])
        self.assertFalse(all(raw[0]["jointProjectionInImage"]))
        self.assertEqual(ptp.select(records)["chosen"]["sourceIndex"], 10)

    def test_a_cropped_sample_is_used_when_nothing_is_whole_and_the_reason_is_recorded(self):
        records = [
            cropped_person(sample=0, source_index=0, height_fraction=1.79),
            sample_record(14, 28, 1.08, [828.0, 147.0, 1553.0, 948.0], False, score=0.66),
        ]
        chosen = ptp.select(records)
        self.assertEqual(chosen["basis"], "cropped-fallback")
        self.assertEqual(chosen["chosen"]["sample"], 14)
        self.assertIn("no sample shows the whole person", chosen["reason"])
        self.assertIn("50% of the body", chosen["reason"])
        self.assertTrue(chosen["chosen"]["portraitMask"])

    def test_a_landscape_mask_box_is_excluded_from_the_fallback(self):
        """The direct predictor of the downstream refusal: a mask wider than it is tall."""
        records = [
            cropped_person(sample=0, source_index=0, height_fraction=1.2, score=0.9),
            sample_record(3, 6, 1.5, [900.0, 100.0, 1200.0, 1000.0], False, score=0.3),
        ]
        chosen = ptp.select(records)
        self.assertEqual(chosen["chosen"]["sample"], 3)
        self.assertEqual(chosen["excluded"], 1)

    def test_a_track_that_is_never_more_than_a_close_up_is_refused_by_name(self):
        records = [
            close_up(sample=i, source_index=2 * i, height_fraction=4.2 + i) for i in range(4)
        ]
        with self.assertRaises(ptp.NoUsableSample) as refusal:
            ptp.select(records)
        message = str(refusal.exception)
        self.assertIn("never fully visible", message)
        self.assertIn("4.20 frame heights", message)
        self.assertIn("24% of the person", message)
        self.assertIn("portrait mask", message)

    def test_an_explicit_frame_is_still_honoured_and_labelled(self):
        records = [cropped_person(), whole_person()]
        chosen = ptp.select(records, frame=0)
        self.assertEqual(chosen["basis"], "requested")
        self.assertEqual(chosen["chosen"]["sourceIndex"], 0)
        self.assertIn("explicitly requested", chosen["reason"])

    def test_an_explicit_frame_outside_the_track_is_refused(self):
        with self.assertRaises(ptp.NoUsableSample):
            ptp.select([whole_person()], frame=999)

    def test_camera_facing_is_still_preferred_among_whole_people(self):
        records = [
            whole_person(sample=1, source_index=2, facing=FACING_AWAY, score=0.9),
            whole_person(sample=2, source_index=4, facing=FACING_CAMERA, score=0.5),
        ]
        self.assertEqual(ptp.select(records)["chosen"]["sample"], 2)


class SampleAuthority(unittest.TestCase):
    # The shape a 59.49 fps phone clip's solve really has: 4- and 5-frame steps starting at 2,
    # which no `round(time * average_fps)` schedule can produce.
    VFR_INDICES = [2, 7, 12, 17, 22, 27, 31, 36, 41, 46, 51, 56, 60]

    def cameras(self, indices, times=None):
        times = [i / 59.49 for i in indices] if times is None else times
        return [dict(sourceIndex=int(i), time=float(t)) for i, t in zip(indices, times)]

    def test_cameras_are_adopted_exactly(self):
        indices, times, authority = la.plan_samples(
            self.cameras(self.VFR_INDICES), None, None, 12, 59.49, 477
        )
        self.assertEqual(authority, la.SAMPLE_AUTHORITY_CAMERAS)
        self.assertEqual([int(i) for i in indices], self.VFR_INDICES)
        np.testing.assert_allclose(times, [i / 59.49 for i in self.VFR_INDICES])

    def test_camera_times_are_used_not_recomputed(self):
        _, times, _ = la.plan_samples(
            self.cameras([0, 5, 9], [0.0, 0.0836, 0.1502]), None, None, 12, 59.49, 477
        )
        np.testing.assert_allclose(times, [0.0, 0.0836, 0.1502])

    def test_the_old_rule_could_not_have_produced_that_plan(self):
        """Why the stage used to abort before any GPU work on a variable-rate clip."""
        legacy, _, authority = la.plan_samples(None, None, None, 12, 59.49, 477)
        self.assertEqual(authority, la.SAMPLE_AUTHORITY_FPS_RULE)
        self.assertNotEqual([int(i) for i in legacy[: len(self.VFR_INDICES)]], self.VFR_INDICES)

    def test_a_seed_track_is_the_authority_when_no_cameras_are_supplied(self):
        indices, times, authority = la.plan_samples(
            None, self.VFR_INDICES, [i / 59.49 for i in self.VFR_INDICES], 12, 59.49, 477
        )
        self.assertEqual(authority, la.SAMPLE_AUTHORITY_SEED)
        self.assertEqual([int(i) for i in indices], self.VFR_INDICES)

    def test_a_seed_track_without_times_falls_back_to_the_container_rate(self):
        _, times, _ = la.plan_samples(None, [2, 7, 12], None, 12, 24.0, 133)
        np.testing.assert_allclose(times, [2 / 24.0, 7 / 24.0, 12 / 24.0])

    def test_neither_supplied_keeps_the_legacy_constant_rate_rule(self):
        indices, times, authority = la.plan_samples(None, None, None, 12, 24.0, 240)
        self.assertEqual(authority, la.SAMPLE_AUTHORITY_FPS_RULE)
        self.assertEqual([int(i) for i in indices], list(range(0, 240, 2)))
        np.testing.assert_allclose(times, np.arange(0, 240, 2) / 24.0)

    def test_only_the_tracked_person_s_own_samples_are_requested(self):
        """A track's gaps are not work this run was asked for and did not do.

        soccer-s2: 96 solved samples, a track present in 90 of them. Counting all 96 as
        requested made a complete animation certify as partial source-motion coverage.
        """
        indices, times, _ = la.plan_samples(
            self.cameras(self.VFR_INDICES), None, None, 12, 59.49, 477
        )
        absent = {3, 4, 9}
        poses = [None if sample in absent else object() for sample in range(len(indices))]
        requested = la.requested_samples(indices, times, poses, True)
        seeded = [s for s in range(len(indices)) if s not in absent]
        self.assertEqual([sample for sample, _, _ in requested], seeded)
        # Cameras stay the authority for what each of those sample numbers addresses.
        self.assertEqual(
            [index for _, index, _ in requested], [self.VFR_INDICES[s] for s in seeded]
        )
        np.testing.assert_allclose(
            [time for _, _, time in requested], [self.VFR_INDICES[s] / 59.49 for s in seeded]
        )
        self.assertFalse([sample for sample, _, _ in requested if sample in absent])

    def test_requested_samples_match_the_frames_a_track_run_can_export(self):
        """The certification rule itself: requestedSamples has to equal the exported frames."""
        indices, times, _ = la.plan_samples(
            self.cameras(self.VFR_INDICES), None, None, 12, 59.49, 477
        )
        poses = [None if sample in (0, 5) else object() for sample in range(len(indices))]
        exported = [sample for sample, pose in enumerate(poses) if pose is not None]
        self.assertEqual(len(la.requested_samples(indices, times, poses, True)), len(exported))

    def test_a_gappy_track_on_a_constant_rate_clip_keeps_the_old_indices_and_times(self):
        indices, times, _ = la.plan_samples(None, None, None, 12, 24.0, 240)
        poses = [None if sample in (71, 80, 81) else object() for sample in range(len(indices))]
        requested = la.requested_samples(indices, times, poses, True)
        self.assertEqual(len(requested), len(indices) - 3)
        for sample, index, time in requested:
            self.assertEqual(index, int(indices[sample]))
            self.assertAlmostEqual(time, float(times[sample]))

    def test_pose_recovery_still_requests_the_samples_it_has_to_re_estimate(self):
        """Without --track-only the gaps are the work: they stay requested."""
        indices, times, _ = la.plan_samples(
            self.cameras(self.VFR_INDICES), None, None, 12, 59.49, 477
        )
        poses = [None if sample in (3, 4) else object() for sample in range(len(indices))]
        self.assertEqual(len(la.requested_samples(indices, times, poses, False)), len(indices))

    def test_every_sample_is_requested_without_a_seed_track(self):
        indices, times, _ = la.plan_samples(None, None, None, 12, 24.0, 240)
        requested = la.requested_samples(indices, times, None, False)
        self.assertEqual([index for _, index, _ in requested], [int(i) for i in indices])

    def test_a_seed_of_the_wrong_length_is_refused(self):
        indices, times, _ = la.plan_samples(
            self.cameras(self.VFR_INDICES), None, None, 12, 59.49, 477
        )
        with self.assertRaises(ValueError):
            la.requested_samples(indices, times, [object()] * 3, True)

    def test_malformed_supplied_samples_are_refused(self):
        for bad in ([3, 3, 9], [3, 9, 8], [-1, 4, 8], []):
            with self.assertRaises(ValueError):
                la.plan_samples(self.cameras(bad), None, None, 12, 24.0, 240)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is needed to build the test clip")
class SequentialDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory(prefix="person-prep-sampling-")
        cls.folder = Path(cls._temp.name)
        cls.clip = build_vfr_clip(cls.folder)
        cls.fps, cls.count, cls.size = tp.source_metadata(cls.clip)
        cls.last = cls.count - 1

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

    def test_one_frame_comes_back_as_the_ordinal_that_was_asked_for(self):
        for index in (0, 3, 7, self.last - 1):
            rgb, fps, count = prep.decode(self.clip, index)
            self.assertEqual(grey_ordinal(rgb), min(index, CLIP_FRAMES - 1))
            self.assertEqual(count, self.count)
            self.assertGreater(fps, 0)

    def test_the_last_frame_is_reachable(self):
        """Seeking could not reach it: `CAP_PROP_POS_FRAMES` fails near the end of such a file."""
        rgb, _, _ = prep.decode(self.clip, self.last)
        self.assertEqual(grey_ordinal(rgb), CLIP_FRAMES - 1)

    def test_several_frames_come_from_one_forward_pass(self):
        wanted = [0, 3, 7, self.last]
        frames, _, count = prep.decode_frames(self.clip, wanted)
        self.assertEqual(sorted(frames), wanted)
        self.assertEqual(count, self.count)
        for index, rgb in frames.items():
            self.assertEqual(grey_ordinal(rgb), min(index, CLIP_FRAMES - 1))

    def test_the_prep_and_the_tracker_count_the_same_ordinals(self):
        """The whole point of the fix: one index means one frame across the stages."""
        wanted = {0, 3, 7, self.last}
        frames, _, _ = prep.decode_frames(self.clip, wanted)
        for index, bgr in tp.opencv_frames(self.clip, wanted):
            np.testing.assert_array_equal(frames[index], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def test_a_frame_past_the_end_is_refused_precisely(self):
        beyond = self.count + 5
        with self.assertRaises(RuntimeError) as refusal:
            prep.decode_frames(self.clip, [0, beyond])
        message = str(refusal.exception)
        self.assertIn(f"cannot decode source frame {beyond}", message)
        self.assertIn(f"{self.count} frames decode", message)
        self.assertIn(f"last source index is {self.last}", message)

    def test_a_negative_index_is_refused(self):
        with self.assertRaises(ValueError):
            prep.decode_frames(self.clip, [-1])


if __name__ == "__main__":
    unittest.main()
