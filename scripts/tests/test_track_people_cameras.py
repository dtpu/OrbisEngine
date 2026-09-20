"""Cameras are matched to sampled frames by source index, not by list position.

A solve records no camera for a frame whose subject it could not reconstruct, and may cover a
shorter span than the tracking schedule when a container overstates its frame count. Both leave
gaps. Requiring exact positional correspondence turned either into a hard failure.
"""

import re
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_policy():
    """Extract the pure-python coverage rule; the module itself needs torch and CUDA."""
    source = (ROOT / "worker/stages/track_people.py").read_text()
    match = re.search(r"^MINIMUM_CAMERA_COVERAGE = .*?^def sample_indices", source, re.S | re.M)
    assert match, "policy block not found"
    module = types.ModuleType("track_people_policy")
    exec(
        compile(match.group(0).rsplit("\ndef sample_indices", 1)[0], "track_people", "exec"),
        module.__dict__,
    )
    return module


MODULE = load_policy()
COVERAGE = MODULE.MINIMUM_CAMERA_COVERAGE
decoded_span_usable = MODULE.decoded_span_usable


def index_cameras(cameras):
    """The mapping the stage builds: source frame -> camera, skipping absences."""
    return {int(c["sourceIndex"]): c for c in cameras if c}


def covered(indices, cameras):
    by_source = index_cameras(cameras)
    return sum(1 for i in indices if int(i) in by_source)


def camera(source_index):
    return {"sourceIndex": source_index, "source_intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


class CameraMatchingTests(unittest.TestCase):
    def test_null_cameras_are_skipped_not_indexed(self):
        cams = [camera(i) for i in range(10)]
        cams[3] = cams[7] = None
        by_source = index_cameras(cams)
        self.assertNotIn(3, by_source)
        self.assertEqual(len(by_source), 8)

    def test_a_shorter_solve_still_covers_its_own_frames(self):
        indices = np.arange(20)
        cams = [camera(i) for i in range(15)]  # solve stopped early
        self.assertEqual(covered(indices, cams), 15)

    def test_gaps_leave_coverage_above_the_floor(self):
        indices = np.arange(179)
        cams = [camera(i) for i in range(179)]
        for gap in (3, 4, 8, 13, 24, 25):  # the frames this clip actually skipped
            cams[gap] = None
        self.assertGreaterEqual(covered(indices, cams) / len(indices), COVERAGE)

    def test_a_different_video_falls_below_the_floor(self):
        indices = np.arange(100)
        cams = [camera(i + 500) for i in range(100)]  # describes another span entirely
        self.assertLess(covered(indices, cams) / len(indices), COVERAGE)

    def test_matching_is_by_source_index_not_position(self):
        indices = np.array([0, 2, 4, 6])
        cams = [camera(0), None, camera(4), camera(6)]
        by_source = index_cameras(cams)
        # Position 2 holds the camera for source frame 4, not source frame 2.
        self.assertEqual(by_source[4]["sourceIndex"], 4)
        self.assertNotIn(2, by_source)
        self.assertEqual(covered(indices, cams), 3)


class DecodeSpanTests(unittest.TestCase):
    """This clip reports 448 frames and decodes 358; the tail must not discard the run."""

    def test_a_truncated_tail_is_tracked_not_discarded(self):
        self.assertTrue(decoded_span_usable(decoded=179, requested=225))

    def test_a_source_that_mostly_fails_is_refused(self):
        self.assertFalse(decoded_span_usable(decoded=12, requested=225))

    def test_zero_requested_is_refused(self):
        self.assertFalse(decoded_span_usable(decoded=0, requested=0))


if __name__ == "__main__":
    unittest.main()
