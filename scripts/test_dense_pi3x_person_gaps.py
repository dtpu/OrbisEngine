#!/usr/bin/env python3
"""Person-gap bookkeeping for the Pi3X dense solve; numpy only, no torch, GPU or media.

A frame with nobody in it is not a failed solve, so `worker/stages/dense_pi3x.py` records
it instead of aborting. These checks cover the decision itself and the metadata it writes.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

import dense_pi3x
from dense_pi3x import MIN_PERSON_POINTS, person_gap, person_gap_summary, person_present


def summary_for(counts):
    """The summary a solve of `len(counts)` frames with those point counts would write."""
    gaps = [
        person_gap(i, 1000 + i, i / 12.0, n) for i, n in enumerate(counts) if not person_present(n)
    ]
    return gaps, person_gap_summary(gaps, len(counts))


class ImportsWithoutTheGpuStack(unittest.TestCase):
    def test_module_loads_without_torch_cv2_or_pillow(self):
        self.assertIsNone(sys.modules.get("torch"))
        self.assertIsNone(sys.modules.get("cv2"))
        self.assertIsNone(sys.modules.get("PIL"))
        self.assertTrue(callable(dense_pi3x.main))


class Threshold(unittest.TestCase):
    def test_named_constant_is_the_documented_hundred_points(self):
        self.assertEqual(MIN_PERSON_POINTS, 100)

    def test_edge_is_inclusive_at_the_constant(self):
        self.assertFalse(person_present(MIN_PERSON_POINTS - 1))
        self.assertTrue(person_present(MIN_PERSON_POINTS))
        self.assertFalse(person_present(99))
        self.assertTrue(person_present(100))
        self.assertTrue(person_present(101))

    def test_empty_and_numpy_counts(self):
        self.assertFalse(person_present(0))
        self.assertFalse(person_present(len(np.zeros((0, 3)))))
        self.assertTrue(person_present(np.int64(400)))

    def test_alternative_minimum_is_honoured(self):
        self.assertTrue(person_present(50, minimum=50))
        self.assertFalse(person_present(49, minimum=50))


class GapRecord(unittest.TestCase):
    def test_record_is_json_safe_even_from_numpy_scalars(self):
        times = np.arange(4) / 12.0
        indices = np.array([0, 3, 5, 8])
        rec = person_gap(2, indices[2], times[2], len(np.zeros((7, 3))))
        self.assertEqual(rec, dict(sample=2, sourceIndex=5, time=times[2], points=7))
        for key, kind in (("sample", int), ("sourceIndex", int), ("time", float), ("points", int)):
            self.assertIs(type(rec[key]), kind, key)


class AllFramesPresent(unittest.TestCase):
    def test_no_gaps_recorded_and_every_ply_listed(self):
        gaps, s = summary_for([4000, 3900, 4100, 4050])
        self.assertEqual(gaps, [])
        self.assertEqual(s["personAbsentFrames"], [])
        self.assertEqual(s["personAbsentCount"], 0)
        self.assertEqual(s["personPresentCount"], 4)
        self.assertEqual(
            s["personFrames"], ["frame_000.ply", "frame_001.ply", "frame_002.ply", "frame_003.ply"]
        )
        self.assertFalse(s["cameraOnly"])
        self.assertEqual(s["minPersonPoints"], MIN_PERSON_POINTS)


class GapInTheMiddle(unittest.TestCase):
    def test_absent_frames_are_named_and_their_plys_dropped(self):
        # Everyone walks into the elevator for two sampled frames, then the shot goes on.
        gaps, s = summary_for([4000, 3800, 12, 0, 3700, 3600])
        self.assertEqual([g["sample"] for g in gaps], [2, 3])
        self.assertEqual([g["sourceIndex"] for g in gaps], [1002, 1003])
        self.assertEqual([g["points"] for g in gaps], [12, 0])
        self.assertAlmostEqual(gaps[0]["time"], 2 / 12.0)
        self.assertEqual(s["personAbsentCount"], 2)
        self.assertEqual(s["personPresentCount"], 4)
        self.assertEqual(
            s["personFrames"],
            ["frame_000.ply", "frame_001.ply", "frame_004.ply", "frame_005.ply"],
        )
        self.assertFalse(s["cameraOnly"])

    def test_summary_sorts_records_and_copies_them(self):
        unordered = [person_gap(5, 105, 5 / 12.0, 0), person_gap(1, 101, 1 / 12.0, 3)]
        s = person_gap_summary(unordered, 6)
        self.assertEqual([g["sample"] for g in s["personAbsentFrames"]], [1, 5])
        s["personAbsentFrames"][0]["points"] = -1
        self.assertEqual(unordered[1]["points"], 3)


class GapAtFrameZero(unittest.TestCase):
    def test_clip_opening_on_an_empty_room_still_reports_the_rest(self):
        gaps, s = summary_for([0, 40, 3900, 4000])
        self.assertEqual([g["sample"] for g in gaps], [0, 1])
        self.assertEqual(gaps[0], dict(sample=0, sourceIndex=1000, time=0.0, points=0))
        self.assertEqual(s["personAbsentCount"], 2)
        self.assertEqual(s["personPresentCount"], 2)
        self.assertNotIn("frame_000.ply", s["personFrames"])
        self.assertEqual(s["personFrames"], ["frame_002.ply", "frame_003.ply"])
        self.assertFalse(s["cameraOnly"])


class NobodyInTheWholeClip(unittest.TestCase):
    def test_camera_only_solve_lists_every_frame_as_a_gap(self):
        gaps, s = summary_for([0, 0, 5, 99])
        self.assertEqual([g["sample"] for g in gaps], [0, 1, 2, 3])
        self.assertEqual(s["personAbsentCount"], 4)
        self.assertEqual(s["personPresentCount"], 0)
        self.assertEqual(s["personFrames"], [])
        self.assertTrue(s["cameraOnly"])

    def test_zero_frames_is_not_reported_as_camera_only(self):
        self.assertFalse(person_gap_summary([], 0)["cameraOnly"])


class MetadataIsWritable(unittest.TestCase):
    def test_summary_survives_json_dumps_from_numpy_inputs(self):
        times, indices = np.arange(5) / 12.0, np.arange(5) * 3
        gaps = [person_gap(np.int64(i), indices[i], times[i], np.int64(0)) for i in (1, 4)]
        doc = json.loads(json.dumps(person_gap_summary(gaps, np.int64(5))))
        self.assertEqual([g["sample"] for g in doc["personAbsentFrames"]], [1, 4])
        self.assertEqual(doc["personPresentCount"], 3)
        self.assertEqual(doc["personFrames"], ["frame_000.ply", "frame_002.ply", "frame_003.ply"])


class DisclosureText(unittest.TestCase):
    def test_note_states_the_threshold_and_that_cameras_survive(self):
        note = person_gap_summary([], 3)["personGapNote"]
        self.assertIn(str(MIN_PERSON_POINTS), note)
        self.assertIn("cameras.json", note)
        self.assertIn("not a failed solve", note)


if __name__ == "__main__":
    unittest.main()
