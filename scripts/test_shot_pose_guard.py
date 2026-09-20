"""Offline scale/reference regressions for the unchanged camera teleport thresholds."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shot_cuts import scene_depth, teleport_check


class PoseGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def anchors(self, first_supported=False):
        points = np.zeros((2, 2, 2, 3))
        points[0, ..., 2] = 100
        points[1, ..., 2] = 12
        # Own-camera distance is ten, not distance from the first camera or world zero.
        poses = np.stack([np.eye(4), np.eye(4)])
        poses[1, 2, 3] = 2
        valid = np.ones((2, 2, 2), bool)
        valid[0] = first_supported
        people = np.zeros_like(valid)
        people[1, 0, 0] = True
        points[1, 0, 0, 2] = 1000
        np.savez(self.root / "anchors.npz", points=points, poses=poses, valid=valid, people=people)

    def cameras(self, reference=None, positions=None):
        positions = positions or [0, 0.01, 0.02, 1.02, 1.03, 1.04]
        doc = dict(
            cameras=[
                dict(position=[p, 0, 0], time=i / 12, sourceIndex=2 * i)
                for i, p in enumerate(positions)
            ]
        )
        if reference is not None:
            doc["reference"] = reference
        (self.root / "cameras.json").write_text(json.dumps(doc))

    def test_missing_first_anchor_and_camera_normalization(self):
        self.anchors()
        self.cameras(dict(scale=0.2, staticScaleReference=dict(anchorOrdinal=1)))
        self.assertEqual(scene_depth(self.root), 2)

    def test_explicit_anchor_wins_even_if_first_view_supported(self):
        self.anchors(first_supported=True)
        self.cameras(dict(scale=0.2, staticScaleReference=dict(anchorOrdinal=1)))
        self.assertEqual(scene_depth(self.root), 2)

    def test_legacy_selects_first_supported_and_preserves_known_scale(self):
        self.anchors()
        self.cameras(dict(scale=0.3))
        self.assertEqual(scene_depth(self.root), 3)
        self.cameras()
        self.assertEqual(scene_depth(self.root), 10)

    def test_static_mask_excludes_dominant_person_depth(self):
        self.anchors()
        with np.load(self.root / "anchors.npz") as z:
            data = dict(z)
        data["people"][1] = [[True, True], [True, False]]
        data["points"][1, :1, :, 2] = 1000
        data["points"][1, 1, 0, 2] = 1000
        np.savez(self.root / "anchors.npz", **data)
        self.cameras(dict(scale=0.2, staticScaleReference=dict(anchorOrdinal=1)))
        self.assertEqual(scene_depth(self.root), 2)

    def test_invalid_explicit_reference_never_falls_back_to_pass(self):
        self.anchors(first_supported=True)
        for reference in (
            dict(scale=0),
            dict(scale=-1),
            dict(scale=float("nan")),
            dict(scale=float("inf")),
            dict(scale=True),
            dict(staticScaleReference=dict(anchorOrdinal=20)),
            dict(staticScaleReference=dict(anchorOrdinal=True)),
        ):
            with self.subTest(reference=reference):
                self.cameras(reference)
                self.assertIsNone(scene_depth(self.root))
                self.assertIsNone(teleport_check(self.root)["ok"])

    def test_all_unsupported_is_unchecked(self):
        self.anchors()
        with np.load(self.root / "anchors.npz") as z:
            data = dict(z)
        data["valid"][:] = False
        np.savez(self.root / "anchors.npz", **data)
        self.cameras()
        self.assertIsNone(teleport_check(self.root)["ok"])

    def test_explicit_unsupported_anchor_is_not_replaced(self):
        self.anchors()
        self.cameras(dict(scale=0.2, staticScaleReference=dict(anchorOrdinal=0)))
        self.assertIsNone(scene_depth(self.root))

    def test_nonfinite_or_behind_camera_geometry_is_not_a_depth(self):
        self.anchors()
        with np.load(self.root / "anchors.npz") as z:
            data = dict(z)
        data["points"][1, ..., 2] = [[np.nan, np.inf], [-3, 1]]
        np.savez(self.root / "anchors.npz", **data)
        self.cameras(dict(scale=0.2))
        self.assertIsNone(scene_depth(self.root))

    def test_scale_correction_rejects_teleport_with_unchanged_thresholds(self):
        self.anchors()
        self.cameras(dict(scale=0.2, staticScaleReference=dict(anchorOrdinal=1)))
        report = teleport_check(self.root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["limits"], dict(depthsPerSecond=3.0, spikeOverMedian=8.0))
        self.assertEqual(report["teleportFrames"], [6])
        self.assertEqual(report["worst"]["depthsPerSecond"], 6)
        self.assertEqual(report["worst"]["spikeOverMedian"], 100)

    def test_continuous_motion_still_passes(self):
        self.anchors()
        self.cameras(dict(scale=0.2), [0, 0.01, 0.02, 0.03, 0.04])
        self.assertTrue(teleport_check(self.root)["ok"])


if __name__ == "__main__":
    unittest.main()
