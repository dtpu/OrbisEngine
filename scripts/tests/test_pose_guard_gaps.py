"""The pose guard must tolerate frames that recorded no camera.

A frame whose subject could not be reconstructed writes a null camera. Those gaps are absences,
not positions: testing a step across one, or crashing on it, would condemn a good solve.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from scripts.shot_cuts import teleport_check


def write_solve(root: Path, cameras: list, depth_points=400):
    """A minimal solve directory: cameras plus the anchors that give the scene its scale."""
    (root / "cameras.json").write_text(json.dumps({"cameras": cameras}))
    generator = np.random.default_rng(3)
    points = generator.normal(size=(1, depth_points, 3)) * 2.0 + np.array([0.0, 0.0, 5.0])
    np.savez(
        root / "anchors.npz",
        points=points,
        valid=np.ones((1, depth_points), bool),
        poses=np.eye(4)[None],
    )
    return root


def camera(index: int, x: float):
    return {"position": [x, 0.0, 0.0], "time": index * 0.1, "sourceIndex": index}


class PoseGuardGapTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wander-pose-"))

    def test_null_cameras_do_not_crash_the_guard(self):
        cams = [camera(i, i * 0.01) for i in range(12)]
        for gap in (3, 7):
            cams[gap] = None
        write_solve(self.root, cams)
        result = teleport_check(self.root)
        self.assertIsNotNone(result.get("ok"), result.get("reason"))
        self.assertEqual(result["cameras"], 10)

    def test_a_continuous_path_with_gaps_still_passes(self):
        cams = [camera(i, i * 0.01) for i in range(40)]
        cams[5] = cams[19] = None
        write_solve(self.root, cams)
        self.assertTrue(teleport_check(self.root)["ok"])

    def test_too_few_real_cameras_is_reported_not_crashed(self):
        cams = [camera(0, 0.0), None, None, camera(3, 0.1)]
        write_solve(self.root, cams)
        result = teleport_check(self.root)
        self.assertIsNone(result["ok"])
        self.assertIn("nothing to test", result["reason"])

    def test_a_real_teleport_is_still_caught_across_a_gap(self):
        cams = [camera(i, i * 0.01) for i in range(40)]
        cams[20] = None
        for i in range(21, 40):
            cams[i] = camera(i, 100.0 + i * 0.01)  # the path jumps the scene
        write_solve(self.root, cams)
        self.assertFalse(teleport_check(self.root)["ok"])


if __name__ == "__main__":
    unittest.main()
