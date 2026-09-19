"""Synthetic timing regression tests for object packaging."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lift_object_3d
import package_objects
from package_objects import bake_orientation, ballistic_position, load_cameras
from source_clock import camera_source_fps, resolve_source_fps


class PackageObjectTimingTests(unittest.TestCase):
    def package_fixture(self, root, fps, fit_fps=None):
        world = root / "world"
        world.mkdir()
        indices = [0, fps, 2 * fps]
        people = {
            "fps": 1,
            "samples": 3,
            "sourceIndices": indices,
            "timestamps": [0, 1, 2],
            "floorFit": {
                "metresPerWorldUnit": 1,
                "sharedCameraDrift": {"samples": [0, 1, 2], "offsetYUnits": [0, 0, 0]},
                "levels": {"C_test": {"translationY": {"person0": 0, "person1": 0}}},
            },
            "people": [{"id": f"person{i}", "track": i, "registrationScale": 1} for i in range(2)],
        }
        (world / "people.json").write_text(json.dumps(people))
        cameras = root / "cameras.json"
        cameras.write_text(json.dumps({"cameras": [self.camera(i, i / fps) for i in indices]}))
        for track in range(2):
            directory = root / "tracks" / f"track_{track:02d}"
            directory.mkdir(parents=True)
            joints = np.tile([track, 0, 2.0], (22, 1))
            torch.save(
                {"sourceIndices": indices, "poses": [{"j3d": joints} for _ in indices]},
                directory / "source-poses.pt",
            )
        fit = root / "fit.json"
        fit.write_text(
            json.dumps(
                {
                    "schema": "wander.object-fit/2",
                    "fps": fps if fit_fps is None else fit_fps,
                    "gravityUnitsPerS2": 9.80665,
                    "flights": [
                        {
                            "flight": [0, 2 * fps],
                            "thrower": 0,
                            "catcher": 1,
                            "frames": list(range(2 * fps + 1)),
                            "flightSec": 2,
                            "spanM": 2,
                            "fits": {
                                "handAnchored": {
                                    "p0": [0, 0, -2],
                                    "v0": [1, 4, 0],
                                    "reprojPxRms": 0,
                                    "reprojPxMax": 0,
                                    "p0SigmaM": [0, 0, 0],
                                    "speedMs": 1,
                                }
                            },
                        }
                    ],
                }
            )
        )
        args = [
            "package_objects.py",
            "--world",
            str(world),
            "--fit",
            str(fit),
            "--tracks-dir",
            str(root / "tracks"),
            "--cameras",
            str(cameras),
        ]
        return args, world / "objects.json"

    def test_real_package_command_preserves_one_second_position_at_each_source_rate(self):
        for fps in (24, 30, 60):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as directory:
                args, output = self.package_fixture(Path(directory), fps)
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    package_objects.main()
                track = json.loads(output.read_text())["objects"][0]["bakedTrack"]
                self.assertEqual(track["fps"], fps)
                self.assertEqual(len(track["positions"]), 2 * fps + 1)
                np.testing.assert_allclose(track["positions"][fps], [1, -0.903325, -2], atol=1e-5)

    def test_real_lift_command_carries_detected_source_rate_into_fit(self):
        for fps in (24, 30, 60):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.package_fixture(root, fps)
                flights = root / "flights.json"
                flights.write_text(json.dumps({"fps": fps, "flights": []}))
                output = root / "lifted.json"
                args = [
                    "lift_object_3d.py",
                    "--flights",
                    str(flights),
                    "--people",
                    str(root / "world/people.json"),
                    "--cameras",
                    str(root / "cameras.json"),
                    "--tracks-dir",
                    str(root / "tracks"),
                    "--out",
                    str(output),
                ]
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    lift_object_3d.main()
                self.assertEqual(json.loads(output.read_text())["fps"], fps)
                output.write_text("previous accepted fit")
                with (
                    patch.object(sys, "argv", [*args, "--fps", str(fps * 2)]),
                    self.assertRaisesRegex(SystemExit, "conflict"),
                ):
                    lift_object_3d.main()
                self.assertEqual(output.read_text(), "previous accepted fit")

    def test_real_package_rejects_conflicting_fit_before_overwriting_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output = self.package_fixture(Path(directory), 60, fit_fps=30)
            output.write_text("existing manifest")
            with patch.object(sys, "argv", args), self.assertRaisesRegex(SystemExit, "conflict"):
                package_objects.main()
            self.assertEqual(output.read_text(), "existing manifest")

    @staticmethod
    def camera(index, time=None):
        row = {
            "sourceIndex": index,
            "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "source_intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        if time is not None:
            row["time"] = time
        return row

    def test_camera_time_recovers_non_thirty_source_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(
                json.dumps(
                    {"cameras": [self.camera(index, index / 60) for index in range(0, 181, 60)]}
                )
            )
            self.assertAlmostEqual(load_cameras(path)["source_fps"], 60.0)

    def test_legacy_camera_without_times_keeps_thirty_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps({"cameras": [self.camera(i) for i in range(3)]}))
            self.assertIsNone(load_cameras(path)["source_fps"])
            self.assertEqual(resolve_source_fps(camera=None), 30.0)

    def test_partial_reversed_variable_and_nonfinite_clocks_are_rejected(self):
        for times in ([0, None, 2], [0, 2, 1], [0, 1, 3], [0, float("nan"), 2], [0, True, 2]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                camera_source_fps([self.camera(i * 60, t) for i, t in enumerate(times)])

    def test_invalid_and_conflicting_metadata_are_rejected(self):
        for invalid in (True, 0, -30, float("nan"), float("inf"), "60"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_source_fps(fit=invalid, camera=60)
        with self.assertRaisesRegex(ValueError, "conflict"):
            resolve_source_fps(fit=30, camera=60)
        self.assertEqual(resolve_source_fps(flights=60, camera=60), 60)

    def test_ballistic_position_is_source_clock_invariant(self):
        expected = [1.0, 2.0 - 0.5 * 9.81, 0.0]
        for fps in (24.0, 30.0, 60.0):
            position = ballistic_position([1, 2, 0], [0, 0, 0], [0, -9.81, 0], fps, 0, fps)
            self.assertEqual(position.tolist(), expected)

    def test_spin_clock_is_source_rate_invariant(self):
        quaternions = []
        for fps in (24.0, 30.0, 60.0):
            frames = [0, fps / 2]
            positions = [[0, 0, 0], [1, 0, 0]]
            segments = [
                {
                    "kind": "free",
                    "fromSourceFrame": 0,
                    "toSourceFrame": int(fps / 2),
                    "t0SourceFrame": 0,
                    "orientation": {
                        "mode": "measuredRate",
                        "spinRevPerSec": 1.0,
                        "spinAxisWorld": [0, 1, 0],
                    },
                }
            ]
            quaternions.append(bake_orientation(frames, positions, {}, segments, fps=fps)[-1])
        for quaternion in quaternions[1:]:
            self.assertAlmostEqual(abs(float(quaternion @ quaternions[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
