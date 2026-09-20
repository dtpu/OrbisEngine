#!/usr/bin/env python3
"""Synthetic calibrated wall observations; no media, model, GPU or network."""

import sys
from pathlib import Path

# The modules under test are this directory's parent; importing them by name is what
# running from scripts/ used to give for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData

from package_static_anchors import package_arrays, track_box_masks, write_gaussian_ply


class StaticAnchorGeometry(unittest.TestCase):
    def fixture(self):
        # Two cameras observe the SAME world marker at raw OpenGL [2,-4,-6].
        # Camera 1 is translated +2 on x; its marker ray is therefore straight ahead.
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        reframe = np.eye(4)
        reframe[:3, :3] = rotation
        reframe[:3, 3] = [10, 20, 30]
        poses, points, raw, packaged = [], [], [], []
        for i in range(2):
            anchor = np.eye(4)
            anchor[:3, :3] = rotation if i else np.eye(3)
            anchor[:3, 3] = [17, -8, 11 + i]
            poses.append(anchor)
            local = np.tile([1 - i, 2, 3], (6, 1)).astype(float)
            local[4, 2] = -3  # Behind the camera, despite a valid flag.
            local[5, 0] = np.nan
            points.append((local @ anchor[:3, :3].T + anchor[:3, 3])[None])
            camera = np.eye(4)
            camera[0, 3] = 2 * i
            row = dict(
                sourceIndex=20 + i * 10,
                time=i * 0.5,
                camera_to_world=camera.tolist(),
                source_intrinsics=[[100, 0, 3], [0, 100, 0.5], [0, 0, 1]],
                source_image_size=[6, 1],
            )
            raw.append(row)
            packaged.append({**row, "camera_to_world": (reframe @ camera).tolist()})
        rgb = np.zeros((2, 1, 6, 3), dtype=np.uint8)
        rgb[0, :, :, 0] = 255
        rgb[1, :, :, 2] = 255
        valid = np.ones((2, 1, 6), dtype=bool)
        valid[:, :, 2] = False
        people = np.zeros_like(valid)
        people[:, :, 1] = True
        conf = np.ones_like(valid, dtype=float)
        conf[:, :, 3] = 0.1
        anchors = dict(
            points=np.array(points),
            poses=np.array(poses),
            rgb=rgb,
            valid=valid,
            people=people,
            conf=conf,
        )
        sequence = dict(anchors=[0, 1], sourceIndices=[20, 30], timestamps=[0, 0.5])
        return (
            anchors,
            sequence,
            dict(reference={"scale": 2}, cameras=raw),
            dict(cameras=packaged[::-1]),
        )

    def run_fixture(self, fixture):
        return package_arrays(
            *fixture, confidence=0.5, min_radius=0.001, max_radius=0.02, pixel_radius=0.6
        )

    def test_shared_marker_color_mask_and_ply_coordinates(self):
        xyz, rgb, radius, report = self.run_fixture(self.fixture())
        # Independently: rotating raw [2,-4,-6] 90 degrees gives [4,2,-6],
        # then translating gives [14,22,24]. Both observations must coincide.
        np.testing.assert_allclose(xyz, [[14, 22, 24], [14, 22, 24]], atol=1e-6)
        np.testing.assert_array_equal(rgb, [[255, 0, 0], [0, 0, 255]])
        np.testing.assert_allclose(radius, [0.02, 0.02])
        self.assertEqual([row["personExcluded"] for row in report["anchors"]], [1, 1])
        self.assertEqual([row["kept"] for row in report["anchors"]], [1, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.ply"
            write_gaussian_ply(path, xyz, rgb, scale=radius)
            vertices = PlyData.read(path)["vertex"].data
            decoded_xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
            decoded_rgb = (
                np.column_stack([vertices[f"f_dc_{i}"] for i in range(3)]) * 0.28209479177387814
                + 0.5
            )
            np.testing.assert_allclose(decoded_xyz, xyz)
            np.testing.assert_allclose(decoded_rgb, rgb / 255, atol=1e-6)
            np.testing.assert_allclose(np.exp(vertices["scale_0"]), radius, atol=1e-7)

    def test_mismatched_camera_frame_is_rejected(self):
        fixture = copy.deepcopy(self.fixture())
        fixture[3]["cameras"][0]["camera_to_world"][0][3] += 1
        with self.assertRaisesRegex(ValueError, "rigid reframe"):
            self.run_fixture(fixture)

    def test_mismatched_source_time_is_rejected(self):
        fixture = copy.deepcopy(self.fixture())
        fixture[3]["cameras"][0]["time"] += 0.1
        with self.assertRaisesRegex(ValueError, "camera time"):
            self.run_fixture(fixture)

    def test_source_bound_mask_and_anchor_selection(self):
        fixture = self.fixture()
        motions = {"track": {"frames": [{"sourceIndex": 30, "time": 0.5, "maskBox": [0, 0, 1, 1]}]}}
        masks, evidence = track_box_masks(motions, fixture[1], fixture[2], (2, 1, 6), 0)
        self.assertFalse(masks[0].any())
        self.assertTrue(masks[1, 0, 0])
        self.assertEqual(evidence[0]["sourceIndex"], 30)
        xyz, rgb, _, report = package_arrays(
            *fixture,
            confidence=0.5,
            min_radius=0.001,
            max_radius=0.02,
            pixel_radius=0.6,
            extra_masks=masks,
            selected_samples=[0, 1],
        )
        np.testing.assert_allclose(xyz, [[14, 22, 24]])
        np.testing.assert_array_equal(rgb, [[255, 0, 0]])
        self.assertEqual(report["anchors"][1]["extraMaskExcluded"], 1)
        _, selected_rgb, _, selected_report = package_arrays(
            *fixture,
            confidence=0.5,
            min_radius=0.001,
            max_radius=0.02,
            pixel_radius=0.6,
            selected_samples=[1],
        )
        np.testing.assert_array_equal(selected_rgb, [[0, 0, 255]])
        self.assertEqual(selected_report["anchors"][0]["sample"], 1)


if __name__ == "__main__":
    unittest.main()
