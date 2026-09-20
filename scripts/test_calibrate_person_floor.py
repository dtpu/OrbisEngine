#!/usr/bin/env python3
"""Synthetic floor contact and pinhole projection regressions; no media or services."""

import unittest

import numpy as np

from calibrate_person_floor import corrected_points, match_cameras, projection_error, solve_scale


class CameraCentredFloor(unittest.TestCase):
    def test_moving_camera_scale_contacts_floor_and_preserves_pixels(self):
        centers = np.array([[0, 2, 0], [1, 2.2, 0.1], [-0.5, 1.8, -0.2]])
        # Ground is y=0. Input foot height is one sixth of camera height,
        # so extending the camera-to-foot ray by 6/5 reaches ground exactly.
        levels = centers[:, 1] / 6
        scale, _, before, after, _ = solve_scale(levels, centers, [0, 1, 0], 0, 1.7, 0.01)
        self.assertAlmostEqual(scale, 1.2)
        self.assertTrue(np.all(before > 0))
        np.testing.assert_allclose(after, 0, atol=1e-15)
        for center, low in zip(centers, levels):
            points = np.array(
                [[center[0] + 0.2, low, center[2] - 3], [center[0] - 0.4, low + 1, center[2] - 2]]
            )
            corrected = corrected_points(points, center, scale)
            self.assertAlmostEqual(corrected[0, 1], 0)
            pose = np.eye(4)
            pose[:3, 3] = center
            camera = dict(
                camera_to_world=pose.tolist(),
                source_intrinsics=[[800, 0, 320], [0, 700, 240], [0, 0, 1]],
            )
            self.assertLess(projection_error(points, corrected, camera), 1e-10)
            # Viewer adds offsets AFTER its global scale. This independent final
            # world-space calculation catches forgetting the g multiplier.
            g = 0.4
            world = g * scale * points + g * (1 - scale) * center
            np.testing.assert_allclose(world, g * corrected, atol=1e-15)
            np.testing.assert_allclose(
                (world - g * center) / (g * scale), points - center, atol=1e-15
            )

    def test_tilted_floor_uses_plane_normal(self):
        normal = np.array([0.1, 1, -0.05])
        normal /= np.linalg.norm(normal)
        centers = np.array([[0, 2, 0], [1, 2, 1], [-1, 3, 2]])
        camera_levels = centers @ normal
        levels = camera_levels + (-0.5 - camera_levels) / 1.1
        scale, _, _, after, _ = solve_scale(levels, centers, normal, -0.5, 1.7, 0.01)
        self.assertAlmostEqual(scale, 1.1)
        np.testing.assert_allclose(after, 0, atol=1e-15)

    def test_degenerate_and_unstable_scales_fail(self):
        centers = np.array([[0, 2, 0]] * 3)
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            solve_scale([2, 2, 2], centers, [0, 1, 0], 0, 1.7, 0.2)
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            solve_scale([-2, 0, 1], centers, [0, 1, 0], 0, 1.7, 0.2)

    def test_sequence_gaps_match_source_indices_not_row_numbers(self):
        def camera(source, time):
            return dict(sourceIndex=source, time=time, camera_to_world=np.eye(4).tolist())

        cameras = dict(cameras=[camera(10, 0.5), camera(30, 1.5), camera(20, 1)])
        sequence = dict(
            frames=["first.ply", "third.ply"], sourceIndices=[10, 30], timestamps=[0.5, 1.5]
        )
        self.assertEqual([c["sourceIndex"] for c in match_cameras(sequence, cameras)], [10, 30])
        sequence["timestamps"][1] = 1.6
        with self.assertRaisesRegex(ValueError, "source index/time"):
            match_cameras(sequence, cameras)


if __name__ == "__main__":
    unittest.main()
