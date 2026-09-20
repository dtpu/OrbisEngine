#!/usr/bin/env python3
"""Synthetic floor contact and pinhole projection regressions; no media or services."""

import unittest

import numpy as np

from calibrate_person_floor import (
    camera_source_binding,
    contact_offsets,
    corrected_points,
    match_cameras,
    projection_error,
    size_error,
    solve_scale,
)


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


class ContactOffsets(unittest.TestCase):
    def test_per_frame_offset_lands_every_foot_and_keeps_its_pixel(self):
        # Two frames whose recorded depth error is NOT a constant multiple, so one scale cannot
        # contact both: the spread gate refuses it, and the offsets reach the floor anyway.
        centers = np.array([[0.0, 2, 0], [1.0, 2, 0], [-1.0, 2, 0]])
        feet = np.array([[0.2, 0.5, -3.0], [1.1, 1.0, -2.0], [-0.9, 0.8, -2.5]])
        # each frame's own contact scale: extend the camera-to-foot ray until it reaches y = 0
        candidates = (0 - centers[:, 1]) / (feet[:, 1] - centers[:, 1])
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            solve_scale(feet[:, 1], centers, [0, 1, 0], 0, 1.7, 0.2)
        scale = float(np.median(candidates))
        offsets = contact_offsets(candidates, scale, feet, centers)
        for i in range(len(centers)):
            moved = corrected_points(feet[i : i + 1], centers[i], scale, offsets[i])
            exact = centers[i] + candidates[i] * (feet[i] - centers[i])
            np.testing.assert_allclose(moved[0], exact, atol=1e-12)
            self.assertAlmostEqual(float(moved[0][1]), 0.0)  # on the y = 0 floor
            camera = dict(
                camera_to_world=np.block(
                    [[np.eye(3), centers[i][:, None]], [np.zeros(3), 1]]
                ).tolist(),
                source_intrinsics=[[800, 0, 320], [0, 700, 240], [0, 0, 1]],
            )
            # the contacting point itself moved along its own ray, so its pixel is unchanged
            self.assertLess(projection_error(feet[i : i + 1], moved, camera), 1e-9)
        np.testing.assert_allclose(
            size_error(candidates, scale), candidates / scale - 1, atol=1e-12
        )
        self.assertGreater(float(np.max(np.abs(size_error(candidates, scale)))), 0.15)

    def test_contact_offsets_reject_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            contact_offsets([1.0, 2.0], 1.0, np.zeros((3, 3)), np.zeros((3, 3)))


class CameraBinding(unittest.TestCase):
    def test_hash_binding_preferred_and_path_binding_named(self):
        people = dict(sourceSha256="a" * 64, clip="/clips/x.mp4")
        self.assertEqual(camera_source_binding(dict(sourceSha256="a" * 64), people), "sha256")
        with self.assertRaisesRegex(ValueError, "source hashes differ"):
            camera_source_binding(dict(sourceSha256="b" * 64), people)
        self.assertEqual(
            camera_source_binding(dict(sourceClip="/clips/x.mp4"), people), "clip-path"
        )
        with self.assertRaisesRegex(ValueError, "neither a source hash"):
            camera_source_binding(dict(sourceClip="/clips/other.mp4"), people)


if __name__ == "__main__":
    unittest.main()
