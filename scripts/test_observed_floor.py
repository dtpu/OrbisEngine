#!/usr/bin/env python3
"""Synthetic floor-measurement regressions: a plane plus a box, a camera path, a foot track."""

import unittest

import numpy as np

from observed_floor import fit_offset, gravity_normal, measure_plane, track_mask

UP = np.array([0.0, 1.0, 0.0])


def plane_and_box(seed=7):
    """A ground plane at y = -1.6 plus a raised box top at y = -0.4 standing off to one side.

    The box is the trap: it holds more points than the walked strip of ground, so any fit that
    takes the densest horizontal surface in the whole cloud lands on the box instead of the floor.
    """
    rng = np.random.default_rng(seed)
    ground = np.stack(
        [rng.uniform(-6, 6, 4000), rng.normal(-1.6, 0.004, 4000), rng.uniform(-6, 6, 4000)], 1
    )
    box = np.stack(
        [
            rng.uniform(3.0, 5.0, 30000),
            rng.normal(-0.4, 0.004, 30000),
            rng.uniform(3.0, 5.0, 30000),
        ],
        1,
    )
    return np.concatenate([ground, box])


class FloorMeasurement(unittest.TestCase):
    def test_gravity_normal_requires_a_levelled_package(self):
        self.assertTrue(np.array_equal(gravity_normal("OpenGL, gravity on +y (x)"), UP))
        with self.assertRaisesRegex(ValueError, "levelled"):
            gravity_normal("OpenGL, camera 0 keeps its own orientation")

    def test_walked_column_rejects_a_higher_denser_surface(self):
        cloud = plane_and_box()
        track = np.stack([np.linspace(-2, 2, 40), np.linspace(-1, 1, 40)], 1)
        got = measure_plane(cloud, track, UP, -1.3, radius=0.6, band=0.05, body_height=1.0)
        self.assertAlmostEqual(got["offset"], -1.6, places=2)
        self.assertLess(got["residualRmsBodyHeights"], 0.01)
        # the same cloud, fitted over the box's own footprint, finds the box: the column is what
        # separates them, not the histogram
        over_box = np.stack([np.linspace(3.5, 4.5, 40), np.linspace(3.5, 4.5, 40)], 1)
        self.assertAlmostEqual(
            measure_plane(cloud, over_box, UP, 0.0, radius=0.6, band=0.05, body_height=1.0)[
                "offset"
            ],
            -0.4,
            places=2,
        )

    def test_a_surface_above_the_feet_is_not_a_floor(self):
        cloud = plane_and_box()
        over_box = np.stack([np.linspace(3.5, 4.5, 40), np.linspace(3.5, 4.5, 40)], 1)
        with self.assertRaisesRegex(ValueError, "no floor was measured"):
            measure_plane(cloud, over_box, UP, -2.0, radius=0.6, band=0.05, body_height=1.0)

    def test_offset_seeds_on_the_mode_not_the_low_tail(self):
        rng = np.random.default_rng(3)
        surface = rng.normal(0.0, 0.003, 4000)
        fuzz = rng.uniform(-0.30, -0.08, 600)  # sub-floor fuzz a low percentile would sit in
        offset, inliers = fit_offset(np.concatenate([surface, fuzz]), band=0.04)
        self.assertAlmostEqual(offset, 0.0, places=3)
        self.assertEqual(int(inliers[4000:].sum()), 0)

    def test_track_mask_keeps_a_neighbourhood_and_drops_the_rest(self):
        points = np.array([[0.0, 0, 0], [0.5, 9, 0.5], [40.0, 0, 40.0]])
        near = track_mask(points, np.array([[0.0, 0.0]]), radius=1.0)
        self.assertEqual(near.tolist(), [True, True, False])
        with self.assertRaisesRegex(ValueError, "Radius"):
            track_mask(points, np.array([[0.0, 0.0]]), radius=0.0)

    def test_too_few_observed_points_refuses_rather_than_guessing(self):
        cloud = plane_and_box()
        far = np.array([[80.0, 80.0]])
        with self.assertRaisesRegex(ValueError, "no floor was measured"):
            measure_plane(cloud, far, UP, -1.3, radius=0.6, band=0.05, body_height=1.0)

    def test_tilted_normal_must_be_a_unit_vector(self):
        with self.assertRaisesRegex(ValueError, "unit vector"):
            measure_plane(
                plane_and_box(),
                np.zeros((1, 2)),
                np.array([0.0, 2.0, 0.0]),
                -1.3,
                radius=0.6,
                band=0.05,
                body_height=1.0,
            )


class CalibratorContact(unittest.TestCase):
    """The measured plane feeds scripts/calibrate_person_floor.py; check the pair end to end."""

    def test_measured_floor_puts_a_synthetic_track_on_the_ground(self):
        from calibrate_person_floor import solve_scale

        cloud = plane_and_box()
        centers = np.stack(
            [np.linspace(-2, 2, 40), np.full(40, 0.4), np.linspace(-1, 1, 40)], 1
        )  # camera path above the ground
        # feet reconstructed too shallow: only three quarters of the way down each camera ray
        levels = centers[:, 1] + 0.75 * (-1.6 - centers[:, 1])
        track = np.stack([centers[:, 0], centers[:, 2]], 1)
        plane = measure_plane(cloud, track, UP, float(np.median(levels)), 0.6, 0.05, 1.0)
        scale, _, before, after, spread = solve_scale(
            levels, centers, UP, plane["offset"], 1.0, 0.05
        )
        self.assertAlmostEqual(scale, 1 / 0.75, places=2)
        self.assertGreater(float(np.median(np.abs(before))), 0.4)
        self.assertLess(float(np.max(np.abs(after))), 0.01)
        self.assertLess(spread, 0.05)


if __name__ == "__main__":
    unittest.main()
