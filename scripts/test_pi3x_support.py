"""Offline camera/person support tests, including the actual dense export loop."""

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker/stages"))
from pi3x_support import (
    fit_ray_intrinsics,
    person_sequence_metadata,
    person_support,
    static_scale_reference,
)


def rays():
    y, x = np.mgrid[:30, :40]
    return np.stack([(x + 0.5 - 20) / 40, (y + 0.5 - 15) / 35, np.ones_like(x)], -1)


def camera_function():
    tree = ast.parse((ROOT / "worker/stages/dense_pi3x.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "viewer_camera"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "viewer_camera", "exec"), namespace)
    return namespace["viewer_camera"]


class StaticReferenceTests(unittest.TestCase):
    def fixture(self):
        points = np.zeros((2, 10, 12, 3))
        points[0, ..., 2] = 6
        points[1, ..., 2] = 15
        poses = np.stack([np.eye(4), np.eye(4)])
        poses[1, 2, 3] = 5
        return dict(
            points=points,
            poses=poses,
            valid=np.ones((2, 10, 12), bool),
            people=np.zeros((2, 10, 12), bool),
        )

    def test_first_supported_anchor_uses_own_camera_depth(self):
        base = self.fixture()
        base["valid"][0] = False
        scale, reference = static_scale_reference(base, [0, 2], [0, 2, 4], [0, 0.1, 0.2])
        self.assertEqual(scale, 0.3)
        self.assertEqual(reference["medianOwnCameraDepth"], 10)
        self.assertEqual(reference["sourceIndex"], 4)
        self.assertEqual(reference["anchorOrdinal"], 1)
        camera = camera_function()(
            base["poses"][0], 1, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), scale
        )
        np.testing.assert_array_equal(camera, np.eye(4))

    def test_supported_first_anchor_preserves_original_scale(self):
        self.assertEqual(static_scale_reference(self.fixture(), [0, 1], [0, 2], [0, 0.1])[0], 0.5)

    def test_no_support_and_invalid_depths_fail(self):
        for value in (0, -1, np.nan, np.inf):
            base = self.fixture()
            base["points"][..., 2] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                static_scale_reference(base, [0, 1], [0, 2], [0, 0.1])

    def test_invalid_pose_and_transform_fail(self):
        base = self.fixture()
        base["poses"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            static_scale_reference(base, [0, 1], [0, 2], [0, 0.1])
        with self.assertRaises(ValueError):
            camera_function()(np.eye(4), 1, np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), np.nan)


class IntrinsicTests(unittest.TestCase):
    def test_depth_missing_fits_predicted_rays_with_provenance(self):
        result = fit_ray_intrinsics(rays(), np.zeros((30, 40), bool), (80, 60))
        self.assertEqual(result["intrinsicsSource"], "predicted-rays-depth-unfiltered")
        self.assertEqual(result["intrinsicDepthValidRayCount"], 0)
        self.assertLess(result["intrinsic_fit_pixel_rmse"], 1e-10)
        np.testing.assert_allclose(
            result["source_intrinsics"], [[80, 0, 40], [0, 70, 30], [0, 0, 1]]
        )

    def test_supported_depth_uses_original_ray_selection(self):
        result = fit_ray_intrinsics(rays(), np.ones((30, 40), bool), (40, 30))
        self.assertEqual(result["intrinsicsSource"], "predicted-rays-depth-valid")
        self.assertIsNone(result["intrinsicFallbackMaxPixelRmse"])

    def test_invalid_degenerate_negative_and_high_residual_rays_fail(self):
        invalid = rays()
        invalid[:] = np.nan
        negative = rays()
        negative[..., 0] *= -1
        noisy = rays()
        noisy[..., :2] += np.random.default_rng(1).normal(0, 0.3, noisy[..., :2].shape)
        for candidate in (invalid, np.ones_like(rays()), negative, noisy, rays() * -1):
            with self.assertRaises(ValueError):
                fit_ray_intrinsics(candidate, np.zeros((30, 40), bool), (40, 30))


class PersonExportTests(unittest.TestCase):
    def test_gap_metadata_preserves_slots(self):
        for counts in ([0, 99, 100, 0, 101], [0, 0], [100, 200]):
            support = [person_support(i, 2 * i, i / 12, count) for i, count in enumerate(counts)]
            meta = person_sequence_metadata(support)
            self.assertEqual(len(meta["frames"]), len(counts))
            self.assertEqual(meta["personPresentCount"], sum(n >= 100 for n in counts))
            for i, count in enumerate(counts):
                self.assertEqual(meta["frames"][i], f"frame_{i:03d}.ply" if count >= 100 else None)
            json.dumps(meta, allow_nan=False)

    def test_actual_export_loop_keeps_camera_zero_without_person_or_depth(self):
        tree = ast.parse((ROOT / "worker/stages/dense_pi3x.py").read_text())
        loop = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == "chunk"
        )
        with tempfile.TemporaryDirectory() as directory:
            cloud = rays().copy()
            cloud[..., 2] = 2
            valid = np.ones((2, 30, 40), bool)
            valid[0] = False
            people = np.ones_like(valid)
            people[0] = False
            written = []

            def write(path, xyz, colors):
                self.assertTrue(np.isfinite(xyz).all())
                written.append(path.name)

            env = dict(
                np=np,
                chunk=[0, 1],
                ids=[0, 1],
                p=dict(
                    points=np.stack([cloud, cloud]),
                    rgb=np.zeros((2, 30, 40, 3)),
                    valid=valid,
                    poses=np.stack([np.eye(4), np.eye(4)]),
                    rays=np.stack([rays(), rays()]),
                ),
                pm=people,
                R=np.eye(3),
                s=1,
                t=np.zeros(3),
                R0=np.eye(3),
                t00=np.zeros(3),
                scale=0.5,
                flip=np.array([1, -1, -1]),
                indices=np.array([0, 2]),
                times=np.array([0, 1 / 12]),
                support=[],
                point_counts=[],
                cams=[None, None],
                out=Path(directory),
                home=None,
                home_frame=None,
                source_size=(40, 30),
                person_support=person_support,
                fit_ray_intrinsics=fit_ray_intrinsics,
                viewer_camera=camera_function(),
                write_point_ply=write,
                home_distance=lambda xyz: 1.0,
            )
            exec(
                compile(ast.Module(body=[loop], type_ignores=[]), "dense_export_loop", "exec"), env
            )
            self.assertEqual(written, ["frame_001.ply"])
            self.assertEqual(env["home_frame"]["sourceIndex"], 2)
            self.assertEqual([c["sourceIndex"] for c in env["cams"]], [0, 2])
            self.assertTrue((Path(directory) / "source-camera-0.npz").exists())
            json.dumps(env["cams"], allow_nan=False)
            # The complete camera-only path must not invent a home/person frame.
            env.update(home=None, home_frame=None, support=[], point_counts=[], cams=[None, None])
            env["pm"][:] = False
            written.clear()
            exec(
                compile(ast.Module(body=[loop], type_ignores=[]), "dense_export_loop", "exec"), env
            )
            self.assertEqual(written, [])
            self.assertIsNone(env["home"])
            self.assertTrue(person_sequence_metadata(env["support"])["cameraOnly"])
            json.dumps(env["cams"], allow_nan=False)
            # Do not silently accept or delete a stale person file for a gap.
            (Path(directory) / "frame_000.ply").write_bytes(b"retained evidence")
            with self.assertRaisesRegex(RuntimeError, "Conflicting retained"):
                exec(
                    compile(ast.Module(body=[loop], type_ignores=[]), "dense_export_loop", "exec"),
                    env,
                )


if __name__ == "__main__":
    unittest.main()
