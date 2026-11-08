"""Synthetic camera and world-point contracts for anchor-constrained stitching."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.wander_worker.camera_stitch import stitch_camera_batch


def pose(rotation=(0, 0, 0), center=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = center
    return result


def fixture():
    times = np.array([0.1, 0.3, 0.9, 1.7, 2.1])
    batch = np.array([pose((0.1 * t, -0.2 * t, 0.05 * t), (t, t**2, -t)) for t in times])
    corrections = np.array(
        [pose((0.25, -0.1, 0.3), (1, 2, -1)), pose((-0.2, 0.35, 0.1), (-2, 1, 3))]
    )
    canonical = corrections @ batch[[0, -1]]
    return batch, times, times[[0, -1]], canonical, corrections


def test_noncommuting_correction_uses_recorded_time_and_preserves_camera_points():
    batch, times, anchors, canonical, corrections = fixture()
    saved = batch.copy()
    corrected, maps = stitch_camera_batch(batch, times, anchors, canonical)
    np.testing.assert_array_equal(batch, saved)
    np.testing.assert_array_equal(corrected[[0, -1]], canonical)
    # Independent fractional axis-angle composition checks interpolation and its
    # left/right ordering. Timestamp 0.3 is 10%, not the ordinal fraction 25%.
    delta = Rotation.from_matrix(corrections[0, :3, :3].T @ corrections[1, :3, :3])
    expected = np.eye(4)
    expected[:3, :3] = (
        corrections[0, :3, :3] @ Rotation.from_rotvec(delta.as_rotvec() * 0.1).as_matrix()
    )
    expected[:3, 3] = 0.9 * corrections[0, :3, 3] + 0.1 * corrections[1, :3, 3]
    np.testing.assert_allclose(corrected[1], expected @ batch[1], atol=2e-14)
    points = np.array([[1.2, -0.5, 3, 1], [-2, 1.1, 0.4, 1]]).T
    for old, new, transform in zip(batch, corrected, maps):
        np.testing.assert_allclose(
            np.linalg.inv(new) @ transform @ points, np.linalg.inv(old) @ points, atol=2e-14
        )
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=2e-14)
        assert np.linalg.det(new[:3, :3]) == pytest.approx(1)


def test_already_shared_batch_has_exact_identity_point_maps():
    batch, times, anchors, _, _ = fixture()
    corrected, maps = stitch_camera_batch(batch, times, anchors, batch[[0, -1]])
    np.testing.assert_array_equal(corrected, batch)
    np.testing.assert_array_equal(maps, np.broadcast_to(np.eye(4), batch.shape))


def test_all_anchor_intervals_are_used():
    batch = np.repeat(np.eye(4)[None], 5, axis=0)
    canonical = np.array([pose(center=(0, 0, 0)), pose(center=(2, 0, 0)), pose(center=(0, 0, 0))])
    corrected, _ = stitch_camera_batch(batch, [0, 1, 2, 3, 4], [0, 2, 4], canonical)
    np.testing.assert_allclose(corrected[:, 0, 3], [0, 1, 2, 1, 0])


def test_small_inference_roundoff_is_repaired_without_losing_point_coordinates():
    batch, times, anchors, canonical, _ = fixture()
    batch[:, :3, 0] *= 1.00002
    corrected, maps = stitch_camera_batch(batch, times, anchors, canonical)
    np.testing.assert_allclose(
        corrected[:, :3, :3].swapaxes(1, 2) @ corrected[:, :3, :3],
        np.broadcast_to(np.eye(3), (len(times), 3, 3)),
        atol=2e-14,
    )
    np.testing.assert_allclose(np.linalg.inv(corrected) @ maps, np.linalg.inv(batch), atol=2e-14)


def test_near_pi_corrections_take_short_arc():
    batch = np.repeat(np.eye(4)[None], 3, axis=0)
    canonical = np.array(
        [pose(rotation=(0, 0, np.deg2rad(179))), pose(rotation=(0, 0, np.deg2rad(-179)))]
    )
    corrected, _ = stitch_camera_batch(batch, [0, 0.5, 1], [0, 1], canonical)
    np.testing.assert_allclose(corrected[1], pose(rotation=(0, 0, np.pi)), atol=2e-14)


@pytest.mark.parametrize("defect", ["scale", "shear"])
def test_rejects_distortion_beyond_numerical_rotation_tolerance(defect):
    batch = np.repeat(np.eye(4)[None], 3, axis=0)
    canonical = batch[[0, -1]].copy()
    if defect == "scale":
        batch[1, 0, 0] += 2e-3
    else:
        batch[1, 0, 1] += 5e-3
    with pytest.raises(ValueError, match="scale or shear"):
        stitch_camera_batch(batch, [0, 0.5, 1], [0, 1], canonical)


@pytest.mark.parametrize(
    "times,anchors",
    [
        ([0, 0, 2], [0, 2]),
        ([2, 1, 0], [0, 2]),
        ([0, np.nan, 2], [0, 2]),
        ([0, np.inf, 2], [0, 2]),
        ([[0, 1, 2]], [0, 2]),
        ([0], [0, 2]),
        ([0, 1, 2], [0, 0]),
        ([0, 1, 2], [2, 0]),
        ([0, 1, 2], [0, np.nan]),
        ([0, 1, 2], [0, 1.5, 2]),
        ([0, 1, 2], [0.5, 2]),
        ([0, 1, 2], [0, 1.5]),
    ],
)
def test_rejects_invalid_timestamp_manifests(times, anchors):
    with pytest.raises(ValueError):
        stitch_camera_batch(
            np.repeat(np.eye(4)[None], 3, axis=0),
            times,
            anchors,
            np.repeat(np.eye(4)[None], 2, axis=0),
        )


@pytest.mark.parametrize("target", ["batch", "canonical"])
@pytest.mark.parametrize(
    "defect", ["shape", "nan", "infinity", "bottom", "reflection", "scale", "shear"]
)
def test_rejects_invalid_poses(target, defect):
    batch, times, anchors, canonical, _ = fixture()
    value = batch if target == "batch" else canonical
    if defect == "shape":
        value = value[:, :3]
    elif defect == "nan":
        value[0, 0, 3] = np.nan
    elif defect == "infinity":
        value[0, 0, 3] = np.inf
    elif defect == "bottom":
        value[0, 3, 0] = 0.1
    elif defect == "reflection":
        value[0, :3, 0] *= -1
    elif defect == "scale":
        value[0, :3, :3] *= 2
    else:
        value[0, 0, 1] += 0.5
    if target == "batch":
        batch = value
    else:
        canonical = value
    with pytest.raises(ValueError):
        stitch_camera_batch(batch, times, anchors, canonical)
