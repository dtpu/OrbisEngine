"""Projection and inverse-transform checks, independent of any pose model."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from worker.stages.pose_crops import crop_bounds, crop_intrinsics, source_joints


@pytest.mark.parametrize(
    "box,expected",
    [
        ([400.25, 100.75, 700.5, 500.25], (340, 20, 761, 581)),
        ([-20, 40, 180, 300], (0, 0, 220, 352)),
        ([1100, 600, 1300, 800], (1060, 560, 1280, 720)),
    ],
)
def test_bounds_round_outward_and_clamp(box, expected):
    original = np.array(box)
    assert crop_bounds(original, 1280, 720) == expected
    np.testing.assert_array_equal(original, box)
    assert crop_bounds([2.2, 3.3, 4.4, 5.5], 1280, 720, padding=0) == (2, 3, 5, 6)


@pytest.mark.parametrize("box", [[400, 100, 700, 500], [-20, 40, 180, 300]])
def test_calibrated_crop_preserves_3d_rays_and_source_projection(box):
    bounds = crop_bounds(box, 1280, 720)
    K = np.array([[920.0, 31.0, 603.5], [0.0, 870.0, 341.25], [0.0, 0.0, 1.0]])
    original = K.copy()
    points = np.array([[-0.7, -0.3, 2.0], [0.4, 0.5, 3.0], [1.2, -0.2, 4.0]])
    source_h = (K @ points.T).T
    source_pixels = source_h[:, :2] / source_h[:, 2:]
    crop_K = crop_intrinsics(K, bounds)
    crop_h = (crop_K @ points.T).T
    crop_pixels = crop_h[:, :2] / crop_h[:, 2:]
    np.testing.assert_allclose(crop_pixels + bounds[:2], source_pixels, atol=1e-12)
    rays = np.linalg.solve(crop_K, np.column_stack((crop_pixels, np.ones(3))).T).T
    np.testing.assert_allclose(rays, points / points[:, 2:], atol=1e-12)
    assert crop_K[0, 1] == K[0, 1]
    np.testing.assert_array_equal(K, original)

    # Asymmetric center padding and preprocessing padding, with a non-unit resize.
    pad_left, pad_top, factor, offset_w, offset_h = 17, 29, 0.73, 43, 11
    preprocessing = np.array(
        [
            [factor, 0, factor * offset_w + pad_left],
            [0, factor, factor * offset_h + pad_top],
            [0, 0, 1],
        ]
    )
    model_h = (preprocessing @ crop_K @ points.T).T
    model_pixels = model_h[:, :2] / model_h[:, 2:]
    before = model_pixels.copy()
    restored = source_joints(model_pixels, pad_left, pad_top, factor, offset_w, offset_h, bounds)
    np.testing.assert_allclose(restored, source_pixels, atol=1e-12)
    np.testing.assert_array_equal(model_pixels, before)


def test_inverse_does_not_clip_predictions_to_crop():
    expected = np.array([[-5.0, 80.0], [900.0, 1200.0]])
    bounds = (100, 200, 400, 600)
    predicted = (expected - bounds[:2] + [9, 21]) * 1.7 + [3, 15]
    np.testing.assert_allclose(source_joints(predicted, 3, 15, 1.7, 9, 21, bounds), expected)


@pytest.mark.parametrize(
    "box",
    [
        [],
        [1, 2, 3],
        [[1, 2, 3, 4]],
        [0, 0, 0, 1],
        [3, 0, 2, 1],
        [0, 0, np.nan, 1],
        [0, 0, np.inf, 1],
        [1300, 0, 1400, 40],
        [-40, 0, 0, 40],
        [0, 720, 10, 800],
    ],
)
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        crop_bounds(box, 1280, 720)


@pytest.mark.parametrize(
    "name,value",
    [
        ("width", 0),
        ("height", -1),
        ("width", 1.5),
        ("width", True),
        ("height", np.inf),
        ("padding", -0.1),
        ("padding", np.nan),
        ("padding", [0.2]),
        ("padding", "0.2"),
    ],
)
def test_invalid_crop_parameters(name, value):
    kwargs = dict(width=1280, height=720, padding=0.2)
    kwargs[name] = value
    with pytest.raises(ValueError):
        crop_bounds([10, 10, 100, 100], **kwargs)


@pytest.mark.parametrize(
    "bounds", [[], [0, 0, 1], [0, 0, 0, 1], [-1, 0, 2, 3], [0.5, 0, 2, 3], [0, 0, np.inf, 3]]
)
def test_invalid_bounds_rejected_by_both_transforms(bounds):
    with pytest.raises(ValueError):
        crop_intrinsics(np.eye(3), bounds)
    with pytest.raises(ValueError):
        source_joints([[1, 2]], 0, 0, 1, 0, 0, bounds)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2),
        np.zeros((3, 3)),
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
        [[1, np.nan, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
    ],
)
def test_invalid_intrinsics(matrix):
    with pytest.raises(ValueError):
        crop_intrinsics(matrix, (10, 20, 30, 40))


@pytest.mark.parametrize(
    "name,value",
    [
        ("joints", []),
        ("joints", [1, 2]),
        ("joints", [[1, 2, 3]]),
        ("joints", [[np.nan, 2]]),
        ("factor", 0),
        ("factor", -1),
        ("factor", np.inf),
        ("pad_left", -1),
        ("pad_top", np.nan),
        ("offset_w", -1),
        ("offset_h", [1]),
    ],
)
def test_invalid_inverse_inputs(name, value):
    kwargs = dict(
        joints=[[1, 2]],
        pad_left=0,
        pad_top=0,
        factor=1,
        offset_w=0,
        offset_h=0,
        bounds=(10, 20, 30, 40),
    )
    kwargs[name] = value
    with pytest.raises(ValueError):
        source_joints(**kwargs)


def test_overflow_fails_instead_of_returning_nonfinite_geometry():
    with pytest.raises(ValueError):
        crop_bounds([-1e308, 0, 1e308, 20], 1280, 720)
    with pytest.raises(ValueError):
        source_joints([[1e308, 2]], 0, 0, 1e-308, 0, 0, (0, 0, 10, 10))
