"""Small geometric/analytic exposure fixtures independent of the private audit."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat_observations import measure_source_exposure


def scene(xyz=((0, 0, 2),), opacity=None, sigma=0.2):
    n = len(xyz)
    return dict(
        n=n,
        xyz=np.asarray(xyz, np.float32),
        alpha=np.full(n, 0.8, np.float32) if opacity is None else np.asarray(opacity, np.float32),
        scale=np.full((n, 3), sigma, np.float32),
        quat=np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1)),
        rgb=np.full((n, 3), 0.5, np.float32),
    )


def inputs(world=None, size=9, frames=1):
    return dict(
        world=scene() if world is None else world,
        c2w_gl=np.tile(np.eye(4), (frames, 1, 1)),
        intrinsics=np.tile(
            np.array([[10, 0, size / 2], [0, 10, size / 2], [0, 0, 1.0]]), (frames, 1, 1)
        ),
        masks={f: np.zeros((size, size), bool) for f in range(frames)},
        origin_c2w=np.eye(4),
        scale=1.0,
        frame_indices=list(range(frames)),
    )


def test_analytic_one_pixel_alpha_cap_and_occlusion():
    args = inputs(scene(((0, 0, 1), (0, 0, 2), (0, 0, 3)), [1, 1, 1]), size=1)
    result = measure_source_exposure(**args)
    # The first center contributes .99; the second receives .01 transmittance.
    # Their combined opacity leaves only .0001, below the third splat cutoff.
    np.testing.assert_allclose(result["rendered_mass"], [0.99, 0.0099, 0], rtol=3e-8)
    np.testing.assert_array_equal(result["clear_views"], [1, 1, 0])


def test_footprint_crosses_mask_while_center_is_masked():
    args = inputs()
    args["masks"][0][:, :5] = True  # Center is in masked column 4.
    result = measure_source_exposure(**args)
    assert result["masked_mass"][0] > result["clear_mass"][0] > 0
    np.testing.assert_allclose(
        result["rendered_mass"], result["masked_mass"] + result["clear_mass"], atol=1e-12
    )
    # An independent isotropic Gaussian sum over the 7x7 retained radius.
    yy, xx = np.mgrid[1:8, 1:8]
    alpha = 0.8 * np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / (2 * 1.3))
    alpha[alpha < 0.004] = 0
    np.testing.assert_allclose(result["rendered_mass"], alpha.sum(), rtol=1e-6)
    np.testing.assert_allclose(result["clear_mass"], alpha[:, 4:].sum(), rtol=1e-6)


def test_masks_partition_mass_and_counts_are_views_not_pixels():
    args = inputs(frames=3)
    args["frame_indices"] = [2, 0, 1]
    args["masks"][1][:] = True
    seen = []
    args["progress"] = lambda *value: seen.append(value)
    result = measure_source_exposure(**args)
    assert result["clear_mass"][0] > 2
    np.testing.assert_array_equal(result["clear_views"], [2])
    np.testing.assert_allclose(result["clear_mass"], 2 * result["masked_mass"])
    np.testing.assert_allclose(
        result["rendered_mass"], result["masked_mass"] + result["clear_mass"]
    )
    assert seen == [(2, 1, 3), (0, 2, 3), (1, 3, 3)]
    np.testing.assert_array_equal(result["frame_indices"], [2, 0, 1])
    assert result["clear_views"].dtype == np.int32
    assert result["rendered_mass"].dtype == np.float64


def test_masks_do_not_remove_occluders():
    args = inputs(scene(((0, 0, 1), (0, 0, 2)), [0.99, 0.8]))
    baseline = measure_source_exposure(**args)
    args["masks"][0][:] = True
    masked = measure_source_exposure(**args)
    np.testing.assert_array_equal(masked["rendered_mass"], baseline["rendered_mass"])
    np.testing.assert_array_equal(masked["masked_mass"], baseline["rendered_mass"])
    np.testing.assert_array_equal(masked["clear_views"], [0, 0])
    alone = measure_source_exposure(**inputs(scene(((0, 0, 2),), [0.8])))
    assert baseline["rendered_mass"][1] < alone["rendered_mass"][0]


def test_offscreen_center_can_contribute_footprint():
    args = inputs(scene(((-1, 0, 2),), sigma=0.5))
    # Projected center u=-.5, outside the image, with footprint extending inside.
    result = measure_source_exposure(**args)
    assert result["clear_mass"][0] > 0
    assert result["clear_views"][0] == 1


def test_rotated_anisotropic_footprint_matches_analytic_axes():
    world = scene()
    world["scale"][0] = [0.4, 0.1, 0.1]
    world["quat"][0] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    args = inputs(world)
    args["masks"][0][:, :5] = True
    result = measure_source_exposure(**args)
    # A quarter turn swaps projected x/y variances to .55 and 4.3.
    yy, xx = np.mgrid[-4:5, -4:5]
    alpha = 0.8 * np.exp(-0.5 * (xx**2 / 0.55 + yy**2 / 4.3))
    alpha[alpha < 0.004] = 0
    np.testing.assert_allclose(result["rendered_mass"], alpha.sum(), rtol=1e-6)
    np.testing.assert_allclose(result["clear_mass"], alpha[:, 5:].sum(), rtol=1e-6)


def test_no_projected_points_returns_zero_without_changing_inputs():
    args = inputs(scene(((0, 0, -1), (0, 0, 0.01), (100, 0, 1))))
    snapshot = {
        key: value.copy() for key, value in args["world"].items() if isinstance(value, np.ndarray)
    }
    result = measure_source_exposure(**args)
    for key in ("rendered_mass", "masked_mass", "clear_mass", "clear_views"):
        np.testing.assert_array_equal(result[key], [0, 0, 0])
    for key, value in snapshot.items():
        np.testing.assert_array_equal(args["world"][key], value)


def test_nonidentity_origin_and_scale_preserve_matching_camera_projection():
    args = inputs(scene(((0.1, -0.2, 2),)))
    expected = measure_source_exposure(**args)
    origin = np.array([[0, 0, 1, 7], [0, 1, 0, -3], [-1, 0, 0, 2], [0, 0, 0, 1.0]])
    args["origin_c2w"] = origin
    args["c2w_gl"][0] = origin
    args["scale"] = 2.0
    actual = measure_source_exposure(**args)
    for key in ("clear_mass", "masked_mass", "rendered_mass", "clear_views"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=1e-8)


def test_progress_exception_cancels_after_first_view():
    args = inputs(frames=2)
    seen = []

    def cancel(frame, completed, total):
        seen.append(frame)
        raise RuntimeError("cancel requested")

    with pytest.raises(RuntimeError, match="cancel requested"):
        measure_source_exposure(**args, progress=cancel)
    assert seen == [0]


@pytest.mark.parametrize("frames", [[], [True], [0.0, 1.0], [-1], [2], [0, 0], "0", None])
def test_invalid_frame_ids(frames):
    args = inputs(frames=2)
    args["frame_indices"] = frames
    with pytest.raises(ValueError):
        measure_source_exposure(**args)


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(scale=0),
        lambda a: a.update(scale=True),
        lambda a: a.update(scale=np.inf),
        lambda a: a.update(c2w_gl=np.eye(4)),
        lambda a: a["c2w_gl"].__setitem__((1, 0, 0), np.nan),
        lambda a: a["c2w_gl"].__setitem__((1, 3, 3), 0),
        lambda a: a["c2w_gl"].__setitem__((1, 0, 0), 0),
        lambda a: a["origin_c2w"].__setitem__((0, 0), 0),
        lambda a: a["intrinsics"].__setitem__((1, 0, 0), -1),
        lambda a: a["intrinsics"].__setitem__((1, 0, 1), 0.1),
        lambda a: a["intrinsics"].__setitem__((1, 2, 2), 0),
        lambda a: a["masks"].pop(1),
        lambda a: a["masks"].update({1: np.zeros((9, 9), np.uint8)}),
        lambda a: a["masks"].update({1: np.zeros((8, 9), bool)}),
        lambda a: a["masks"].update({1: np.zeros((0, 9), bool)}),
        lambda a: a["world"].update(n=0),
        lambda a: a["world"].update(n=True),
        lambda a: a["world"].update(xyz=np.zeros((2, 3))),
        lambda a: a["world"]["xyz"].__setitem__((0, 0), np.inf),
        lambda a: a["world"]["scale"].__setitem__((0, 0), 0),
        lambda a: a["world"]["alpha"].__setitem__(0, 1.1),
        lambda a: a["world"]["quat"].__setitem__(0, 0),
        lambda a: a["world"].pop("rgb"),
    ],
)
def test_bad_inputs_rejected_before_any_progress(change):
    args = inputs(frames=2)
    change(args)
    seen = []
    with pytest.raises(ValueError):
        measure_source_exposure(**args, progress=lambda *x: seen.append(x))
    assert seen == []
