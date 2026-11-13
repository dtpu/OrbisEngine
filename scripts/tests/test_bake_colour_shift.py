"""Opt-in colour-shift propagation and anchor-cloud cache keying for the video colour bake."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bake_video_colours as bake  # noqa: E402


def scene():
    xyz = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.5, 0, 0], [0.95, 0, 0], [3.0, 0, 0]])
    base = np.full((5, 3), 0.4)
    baked = base.copy()
    baked[:2] = [0.6, 0.5, 0.3]
    observed = np.array([True, True, False, False, False])
    return xyz, base, baked, observed


def test_shift_fades_with_distance_and_leaves_observed_and_far_splats_exact():
    xyz, base, baked, observed = scene()
    out, changed = bake.propagate_colour_shift(xyz, base, baked, observed, np.ones(5, bool), 1.0)
    np.testing.assert_array_equal(out[:2], baked[:2])
    np.testing.assert_array_equal(out[4], base[4])
    assert changed.tolist() == [False, False, True, True, False]
    shift = baked[0] - base[0]
    np.testing.assert_allclose(out[2], base[2] + (1 - 0.4) * shift)
    np.testing.assert_allclose(out[3], base[3] + (1 - 0.85) * shift)


def test_ineligible_splats_keep_their_colour():
    xyz, base, baked, observed = scene()
    eligible = np.array([True, True, False, True, True])
    out, changed = bake.propagate_colour_shift(xyz, base, baked, observed, eligible, 1.0)
    np.testing.assert_array_equal(out[2], base[2])
    assert not changed[2] and changed[3]


def test_nothing_observed_changes_nothing():
    xyz, base, baked, _ = scene()
    out, changed = bake.propagate_colour_shift(
        xyz, base, base, np.zeros(5, bool), np.ones(5, bool), 1.0
    )
    np.testing.assert_array_equal(out, base)
    assert not changed.any()


@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan")])
def test_invalid_radius_is_refused(radius):
    xyz, base, baked, observed = scene()
    with pytest.raises(ValueError):
        bake.propagate_colour_shift(xyz, base, baked, observed, np.ones(5, bool), radius)


def test_mismatched_inputs_are_refused():
    xyz, base, baked, observed = scene()
    with pytest.raises(ValueError):
        bake.propagate_colour_shift(xyz, base[:4], baked, observed, np.ones(5, bool), 1.0)


def test_anchor_cloud_cache_is_shared_only_beside_the_anchors(tmp_path):
    pi3x = tmp_path / "pi3x"
    pi3x.mkdir()
    anchors = pi3x / "anchors.npz"
    native = pi3x / "cameras.json"
    native.write_text('{"cameras": []}')
    assert bake.anchor_cloud_cache(anchors, native, "0,1") == pi3x / "static-scene.npy"

    packaged = tmp_path / "packaged.json"
    packaged.write_text('{"cameras": [1]}')
    keyed = bake.anchor_cloud_cache(anchors, packaged, "0,1")
    assert keyed.parent == pi3x and keyed.name.startswith("static-scene-")
    assert keyed != pi3x / "static-scene.npy"
    assert bake.anchor_cloud_cache(anchors, packaged, "0,2") != keyed
    packaged.write_text('{"cameras": [2]}')
    assert bake.anchor_cloud_cache(anchors, packaged, "0,1") != keyed


def test_depth_gate_keeps_only_splats_on_the_source_surface():
    observed = np.array([True, True, True, True, False])
    weight = np.array([2.0, 2.0, 2.0, 0.0, 2.0])
    log_ratio = np.array([0.05, -0.3, 0.12, 0.0, 0.0])
    kept, gated, mean = bake.depth_agreement_gate(observed, weight * log_ratio, weight, 0.1)
    assert kept.tolist() == [True, False, False, False, False]
    # Too near, too far and no supported source depth are all gated; unobserved stays unobserved.
    assert gated.tolist() == [False, True, True, True, False]
    np.testing.assert_allclose(mean[:3], log_ratio[:3], rtol=1e-6)
    assert np.isnan(mean[3])


def test_depth_gate_off_changes_nothing():
    observed = np.array([True, False, True])
    kept, gated, _ = bake.depth_agreement_gate(observed, np.array([5.0, 0, -5]), np.ones(3), 0.0)
    assert kept.tolist() == observed.tolist() and not gated.any()


@pytest.mark.parametrize("tolerance", [-0.1, float("nan")])
def test_depth_gate_refuses_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        bake.depth_agreement_gate(np.ones(2, bool), np.zeros(2), np.ones(2), tolerance)


def test_depth_gate_refuses_mismatched_inputs():
    with pytest.raises(ValueError):
        bake.depth_agreement_gate(np.ones(2, bool), np.zeros(3), np.ones(2), 0.1)


def test_chroma_transfer_keeps_generated_luminance_and_takes_observed_hue():
    base = np.array([[0.35, 0.53, 0.22], [0.26, 0.31, 0.38]])  # lime, bluish
    raw = np.array([[0.43, 0.43, 0.39], [0.45, 0.45, 0.41]])  # grey-olive, grey
    out, changed = bake.transfer_chroma(base, raw)
    assert changed.all()
    np.testing.assert_allclose(out @ bake.LUMA, base @ bake.LUMA, atol=1e-9)
    # Chromaticity follows the observation: channel ratios match the observed colour.
    np.testing.assert_allclose(out / out.sum(1, keepdims=True), raw / raw.sum(1, keepdims=True))


def test_chroma_transfer_ignores_observations_of_another_surface():
    dark_leaf = np.array([[0.13, 0.14, 0.11]])
    sky = np.array([[0.85, 0.80, 0.88]])
    unlit = np.array([[0.0, 0.0, 0.0]])
    out, changed = bake.transfer_chroma(np.vstack([dark_leaf, dark_leaf]), np.vstack([sky, unlit]))
    assert not changed.any()
    np.testing.assert_array_equal(out, np.vstack([dark_leaf, dark_leaf]))


def test_chroma_transfer_refuses_mismatched_colours():
    with pytest.raises(ValueError):
        bake.transfer_chroma(np.zeros((2, 3)), np.zeros((3, 3)))
