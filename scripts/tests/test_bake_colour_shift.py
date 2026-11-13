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
