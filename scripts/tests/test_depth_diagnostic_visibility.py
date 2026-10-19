"""Depth comparisons use visible generated centers, never agreement-based filtering."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bake_video_colours import diagnostic_depth_ratios
from run_clip import SCALE_RATIO_BAND, SCALE_SPREAD_BAND


def compare(pixels, depths, source=None, support=None, unmasked=None):
    source = np.ones(8) if source is None else np.asarray(source)
    support = np.ones(source.shape, dtype=bool) if support is None else support
    unmasked = np.ones(len(pixels), dtype=bool) if unmasked is None else unmasked
    return diagnostic_depth_ratios(np.asarray(pixels, dtype=int), depths, source, support, unmasked)


@pytest.mark.parametrize("copies", [1, 20])
def test_hidden_rear_geometry_and_density_do_not_change_front_comparison(copies):
    pixels = [*range(8), *([0, 1] * copies)]
    depths = [1] * 8 + [1.5] * (copies * 2)
    front, legacy, counts = compare(pixels, depths)
    np.testing.assert_array_equal(front, np.ones(8))
    assert np.percentile(legacy, 90) == 1.5
    assert np.median(legacy) == (1 if copies == 1 else 1.5)
    assert counts == {
        "sourceSupportedPixels": 8,
        "generatedFrontPixels": 8,
        "comparedPixels": 8,
        "legacySplats": 8 + 2 * copies,
    }


def test_revealing_a_wrong_rear_surface_still_fails_upper_spread_band():
    # Remove its front occluders: the same rear samples now really are the visible depth.
    ratios, _, _ = compare(range(8), [1.5, 1.5, 1, 1, 1, 1, 1, 1])
    assert np.median(ratios) == 1
    assert np.percentile(ratios, 90) == 1.5 > SCALE_SPREAD_BAND[1]


def test_wrong_foreground_is_not_filtered_out_for_disagreeing_with_source():
    ratios, _, _ = compare([*range(8), 0, 1], [1] * 8 + [0.75, 0.75])
    assert np.median(ratios) == 1
    assert np.percentile(ratios, 10) == 0.75 < SCALE_SPREAD_BAND[0]


def test_global_visible_scale_error_still_fails_median_band():
    ratios, _, _ = compare(range(8), [1.25] * 8)
    assert np.median(ratios) == 1.25 > SCALE_RATIO_BAND[1]


def test_duplicate_front_centers_do_not_reweight_image_rays():
    pixels = [*range(8), *([0] * 100)]
    ratios, _, counts = compare(pixels, [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5] + [1] * 100)
    assert len(ratios) == counts["comparedPixels"] == 8
    assert np.median(ratios) == 1.5


def test_unsupported_and_person_masked_rays_are_excluded_with_coverage_counts():
    ratios, legacy, counts = compare(
        [0, 1, 2, 3],
        [1, 5, 6, 2],
        source=[1, np.nan, 1, 2],
        support=[True, False, True, True],
        unmasked=[True, True, False, True],
    )
    np.testing.assert_array_equal(ratios, [1, 1])
    np.testing.assert_array_equal(legacy, [1, 1])
    assert counts["comparedPixels"] == 2
    assert counts["sourceSupportedPixels"] == 3
    assert counts["generatedFrontPixels"] == 4


def test_masking_a_front_center_does_not_reveal_its_hidden_background():
    ratios, legacy, counts = compare([0, 0, 1], [1, 1.5, 1], unmasked=[False, True, True])
    np.testing.assert_array_equal(ratios, [1])
    np.testing.assert_array_equal(legacy, [1.5, 1])
    assert counts["comparedPixels"] == 1


@pytest.mark.parametrize("case", ["empty-world", "unsupported", "masked", "disjoint"])
def test_no_comparable_ray_fails_instead_of_producing_a_pass(case):
    pixels, depths, support, unmasked = [0], [1], [True, False], [True]
    if case == "empty-world":
        pixels, depths, unmasked = [], [], []
    elif case == "unsupported":
        support = [False, False]
    elif case == "masked":
        unmasked = [False]
    else:
        pixels = [1]
    with pytest.raises(ValueError, match="No supported, unmasked front-depth overlap"):
        compare(pixels, depths, source=[1, 1], support=support, unmasked=unmasked)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
@pytest.mark.parametrize("side", ["source", "generated"])
def test_invalid_compared_depth_fails_clearly(value, side):
    with pytest.raises(ValueError, match=f"Nonfinite or nonpositive {side} depth"):
        compare(
            [0], [value if side == "generated" else 1], source=[value if side == "source" else 1]
        )


def test_report_counts_expose_partial_overlap_without_inventing_a_coverage_threshold():
    ratios, _, counts = compare([0], [1])
    np.testing.assert_array_equal(ratios, [1])
    assert counts["comparedPixels"] == 1
    assert counts["sourceSupportedPixels"] == 8


def test_frozen_acceptance_bands_are_unchanged():
    assert SCALE_RATIO_BAND == (0.95, 1.05)
    assert SCALE_SPREAD_BAND == (0.9, 1.1)
