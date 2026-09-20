#!/usr/bin/env python3
"""Anchor selection for the Pi3X dense solve; numpy only, no torch, GPU or media.

Four real clips failed in `worker/stages/dense_pi3x.py` at the anchor setup and the first
batch alignment, and in every case the cause was a set that was silently empty: a fixed
confidence cut that no view of those clips reached, a semantic person mask that labelled a
whole stadium crowd, or an anchor set whose views never saw the same surface. These checks
cover the pure decisions behind each of those, so an empty set is named rather than handed
to an SVD.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker" / "stages"))

from dense_pi3x import (  # noqa: E402
    ANCHOR_MATCH_MIN,
    ANCHOR_MATCH_TARGET,
    ANCHOR_SPREAD_LIMIT,
    CONF_ABSOLUTE,
    CONF_FLOOR,
    CONF_MODE_ABSOLUTE,
    CONF_MODE_RANK,
    CROWD_PERSON_FRACTION,
    MASK_FOREGROUND,
    MASK_SEMANTIC,
    MIN_SIMILARITY_POINTS,
    anchor_matches,
    anchor_spread,
    confidence_threshold,
    mask_mode,
    similarity,
)


def view(height=8, width=8, value=0.4):
    return np.full((height, width), float(value))


def anchor_set(confidences, height=40, width=40, static=True):
    """(A,H,W) confidence and static masks with one constant confidence per view."""
    conf = np.stack([np.full((height, width), float(c)) for c in confidences])
    return conf, np.full(conf.shape, bool(static))


def pose_at(x, y=0.0, z=0.0):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


class ConfidenceThreshold(unittest.TestCase):
    def test_confident_view_keeps_the_absolute_cut_it_always_used(self):
        conf = np.linspace(0.3, 0.9, 1000).reshape(10, 100)
        cut, mode = confidence_threshold(conf)
        self.assertEqual(mode, CONF_MODE_ABSOLUTE)
        self.assertEqual(cut, CONF_ABSOLUTE)

    def test_a_view_the_model_is_unsure_of_falls_back_to_its_own_rank(self):
        # soccer-s1's anchor 0: the whole map sits under the fixed 0.3, so `valid` was empty.
        conf = np.linspace(0.06, 0.30, 1000).reshape(10, 100)
        cut, mode = confidence_threshold(conf)
        self.assertEqual(mode, CONF_MODE_RANK)
        self.assertLess(cut, CONF_ABSOLUTE)
        self.assertGreater(cut, CONF_FLOOR)
        self.assertGreater(int((conf > cut).sum()), 0)

    def test_the_rank_cut_keeps_a_usable_share_of_the_view(self):
        conf = np.linspace(0.06, 0.30, 1000).reshape(10, 100)
        cut, _ = confidence_threshold(conf)
        self.assertGreater(float((conf > cut).mean()), 0.2)

    def test_a_view_confident_about_nothing_is_floored_not_zeroed(self):
        conf = np.full((10, 10), 1e-9)
        cut, mode = confidence_threshold(conf)
        self.assertEqual(mode, CONF_MODE_RANK)
        self.assertEqual(cut, CONF_FLOOR)

    def test_non_finite_confidence_never_produces_a_non_finite_cut(self):
        conf = np.full((4, 4), np.nan)
        cut, mode = confidence_threshold(conf)
        self.assertTrue(np.isfinite(cut))
        self.assertEqual(mode, CONF_MODE_RANK)


class MaskModeChoice(unittest.TestCase):
    def test_ordinary_subjects_keep_the_semantic_mask(self):
        # img5594, movie-s26, hp-fly-s63 and game-s1 all measure well under the threshold.
        mode, mean = mask_mode([0.10, 0.12, 0.15, 0.06])
        self.assertEqual(mode, MASK_SEMANTIC)
        self.assertAlmostEqual(mean, 0.1075)

    def test_a_crowd_switches_to_foreground_instances(self):
        # soccer-s1's eight anchors, as measured from its saved prediction.
        mode, mean = mask_mode([0.149, 0.322, 0.365, 0.517, 0.501, 0.467, 0.438, 0.082])
        self.assertEqual(mode, MASK_FOREGROUND)
        self.assertGreater(mean, CROWD_PERSON_FRACTION)

    def test_one_close_up_view_does_not_decide_the_clip(self):
        mode, _ = mask_mode([0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.assertEqual(mode, MASK_SEMANTIC)

    def test_no_views_is_not_a_crowd(self):
        mode, mean = mask_mode([])
        self.assertEqual(mode, MASK_SEMANTIC)
        self.assertEqual(mean, 0.0)


class AdaptiveMatchSelection(unittest.TestCase):
    def test_every_view_contributes_the_target_when_all_are_comparable(self):
        conf, static = anchor_set([0.8] * 4, height=80, width=80)
        conf += np.random.default_rng(0).normal(0, 0.01, conf.shape)
        kept, records = anchor_matches(conf, static)
        self.assertEqual([j for j, _ in kept], [0, 1, 2, 3])
        for _, pick in kept:
            self.assertEqual(len(pick), ANCHOR_MATCH_TARGET)
        self.assertTrue(all("dropped" not in record for record in records))

    def test_a_uniformly_low_confidence_clip_still_selects_its_matches(self):
        # The whole failing set: no view of soccer-s1, creed-v1, game-s1 or movie-s17 ever
        # reached the 0.5 the old fixed cut asked for, so every match set came out empty.
        rng = np.random.default_rng(1)
        conf = rng.uniform(0.05, 0.32, (6, 80, 80))
        kept, _ = anchor_matches(conf, np.ones(conf.shape, bool))
        self.assertEqual(len(kept), 6)
        self.assertEqual(sum(len(pick) for _, pick in kept), 6 * ANCHOR_MATCH_TARGET)

    def test_matches_index_the_flattened_view_and_are_unique(self):
        conf, static = anchor_set([0.5] * 2, height=60, width=60)
        conf += np.random.default_rng(2).normal(0, 0.01, conf.shape)
        kept, _ = anchor_matches(conf, static)
        for _, pick in kept:
            self.assertEqual(len(np.unique(pick)), len(pick))
            self.assertTrue((pick >= 0).all() and (pick < conf[0].size).all())

    def test_selection_prefers_the_confident_pixels_it_is_given(self):
        conf = np.full((2, 200, 100), 0.05)
        conf[:, :70, :] = 0.9  # a confident region larger than the selection pool
        kept, _ = anchor_matches(conf, np.ones(conf.shape, bool))
        for _, pick in kept:
            self.assertTrue((pick // 100 < 70).all())

    def test_an_anchor_with_almost_no_static_pixels_is_dropped_and_named(self):
        conf, static = anchor_set([0.6] * 3, height=80, width=80)
        static[1] = False
        static[1].reshape(-1)[: ANCHOR_MATCH_MIN - 1] = True
        kept, records = anchor_matches(conf, static)
        self.assertEqual([j for j, _ in kept], [0, 2])
        self.assertIn("static pixels", records[1]["dropped"])
        self.assertEqual(records[1]["staticPixels"], ANCHOR_MATCH_MIN - 1)

    def test_an_anchor_sharing_no_confident_content_is_dropped_and_named(self):
        # game-s1 anchor 7 measured 0.157 against a best anchor of 0.406; soccer-s1 anchor 7
        # measured 0.049 against 0.242. Aligning to such a view drags the whole batch.
        conf, static = anchor_set([0.41, 0.39, 0.40, 0.10], height=80, width=80)
        kept, records = anchor_matches(conf, static)
        self.assertEqual([j for j, _ in kept], [0, 1, 2])
        self.assertIn("shares no confident content", records[3]["dropped"])

    def test_too_few_usable_anchors_raises_with_the_counts(self):
        conf, static = anchor_set([0.6, 0.01, 0.01, 0.01], height=80, width=80)
        with self.assertRaises(RuntimeError) as caught:
            anchor_matches(conf, static)
        message = str(caught.exception)
        self.assertIn("1 usable anchor view(s) of 4", message)
        self.assertIn("static", message)
        self.assertIn("confidence", message)

    def test_no_static_pixels_at_all_raises_instead_of_returning_empty_sets(self):
        conf, static = anchor_set([0.6] * 4, height=80, width=80, static=False)
        with self.assertRaises(RuntimeError) as caught:
            anchor_matches(conf, static)
        self.assertIn("0 usable anchor view(s) of 4", str(caught.exception))
        self.assertIn("0 matched points", str(caught.exception))

    def test_a_small_but_sufficient_anchor_keeps_every_pixel_it_has(self):
        conf, static = anchor_set([0.6] * 2, height=80, width=80)
        static[1] = False
        static[1].reshape(-1)[:400] = True
        kept, records = anchor_matches(conf, static)
        self.assertEqual(dict(kept)[1].size, 400)
        self.assertEqual(records[1]["matches"], 400)

    def test_confident_views_alone_carry_a_clip_that_has_enough_of_them(self):
        # movie-s26 already solved: six of its eight anchors clear the absolute cut, and the
        # rank fallback must not hand its alignment two views it never used.
        conf, static = anchor_set([0.6] * 6 + [0.28, 0.28], height=80, width=80)
        preferred = np.array([True] * 6 + [False, False])
        kept, records = anchor_matches(conf, static, preferred=preferred)
        self.assertEqual([j for j, _ in kept], list(range(6)))
        self.assertIn("absolute confidence cut", records[7]["dropped"])

    def test_too_few_confident_views_align_on_everything_usable(self):
        # creed-v1: two of eight, and adjacent, so the fit needs the rank-mode views too.
        conf, static = anchor_set([0.3] * 8, height=80, width=80)
        preferred = np.array([False, False, False, True, True, False, False, False])
        kept, _ = anchor_matches(conf, static, preferred=preferred)
        self.assertEqual(len(kept), 8)

    def test_no_confident_views_at_all_falls_back_without_raising(self):
        conf, static = anchor_set([0.2] * 5, height=80, width=80)
        kept, _ = anchor_matches(conf, static, preferred=np.zeros(5, bool))
        self.assertEqual(len(kept), 5)

    def test_the_selection_is_reproducible(self):
        rng = np.random.default_rng(3)
        conf = rng.uniform(0.1, 0.4, (4, 70, 70))
        static = np.ones(conf.shape, bool)
        first = [pick for _, pick in anchor_matches(conf, static)[0]]
        second = [pick for _, pick in anchor_matches(conf, static)[0]]
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


class AnchorSpread(unittest.TestCase):
    def test_a_clip_whose_anchors_look_at_one_scene_passes(self):
        poses = [pose_at(x) for x in np.linspace(0, 1.6, 8)]
        spread, depth = anchor_spread(poses, [2.4] * 8)
        self.assertAlmostEqual(depth, 2.4)
        self.assertLess(spread, ANCHOR_SPREAD_LIMIT)

    def test_a_traversal_is_refused_with_the_measurement_and_the_remedy(self):
        # game-s1: 22 s across several areas, largest baseline 11.44 median depths.
        poses = [pose_at(x) for x in np.linspace(0, 51.3, 8)]
        with self.assertRaises(RuntimeError) as caught:
            anchor_spread(poses, [4.485] * 8)
        message = str(caught.exception)
        self.assertIn("scene depths", message)
        self.assertIn("shorter segment", message)

    def test_anchors_without_a_measurable_depth_raise(self):
        with self.assertRaises(RuntimeError) as caught:
            anchor_spread([pose_at(0), pose_at(1)], [np.nan, -1.0])
        self.assertIn("positive median depth", str(caught.exception))


class SimilarityGuards(unittest.TestCase):
    def cloud(self, count=400, seed=4):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, (count, 3))

    def test_a_clean_similarity_still_recovers_the_transform(self):
        src = self.cloud()
        angle = 0.3
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        dst = src @ R.T * 2.0 + np.array([1.0, -2.0, 0.5])
        s, fitted, t, rms = similarity(src, dst)
        self.assertAlmostEqual(s, 2.0, places=5)
        np.testing.assert_allclose(fitted, R, atol=1e-6)
        np.testing.assert_allclose(t, [1.0, -2.0, 0.5], atol=1e-5)
        self.assertLess(rms, 1e-6)

    def test_the_trim_still_rejects_outliers(self):
        src = self.cloud()
        dst = src * 1.5
        dst[:40] += np.random.default_rng(5).normal(0, 20, (40, 3))
        s, _, _, rms = similarity(src, dst)
        self.assertAlmostEqual(s, 1.5, places=3)
        self.assertLess(rms, 1e-3)

    def test_empty_inputs_raise_a_named_error_not_an_svd_crash(self):
        with self.assertRaises(RuntimeError) as caught:
            similarity(np.zeros((0, 3)), np.zeros((0, 3)))
        message = str(caught.exception)
        self.assertIn("0 finite paired points", message)
        self.assertIn(str(MIN_SIMILARITY_POINTS), message)

    def test_non_finite_points_are_dropped_and_then_counted(self):
        src, dst = self.cloud(), self.cloud()
        src[:] = np.nan
        with self.assertRaises(RuntimeError) as caught:
            similarity(src, dst)
        self.assertIn("dropped as NaN", str(caught.exception))

    def test_a_few_non_finite_points_do_not_stop_a_good_fit(self):
        src = self.cloud()
        dst = src * 3.0 + 1.0
        src[:5] = np.nan
        dst[10:15, 0] = np.inf
        s, _, _, rms = similarity(src, dst)
        self.assertAlmostEqual(s, 3.0, places=5)
        self.assertLess(rms, 1e-6)

    def test_collinear_correspondences_raise_instead_of_an_unconstrained_rotation(self):
        line = np.linspace(0, 1, 500)[:, None] * np.array([1.0, 2.0, -0.5])
        with self.assertRaises(RuntimeError) as caught:
            similarity(line, line * 2.0)
        self.assertIn("degenerate", str(caught.exception))

    def test_a_single_repeated_point_raises(self):
        point = np.tile([1.0, 2.0, 3.0], (300, 1))
        with self.assertRaises(RuntimeError) as caught:
            similarity(point, point)
        self.assertIn("spread", str(caught.exception))

    def test_mismatched_pair_counts_raise(self):
        with self.assertRaises(RuntimeError) as caught:
            similarity(self.cloud(30), self.cloud(20))
        self.assertIn("paired points", str(caught.exception))

    def test_the_minimum_sized_input_fits_without_trimming_itself_away(self):
        src = self.cloud(MIN_SIMILARITY_POINTS, seed=6)
        s, _, _, rms = similarity(src, src * 1.25)
        self.assertAlmostEqual(s, 1.25, places=5)
        self.assertLess(rms, 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
