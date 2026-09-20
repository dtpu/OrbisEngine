"""The anchor alignment must solve, or say what was missing; never fail as a numerical mystery."""

import re
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    """Import similarity() without the GPU-only module imports around it.

    dense_pi3x.py imports torch-backed helpers at module scope, which are absent locally. The
    function under test is pure numpy, so the source is extracted and executed on its own.
    """
    source = (ROOT / "worker/stages/dense_pi3x.py").read_text()
    match = re.search(r"^# A similarity transform.*?^def viewer_camera", source, re.S | re.M)
    assert match, "similarity() block not found"
    module = types.ModuleType("dense_pi3x_numeric")
    module.__dict__["np"] = np
    exec(
        compile(match.group(0).rsplit("\ndef viewer_camera", 1)[0], "dense_pi3x", "exec"),
        module.__dict__,
    )
    return module


MODULE = load_module()
similarity = MODULE.similarity
MINIMUM = MODULE.MINIMUM_CORRESPONDENCES
anchor_indices = MODULE.anchor_indices
confidence_cutoff = MODULE.confidence_cutoff
person_frames_usable = MODULE.person_frames_usable
decoded_span_usable = MODULE.decoded_span_usable


def transformed(points, scale, axis_angle, offset):
    x, y, z = axis_angle
    rotation = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return points @ rotation.T * scale + offset


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.generator = np.random.default_rng(7)

    def test_recovers_a_known_transform(self):
        source = self.generator.normal(size=(200, 3))
        target = transformed(source, 2.5, (0, 0, 0.4), np.array([1.0, -2.0, 0.5]))
        scale, rotation, offset, rms = similarity(source, target)
        self.assertAlmostEqual(scale, 2.5, places=6)
        self.assertLess(rms, 1e-6)
        np.testing.assert_allclose(source @ rotation.T * scale + offset, target, atol=1e-6)

    def test_outliers_are_trimmed_without_losing_the_fit(self):
        source = self.generator.normal(size=(200, 3))
        target = transformed(source, 1.5, (0, 0, 0.2), np.array([0.5, 0.5, 0.5]))
        target[:20] += self.generator.normal(size=(20, 3)) * 50  # gross outliers
        scale, _, _, rms = similarity(source, target)
        self.assertAlmostEqual(scale, 1.5, places=3)
        self.assertLess(rms, 1.0)

    def test_too_few_correspondences_reports_the_cause(self):
        source = self.generator.normal(size=(2, 3))
        with self.assertRaisesRegex(ValueError, "at least 3 finite correspondences, got 2"):
            similarity(source, source * 2)

    def test_non_finite_points_are_dropped_not_propagated_into_the_solve(self):
        source = self.generator.normal(size=(60, 3))
        target = transformed(source, 2.0, (0, 0, 0.1), np.zeros(3))
        target[5] = np.nan
        source[9, 1] = np.inf
        scale, _, _, rms = similarity(source, target)
        self.assertAlmostEqual(scale, 2.0, places=5)
        self.assertTrue(np.isfinite(rms))

    def test_mostly_non_finite_input_raises_instead_of_svd_failure(self):
        source = self.generator.normal(size=(40, 3))
        target = transformed(source, 2.0, (0, 0, 0.1), np.zeros(3))
        target[2:] = np.nan
        with self.assertRaisesRegex(ValueError, "finite correspondences"):
            similarity(source, target)

    def test_empty_input_raises_instead_of_dividing_by_zero(self):
        empty = np.zeros((0, 3))
        with self.assertRaisesRegex(ValueError, "finite correspondences"):
            similarity(empty, empty)

    def test_trimming_never_drops_below_a_solvable_system(self):
        # Exactly the minimum: the refinement loop must not trim any of them away.
        source = self.generator.normal(size=(MINIMUM, 3))
        target = transformed(source, 3.0, (0, 0, 0.0), np.array([1.0, 1.0, 1.0]))
        scale, _, _, rms = similarity(source, target)
        self.assertTrue(np.isfinite(scale) and np.isfinite(rms))


class AnchorSelectionTests(unittest.TestCase):
    """Anchors must never hand an empty point set to the solver."""

    def masks(self, n, *, people_frac=0.0, conf_value=1.0, valid_frac=1.0):
        valid = np.zeros(n, bool)
        valid[: int(n * valid_frac)] = True
        people = np.zeros(n, bool)
        people[: int(n * people_frac)] = True
        return valid, people, np.full(n, conf_value)

    def test_prefers_static_confident_non_person_points(self):
        valid, people, conf = self.masks(100, people_frac=0.2)
        idx, basis = anchor_indices(valid=valid, people=people, conf=conf)
        self.assertEqual(basis, "static")
        self.assertEqual(len(idx), 80)

    def test_relaxes_confidence_when_nothing_is_confident(self):
        valid, people, conf = self.masks(100, people_frac=0.2, conf_value=0.1)
        idx, basis = anchor_indices(valid=valid, people=people, conf=conf)
        self.assertEqual(basis, "low-confidence static")
        self.assertEqual(len(idx), 80)

    def test_allows_person_points_when_the_frame_is_all_people(self):
        """The soccer and UFC clips: people cover the frame, so nothing static remains."""
        valid, people, conf = self.masks(100, people_frac=1.0)
        idx, basis = anchor_indices(valid=valid, people=people, conf=conf)
        self.assertEqual(basis, "confident including people")
        self.assertEqual(len(idx), 100)

    def test_falls_back_to_any_valid_point(self):
        valid, people, conf = self.masks(100, people_frac=1.0, conf_value=0.0)
        idx, basis = anchor_indices(valid=valid, people=people, conf=conf)
        self.assertEqual(basis, "any valid")
        self.assertEqual(len(idx), 100)

    def test_genuinely_empty_input_returns_empty_for_the_caller_to_report(self):
        valid, people, conf = self.masks(100, valid_frac=0.0)
        idx, _ = anchor_indices(valid=valid, people=people, conf=conf)
        self.assertEqual(len(idx), 0)

    def test_the_result_always_indexes_valid_points(self):
        valid, people, conf = self.masks(100, people_frac=0.9, conf_value=0.2, valid_frac=0.5)
        idx, _ = anchor_indices(valid=valid, people=people, conf=conf)
        self.assertTrue(valid[idx].all())


class ConfidenceCutoffTests(unittest.TestCase):
    """A view with no confident points must still yield a reconstruction to judge."""

    def test_the_absolute_bar_is_used_when_anything_clears_it(self):
        confidence = np.array([0.1, 0.4, 0.9])
        cutoff, relaxed = confidence_cutoff(confidence)
        self.assertEqual(cutoff, 0.3)
        self.assertFalse(relaxed)

    def test_falls_back_to_the_most_confident_share_when_nothing_clears_it(self):
        """The soccer broadcast case: every point scored under 0.3."""
        confidence = np.linspace(0.0, 0.29, 1000)
        cutoff, relaxed = confidence_cutoff(confidence, fraction=0.2)
        self.assertTrue(relaxed)
        self.assertLess(cutoff, 0.3)
        kept = (confidence >= cutoff).sum()
        self.assertAlmostEqual(kept / confidence.size, 0.2, places=2)

    def test_non_finite_confidence_is_ignored(self):
        confidence = np.array([np.nan, np.inf, 0.05, 0.06, 0.07])
        cutoff, relaxed = confidence_cutoff(confidence, fraction=0.5)
        self.assertTrue(relaxed)
        self.assertTrue(np.isfinite(cutoff))

    def test_an_all_non_finite_view_does_not_relax(self):
        cutoff, relaxed = confidence_cutoff(np.array([np.nan, np.nan]))
        self.assertFalse(relaxed)
        self.assertEqual(cutoff, 0.3)

    def test_a_uniformly_low_view_still_keeps_points(self):
        confidence = np.full(500, 0.01)
        cutoff, relaxed = confidence_cutoff(confidence)
        self.assertTrue(relaxed)
        self.assertGreater((confidence >= cutoff).sum(), 0)


class PersonFrameTests(unittest.TestCase):
    """One sparse frame must not throw away every frame that did reconstruct."""

    def test_a_single_sparse_frame_does_not_condemn_the_run(self):
        self.assertTrue(person_frames_usable(reconstructed=19, skipped=1))

    def test_a_mostly_empty_run_is_refused(self):
        self.assertFalse(person_frames_usable(reconstructed=2, skipped=18))

    def test_exactly_at_the_floor_is_accepted(self):
        self.assertTrue(person_frames_usable(reconstructed=10, skipped=10))

    def test_just_below_the_floor_is_refused(self):
        self.assertFalse(person_frames_usable(reconstructed=9, skipped=11))

    def test_no_frames_at_all_is_refused_rather_than_dividing_by_zero(self):
        self.assertFalse(person_frames_usable(reconstructed=0, skipped=0))

    def test_the_floor_is_adjustable(self):
        self.assertTrue(person_frames_usable(reconstructed=3, skipped=17, floor=0.1))


class DecodedSpanTests(unittest.TestCase):
    """Container frame counts overstate; a short unreadable tail is not a broken source."""

    def test_a_short_unreadable_tail_is_tolerated(self):
        self.assertTrue(decoded_span_usable(decoded=143, requested=144))

    def test_a_source_that_mostly_fails_is_refused(self):
        self.assertFalse(decoded_span_usable(decoded=10, requested=144))

    def test_exactly_at_the_floor_is_accepted(self):
        self.assertTrue(decoded_span_usable(decoded=72, requested=144))

    def test_zero_requested_frames_is_refused_rather_than_dividing_by_zero(self):
        self.assertFalse(decoded_span_usable(decoded=0, requested=0))

    def test_the_floor_is_adjustable(self):
        self.assertTrue(decoded_span_usable(decoded=20, requested=144, floor=0.1))


if __name__ == "__main__":
    unittest.main()
