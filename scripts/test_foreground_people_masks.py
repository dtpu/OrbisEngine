"""CPU-only tests for foreground person selection in the cleaner's mask layer.

The selection rule is a pure function over instance boxes/masks and the semantic person map, so
every case here is a synthetic mask: no model, no weights, no GPU. One optional smoke test runs
the real Mask R-CNN and only checks shapes and dtypes; it skips when torchvision or its weights
are unavailable (`uv run --locked --group inference scripts/test_foreground_people_masks.py`).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from wander_worker.masks import (
    FOREGROUND_GROWTH_BOX_FRAC,
    FOREGROUND_MAX_GROWTH_FRAC,
    FOREGROUND_MIN_HEIGHT_FRAC,
    select_foreground_mask,
)

H, W = 200, 400
# one foreground subject: a 150 px box, 0.75 of frame height, well over the height cut
SUBJECT = (120.0, 40.0, 160.0, 190.0)


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def subject_mask():
    x0, y0, x1, y1 = (int(v) for v in SUBJECT)
    return rect(y0, y1, x0, x1)


class SelectionTests(unittest.TestCase):
    def test_distant_crowd_is_left_in_the_plate(self):
        """The stadium failure: a large semantic crowd that reaches no instance stays untouched."""
        crowd = rect(0, 30, 200, 390)
        semantic = subject_mask() | crowd
        mask, counts = select_foreground_mask([SUBJECT], subject_mask()[None], semantic)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.shape, (H, W))
        self.assertFalse((mask & crowd).any(), "crowd pixels were inpainted away")
        self.assertTrue(mask[subject_mask()].all())
        self.assertEqual((counts["selected"], counts["rejectedSmall"]), (1, 0))

    def test_crowd_near_the_subject_but_unconnected_stays_out(self):
        """Inside the growth band is not enough: the growth is geodesic, not a dilation."""
        x0, y0, x1, y1 = (int(v) for v in SUBJECT)
        gap = rect(y0, y0 + 20, x1 + 4, x1 + 24)  # 4 px clear of the body, well inside the band
        semantic = subject_mask() | gap
        mask, _ = select_foreground_mask([SUBJECT], subject_mask()[None], semantic)
        self.assertFalse((mask & gap).any())

    def test_touching_crowd_is_clipped_to_the_band(self):
        x0, y0, x1, y1 = (int(v) for v in SUBJECT)
        band = round(FOREGROUND_GROWTH_BOX_FRAC * (y1 - y0))
        crowd = rect(y0, y0 + 20, x1, W)  # a wide blob welded to the subject's shoulder
        semantic = subject_mask() | crowd
        mask, counts = select_foreground_mask([SUBJECT], subject_mask()[None], semantic)
        self.assertEqual(counts["bandPx"], band)
        self.assertEqual(counts["rejectedGrowth"], 0)
        self.assertTrue((mask & crowd).any(), "nothing of the touching blob was taken")
        self.assertFalse(
            mask[y0 : y0 + 20, x1 + band + 2 :].any(), "the blob was swallowed past the band"
        )
        # what is taken is a band-sized bite, not the blob
        self.assertLess((mask & crowd).sum(), 0.25 * crowd.sum())

    def test_crowd_wrapped_around_the_subject_drops_the_growth_entirely(self):
        """The stadium close-up: the terrace is one sheet of person pixels welded to the body."""
        thin_box = (200.0, 40.0, 210.0, 190.0)
        thin = rect(40, 190, 200, 210)
        crowd = rect(30, 200, 190, 260) & ~thin
        mask, counts = select_foreground_mask([thin_box], thin[None], thin | crowd)
        self.assertEqual(counts["rejectedGrowth"], 1)
        self.assertEqual(counts["grownPx"], 0)
        np.testing.assert_array_equal(mask, thin)

    def test_growth_under_the_guard_is_kept(self):
        """The guard is a ceiling on the growth, not a ban on it."""
        thin_box = (200.0, 40.0, 210.0, 190.0)
        thin = rect(40, 190, 200, 210)
        limb = rect(100, 110, 210, 213)
        self.assertLess(limb.sum(), FOREGROUND_MAX_GROWTH_FRAC * thin.sum())
        mask, counts = select_foreground_mask([thin_box], thin[None], thin | limb)
        self.assertEqual(counts["rejectedGrowth"], 0)
        self.assertEqual(counts["grownPx"], int(limb.sum()))
        self.assertTrue(mask[limb].all())

    def test_limb_connected_to_the_subject_is_covered(self):
        """The phone-clip failure: an arm the instance mask cut off is still person."""
        x0, y0, x1, y1 = (int(v) for v in SUBJECT)
        band = round(FOREGROUND_GROWTH_BOX_FRAC * (y1 - y0))
        limb = rect(y0 + 10, y0 + 18, x1, x1 + band - 4)
        semantic = subject_mask() | limb
        mask, _ = select_foreground_mask([SUBJECT], subject_mask()[None], semantic)
        self.assertTrue(mask[limb].all(), "connected limb pixels were dropped")

    def test_instance_survives_a_semantic_map_that_misses_it_entirely(self):
        mask, counts = select_foreground_mask(
            [SUBJECT], subject_mask()[None], np.zeros((H, W), bool)
        )
        self.assertTrue(mask[subject_mask()].all())
        self.assertEqual(counts["semanticPx"], 0)

    def test_small_instances_are_rejected_with_their_semantic_pixels(self):
        small_box = (10.0, 100.0, 30.0, 100.0 + FOREGROUND_MIN_HEIGHT_FRAC * H - 2)
        small = rect(100, int(small_box[3]), 10, 30)
        semantic = subject_mask() | small
        mask, counts = select_foreground_mask(
            [SUBJECT, small_box], np.stack([subject_mask(), small]), semantic
        )
        self.assertEqual((counts["instances"], counts["selected"]), (2, 1))
        self.assertEqual(counts["rejectedSmall"], 1)
        self.assertFalse((mask & small).any())

    def test_instance_exactly_at_the_height_cut_is_kept(self):
        box = (10.0, 0.0, 30.0, FOREGROUND_MIN_HEIGHT_FRAC * H)
        inst = rect(0, int(box[3]), 10, 30)
        mask, counts = select_foreground_mask([box], inst[None], inst)
        self.assertEqual(counts["selected"], 1)
        self.assertTrue(mask[inst].all())

    def test_empty_frame(self):
        for semantic in (np.zeros((H, W), bool), rect(0, 30, 200, 390)):
            with self.subTest(semanticPx=int(semantic.sum())):
                mask, counts = select_foreground_mask(
                    np.zeros((0, 4)), np.zeros((0, H, W), bool), semantic
                )
                self.assertFalse(mask.any())
                self.assertEqual(counts["selected"], 0)
                self.assertEqual(counts["rejectedSmall"], 0)
                self.assertEqual(counts["bandPx"], 0)
                self.assertEqual(counts["maskPx"], 0)

    def test_empty_instance_mask_contributes_nothing(self):
        mask, counts = select_foreground_mask([SUBJECT], np.zeros((1, H, W), bool), subject_mask())
        self.assertFalse(mask.any())
        self.assertEqual(counts["selected"], 1)

    def test_each_instance_gets_its_own_band(self):
        """A short-but-selected person must not inherit the tall subject's reach."""
        short_box = (300.0, 130.0, 330.0, 190.0)
        short = rect(130, 190, 300, 330)
        crowd = rect(130, 150, 330, W)
        semantic = subject_mask() | short | crowd
        mask, counts = select_foreground_mask(
            [SUBJECT, short_box], np.stack([subject_mask(), short]), semantic
        )
        self.assertEqual(counts["selected"], 2)
        reach = round(FOREGROUND_GROWTH_BOX_FRAC * 60)
        self.assertFalse(mask[130:150, 330 + reach + 2 :].any())


class DetectorSmokeTest(unittest.TestCase):
    def test_real_detector_shapes_and_dtypes(self):
        try:
            import torch  # noqa: F401
            from wander_worker.masks import _load_detector, person_instances

            _load_detector()
        except Exception as error:  # no torchvision, no weights, no network
            self.skipTest(f"Mask R-CNN unavailable: {type(error).__name__}: {error}")
        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(2, 128, 160, 3), dtype=np.uint8)
        frames = list(person_instances(images, batch_size=1))
        self.assertEqual(len(frames), len(images))
        for boxes, instances in frames:
            self.assertEqual(boxes.ndim, 2)
            self.assertEqual(boxes.shape[1], 4)
            self.assertEqual(boxes.dtype, np.dtype(np.float32))
            self.assertEqual(instances.shape, (len(boxes),) + images.shape[1:3])
            self.assertEqual(instances.dtype, np.dtype(bool))
            mask, counts = select_foreground_mask(
                boxes, instances, np.zeros(images.shape[1:3], bool)
            )
            self.assertEqual(mask.shape, images.shape[1:3])
            self.assertEqual(mask.dtype, np.dtype(bool))
            self.assertEqual(counts["instances"], len(boxes))


if __name__ == "__main__":
    unittest.main()
