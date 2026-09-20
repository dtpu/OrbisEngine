"""Saved dense anchor IDs must bind all CPU alignment consumers to their cameras."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import run_clip


class SavedAnchorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.pipeline = object.__new__(run_clip.Pipeline)
        self.pipeline.ctx = Path(temporary.name)
        self.folder = self.pipeline.ctx / "pi3x"
        self.folder.mkdir()
        self.anchors = [0, 1, 3, 5, 6, 8, 11, 12, 14, 17, 18, 20, 22, 24, 26, 29]
        self.sequence = {"anchors": self.anchors, "sourceIndices": list(range(0, 60, 2))}
        self.cameras = [{"sourceIndex": index} for index in self.sequence["sourceIndices"]]
        self.save()

    def save(self, count=None):
        (self.folder / "sequence.json").write_text(json.dumps(self.sequence))
        (self.folder / "cameras.json").write_text(json.dumps({"cameras": self.cameras}))
        n = len(self.anchors) if count is None else count
        np.savez(
            self.folder / "anchors.npz",
            poses=np.zeros((n, 4, 4)),
            points=np.zeros((n, 2, 2, 3)),
            valid=np.zeros((n, 2, 2)),
            people=np.zeros((n, 2, 2)),
            conf=np.zeros((n, 2, 2)),
        )

    def test_saved_nonuniform_sixteen_anchor_order_is_preserved(self):
        self.assertEqual(self.pipeline.anchor_samples(), ",".join(map(str, self.anchors)))

    def test_legacy_eight_works_only_with_matching_saved_metadata(self):
        self.anchors = self.anchors[::2]
        self.sequence["anchors"] = self.anchors
        self.save()
        self.assertEqual(self.pipeline.anchor_samples(), ",".join(map(str, self.anchors)))
        del self.sequence["anchors"]
        self.save()
        with self.assertRaisesRegex(ValueError, "saved anchor"):
            self.pipeline.anchor_samples()

    def test_malformed_anchor_ids_fail_closed(self):
        for anchors in (None, [], [0], [0, 0], [-1, 1], [0, 30], [0, True], [0, "1"]):
            with self.subTest(anchors=anchors):
                self.sequence["anchors"] = anchors
                self.save()
                with self.assertRaisesRegex(ValueError, "saved anchor"):
                    self.pipeline.anchor_samples()

    def test_npz_count_must_match_metadata(self):
        self.save(count=8)
        with self.assertRaisesRegex(ValueError, "anchor count"):
            self.pipeline.anchor_samples()

    def test_each_npz_array_and_required_arrays_are_checked(self):
        np.savez(self.folder / "anchors.npz", poses=np.zeros((16, 4, 4)))
        with self.assertRaisesRegex(ValueError, "anchor count"):
            self.pipeline.anchor_samples()
        self.save()
        with np.load(self.folder / "anchors.npz") as saved:
            arrays = dict(saved)
        arrays["conf"] = np.zeros((8, 2, 2))
        np.savez(self.folder / "anchors.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "anchor count"):
            self.pipeline.anchor_samples()

    def test_source_camera_correspondence_is_required(self):
        for indices in (None, [0], [0] * 30, list(range(30))):
            with self.subTest(indices=indices):
                self.sequence["sourceIndices"] = indices
                self.save()
                with self.assertRaisesRegex(ValueError, "source indices"):
                    self.pipeline.anchor_samples()

    def test_alignment_validates_before_subprocess(self):
        self.pipeline.name = "test-source"
        self.save(count=8)
        with (
            patch.object(run_clip, "run") as execute,
            self.assertRaisesRegex(ValueError, "anchor count"),
        ):
            self.pipeline.frame_align()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
