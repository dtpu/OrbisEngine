"""No-spend preflight checks for externally reviewed clean masks."""

from __future__ import annotations

import sys
from pathlib import Path

# The modules under test are this directory's parent; importing them by name is what
# running from scripts/ used to give for free.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker"))
import run_clip


class CleanMaskPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pipeline = object.__new__(run_clip.Pipeline)
        self.pipeline.clip = self.root / "source.mp4"
        self.pipeline.info = {"width": 11, "height": 4, "frames": 448}
        self.pipeline.ctx = self.root / "run"
        self.pipeline.ctx.mkdir()
        self.pipeline.clean_mp4 = self.root / "clean.mp4"
        self.pipeline.a = SimpleNamespace(
            clean_masks=None,
            moved_mask=False,
            fps=12.0,
            dilate=20,
            bottom_extra=40,
            lama_px=960,
        )

    def write_masks(self, samples, *, declared=True):
        path = self.root / "reviewed-masks.npz"
        payload = {
            "masks": np.zeros((samples, 4, math.ceil(11 / 8)), dtype=np.uint8),
        }
        if declared:
            payload["shape"] = np.array([samples, 4, 11])
        np.savez_compressed(path, **payload)
        return path

    @staticmethod
    def provenance(samples):
        return {"frames": [{} for _ in range(samples)]}

    def test_forwards_valid_masks_using_observed_sample_count_not_metadata_count(self):
        masks = self.write_masks(5)
        self.pipeline.a.clean_masks = str(masks)
        self.pipeline.paid_run = Mock()
        with patch("run_clip.clean_sample_count", return_value=5):
            self.pipeline.clean()
        command = self.pipeline.paid_run.call_args.args[1]
        self.assertEqual(command[command.index("--masks-in") + 1], str(masks.resolve()))

    def test_wrong_sample_count_blocks_before_paid_clean_launch(self):
        self.pipeline.a.clean_masks = str(self.write_masks(4))
        self.pipeline.paid_run = Mock()
        with (
            patch(
                "run_clip.clean_sample_count",
                return_value=5,
            ),
            self.assertRaisesRegex(ValueError, "observed source sampling"),
        ):
            self.pipeline.clean()
        self.pipeline.paid_run.assert_not_called()

    def test_isolated_inpainting_mode_reaches_paid_worker(self):
        self.pipeline.a.inpaint_mode = "isolated"
        self.pipeline.paid_run = Mock()
        self.pipeline.clean()
        command = self.pipeline.paid_run.call_args.args[1]
        self.assertEqual(command[command.index("--inpaint-mode") + 1], "isolated")

    def test_default_inpainting_mode_remains_legacy(self):
        flags = self.pipeline.clean_flags()
        self.assertEqual(flags[flags.index("--inpaint-mode") + 1], "legacy")

    def test_invalid_inpainting_mode_blocks_before_paid_launch(self):
        self.pipeline.a.inpaint_mode = "typo"
        self.pipeline.paid_run = Mock()
        with self.assertRaisesRegex(ValueError, "inpaint-mode"):
            self.pipeline.clean()
        self.pipeline.paid_run.assert_not_called()

    def test_orchestrated_submission_cannot_silently_discard_isolated_mode(self):
        with (
            patch.object(
                sys,
                "argv",
                [
                    "run_clip.py",
                    "--clip",
                    "fixture.mp4",
                    "--name",
                    "fixture",
                    "--orchestrated",
                    "--inpaint-mode",
                    "isolated",
                ],
            ),
            patch.object(run_clip, "resolve_shots") as resolve,
            patch("sys.stderr"),
            self.assertRaises(SystemExit) as exited,
        ):
            run_clip.main()
        self.assertEqual(exited.exception.code, 2)
        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
