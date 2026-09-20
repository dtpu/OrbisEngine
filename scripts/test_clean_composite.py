"""Offline compositing invariants and first-frame diagnostic delivery (no inference)."""

import ast
import hashlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))


def worker_functions():
    tree = ast.parse((ROOT / "worker/modal_clean_video.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"composite_fill", "clean"}
    ]
    for node in nodes:
        node.decorator_list = []
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "clean-worker", "exec"), namespace)
    return namespace


class CleanCompositeTests(unittest.TestCase):
    def setUp(self):
        self.functions = worker_functions()
        self.composite = self.functions["composite_fill"]
        self.mask = np.zeros((24, 32), bool)
        self.mask[6:18, 12:15] = True  # Thin support exposes inward feather leakage.
        self.fill = np.full((24, 32, 3), 127, np.uint8)

    def test_masked_pixels_are_independent_of_original_source(self):
        black = self.composite(np.zeros_like(self.fill), self.fill, self.mask)
        white = self.composite(np.full_like(self.fill, 255), self.fill, self.mask)
        np.testing.assert_array_equal(black[self.mask], self.fill[self.mask])
        np.testing.assert_array_equal(white[self.mask], self.fill[self.mask])

    def test_exterior_feather_is_unchanged(self):
        source = np.full_like(self.fill, 220)
        alpha = np.clip(cv2.GaussianBlur(self.mask.astype(np.float32), (0, 0), 3) * 1.5, 0, 1)
        old = (source * (1 - alpha[..., None]) + self.fill * alpha[..., None]).astype(np.uint8)
        actual = self.composite(source, self.fill, self.mask)
        np.testing.assert_array_equal(actual[~self.mask], old[~self.mask])
        self.assertTrue(np.any(old[self.mask] != self.fill[self.mask]))

    def test_empty_and_full_masks(self):
        source = np.full_like(self.fill, 201)
        np.testing.assert_array_equal(
            self.composite(source, self.fill, np.zeros_like(self.mask)), source
        )
        np.testing.assert_array_equal(
            self.composite(source, self.fill, np.ones_like(self.mask)), self.fill
        )

    def test_invalid_shapes_nonfinite_and_nonbinary_inputs_fail(self):
        for image, fill, mask in [
            (self.fill, self.fill[:, :-1], self.mask),
            (self.fill, self.fill, self.mask[..., None]),
            (self.fill, np.full(self.fill.shape, np.nan), self.mask),
            (np.full(self.fill.shape, np.inf), self.fill, self.mask),
            (self.fill, self.fill, np.full(self.mask.shape, np.nan)),
            (self.fill, self.fill, np.full(self.mask.shape, 0.5)),
            (self.fill, np.full(self.fill.shape, 256), self.mask),
            (np.empty((0, 32, 3)), np.empty((0, 32, 3)), np.empty((0, 32))),
        ]:
            with self.subTest(image=image.shape, fill=fill.shape, mask=mask.shape):
                with self.assertRaises(ValueError):
                    self.composite(image, fill, mask)

    def test_worker_returns_source_bound_lossless_diagnostics_for_first_selected_frame(self):
        from wander_worker import source_timing

        source = np.full((3, 16, 16, 3), 230, np.uint8)
        masks = np.zeros((3, 16, 16), bool)
        masks[1:, 4:12, 4:12] = True
        packed = io.BytesIO()
        np.savez_compressed(packed, masks=np.packbits(masks, axis=-1))
        frames = [{"frameIndex": i, "sourceIndex": i * 2, "sourcePts": i * 2002} for i in range(3)]
        provenance = {
            "frames": frames,
            "sourceSha256": hashlib.sha256(b"source").hexdigest(),
            "sourceTimeBase": "1/24000",
            "sourceAverageFrameRate": "24000/1001",
        }
        calls = []

        class FakeLama:
            def __init__(self, device):
                pass

            def __call__(self, image, mask):
                calls.append((image.size, mask.size))
                return Image.new("RGB", image.size, (127, 127, 127))

        fake_package = types.ModuleType("simple_lama_inpainting")
        fake_package.SimpleLama = FakeLama
        self.functions["cache"] = types.SimpleNamespace(commit=lambda: None)
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / "fake-weights"
            weights.write_bytes(b"not a model; never loaded")
            self.functions["LAMA_PT"] = str(weights)
            with (
                patch.dict(sys.modules, {"simple_lama_inpainting": fake_package}),
                patch("torch.cuda.is_available", return_value=True),
                patch("importlib.metadata.version", return_value="test-package"),
                patch.object(
                    source_timing, "resample_source", return_value=(source.tobytes(), provenance)
                ),
            ):
                result = self.functions["clean"](
                    b"source",
                    width=16,
                    height=16,
                    lama_px=16,
                    only="1,2",
                    encode=False,
                    masks_npz=packed.getvalue(),
                )
                empty_first = self.functions["clean"](
                    b"source",
                    width=16,
                    height=16,
                    lama_px=16,
                    only="0",
                    encode=False,
                    masks_npz=packed.getvalue(),
                )
        self.assertIsNone(empty_first["report"]["error"])
        self.assertFalse(empty_first["report"]["diagnostics"]["inpaintingApplied"])
        self.assertEqual(
            set(empty_first["diagnostics"]),
            {
                "source.png",
                "removal-mask.png",
                "composite.png",
            },
        )
        self.assertIsNone(result["report"]["error"])
        self.assertEqual(len(calls), 2)
        diagnostic = result["report"]["diagnostics"]
        self.assertEqual(diagnostic["source"], frames[1])
        self.assertEqual(diagnostic["sourceSha256"], provenance["sourceSha256"])
        self.assertEqual(diagnostic["packageVersion"], "test-package")
        self.assertEqual(
            diagnostic["modelSha256"], hashlib.sha256(b"not a model; never loaded").hexdigest()
        )
        self.assertEqual(
            set(result["diagnostics"]),
            {
                "source.png",
                "removal-mask.png",
                "pass-00-model-rgb.png",
                "pass-00-model-mask.png",
                "pass-00-raw-fill.png",
                "pass-00-composite.png",
                "composite.png",
            },
        )
        for name, data in result["diagnostics"].items():
            self.assertEqual(diagnostic["images"][name]["sha256"], hashlib.sha256(data).hexdigest())
        composite = np.asarray(Image.open(io.BytesIO(result["diagnostics"]["composite.png"])))
        np.testing.assert_array_equal(composite[masks[1]], np.full((64, 3), 127))


if __name__ == "__main__":
    unittest.main()
