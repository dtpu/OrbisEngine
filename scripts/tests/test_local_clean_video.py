"""Offline CPU helper contract; no real model load or inference."""

import contextlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_clean_video import clean_video, load_cpu_model
from worker.wander_worker.source_timing import resample_source


class LocalCleanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clip = self.root / "source.mkv"
        raw = b"".join(bytes([i * 20, 150, 200]) * 16 * 16 for i in range(6))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "16x16",
                "-r",
                "12",
                "-i",
                "-",
                "-c:v",
                "ffv1",
                str(self.clip),
            ],
            input=raw,
            capture_output=True,
            check=True,
            timeout=30,
        )
        _, provenance = resample_source(self.clip, 12)
        mask = np.zeros((6, 16, 16), dtype=bool)
        mask[:, 5:11, 5:11] = True
        self.data = dict(
            shape=np.array(mask.shape),
            masks=np.packbits(mask, axis=2),
            source_indices=np.array([f["sourceIndex"] for f in provenance["frames"]]),
            source_pts=np.array([f["sourcePts"] for f in provenance["frames"]]),
            source_sha256=np.array(provenance["sourceSha256"]),
            source_time_base=np.array(provenance["sourceTimeBase"]),
        )
        self.masks = self.root / "masks.npz"
        np.savez_compressed(self.masks, **self.data)
        self.weights = self.root / "cached.pt"
        self.weights.write_bytes(b"injected model fixture, not TorchScript")
        self.model = Mock(
            side_effect=lambda image, mask: Image.new("RGB", image.size, (230, 10, 40))
        )
        self.loader = Mock(return_value=self.model)
        self.out = self.root / "output"

    def run_clean(self, **options):
        return clean_video(
            clip=self.clip,
            masks=self.masks,
            weights=self.weights,
            out=self.out,
            lama_px=16,
            model_loader=self.loader,
            **options,
        )

    def test_torchscript_adapter_uses_cpu_float_rgb_and_integer_binary_mask(self):
        class Tensor:
            def __init__(self, values):
                self.values = values
                self.shape = values.shape

            def __getitem__(self, key):
                return Tensor(self.values[key])

            def permute(self, *axes):
                return Tensor(self.values.transpose(axes))

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        def inference(rgb, mask):
            self.assertEqual(rgb.shape, (1, 3, 8, 16))
            self.assertEqual(rgb.values.dtype, np.float32)
            self.assertEqual(mask.shape, (1, 1, 8, 16))
            self.assertEqual(mask.values.dtype, np.int64)
            self.assertEqual(set(np.unique(mask.values)), {0, 1})
            self.assertTrue(np.all(rgb.values == 1))
            return Tensor(np.full((1, 3, 8, 16), 0.5, dtype=np.float32))

        model = Mock(side_effect=inference)
        torch = SimpleNamespace(
            jit=SimpleNamespace(load=Mock(return_value=model)),
            from_numpy=Tensor,
            inference_mode=contextlib.nullcontext,
            set_num_threads=Mock(),
        )

        def load(*args, **kwargs):
            torch.set_num_threads.assert_called_once_with(4)
            return model

        torch.jit.load.side_effect = load
        mask = np.zeros((8, 16), np.uint8)
        mask[:, 8:] = 1
        with patch.dict(sys.modules, {"torch": torch}):
            predict = load_cpu_model(self.weights)
            result = predict(Image.new("RGB", (16, 8), "white"), Image.fromarray(mask))
        torch.set_num_threads.assert_called_once_with(4)
        torch.jit.load.assert_called_once_with(str(self.weights), map_location="cpu")
        model.eval.assert_called_once_with()
        self.assertTrue(np.all(np.array(result) == 127))

    def make_recipe(self):
        # Frame zero is a genuinely unmasked source donor, not generated model output.
        self.data["masks"][0] = 0
        np.savez_compressed(self.masks, **self.data)
        return {
            "schema": "wander.source-patches/1",
            "sourceSha256": str(self.data["source_sha256"]),
            "sourceTimeBase": str(self.data["source_time_base"]),
            "sourceIndices": self.data["source_indices"].tolist(),
            "sourcePts": self.data["source_pts"].tolist(),
            "fps": 12,
            "width": 16,
            "height": 16,
            "patches": [
                {
                    "donorSample": 0,
                    "targetSample": 2,
                    "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "donorRoiXYXY": [0, 0, 16, 16],
                    "featherPx": 8,
                    "maskMarginPx": 8,
                }
            ],
        }

    def test_source_patch_uses_extra_decoded_donor_and_retains_truthful_lama_base(self):
        recipe = self.make_recipe()
        path = self.root / "recipe.json"
        path.write_text(json.dumps(recipe))
        self.model.side_effect = lambda image, mask: Image.new("RGB", image.size, "black")
        report = self.run_clean(only=[2], source_patches=path)
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(report["sourcePatches"], recipe)
        self.assertEqual((self.out / "source-patches.json").read_bytes(), path.read_bytes())
        check = report["compositeChecks"][0]
        self.assertEqual(check["frameIndex"], 2)
        evidence = check["sourcePatch"]
        self.assertEqual(evidence["donorProvenance"]["frameIndex"], 0)
        self.assertEqual(evidence["donorProvenance"]["sourcePts"], int(self.data["source_pts"][0]))
        self.assertEqual(evidence["sourceSupportPixels"], 36)
        self.assertEqual(evidence["outsideMaskChangedPixels"], 0)
        lama = np.array(Image.open(self.out / check["lamaPath"]))
        final = np.array(Image.open(self.out / check["path"]))
        self.assertTrue(np.all(lama[5:11, 5:11] == 0))
        self.assertGreater(final[7, 7, 1], 0)
        self.assertTrue(np.array_equal(final[0, 0], [40, 150, 200]))
        self.assertEqual(
            check["lamaPngSha256"],
            hashlib.sha256((self.out / check["lamaPath"]).read_bytes()).hexdigest(),
        )

    def test_invalid_source_recipes_fail_before_model_load(self):
        recipe = self.make_recipe()
        changes = {
            "wrong-source": lambda r: r.update(sourceSha256="0" * 64),
            "wrong-grid": lambda r: r["sourceIndices"].__setitem__(1, 0),
            "wrong-pts": lambda r: r["sourcePts"].__setitem__(1, 0),
            "wrong-fps": lambda r: r.update(fps=True),
            "sample-out-of-range": lambda r: r["patches"][0].update(targetSample=6),
            "bool-sample": lambda r: r["patches"][0].update(donorSample=True),
            "nonfinite-H": lambda r: r["patches"][0]["homography"][0].__setitem__(0, float("nan")),
            "singular-H": lambda r: r["patches"][0].update(
                homography=[[0, 0, 0], [0, 0, 0], [0, 0, 1]]
            ),
            "unnormalized-H": lambda r: r["patches"][0]["homography"][2].__setitem__(2, 2),
            "bad-H-shape": lambda r: r["patches"][0].update(homography=[[1, 0], [0, 1]]),
            "projective-pole": lambda r: r["patches"][0].update(
                homography=[[1, 0, 0], [0, 1, 0], [-0.2, 0, 1]]
            ),
            "bad-ROI": lambda r: r["patches"][0].update(donorRoiXYXY=[0, 0, 17, 16]),
            "overlap": lambda r: r["patches"][0].update(donorSample=1),
            "margin": lambda r: r["patches"][0].update(maskMarginPx=0),
            "feather": lambda r: r["patches"][0].update(featherPx=0),
            "duplicate-target": lambda r: r["patches"].append(copy.deepcopy(r["patches"][0])),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                self.out = self.root / name
                modified = copy.deepcopy(recipe)
                change(modified)
                path = self.root / f"{name}.json"
                path.write_text(json.dumps(modified))
                with self.assertRaises(ValueError):
                    self.run_clean(source_patches=path)
                self.loader.assert_not_called()
                self.assertFalse((self.out / "clean.json").exists())

    def test_thread_limit_forwarded_and_recorded(self):
        report = self.run_clean(only=[0], threads=2)
        self.loader.assert_called_once_with(self.weights.resolve(), threads=2)
        self.assertEqual(report["parameters"]["threads"], 2)

    def test_invalid_threads_refused_before_output_or_model(self):
        for threads in (0, -1, 33, True, 1.5, "4"):
            with self.subTest(threads=threads), self.assertRaisesRegex(ValueError, "threads"):
                self.run_clean(threads=threads)
        self.loader.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_selected_indices_and_source_provenance_preserved(self):
        report = self.run_clean(only=[1, 4])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["qualityStatus"], "unreviewed")
        self.assertNotIn("sourcePatches", report)
        self.assertEqual(report["selectedSampleIndices"], [1, 4])
        self.assertEqual([f["frameIndex"] for f in report["sourceProvenance"]["frames"]], [1, 4])
        self.assertEqual([f["sourceIndex"] for f in report["compositeChecks"]], [1, 4])
        self.assertEqual(self.model.call_count, 2)
        for index, check in zip([1, 4], report["compositeChecks"]):
            self.assertEqual(check["sourcePts"], int(self.data["source_pts"][index]))
            self.assertEqual(check["outsideMaskChangedPixels"], 0)
            self.assertEqual(check["sourceOutsideRgbSha256"], check["cleanOutsideRgbSha256"])
            pixels = np.array(Image.open(self.out / check["path"]))
            self.assertTrue(np.array_equal(pixels[0, 0], [index * 20, 150, 200]))
            self.assertTrue(np.array_equal(pixels[7, 7], [230, 10, 40]))
            self.assertEqual(
                check["pngSha256"],
                hashlib.sha256((self.out / check["path"]).read_bytes()).hexdigest(),
            )
        self.assertFalse((self.out / "f_0000.png").exists())
        self.assertTrue((self.out / "clean.json").is_file())

    def test_every_metadata_failure_prevents_model_load_including_unselected_pts(self):
        variants = {
            "hash": dict(source_sha256=np.array("0" * 64)),
            "pts": dict(source_pts=self.data["source_pts"] + 1),
            "timebase": dict(source_time_base=np.array("1/99")),
            "dimensions": dict(shape=np.array([6, 16, 15])),
            "packed_dtype": dict(masks=self.data["masks"].astype(np.int64)),
            "packed_shape": dict(masks=self.data["masks"][:, :, :1]),
            "indices": dict(source_indices=np.array([0, 1, 2, 3, 4, 99])),
        }
        bad_pts = self.data["source_pts"].copy()
        bad_pts[5] += 1
        variants["unselected_pts"] = dict(source_pts=bad_pts)
        for name, changes in variants.items():
            with self.subTest(name=name):
                self.out = self.root / name
                np.savez_compressed(self.masks, **(self.data | changes))
                with self.assertRaises(ValueError):
                    self.run_clean(only=[0])
                self.loader.assert_not_called()
                self.assertEqual(
                    json.loads((self.out / "progress.json").read_text())["status"], "failed"
                )
                self.assertFalse((self.out / "clean.json").exists())

    def test_outside_mutation_rejected_before_png_with_precall_bindings(self):
        def corrupt(image, mask, model, **kwargs):
            image[0, 0, 0] ^= 1
            mask[:] = True
            return image

        with patch("local_clean_video.inpaint_frame", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "Outside-mask pixels changed"):
                self.run_clean(only=[2])
        self.assertEqual(list(self.out.glob("*.png")), [])
        self.assertFalse((self.out / "clean.json").exists())
        failed = json.loads((self.out / "progress.json").read_text())
        check = failed["compositeChecks"][0]
        self.assertEqual(check["outsideMaskChangedPixels"], 1)
        self.assertEqual(check["maskPixels"], 36)
        self.assertEqual(check["sourceIndex"], 2)
        self.assertEqual(len(check["decodedRgbSha256"]), 64)
        self.assertEqual(len(check["removalMaskSha256"]), 64)

    def test_no_overwrite(self):
        self.out.mkdir()
        sentinel = self.out / "existing"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_clean()
        self.loader.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_dangling_output_symlink_is_not_followed(self):
        target = self.root / "must-not-create"
        self.out.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.run_clean()
        self.assertFalse(target.exists())
        self.loader.assert_not_called()

    def test_empty_and_full_masks_have_explicit_preservation_counts(self):
        for full in (False, True):
            self.out = self.root / f"mask-{full}"
            mask = np.full((6, 16, 16), full, dtype=bool)
            np.savez_compressed(self.masks, **(self.data | {"masks": np.packbits(mask, axis=2)}))
            report = self.run_clean(only=[0])
            check = report["compositeChecks"][0]
            self.assertEqual(check["maskPixels"], 256 if full else 0)
            self.assertEqual(check["outsideMaskPixels"], 0 if full else 256)
            self.assertEqual(check["outsideMaskChangedPixels"], 0)
        self.assertEqual(self.model.call_count, 1)

    def test_changed_source_during_model_call_withholds_final_report(self):
        def mutate(image, mask):
            self.clip.write_bytes(self.clip.read_bytes() + b"changed")
            return Image.new("RGB", image.size)

        self.model.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "Inputs changed during cleaning"):
            self.run_clean(only=[0])
        self.assertFalse((self.out / "clean.json").exists())
        self.assertEqual(json.loads((self.out / "progress.json").read_text())["status"], "failed")

    def test_subset_encode_refused_before_model(self):
        with self.assertRaisesRegex(ValueError, "full sample selection"):
            self.run_clean(only=[0, 2], encode=True)
        self.loader.assert_not_called()

    def test_full_encode_has_every_frame_and_no_invented_audio(self):
        report = self.run_clean(encode=True)
        video = self.out / report["video"]["path"]
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-of",
                    "json",
                    str(video),
                ],
                capture_output=True,
                check=True,
            ).stdout
        )
        self.assertEqual(len(probe["streams"]), 1)
        self.assertEqual(int(probe["streams"][0]["nb_read_frames"]), 6)
        self.assertEqual(report["video"]["audio"], "absent")

    def test_model_failure_keeps_incremental_completed_frames_without_final_success(self):
        self.model.side_effect = [
            Image.new("RGB", (16, 16), "black"),
            RuntimeError("injected failure"),
        ]
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self.run_clean(only=[0, 2])
        self.assertTrue((self.out / "f_0000.png").exists())
        self.assertFalse((self.out / "f_0002.png").exists())
        report = json.loads((self.out / "progress.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual([c["status"] for c in report["compositeChecks"]], ["complete", "running"])
        self.assertFalse((self.out / "clean.json").exists())


if __name__ == "__main__":
    unittest.main()
