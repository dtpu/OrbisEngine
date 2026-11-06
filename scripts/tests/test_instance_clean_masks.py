"""Offline fixtures for cached-only person masks; never downloads or runs a learned model."""

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import instance_clean_masks as masks
import run_clip


def prediction(height=20, width=24):
    result = np.zeros((3, 1, height, width), np.float32)
    result[0, 0, 12:15, 5:8] = 1
    result[1, 0, 3:6, 17:20] = 1
    result[2] = 1  # A non-person must never be included.
    return {
        "labels": np.array([1, 1, 3]),
        "scores": np.array([0.99, 0.7, 0.99]),
        "boxes": np.array([[5, 12, 8, 15], [17, 3, 20, 6], [0, 0, width, height]], float),
        "masks": result,
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    clip = tmp_path / "source.mp4"
    clip.write_bytes(b"source fixture")
    provenance = {
        "sourceSha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "sourceTimeBase": "1/12000",
        "frames": [
            {
                "frameIndex": i,
                "sourceIndex": i * 2 + 1,
                "sourcePts": (i * 2 + 1) * 1000,
                "sourceTimeSeconds": (i * 2 + 1) / 12,
            }
            for i in range(2)
        ],
    }
    monkeypatch.setattr(masks, "source_dimensions", lambda clip: (24, 20))
    monkeypatch.setattr(
        masks,
        "resample_source",
        lambda *a, **k: (np.zeros((2, 20, 24, 3), np.uint8).tobytes(), provenance),
    )
    predictor = Mock(side_effect=lambda rgb: prediction(*rgb.shape[:2]))
    monkeypatch.setattr(
        masks,
        "cached_predictor",
        lambda weights=None: (predictor, {"checkpointSha256": "a" * 64, "device": "cpu"}),
    )
    return clip, tmp_path / "new-output", provenance, predictor


def test_all_person_instances_exact_metadata_and_small_padding_without_column_extension(fixture):
    clip, out, provenance, predictor = fixture
    report = masks.build_instance_clean_masks(clip, out, contact_sheets=True)
    assert report["status"] == "complete" and report["acceptance"] == "none"
    assert report["sourceProvenance"] == provenance
    assert report["parameters"]["dilatePixels"] == 2
    assert [row["instances"] for row in report["frames"]] == [2, 2]
    assert predictor.call_count == 2
    with np.load(out / "masks.npz", allow_pickle=False) as archive:
        bits = np.unpackbits(archive["masks"], axis=-1)[:, :, :24]
        assert archive["shape"].tolist() == [2, 20, 24]
        assert bits[0, 13, 6] and bits[0, 4, 18]
        assert not bits[0, 19].any(), "A lower-body component must not be projected to image bottom"
        assert not bits[0, :, 23].any(), "Non-person class must not enter the union"
        assert archive["source_indices"].tolist() == [1, 3]
        assert archive["source_pts"].tolist() == [1000, 3000]
        assert archive["source_time_base"].item() == "1/12000"
        assert archive["source_sha256"].item() == provenance["sourceSha256"]
    assert len(list((out / "frames").glob("*.npz"))) == 2
    assert (out / "contact-000.jpg").is_file()


def test_output_overwrite_is_refused_before_loading_or_prediction(fixture, monkeypatch):
    clip, out, _, predictor = fixture
    out.mkdir()
    (out / "retained.txt").write_text("retain")
    loader = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(masks, "cached_predictor", loader)
    with pytest.raises(ValueError, match="already exists"):
        masks.build_instance_clean_masks(clip, out)
    assert (out / "retained.txt").read_text() == "retain"
    loader.assert_not_called()
    predictor.assert_not_called()


@pytest.mark.parametrize("change", ["no_person", "empty_mask", "nan_mask", "wrong_geometry"])
def test_bad_or_empty_predictions_preserve_failed_report_without_usable_archive(fixture, change):
    clip, out, _, predictor = fixture
    value = prediction()
    if change == "no_person":
        value["labels"][:] = 3
    elif change == "empty_mask":
        value["masks"][0] = 0
    elif change == "nan_mask":
        value["masks"][0, 0, 0, 0] = np.nan
    else:
        value["masks"] = value["masks"][:, :, :-1]
    predictor.side_effect = [prediction(), value]
    with pytest.raises(ValueError):
        masks.build_instance_clean_masks(clip, out)
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "failed" and report["failedFrame"] == 1
    assert len(report["frames"]) == 1
    assert not (out / "masks.npz").exists()
    assert (out / "frames/000000.npz").exists()


def test_source_metadata_mismatch_stops_before_predictions(fixture):
    clip, out, provenance, predictor = fixture
    provenance["sourceSha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        masks.build_instance_clean_masks(clip, out)
    predictor.assert_not_called()
    assert not out.exists()


def test_prediction_phase_cannot_open_network_socket(fixture):
    clip, out, _, predictor = fixture
    predictor.side_effect = lambda rgb: socket.create_connection(("example.com", 443))
    with pytest.raises(RuntimeError, match="Network disabled"):
        masks.build_instance_clean_masks(clip, out)
    assert json.loads((out / "report.json").read_text())["status"] == "failed"


def fake_model_modules(monkeypatch, tmp_path):
    torch = ModuleType("torch")
    torch.hub = SimpleNamespace(get_dir=lambda: str(tmp_path))
    torch.set_num_threads, torch.set_num_interop_threads, torch.load = (
        Mock(),
        Mock(),
        Mock(return_value={}),
    )
    torch.__version__ = "fixture"
    torchvision = ModuleType("torchvision")
    torchvision.__version__ = "fixture"
    detection = ModuleType("torchvision.models.detection")
    model = Mock()
    model.cpu.return_value = model
    model.eval.return_value = model
    detection.maskrcnn_resnet50_fpn_v2 = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", ModuleType("torchvision.models"))
    monkeypatch.setitem(sys.modules, "torchvision.models.detection", detection)
    return torch, detection


def test_missing_or_wrong_cache_never_constructs_model_or_downloads(tmp_path, monkeypatch):
    torch, detection = fake_model_modules(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no download attempted"):
        masks.cached_predictor(tmp_path / "missing.pth")
    wrong = tmp_path / "wrong.pth"
    wrong.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        masks.cached_predictor(wrong)
    detection.maskrcnn_resnet50_fpn_v2.assert_not_called()
    torch.load.assert_not_called()


def test_cached_loader_disables_both_weight_download_paths_and_selects_cpu(tmp_path, monkeypatch):
    torch, detection = fake_model_modules(monkeypatch, tmp_path)
    weights = tmp_path / "cached.pth"
    weights.write_bytes(b"mock cached state")
    monkeypatch.setattr(masks, "file_hash", lambda path: "73cbd019" + "0" * 56)
    _, report = masks.cached_predictor(weights)
    detection.maskrcnn_resnet50_fpn_v2.assert_called_once_with(weights=None, weights_backbone=None)
    torch.load.assert_called_once_with(weights, map_location="cpu", weights_only=True)
    torch.set_num_threads.assert_called_once_with(4)
    assert report["device"] == "cpu"
    assert report["checkpointSha256"].startswith("73cbd019")


def test_real_ffmpeg_archive_passes_pipeline_validation_and_tampered_pts_fail(
    tmp_path, monkeypatch
):
    clip = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x32:r=12:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    monkeypatch.setattr(
        masks,
        "cached_predictor",
        lambda weights=None: (
            lambda rgb: prediction(*rgb.shape[:2]),
            {"checkpointSha256": "b" * 64},
        ),
    )
    out = tmp_path / "masks"
    report = masks.build_instance_clean_masks(clip, out, fps=6, dilate=0)
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.clip, pipeline.info = clip, {"width": 64, "height": 32}
    pipeline.a = SimpleNamespace(fps=6, marble="video")
    pipeline.validate_clean_masks(out / "masks.npz")
    assert report["shape"][0] == 3
    with np.load(out / "masks.npz", allow_pickle=False) as archive:
        changed = {key: archive[key] for key in archive.files}
    changed["source_pts"] = changed["source_pts"] + 1
    np.savez_compressed(tmp_path / "tampered.npz", **changed)
    with pytest.raises(ValueError, match="selected source pixels"):
        pipeline.validate_clean_masks(tmp_path / "tampered.npz")
