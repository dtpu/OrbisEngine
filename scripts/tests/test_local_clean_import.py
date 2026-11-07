"""Receipt-backed offline clean import uses real source/PNG/video validation, no inference."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_clip
from local_clean_video import clean_video, pixel_hash
from worker.wander_worker.source_timing import resample_source


@pytest.fixture
def local_pipeline(tmp_path, monkeypatch):
    clip = tmp_path / "source.mkv"
    raw = b"".join(bytes([i * 15, 130, 200]) * 16 * 16 for i in range(12))
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
            str(clip),
        ],
        input=raw,
        check=True,
        capture_output=True,
    )
    _, provenance = resample_source(clip, 12)
    masks = np.zeros((12, 16, 16), bool)
    masks[:, 5:11, 5:11] = True
    archive = tmp_path / "masks.npz"
    np.savez_compressed(
        archive,
        masks=np.packbits(masks, axis=2),
        shape=np.array(masks.shape),
        source_indices=np.array([f["sourceIndex"] for f in provenance["frames"]]),
        source_pts=np.array([f["sourcePts"] for f in provenance["frames"]]),
        source_sha256=np.array(provenance["sourceSha256"]),
        source_time_base=np.array(provenance["sourceTimeBase"]),
    )
    weights = tmp_path / "injected.pt"
    weights.write_bytes(b"not a real model")
    offline = tmp_path / "offline"
    clean_video(
        clip=clip,
        masks=archive,
        weights=weights,
        out=offline,
        lama_px=16,
        encode=True,
        model_loader=lambda weights, **kwargs: (
            lambda image, mask: Image.new("RGB", image.size, "black")
        ),
    )
    p = object.__new__(run_clip.Pipeline)
    p.ctx = tmp_path / "run"
    p.ctx.mkdir()
    p.name = "fixture"
    p.clip = clip
    p.info = dict(width=16, height=16)
    p.a = SimpleNamespace(
        import_local_clean=str(offline / "clean.json"),
        clean_masks=str(archive),
        clean_candidate="local-v1",
        only="clean",
        force="clean",
        fps=12,
        moved_mask=False,
        orchestrated=False,
        skip_marble=False,
        marble="video",
        reuse_world=None,
        retained_scene=None,
        skip_finetune=True,
    )
    p.selected_clean_candidate = None
    p.clean_mp4 = tmp_path / "original.mp4"
    p.clean_report = p.ctx / "clean.json"
    p.clean_masks_path = p.ctx / "masks.npz"
    p.marble = "video"
    p.state = run_clip.State(p.ctx / "state.json")
    p.state.record("clean", status="ok")
    p.state.record("_paid_clean", status="failed", attempt="preserved-failed-paid-attempt")
    p.state.record("review", status="blocked", accepted=False)
    p.state.record("verify", status="ok")
    p.state.record("_verify", status="passed", passed=True)
    p.paid_run = Mock(side_effect=AssertionError("must not submit paid work"))
    p.summary = Mock()
    p.stages = ["clean", "review", "marble_video", "verify"]
    p.deps = dict(clean=[], review=["clean"], marble_video=["review"], verify=["marble_video"])
    p.ledger = tmp_path / "shared-ledger.json"
    p.ledger.write_text('{"retained failed history":"do not modify"}\n')
    monkeypatch.setenv("WANDER_STAGE_LEDGER", str(p.ledger))
    monkeypatch.setattr(run_clip, "MARBLE_DIR", tmp_path / "marble")
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(tmp_path / "public"))
    return p


def update_receipt(p, change):
    path = Path(p.a.import_local_clean)
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))


def test_import_selects_real_outputs_without_paid_claim_and_requires_new_review(local_pipeline):
    p = local_pipeline
    paid_history = copy.deepcopy(p.state.data["stages"]["_paid_clean"])
    ledger = p.ledger.read_bytes()
    original_receipt = Path(p.a.import_local_clean).read_bytes()
    assert p.go() == 0
    p.paid_run.assert_not_called()
    assert p.ledger.read_bytes() == ledger
    assert p.state.data["stages"]["_paid_clean"] == paid_history
    clean = p.state.data["stages"]["clean"]
    assert clean["provider"] == "local"
    assert clean["execution"] == "imported_local"
    assert clean["freshInference"] is False
    assert p.state.data["stages"]["review"]["status"] == "pending"
    assert p.state.data["stages"]["review"]["accepted"] is False
    assert p.state.data["stages"]["verify"]["status"] == "pending"
    assert p.state.data["stages"]["_verify"]["passed"] is False
    directory = p.ctx / "clean-candidates/local-v1"
    assert (directory / "clean.log").read_bytes() == original_receipt
    report = json.loads((directory / "clean.json").read_text())
    assert report["provider"] == "local"
    assert report["qualityStatus"] == "unreviewed"
    assert report["outputVideoSha256"] == run_clip.file_sha256(p.clean_mp4)
    p.load_clean_selection()
    assert p.clean_masks_path == directory / "masks.npz"
    assert len(list(directory.glob("f_*.png"))) == 12
    # The original receipt is part of the hash-pinned selection, not an unpinned sidecar.
    (directory / "clean.log").write_text("tampered receipt")
    with pytest.raises(ValueError, match="changed"):
        p.load_clean_selection()


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="failed"),
        lambda r: r.update(selectedSampleIndices=[0]),
        lambda r: r.update(sourceSha256="0" * 64),
        lambda r: r["parameters"].update(fps=float("nan")),
        lambda r: r["parameters"].update(fps=True),
        lambda r: r["parameters"].update(threads=True),
        lambda r: r["parameters"].update(lamaPx=float("nan")),
        lambda r: r["parameters"].update(passes=True),
        lambda r: r.update(width=True),
        lambda r: r["sourceProvenance"]["frames"][0].update(sourcePts=True),
        lambda r: r["compositeChecks"][0].update(outsideMaskChangedPixels=False),
        lambda r: r["compositeChecks"][0].update(maskPixels=0),
        lambda r: r["compositeChecks"][0].update(removalMaskSha256="0" * 64),
        lambda r: r["compositeChecks"][0].update(path="../outside.png"),
        lambda r: r["compositeChecks"][0].update(decodedRgbSha256="0" * 64),
        lambda r: r["video"].update(sha256="0" * 64),
        lambda r: r.pop("codeHashes"),
    ],
)
def test_invalid_or_tampered_receipt_never_selected(local_pipeline, change):
    p = local_pipeline
    update_receipt(p, change)
    with pytest.raises(ValueError):
        p.clean()
    p.paid_run.assert_not_called()
    assert not (p.ctx / "clean-selection.json").exists()


def test_changed_png_rejected_even_with_updated_self_reported_hashes(local_pipeline):
    p = local_pipeline
    png = Path(p.a.import_local_clean).parent / "f_0000.png"
    pixels = np.array(Image.open(png))
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels).save(png)

    def change(report):
        check = report["compositeChecks"][0]
        check["pngSha256"] = run_clip.file_sha256(png)
        check["cleanedRgbSha256"] = pixel_hash(pixels)
        mask = np.zeros((16, 16), bool)
        mask[5:11, 5:11] = True
        check["cleanOutsideRgbSha256"] = pixel_hash(pixels[~mask])

    update_receipt(p, change)
    with pytest.raises(ValueError, match="outside pixels"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def test_missing_png_and_symlink_outputs_are_rejected(local_pipeline):
    p = local_pipeline
    png = Path(p.a.import_local_clean).parent / "f_0000.png"
    saved = png.with_name("original.png")
    png.rename(saved)
    with pytest.raises(ValueError, match="sibling file"):
        p.clean()
    p.a.clean_candidate = "symlink"
    png.symlink_to(saved)
    with pytest.raises(ValueError, match="sibling file"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def test_sparse_archive_cannot_claim_full_clip(local_pipeline):
    p = local_pipeline
    archive = Path(p.a.clean_masks)
    with np.load(archive) as data:
        arrays = {k: data[k] for k in data.files}
    for key in ("masks", "source_indices", "source_pts"):
        arrays[key] = arrays[key][:-1]
    arrays["shape"] = np.array([11, 16, 16])
    np.savez_compressed(archive, **arrays)

    def change(report):
        report.update(
            suppliedMasksSha256=run_clip.file_sha256(archive),
            frames=11,
            selectedSampleIndices=list(range(11)),
        )

    update_receipt(p, change)
    with pytest.raises(ValueError, match="full-clip sample coverage"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def test_world_submission_blocks_import_before_any_selection(local_pipeline):
    p = local_pipeline
    (p.ctx / "marble_video.log").write_text("legacy alias does not represent canonical log")
    (p.ctx / "marble_clean.log").write_text("uncertain original paid submission")
    with pytest.raises(ValueError, match="after a Marble submission"):
        p.clean()
    assert not (p.ctx / "clean-candidates").exists()


def test_import_requires_explicit_candidate_masks_and_isolated_stage_selection(local_pipeline):
    p = local_pipeline
    run_clip.validate_clean_candidate(p.a)
    run_clip.validate_local_clean_import(p.a)
    for field, bad in (
        ("clean_candidate", None),
        ("clean_masks", None),
        ("only", None),
        ("only", "clean,marble_video"),
        ("moved_mask", True),
    ):
        args = copy.copy(p.a)
        setattr(args, field, bad)
        with pytest.raises(ValueError):
            run_clip.validate_local_clean_import(args)


def test_cached_clean_import_requires_explicit_force_even_same_candidate(local_pipeline):
    p = local_pipeline
    p.a.force = None
    p.selected_clean_candidate = p.a.clean_candidate
    with pytest.raises(ValueError, match="requires --force clean"):
        p.go()
    assert not (p.ctx / "clean-selection.json").exists()


def test_changed_masks_archive_rejected(local_pipeline):
    p = local_pipeline
    masks = Path(p.a.clean_masks)
    masks.write_bytes(masks.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="complete encoded CPU work"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def test_correct_count_but_clustered_source_grid_is_not_full_coverage(local_pipeline):
    p = local_pipeline
    archive = Path(p.a.clean_masks)
    with np.load(archive) as source:
        data = {key: source[key] for key in source.files}
    for key in ("masks", "source_indices", "source_pts"):
        data[key] = data[key][:6]
    data["shape"] = np.array([6, 16, 16])
    np.savez_compressed(archive, **data)
    weights = p.ctx.parent / "injected.pt"
    weights.write_bytes(b"fixture, no real model")
    clustered = p.ctx.parent / "clustered"
    clean_video(
        clip=p.clip,
        masks=archive,
        weights=weights,
        out=clustered,
        fps=6,
        lama_px=16,
        encode=True,
        model_loader=lambda weights, **kwargs: lambda image, mask: Image.new("RGB", image.size),
    )
    p.a.fps = 6
    p.a.import_local_clean = str(clustered / "clean.json")
    # Both the archive and source at 6 FPS contain six samples, but these masks
    # cover only source frames 0..5 instead of the full observed 1,3,5,7,9,11 grid.
    with pytest.raises(ValueError, match="complete observed FPS sampling"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def add_source_patch_receipt(p):
    archive = Path(p.a.clean_masks)
    with np.load(archive) as source:
        data = {key: source[key] for key in source.files}
    data["masks"][0] = 0
    np.savez_compressed(archive, **data)
    recipe = {
        "schema": "wander.source-patches/1",
        "sourceSha256": str(data["source_sha256"]),
        "sourceTimeBase": str(data["source_time_base"]),
        "sourceIndices": data["source_indices"].tolist(),
        "sourcePts": data["source_pts"].tolist(),
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
    recipe_path = p.ctx.parent / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    offline = p.ctx.parent / "patched"
    clean_video(
        clip=p.clip,
        masks=archive,
        weights=p.ctx.parent / "injected.pt",
        out=offline,
        lama_px=16,
        encode=True,
        source_patches=recipe_path,
        model_loader=lambda weights, **kwargs: (
            lambda image, mask: Image.new("RGB", image.size, "black")
        ),
    )
    p.a.import_local_clean = str(offline / "clean.json")
    return recipe


def test_import_replays_source_composite_and_retains_baseline_and_recipe(local_pipeline):
    p = local_pipeline
    recipe = add_source_patch_receipt(p)
    p.clean()
    directory = p.ctx / "clean-candidates/local-v1"
    report = json.loads((directory / "clean.json").read_text())
    assert report["sourcePatches"] == recipe
    assert report["compositionMethod"] == "LaMa plus measured source-guided composite"
    assert (directory / "source-patches.json").read_bytes() == (
        Path(p.a.import_local_clean).parent / "source-patches.json"
    ).read_bytes()
    base = directory / "lama_0002.png"
    assert run_clip.file_sha256(base) == report["compositeChecks"][2]["lamaPngSha256"]
    assert np.any(np.array(Image.open(base)) != np.array(Image.open(directory / "f_0002.png")))
    p.paid_run.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(sourcePatchesSha256="0" * 64),
        lambda r: r.update(sourcePatchAlgorithm="unknown"),
        lambda r: r["compositeChecks"][2]["sourcePatch"].update(sourceSupportPixels=999),
        lambda r: r["compositeChecks"][2]["sourcePatch"].update(outsideMaskChangedPixels=False),
        lambda r: r["compositeChecks"][2].update(lamaPath="../escape.png"),
        lambda r: r["compositeChecks"][2]["sourcePatch"]["donorProvenance"].update(sourcePts=999),
    ],
)
def test_source_patch_receipt_tampering_is_rejected(local_pipeline, change):
    p = local_pipeline
    add_source_patch_receipt(p)
    update_receipt(p, change)
    with pytest.raises(ValueError):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def test_source_patch_final_pixel_forgery_detected_by_replay(local_pipeline):
    p = local_pipeline
    add_source_patch_receipt(p)
    path = Path(p.a.import_local_clean).parent / "f_0002.png"
    pixels = np.array(Image.open(path))
    pixels[7, 7, 1] ^= 1
    Image.fromarray(pixels).save(path)

    def change(report):
        report["compositeChecks"][2].update(
            pngSha256=run_clip.file_sha256(path), cleanedRgbSha256=pixel_hash(pixels)
        )

    update_receipt(p, change)
    with pytest.raises(ValueError, match="replay/provenance"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()


def test_prepatch_lama_outside_forgery_rejected_even_with_updated_hashes(local_pipeline):
    p = local_pipeline
    add_source_patch_receipt(p)
    path = Path(p.a.import_local_clean).parent / "lama_0002.png"
    pixels = np.array(Image.open(path))
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels).save(path)

    def change(report):
        report["compositeChecks"][2].update(
            lamaPngSha256=run_clip.file_sha256(path), lamaRgbSha256=pixel_hash(pixels)
        )

    update_receipt(p, change)
    with pytest.raises(ValueError, match="pre-patch LaMa hashes/outside"):
        p.clean()
    assert not (p.ctx / "clean-selection.json").exists()
