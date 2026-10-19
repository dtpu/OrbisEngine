"""No-spend contracts for cleaning on a complete, exact tracked source grid."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_clip
import tracks_to_clean_masks


def track_fixture(root, clip, samples=5, width=11, height=4):
    clip.write_bytes(b"source pixels")
    source_sha = hashlib.sha256(clip.read_bytes()).hexdigest()
    indices = list(range(0, 2 * samples, 2))
    provenance = {
        "sourceSha256": source_sha,
        "sourceTimeBase": "1/240",
        "frames": [
            {"sourceIndex": index, "sourcePts": index * 10, "sourceRelativeTimeSeconds": index / 24}
            for index in indices
        ],
    }
    doc = {
        "sourceSha256": source_sha,
        "width": width,
        "height": height,
        "fps": 12.0,
        "samples": samples,
        "sourceIndices": indices,
        "timestamps": [index / 24 for index in indices],
        "trackCount": 2,
        "tracks": [{"track": track, "samples": list(range(samples))} for track in range(2)],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "tracks.json").write_text(json.dumps(doc))
    payload = {"shape": np.array([height, width, 1.0])}
    for track in range(2):
        for sample in range(samples):
            mask = np.zeros((height, width), dtype=bool)
            mask[0, track + sample] = True
            payload[f"t{track}_s{sample:03d}"] = np.packbits(mask)
    np.savez_compressed(root / "masks.npz", **payload)
    return doc, payload, provenance


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    clip, root, out = tmp_path / "clip.mp4", tmp_path / "tracks", tmp_path / "clean-masks.npz"
    doc, payload, provenance = track_fixture(root, clip)
    monkeypatch.setattr(
        tracks_to_clean_masks, "resample_source", lambda *a, **kw: (b"", provenance)
    )
    monkeypatch.setattr(
        tracks_to_clean_masks, "select_source_frames", lambda *a, **kw: (b"", provenance)
    )
    return root, clip, out, doc, payload, provenance


def build(tracking, **options):
    root, clip, out, *_ = tracking
    return tracks_to_clean_masks.build_clean_masks(
        root,
        clip=clip,
        out=out,
        width=11,
        height=4,
        fps=12,
        dilate=options.get("dilate", 0),
        bottom_extra=options.get("bottom_extra", 0),
    )


def test_unions_all_tracks_without_changing_source_grid(tracking):
    _, _, out, doc, _, provenance = tracking
    report = build(tracking)
    assert report["framesCovered"] == 5 and report["samplesWithoutMask"] == 0
    with np.load(out, allow_pickle=False) as archive:
        assert archive["source_indices"].tolist() == doc["sourceIndices"]
        assert archive["source_pts"].tolist() == [
            frame["sourcePts"] for frame in provenance["frames"]
        ]
        assert archive["source_time_base"].item() == "1/240"
        assert archive["source_sha256"].item() == doc["sourceSha256"]
        masks = np.unpackbits(archive["masks"], axis=-1)[:, :, :11]
        for sample, mask in enumerate(masks):
            assert mask.sum() == 2
            assert mask[0, sample] and mask[0, sample + 1]


@pytest.mark.parametrize(
    "change", ["source", "dimensions", "fps", "count", "timestamp", "coverage", "empty-tracks"]
)
def test_rejects_incompatible_track_metadata_without_writing_archive(tracking, change):
    root, _, out, doc, *_ = tracking
    if change == "source":
        doc["sourceSha256"] = "0" * 64
    elif change == "dimensions":
        doc["width"] = 12
    elif change == "fps":
        doc["fps"] = 24
    elif change == "count":
        doc["samples"] = 4
    elif change == "timestamp":
        doc["timestamps"][2] += 0.01
    elif change == "coverage":
        for track in doc["tracks"]:
            track["samples"].pop()
    else:
        doc["tracks"], doc["trackCount"] = [], 0
    (root / "tracks.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        build(tracking)
    assert not out.exists()


@pytest.mark.parametrize(
    "change",
    [
        "missing-file",
        "missing-mask",
        "empty",
        "wrong-shape",
        "wrong-length",
        "wrong-dtype",
        "undeclared",
    ],
)
def test_rejects_missing_or_incomplete_instance_masks(tracking, change):
    root, _, out, _, payload, _ = tracking
    if change == "missing-mask":
        payload.pop("t0_s002")
    elif change == "empty":
        payload["t0_s002"][:] = 0
    elif change == "wrong-shape":
        payload["shape"] = np.array([4, 12, 1])
    elif change == "wrong-length":
        payload["t0_s002"] = np.array([255], np.uint8)
    elif change == "wrong-dtype":
        payload["t0_s002"] = payload["t0_s002"].astype(float)
    elif change == "undeclared":
        payload["t2_s002"] = payload["t0_s002"]
    np.savez_compressed(root / "masks.npz", **payload)
    if change == "missing-file":
        (root / "masks.npz").unlink()
    with pytest.raises(ValueError):
        build(tracking)
    assert not out.exists()


def test_dilation_and_downward_extension_are_applied_once(tracking):
    _, _, out, *_ = tracking
    build(tracking, dilate=1, bottom_extra=1)
    with np.load(out) as archive:
        mask = np.unpackbits(archive["masks"], axis=-1)[0, :, :11]
    assert mask[:3, :3].all()
    assert not mask[3].any()
    assert not mask[:, 4:].any()


def test_exact_tracker_grid_is_preserved_when_fps_filter_selects_other_frames(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for decoded source fixture")
    clip, root, out = tmp_path / "clip.mp4", tmp_path / "tracks", tmp_path / "clean.npz"
    _, payload, _ = track_fixture(root, clip, samples=2, width=16, height=12)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=16x12:rate=6:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    _, sampled = tracks_to_clean_masks.resample_source(clip, 2)
    indices = [0, 3]
    _, exact = tracks_to_clean_masks.select_source_frames(clip, 2, source_indices=indices)
    assert [frame["sourceIndex"] for frame in sampled["frames"]] != indices
    doc = {
        "sourceSha256": exact["sourceSha256"],
        "width": 16,
        "height": 12,
        "fps": 2,
        "samples": 2,
        "sourceIndices": indices,
        "timestamps": [frame["sourceRelativeTimeSeconds"] for frame in exact["frames"]],
        "trackCount": 2,
        "tracks": [{"track": i, "samples": [0, 1]} for i in range(2)],
    }
    (root / "tracks.json").write_text(json.dumps(doc))
    tracks_to_clean_masks.build_clean_masks(
        root, clip=clip, out=out, width=16, height=12, fps=2, dilate=0, bottom_extra=0
    )
    with np.load(out) as archive:
        assert archive["source_indices"].tolist() == indices
        assert archive["source_pts"].tolist() == [frame["sourcePts"] for frame in exact["frames"]]


@pytest.fixture
def runner(tracking, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from worker.wander_worker import source_timing

    root, clip, _, _, _, provenance = tracking
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.ctx = root.parent
    pipeline.clip = clip
    pipeline.clean_mp4 = root.parent / "clean.mp4"
    pipeline.multi = True
    pipeline.info = {"width": 11, "height": 4}
    pipeline.a = SimpleNamespace(
        clean_masks=None,
        moved_mask=False,
        fps=12,
        dilate=0,
        bottom_extra=0,
        lama_px=960,
    )
    pipeline.paid_run = Mock()
    monkeypatch.setattr(run_clip, "clean_sample_count", lambda *a: 5)
    monkeypatch.setattr(source_timing, "select_source_frames", lambda *a, **kw: (b"", provenance))
    return pipeline


def test_default_tracked_clean_forwards_generated_complete_masks(runner):
    runner.clean()
    runner.paid_run.assert_called_once()
    command = runner.paid_run.call_args.args[1]
    masks = Path(command[command.index("--masks-in") + 1])
    assert masks == runner.ctx / "track-clean-masks.npz"
    with np.load(masks) as archive:
        assert archive["source_indices"].tolist() == [0, 2, 4, 6, 8]
    report = json.loads((runner.ctx / "track-clean-masks.json").read_text())
    assert report["framesCovered"] == 5


@pytest.mark.parametrize("invalid", ["absent", "source", "mask", "timestamp"])
def test_invalid_tracking_stops_before_paid_clean_without_semantic_fallback(
    runner, tracking, invalid
):
    root, _, _, doc, payload, _ = tracking
    if invalid == "absent":
        (root / "masks.npz").unlink()
    elif invalid == "mask":
        payload.pop("t1_s001")
        np.savez_compressed(root / "masks.npz", **payload)
    else:
        if invalid == "source":
            doc["sourceSha256"] = "0" * 64
        else:
            doc["timestamps"][1] = 99
        (root / "tracks.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        runner.clean()
    runner.paid_run.assert_not_called()


@pytest.mark.parametrize("explicit", ["reviewed", "moved", "single"])
def test_explicit_mask_choice_and_single_person_preserve_existing_path(runner, explicit):
    if explicit == "reviewed":
        reviewed = runner.ctx / "reviewed.npz"
        np.savez_compressed(reviewed, masks=np.zeros((5, 4, 2), np.uint8))
        runner.a.clean_masks = str(reviewed)
    elif explicit == "moved":
        runner.a.moved_mask = True
    else:
        runner.multi = False
    with patch.object(tracks_to_clean_masks, "build_clean_masks") as build_masks:
        runner.clean()
    build_masks.assert_not_called()
    command = runner.paid_run.call_args.args[1]
    assert ("--masks-in" in command) == (explicit == "reviewed")
    assert ("--moved-mask" in command) == (explicit == "moved")


@pytest.mark.parametrize("corrupt", ["hash", "pts", "timebase", "partial", "shape-dtype"])
def test_explicit_grid_metadata_is_validated_before_paid_submission(runner, tracking, corrupt):
    build(tracking)
    path = tracking[2]
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    if corrupt == "hash":
        payload["source_sha256"] = np.array("0" * 64)
    elif corrupt == "pts":
        payload["source_pts"][0] += 1
    elif corrupt == "timebase":
        payload["source_time_base"] = np.array("1/30")
    elif corrupt == "partial":
        payload.pop("source_indices")
    else:
        payload["shape"] = payload["shape"].astype(float)
    np.savez_compressed(path, **payload)
    runner.a.clean_masks = str(path)
    with pytest.raises(ValueError):
        runner.clean()
    runner.paid_run.assert_not_called()


def test_exact_source_masks_rejected_before_paid_multi_image_clean(runner, tracking):
    build(tracking)
    runner.a.clean_masks = str(tracking[2])
    runner.marble = "multi"
    with pytest.raises(ValueError, match="not supported by --marble multi"):
        runner.clean_flags()
    runner.paid_run.assert_not_called()
