"""Projected body bounds censor cropped edges without inventing silhouette measurements."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import anchor_checks
import place_solve
import silhouette_rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def write_masks(path, masks, height=200, width=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shape=[height, width, 1],
        **{f"t000_s{sample:03d}": np.packbits(mask) for sample, mask in masks.items()},
    )
    return path


def mask(top=50, bottom=140, height=200, width=100):
    result = np.zeros((height, width), bool)
    result[top:bottom, 20:80] = True
    return result


def test_body_crop_is_censored_even_when_mask_stops_short_of_image_boundary():
    observed = mask(top=20, bottom=190)
    baseline = silhouette_rows.mask_rows(observed, 1, 200)
    assert baseline["croppedBottom"] is False
    censored = silhouette_rows.mask_rows(observed, 1, 200, [10, 15, 90, 270])
    assert censored["croppedBottom"] is True
    assert censored["croppedTop"] is False
    assert censored["top"] == baseline["top"] == 20
    assert censored["bottom"] == baseline["bottom"] == 190
    assert censored["areaPx"] == baseline["areaPx"]
    assert silhouette_rows.EDGE_PX == 2
    assert silhouette_rows.mask_rows(observed, 1, 200, [10, -5, 90, 190])["croppedTop"] is True


def test_fully_visible_body_keeps_mask_measurements_and_boundary_rules():
    observed = mask(top=20, bottom=190)
    assert silhouette_rows.mask_rows(
        observed, 1, 200, [10, 15, 90, 195]
    ) == silhouette_rows.mask_rows(observed, 1, 200)
    observed[0:20] = True
    assert silhouette_rows.mask_rows(observed, 1, 200, [10, 15, 90, 195])["croppedTop"] is True


@pytest.mark.parametrize("box", [[0, 1, 2], [0, 1, 2, float("nan")], [0, 1, 0, 3], [0, 3, 2, 1]])
def test_invalid_projected_bounds_fail(box):
    with pytest.raises(ValueError, match="finite ordered"):
        silhouette_rows.mask_rows(mask(), 1, 200, box)


def test_motion_loader_keeps_original_sample_mapping_and_marks_absent_legacy(tmp_path):
    path = tmp_path / "tracks/masks.npz"
    assert silhouette_rows.load_track_motion(path, 0) is None
    frames = [
        {"sample": 8, "sourceIndex": 90, "box": [0, 10, 100, 270]},
        {"sample": 2, "sourceIndex": 10, "box": [0, 20, 100, 190]},
    ]
    write_json(path.parent / "track_00/motion.json", {"frames": frames})
    loaded = silhouette_rows.load_track_motion(path, 0)
    assert loaded[90]["sample"] == 8
    assert loaded[10]["sample"] == 2
    assert 40 not in loaded
    assert silhouette_rows.load_track_motion(path, 1) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-source",
        "duplicate-sample",
        "fractional-index",
        "missing-box",
        "invalid-box",
        "not-frames",
        "invalid-json",
    ],
)
def test_malformed_or_conflicting_motion_metadata_fails(tmp_path, mutation):
    path = tmp_path / "tracks/masks.npz"
    frames = [
        {"sample": 0, "sourceIndex": 7, "box": [0, 10, 100, 190]},
        {"sample": 1, "sourceIndex": 42, "box": [0, 10, 100, 190]},
    ]
    payload = {"frames": frames}
    if mutation == "duplicate-source":
        frames[1]["sourceIndex"] = 7
    elif mutation == "duplicate-sample":
        frames[1]["sample"] = 0
    elif mutation == "fractional-index":
        frames[0]["sample"] = 0.5
    elif mutation == "missing-box":
        del frames[0]["box"]
    elif mutation == "invalid-box":
        frames[0]["box"][-1] = float("inf")
    elif mutation == "not-frames":
        payload["frames"] = {}
    motion_path = write_json(path.parent / "track_00/motion.json", payload)
    if mutation == "invalid-json":
        motion_path.write_text("not JSON")
    with pytest.raises(ValueError, match="invalid motion metadata"):
        silhouette_rows.load_track_motion(path, 0)


def test_duplicate_track_directories_are_rejected(tmp_path):
    path = tmp_path / "masks.npz"
    for directory in ("track_00", "track_000"):
        write_json(path.parent / directory / "motion.json", {"frames": []})
    with pytest.raises(ValueError, match="ambiguous"):
        silhouette_rows.load_track_motion(path, 0)


def fit_inputs(tmp_path, source_indices=(10, 40, 90), manifest_indices=None):
    points = np.column_stack([np.zeros(200), np.linspace(-3, 1, 200), np.full(200, -2)])
    frames = [(points.copy(), points.copy()) for _ in source_indices]
    seq = {"sourceIndices": list(source_indices)}
    manifest = {
        "sourceIndices": list(manifest_indices or source_indices),
        "people": [{"id": "person", "track": 0}],
    }
    feats = [("person", frames, np.zeros(3), None, seq)]
    tracks = [(None, None, None, manifest)]
    cameras = [
        {
            "sourceIndex": source,
            "camera_to_world": np.eye(4).tolist(),
            "source_intrinsics": [[100, 0, 50], [0, 100, 100], [0, 0, 1]],
            "source_image_size": [100, 200],
        }
        for source in source_indices
    ]
    feet = [[{"x": 0, "y": -3, "z": -2, "res": 0} for _ in source_indices]]
    return (
        feats,
        tracks,
        cameras,
        feet,
        1,
        np.zeros(3),
        np.asarray([0, 1, 0]),
        tmp_path / "tracks/masks.npz",
        8,
        0.03,
        0.05,
        4,
        0.15,
    ), points


def test_real_fit_does_not_shrink_full_body_into_a_cropped_torso_mask(tmp_path):
    args, points = fit_inputs(tmp_path)
    path = args[7]
    write_masks(path, {sample: mask() for sample in range(3)})
    legacy = place_solve.silhouette_fit(*args)["person"]
    write_json(
        path.parent / "track_00/motion.json",
        {
            "frames": [
                {"sourceIndex": source, "sample": sample, "box": [20, 50, 80, 250]}
                for sample, source in enumerate((10, 40, 90))
            ]
        },
    )
    fitted = place_solve.silhouette_fit(*args)["person"]
    cam = args[2][0]
    R, C, K = silhouette_rows.cam_matrices(cam, 1, [0, 0, 0])
    legacy_rows = silhouette_rows.avatar_rows(
        points * legacy["sizeScale"] + legacy["offsetUnits"][0], R, C, K
    )
    corrected_rows = silhouette_rows.avatar_rows(
        points * fitted["sizeScale"] + fitted["offsetUnits"][0], R, C, K
    )
    assert legacy_rows["bottom"] < 160  # The uncensored torso mask used to pull the feet into view.
    assert corrected_rows["bottom"] > 200
    assert fitted["heightRatioMedian"] is None
    assert fitted["croppedBottomFrames"] == 3
    assert fitted["croppedTopFrames"] == 0
    assert fitted["measuredFrames"] == 3
    assert "censoring only" in fitted["measurementProvenance"]["cropBounds"]
    assert "legacy" in legacy["measurementProvenance"]["cropBounds"]


def test_fit_routes_reordered_gapped_sequence_to_original_mask_samples(tmp_path, monkeypatch):
    args, _ = fit_inputs(
        tmp_path, source_indices=(40, 90, 10, 999), manifest_indices=(90, 10, 40, 999)
    )
    path = args[7]
    write_masks(path, {0: mask(top=70), 2: mask(top=40), 5: mask(top=50), 8: mask(top=60)})
    write_json(
        path.parent / "track_00/motion.json",
        {
            "frames": [
                {"sourceIndex": source, "sample": sample, "box": [20, 10, 80, 250]}
                for source, sample in ((10, 2), (40, 5), (90, 8))
            ]
        },
    )
    original_rows = silhouette_rows.mask_rows
    measured_tops = []

    def rows(*values, **kwargs):
        result = original_rows(*values, **kwargs)
        measured_tops.append(result["top"])
        return result

    monkeypatch.setattr(silhouette_rows, "mask_rows", rows)
    fitted = place_solve.silhouette_fit(*args)["person"]
    assert measured_tops == [50, 60, 40]
    assert (
        fitted["measuredFrames"] == 3
    )  # Source 999 stays missing despite a stale mask at sample 0.
    assert fitted["sourceIndices"] == [40, 90, 10, 999]
    assert len(fitted["offsetUnits"]) == 4


def test_fit_does_not_guess_masks_for_missing_motion_records(tmp_path):
    args, _ = fit_inputs(tmp_path)
    path = args[7]
    write_masks(path, {sample: mask() for sample in range(3)})
    write_json(path.parent / "track_00/motion.json", {"frames": []})
    assert place_solve.silhouette_fit(*args) == {}


def test_fitted_anchor_gate_uses_original_samples_after_subsetting(tmp_path, monkeypatch):
    import motion_silhouette

    path = tmp_path / "tracks/masks.npz"
    samples = [2, 5, 8, 11, 14, 17, 20, 23]
    sources = [80, 20, 100, 40, 120, 60, 140, 160]
    all_masks = {sample: np.zeros((40, 40), bool) for sample in range(8)}
    all_masks.update({sample: np.ones((40, 40), bool) for sample in samples})
    write_masks(path, all_masks, 40, 40)
    write_json(
        path.parent / "track_00/motion.json",
        {
            "frames": [
                {"sourceIndex": source, "sample": sample, "box": [0, 1, 40, 39]}
                for source, sample in zip(sources, samples)
            ]
        },
    )
    wd = tmp_path / "world"
    write_json(
        wd / "people.json",
        {
            "sourceIndices": sources,
            "people": [{"id": "person", "track": 0, "sequence": "person/sequence.json"}],
        },
    )
    write_json(
        wd / "person/sequence.json", {"sourceIndices": sources, "frames": ["unused.ply"] * 8}
    )
    write_json(
        wd / "cameras.json",
        {"cameras": [{"sourceIndex": source, "source_image_size": [40, 40]} for source in sources]},
    )
    write_json(
        wd / "placement.json",
        {
            "registrationScale": 1,
            "silhouetteFit": {"masks": str(path)},
            "pos0": [0, 0, 0],
            "offsetUnits": [[0, 0, 0]] * 8,
            "perPerson": {"person": {"constantUnits": [0, 0, 0]}},
        },
    )
    monkeypatch.setattr(
        place_solve,
        "read_ply",
        lambda _: {
            key: np.zeros(1) for key in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")
        },
    )
    monkeypatch.setattr(
        motion_silhouette,
        "splat_silhouette",
        lambda xyz, opacity, sizes, camera, hw, subsample: np.ones(hw, bool),
    )
    result = anchor_checks.check_silhouette("fixture", world_dir=wd)
    assert result["ok"] is True
    assert result["n"] == 8
    assert result["iou"] == result["areaRatio"] == 1


def test_projected_bounds_remain_in_source_pixels_for_downsampled_masks():
    observed = np.zeros((100, 50), bool)
    observed[10:95, 10:40] = True
    measured = silhouette_rows.mask_rows(observed, 0.5, 200, [20, 20, 80, 200])
    assert measured["top"] == 20
    assert measured["bottom"] == 190
    assert measured["croppedBottom"] is True


def test_real_fit_is_unchanged_when_projected_body_bounds_are_fully_visible(tmp_path):
    args, _ = fit_inputs(tmp_path)
    path = args[7]
    write_masks(path, {sample: mask() for sample in range(3)})
    legacy = place_solve.silhouette_fit(*args)["person"]
    write_json(
        path.parent / "track_00/motion.json",
        {
            "frames": [
                {"sourceIndex": source, "sample": sample, "box": [20, 50, 80, 190]}
                for sample, source in enumerate((10, 40, 90))
            ]
        },
    )
    fitted = place_solve.silhouette_fit(*args)["person"]
    np.testing.assert_allclose(fitted["offsetUnits"], legacy["offsetUnits"], atol=1e-10)
    assert fitted["sizeScale"] == pytest.approx(legacy["sizeScale"], abs=1e-10)
    assert fitted["heightRatioMedian"] == pytest.approx(legacy["heightRatioMedian"], abs=1e-10)
    assert fitted["croppedTopFrames"] == fitted["croppedBottomFrames"] == 0
