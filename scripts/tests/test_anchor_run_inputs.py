"""Anchor diagnostics must measure the candidate whose paths the runner supplied."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import anchor_checks
import world_ruler


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def write_person(path):
    names = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    path.write_bytes(header.encode() + np.array([[0, 0, -2, 2, -5, -5, -5]], dtype="<f4").tobytes())


@pytest.fixture
def inputs(tmp_path):
    wd = tmp_path / "custom-public/worlds/gym-4d"
    wd.mkdir(parents=True)
    camera = {
        "camera_to_world": np.eye(4).tolist(),
        "source_intrinsics": [[40, 0, 20], [0, 40, 20], [0, 0, 1]],
        "source_image_size": [40, 40],
        "tag": "current",
    }
    indices = list(range(41, 49))
    cameras = write_json(
        tmp_path / "current-cameras.json",
        {"cameras": [dict(camera, sourceIndex=i) for i in indices]},
    )
    write_json(wd / "cameras.json", {"cameras": []})
    write_json(
        wd / "person/sequence.json",
        {"sourceIndices": indices, "frames": ["frame.ply"] * 8, "sourceClip": "/stale/source.mp4"},
    )
    write_person(wd / "person/frame.ply")
    place = {
        "registrationScale": 3,
        "metresPerWorldUnit": 0.7,
        "surface": {
            "sizeCheck": {
                "cameraHeightTrace": {
                    "verdict": "flat in band",
                    "medianM": 1.4,
                    "endToEndSlopeM": 0.01,
                }
            }
        },
        "noiseFloor": {"ok": True, "refusals": [], "contactCm": 2},
    }
    placement = write_json(tmp_path / "reviewed-placement.json", place)
    write_json(wd / "placement.json", {"registrationScale": 999})
    world = tmp_path / "retained-world.spz"
    world.write_bytes(b"retained world; geometry is not read by no-metric-depth")
    video = tmp_path / "exact-source.mp4"
    video.write_bytes(b"local-source")
    masks = tmp_path / "tight-masks.npz"
    ground = np.ones((8, 40, 40), bool)
    np.savez_compressed(
        masks, shape=ground.shape, masks=np.packbits(ground, axis=-1), indices=indices
    )
    return dict(
        world_dir=wd, world=world, cameras=cameras, video=video, masks=masks, placement=placement
    )


def test_ruler_subprocess_uses_exact_candidate_and_placement(inputs, tmp_path):
    report = tmp_path / "ruler.json"
    result = anchor_checks.check_ruler("gym", ["--no-metric-depth"], 0.15, report=report, **inputs)
    assert result["ok"] is False  # No independent ruler was measured; routing is not acceptance.
    doc = json.loads(report.read_text())
    assert doc["world"] == str(inputs["world"])
    assert doc["scale0"] == 3
    assert doc["mpuPerson"] == 0.7
    assert doc["inputs"] == {
        "worldDir": str(inputs["world_dir"]),
        "cameras": str(inputs["cameras"]),
        "video": str(inputs["video"]),
        "placement": str(inputs["placement"]),
        "masks": str(inputs["masks"]),
    }


def test_failed_ruler_never_reuses_previous_passing_report(tmp_path, monkeypatch):
    report = tmp_path / "old-ruler.json"
    report.write_text('{"verdict":{"ok":true},"anchors":{}}')
    monkeypatch.setattr(
        anchor_checks.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=2, stderr="missing current input"),
    )
    result = anchor_checks.check_ruler("gym", [], 0.15, report=report)
    assert result["ok"] is False
    assert "missing current input" in result["why"][0]


def test_single_person_uses_exact_cameras_masks_and_sequence(inputs, monkeypatch):
    import motion_silhouette

    seen = []

    def silhouette(xyz, opacity, sizes, camera, hw, subsample):
        seen.append(camera)
        return np.ones(hw, bool)

    monkeypatch.setattr(motion_silhouette, "splat_silhouette", silhouette)
    result = anchor_checks.check_silhouette(
        "gym", **{k: v for k, v in inputs.items() if k != "world"}
    )
    assert result["ok"] is True
    assert result["n"] == 8
    assert result["iou"] == result["areaRatio"] == 1
    assert [camera["sourceIndex"] for camera in seen] == list(range(41, 49))
    assert all(camera["tag"] == "current" for camera in seen)


def test_single_person_reads_explicit_video_instead_of_sequence_source(
    inputs, tmp_path, monkeypatch
):
    import motion_audit
    import motion_silhouette
    import person_masks_tight

    seen = []
    monkeypatch.setattr(anchor_checks, "ROOT", tmp_path / "isolated-cache")
    monkeypatch.setattr(
        motion_audit,
        "seed_boxes",
        lambda wd, cams, seq, hw: {c["sourceIndex"]: [0, 0, 40, 40] for c in cams},
    )

    def frames(video, indices):
        seen.append(video)
        return {i: np.zeros((40, 40, 3), np.uint8) for i in indices}

    monkeypatch.setattr(person_masks_tight, "read_frames", frames)
    monkeypatch.setattr(
        person_masks_tight, "masks_for", lambda frames, seeds: np.ones(frames.shape[:3], bool)
    )
    monkeypatch.setattr(
        motion_silhouette,
        "splat_silhouette",
        lambda xyz, opacity, sizes, camera, hw, subsample: np.ones(hw, bool),
    )
    kwargs = {k: v for k, v in inputs.items() if k not in {"world", "masks"}}
    assert anchor_checks.check_silhouette("gym", **kwargs)["ok"] is True
    assert anchor_checks.check_silhouette("gym", **kwargs)["ok"] is True
    assert seen == [str(inputs["video"])]  # Reuse only this source/camera/sequence cache.
    changed = tmp_path / "another-source.mp4"
    changed.write_bytes(b"new source")
    kwargs["video"] = changed
    assert anchor_checks.check_silhouette("gym", **kwargs)["ok"] is True
    assert seen == [str(inputs["video"]), str(changed)]


def test_camera_trace_and_noise_read_explicit_placement(inputs, tmp_path, monkeypatch):
    report = tmp_path / "anchors.json"
    argv = [
        "anchor_checks.py",
        "--clip",
        "gym",
        "--skip",
        "ruler,silhouette,obsMask",
        "--json-out",
        str(report),
    ]
    for key, value in inputs.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    monkeypatch.setattr(sys, "argv", argv)
    assert anchor_checks.main() == 0
    checks = json.loads(report.read_text())["checks"]
    assert checks["cameraTrace"]["medianM"] == 1.4
    assert checks["noiseFloor"]["contactCm"] == 2


@pytest.mark.parametrize("module", [anchor_checks, world_ruler])
@pytest.mark.parametrize("option", ["world-dir", "world", "cameras", "video", "masks", "placement"])
def test_invalid_explicit_path_fails_before_any_default_can_be_used(
    module, option, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys, "argv", [module.__name__, "--clip", "gym", "--" + option, str(tmp_path / "missing")]
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_explicit_world_directory_does_not_fall_back_to_raw_cameras(inputs):
    with pytest.raises(ValueError, match="no packaged cameras"):
        world_ruler.resolve_paths(
            "gym", world=inputs["world"], world_dir=inputs["world_dir"] / "person"
        )


def test_historical_path_defaults_remain_unchanged():
    world, directory, cameras, video = world_ruler.resolve_paths("gym")
    assert world == ROOT / "public/marble-gym-clean2.spz"
    assert directory == ROOT / "public/worlds/gym-4d"
    assert cameras == directory / "cameras.json"
    assert video == ROOT / "public/clips/gym.mp4"


def test_tracking_masks_use_packaged_source_indices_without_historical_cache(inputs, tmp_path):
    tracked = tmp_path / "current-tracks/masks.npz"
    tracked.parent.mkdir()
    np.savez_compressed(tracked, shape=[40, 40, 1], t000_s000=np.packbits(np.ones((40, 40), bool)))
    write_json(inputs["world_dir"] / "people.json", {"sourceIndices": [41]})
    masks = world_ruler.person_masks_for(
        "gym", masks=tracked, world_dir=inputs["world_dir"], dilate=False
    )
    assert list(masks) == [41]
    assert masks[41].all()
    assert world_ruler.person_masks_for("gym", world_dir=inputs["world_dir"]) is None
    with pytest.raises(ValueError, match="not a file"):
        world_ruler.person_masks_for("gym", masks=tracked.parent / "absent.npz")


def test_observation_mask_uses_exact_world_placement_camera_and_masks(
    inputs, tmp_path, monkeypatch
):
    import export_observation_confidence as observation

    camera = {
        "sourceIndex": 7,
        "source_image_size": [8, 8],
        "camera_to_world": np.eye(4).tolist(),
        "source_intrinsics": [[4, 0, 0], [0, 4, 0], [0, 0, 1]],
    }
    write_json(inputs["cameras"], {"cameras": [camera]})
    write_json(inputs["placement"], {"registrationScale": 1})
    mask = np.zeros((1, 8, 8), bool)
    mask[0, :4, :4] = True
    np.savez_compressed(
        inputs["masks"], shape=mask.shape, masks=np.packbits(mask, axis=-1), indices=[7]
    )
    sidecar = tmp_path / "arbitrary-sidecar.bin"
    sidecar.write_bytes(bytes([0, 255]))
    seen = []

    def world(path):
        seen.append(path)
        return np.asarray([[0, 0, 1], [1, 1, 1]], dtype=np.float32)

    monkeypatch.setattr(observation, "read_spz_xyz", world)
    result = anchor_checks.check_obs_mask(
        "gym", sidecar, **{k: v for k, v in inputs.items() if k != "video"}
    )
    assert seen == [inputs["world"]]
    assert result["ok"] is True
    assert result["inMask"] == 0
    assert result["outMask"] == 1


def test_tracked_silhouette_uses_supplied_fit_and_masks_instead_of_embedded_paths(
    inputs, tmp_path, monkeypatch
):
    import motion_silhouette

    indices = list(range(41, 49))
    tracked = tmp_path / "current-tracks/masks.npz"
    tracked.parent.mkdir()
    np.savez_compressed(
        tracked,
        shape=[40, 40, 1],
        **{f"t000_s{i:03d}": np.packbits(np.ones((40, 40), bool)) for i in range(8)},
    )
    write_json(
        inputs["world_dir"] / "people.json",
        {
            "sourceIndices": indices,
            "people": [{"id": "actor", "track": 0, "sequence": "person/sequence.json"}],
        },
    )
    place = json.loads(inputs["placement"].read_text())
    place.update(
        silhouetteFit={"masks": "/stale/tracks/masks.npz"},
        pos0=[3, 0, 0],
        offsetUnits=[[0, 0, 0]] * 8,
        perPerson={"actor": {"sizeScale": 2, "constantUnits": [0, 0, 0]}},
    )
    write_json(inputs["placement"], place)
    seen = []

    def silhouette(xyz, opacity, sizes, camera, hw, subsample):
        seen.append((xyz, camera))
        return np.ones(hw, bool)

    monkeypatch.setattr(motion_silhouette, "splat_silhouette", silhouette)
    kwargs = {k: v for k, v in inputs.items() if k != "world"}
    kwargs["masks"] = tracked
    result = anchor_checks.check_silhouette("gym", **kwargs)
    assert result["ok"] is True
    assert result["inSample"] is True
    assert result["n"] == 8
    assert [camera["sourceIndex"] for _, camera in seen] == indices
    for xyz, _ in seen:
        np.testing.assert_allclose(xyz, [[1, 0, -4]])  # Explicit scale/pos0 and per-person fit.


def test_tracking_masks_reject_conflicting_source_frame_mappings(inputs, tmp_path):
    tracked = tmp_path / "tracks/masks.npz"
    tracked.parent.mkdir()
    np.savez_compressed(tracked, shape=[40, 40, 1], t000_s000=np.packbits(np.ones((40, 40), bool)))
    write_json(
        tracked.parent / "track_000/motion.json",
        {"frames": [{"sample": 0, "sourceIndex": 41, "box": [1, 1, 39, 39]}]},
    )
    write_json(inputs["world_dir"] / "people.json", {"sourceIndices": [999]})
    with pytest.raises(ValueError, match="source indices disagree"):
        world_ruler.person_masks_for("gym", masks=tracked, world_dir=inputs["world_dir"])
