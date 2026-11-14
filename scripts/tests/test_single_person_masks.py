"""Exact source observations reach untracked placement without inventing identities/background."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(Path(__file__).parent)]

import anchor_checks
import export_observation_confidence as exporter
import place_solve
import single_person_masks as spm
from test_silhouette_cropping import fit_inputs, mask


def archive_fixture(tmp_path, indices=(10, 40, 90), statuses=None):
    args, _ = fit_inputs(tmp_path, source_indices=indices)
    seq, cameras = args[0][0][4], args[2]
    seq["sourceSha256"] = "a" * 64
    statuses = statuses or ["known"] * len(indices)
    identity = dict(
        sourceSha256=seq["sourceSha256"],
        sourceIndices=list(indices),
        sourceImageSize=[100, 200],
        model=spm.MODEL,
        checkpointSha256=spm.CHECKPOINT_SHA256,
        extractorSha256="c" * 64,
        decoderSha256="d" * 64,
        scoreThreshold=spm.SCORE,
        maskThreshold=spm.PROBABILITY,
        selection=spm.SELECTION,
        dilationPixels=0,
    )
    frames, packed = [], []
    for i, (index, status) in enumerate(zip(indices, statuses)):
        observed = mask(top=40 + i * 10) if status == "known" else np.zeros((200, 100), bool)
        packed.append(np.packbits(observed, axis=-1))
        frames.append(
            dict(
                sourceIndex=index,
                status=status,
                rgbSha256="e" * 64,
                personScores={
                    "known": [0.99],
                    "empty": [0.99],
                    "missing": [],
                    "ambiguous": [0.99, 0.98],
                }[status],
                confidentPersonCount={"known": 1, "empty": 1, "missing": 0, "ambiguous": 2}[status],
            )
        )
    metadata = dict(schema=spm.SCHEMA, identity=identity, frames=frames)
    path = tmp_path / "single-person.npz"
    spm.write_archive(path, packed, metadata)
    return path, seq, cameras, metadata, packed, args


def test_single_person_fit_uses_exact_source_masks_without_people_manifest(tmp_path, monkeypatch):
    path, seq, cameras, metadata, packed, args = archive_fixture(
        tmp_path, (40, 90, 10, 999), ["known"] * 3 + ["ambiguous"]
    )
    args[1][0] = ("person", None, np.zeros(3), None)
    import silhouette_rows

    original = silhouette_rows.mask_rows
    tops = []

    def rows(*values, **kwargs):
        result = original(*values, **kwargs)
        tops.append(result["top"])
        return result

    monkeypatch.setattr(silhouette_rows, "mask_rows", rows)
    fitted = place_solve.silhouette_fit(*args, single_person_masks=path)["person"]
    assert fitted["sourceIndices"] == [40, 90, 10, 999]
    assert fitted["measuredFrames"] == 3
    assert tops == [40, 50, 60]
    assert len(fitted["offsetUnits"]) == 4
    assert (
        fitted["measurementProvenance"]["sampleMapping"]
        == "explicit sourceIndex (single-person archive)"
    )
    assert fitted["sizeScale"] != 1  # Real optimizer consumed rows, rather than silently skipping.


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("hash", "source hash"),
        ("indices", "source indices"),
        ("grid", "grid"),
        ("dilate", "undilated"),
        ("unknown-pixels", "invalid single-person observation"),
        ("missing-camera", "no camera"),
        ("duplicate-camera", "unique"),
    ],
)
def test_archive_rejects_mismatched_or_fabricated_observations(tmp_path, mutation, match):
    path, seq, cameras, metadata, packed, _ = archive_fixture(tmp_path)
    if mutation == "hash":
        seq["sourceSha256"] = "f" * 64
    elif mutation == "indices":
        seq["sourceIndices"] = [90, 40, 10]
    elif mutation == "grid":
        for camera in cameras:
            camera["source_image_size"] = [101, 200]
    elif mutation == "dilate":
        metadata["identity"]["dilationPixels"] = 1
    elif mutation == "unknown-pixels":
        metadata["frames"][0].update(
            status="ambiguous", confidentPersonCount=2, personScores=[0.99, 0.98]
        )
    elif mutation == "missing-camera":
        cameras.pop()
    else:
        cameras.append(copy.deepcopy(cameras[0]))
    spm.write_archive(path, packed, metadata)
    with pytest.raises(ValueError, match=match):
        spm.load_archive(path, seq, cameras)


def test_clean_archive_is_not_a_tight_single_person_observation(tmp_path):
    path, seq, cameras, *_ = archive_fixture(tmp_path)
    np.savez_compressed(
        path, masks=np.zeros((3, 200, 13), np.uint8), shape=[3, 200, 100], indices=[10, 40, 90]
    )
    with pytest.raises(ValueError, match="invalid single-person masks"):
        spm.load_archive(path, seq, cameras)


def test_unknown_masks_are_unavailable_and_cache_identity_must_match(tmp_path):
    path, seq, cameras, metadata, *_ = archive_fixture(
        tmp_path, statuses=["known", "missing", "ambiguous"]
    )
    masks, _ = spm.load_archive(path, seq, cameras)
    assert list(masks) == [10]
    changed = dict(metadata["identity"], extractorSha256="f" * 64)
    with pytest.raises(ValueError, match="cache identity changed"):
        spm.load_archive(path, seq, cameras, expected_identity=changed)


@pytest.mark.parametrize(
    "scores,labels,status,count",
    [
        ([0.99], [1], "known", 1),
        ([0.99, 0.98], [1, 1], "ambiguous", 2),
        ([0.6], [1], "missing", 0),
        ([0.99], [3], "missing", 0),
        ([0.7], [1], "missing", 0),
        ([0.99, 0.5], [1, 1], "known", 1),
    ],
)
def test_detection_never_unions_or_selects_an_ambiguous_person(scores, labels, status, count):
    masks = np.zeros((len(scores), 1, 4, 8))
    masks[0, 0, :, :3] = 1
    if len(scores) > 1:
        masks[1, 0, :, 5:] = 1
    selected, record = spm.select_instance(dict(labels=labels, scores=scores, masks=masks))
    assert record["status"] == status
    assert record["confidentPersonCount"] == count
    assert int(selected.sum()) == (12 if status == "known" else 0)


def test_exporter_blocks_unknown_source_coverage(tmp_path):
    path, seq, cameras, *_ = archive_fixture(tmp_path, statuses=["known", "ambiguous", "missing"])
    world = tmp_path / "world"
    (world / "person").mkdir(parents=True)
    (world / "person/sequence.json").write_text(json.dumps(seq))
    with pytest.raises(ValueError, match="no coverage for camera sourceIndex 40"):
        exporter.load_source_masks(path, world, cameras)


def test_exporter_accepts_all_known_source_masks(tmp_path):
    path, seq, cameras, *_ = archive_fixture(tmp_path)
    world = tmp_path / "world"
    (world / "person").mkdir(parents=True)
    (world / "person/sequence.json").write_text(json.dumps(seq))
    assert list(exporter.load_source_masks(path, world, cameras)) == [10, 40, 90]


def test_ruler_and_anchor_checks_read_the_single_person_archive(tmp_path):
    import world_ruler

    path, seq, cameras, *_ = archive_fixture(tmp_path)
    world = tmp_path / "world"
    (world / "person").mkdir(parents=True)
    (world / "person/sequence.json").write_text(json.dumps(seq))
    (world / "cameras.json").write_text(json.dumps({"cameras": cameras}))
    exact = world_ruler.person_masks_for("fixture", masks=path, world_dir=world, dilate=False)
    assert list(exact) == [10, 40, 90]
    assert all(exact[i].sum() == mask(top=40 + k * 10).sum() for k, i in enumerate([10, 40, 90]))
    dilated = world_ruler.person_masks_for("fixture", masks=path, world_dir=world)
    assert all(dilated[i].sum() > exact[i].sum() for i in exact)
    with pytest.raises(ValueError, match="need the packaged world directory"):
        world_ruler.person_masks_for("fixture", masks=path)


def test_explicit_person_has_viewer_id_and_legacy_presets_keep_ids():
    cfg = {"seq": ["public/worlds/legacy-4d/person/sequence.json"]}
    assert place_solve.load_tracks(cfg)[0][0] == "legacy-4d"
    assert place_solve.load_tracks(dict(cfg, personId="person"))[0][0] == "person"


def test_fitted_single_anchor_applies_person_placement_and_cannot_pass_unknowns(
    tmp_path, monkeypatch
):
    path, seq, cameras, *_ = archive_fixture(tmp_path, indices=tuple(range(8)))
    world = tmp_path / "world"
    (world / "person").mkdir(parents=True)
    seq["frames"] = ["unused.ply"] * 8
    (world / "person/sequence.json").write_text(json.dumps(seq))
    camera_path = world / "cameras.json"
    camera_path.write_text(json.dumps({"cameras": cameras}))
    placement = dict(
        registrationScale=2,
        pos0=[1, 2, 3],
        offsetUnits=[[0, 0, 0]] * 8,
        silhouetteFit=dict(masks=str(path), observationKind="single-person-source"),
        perPerson={
            "person": dict(constantUnits=[0, 0, 0], sizeScale=1.5, offsetUnits=[[2, 0, 0]] * 8)
        },
    )
    (world / "placement.json").write_text(json.dumps(placement))
    monkeypatch.setattr(
        place_solve,
        "read_ply",
        lambda _: {
            k: np.zeros(1) for k in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")
        },
    )
    import motion_silhouette

    def silhouette(vertex, xyz, sizes, camera, hw):
        np.testing.assert_allclose(xyz, [[1.5, 1, 1.5]])
        np.testing.assert_allclose(sizes, np.log(1.5))
        return mask(top=40 + camera["sourceIndex"] * 10)

    monkeypatch.setattr(motion_silhouette, "drawn_silhouette", silhouette)
    result = anchor_checks.check_silhouette("fixture", world_dir=world)
    assert result["ok"] is True
    assert result["n"] == 8
    assert result["iou"] == result["areaRatio"] == 1


def test_runner_prepares_masks_before_receipt_and_solve_and_tracks_source_changes(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import run_clip
    from test_placement_run_inputs import package_inputs

    public = tmp_path / "public"
    world = public / "worlds/example-4d"
    world.mkdir(parents=True)
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.name = "example"
    pipeline.ctx = tmp_path / "runs/example"
    pipeline.clip = tmp_path / "source.mp4"
    pipeline.clip.write_bytes(b"source")

    def confined_media_path(path):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve()), (
            "code reached media resolver"
        )
        return path

    pipeline.review_path = confined_media_path
    package_inputs(world, pipeline.ctx)
    pipeline.state = SimpleNamespace(data={"stages": {"_scale": {"scale0": 1.0}}}, record=Mock())
    pipeline.a = SimpleNamespace(allow_bad_size=False)
    pipeline.placement = Mock(return_value={"floor": -1.0})
    world_file = public / "world.spz"
    world_file.write_bytes(b"world")
    pipeline.world_spz = Mock(return_value=world_file)
    commands = []

    def run(command, *args, **kwargs):
        commands.append(command)
        if command[1] == "scripts/single_person_masks.py":
            (pipeline.ctx / "single-person-masks.npz").write_bytes(b"source observations")
        else:
            assert command[1] == "scripts/place_solve.py"
            assert "--single-person-masks" in command
            assert "--person" in command and "--people" not in command
            assert "--fail-on-size" in command
            (world / "placement.json").write_text(
                json.dumps(
                    dict(
                        samples=3,
                        metresPerWorldUnit=1,
                        surface={
                            "sizeCheck": dict(cameraHeightM=1.4, floorToCeilingM=3, pass_=True)
                        },
                        residualCm={"before": {}, "after": {}},
                        reprojectionCostPx=None,
                    )
                )
            )

    monkeypatch.setattr(run_clip, "run", run)
    pipeline.place_fit()
    assert len(commands) == 2
    receipt = pipeline.state.record.call_args.kwargs["inputReceipt"]
    assert str((pipeline.ctx / "single-person-masks.npz").resolve()) in receipt["files"]
    assert receipt["files"][str((ROOT / "scripts/single_person_masks.py").resolve())] == spm.sha256(
        ROOT / "scripts/single_person_masks.py"
    )
    decoder = ROOT / "worker/wander_worker/source_timing.py"
    assert receipt["files"][str(decoder.resolve())] == spm.sha256(decoder)
    assert receipt == pipeline.placement_inputs(world_file)
    pipeline.clip.write_bytes(b"different source")
    assert receipt != pipeline.placement_inputs(world_file)


def test_cached_checkpoint_missing_never_requests_download(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("download attempted")
    )
    with pytest.raises(FileNotFoundError, match="downloads disabled"):
        spm.cached_checkpoint()


def test_explicit_single_person_fit_fails_when_observations_are_unavailable(tmp_path):
    path, seq, cameras, metadata, packed, args = archive_fixture(
        tmp_path, statuses=["known", "ambiguous", "missing"]
    )
    args[1][0] = ("person", None, np.zeros(3), None)
    with pytest.raises(ValueError, match="at least 3 usable source silhouette"):
        place_solve.silhouette_fit(*args, single_person_masks=path)


def test_cached_checkpoint_rejects_wrong_bytes_without_loading_model(tmp_path, monkeypatch):
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    checkpoint = tmp_path / "checkpoints" / Path(MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url).name
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not the official model")
    with pytest.raises(ValueError, match="checkpoint hash does not match COCO_V1"):
        spm.cached_checkpoint()


def test_archive_rejects_confidence_count_disagreement(tmp_path):
    path, seq, cameras, metadata, packed, _ = archive_fixture(tmp_path)
    metadata["frames"][0]["personScores"] = [0.99, 0.98]
    spm.write_archive(path, packed, metadata)
    with pytest.raises(ValueError, match="score/count provenance at sourceIndex 10"):
        spm.load_archive(path, seq, cameras)
