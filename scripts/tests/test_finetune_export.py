"""Exact-source exclusion mask admission for the local finetune exporter."""

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
import finetune_export as export


@pytest.fixture
def inputs(tmp_path):
    clip = tmp_path / "source.mkv"
    width, height = 17, 12
    pixels = b"".join(bytes((i * 55, 150, 200)) * width * height for i in range(4))
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "5",
            "-i",
            "-",
            "-c:v",
            "ffv1",
            str(clip),
        ],
        input=pixels,
        check=True,
    )
    indices = [0, 2]
    _, provenance = export.select_source_frames(clip, source_indices=indices)
    timestamps = [frame["sourceRelativeTimeSeconds"] for frame in provenance["frames"]]
    masks = np.zeros((2, height, width), np.uint8)
    masks[1, 5, 9] = 1
    archive = {
        "shape": masks.shape,
        "masks": np.packbits(masks, axis=2),
        "source_indices": indices,
        "source_pts": [frame["sourcePts"] for frame in provenance["frames"]],
        "source_sha256": provenance["sourceSha256"],
        "source_time_base": provenance["sourceTimeBase"],
    }
    cameras = {
        "cameras": [
            {
                "sourceIndex": index,
                "time": time,
                "source_image_size": [width, height],
                "camera_to_world": np.eye(4).tolist(),
                "source_intrinsics": np.eye(3).tolist(),
            }
            for index, time in zip(indices, timestamps, strict=True)
        ]
    }
    sequence = {
        "sourceSha256": provenance["sourceSha256"],
        "sourceIndices": indices,
        "timestamps": timestamps,
        "fps": 5.0,
    }
    world = tmp_path / "world.spz"
    world.write_bytes(b"world fixture")
    anchors = tmp_path / "anchors.npz"
    np.savez(anchors, valid=np.ones((2, height, width), dtype=bool))
    out = tmp_path / "export"
    argv = [
        "--clip",
        str(clip),
        "--cameras",
        str(tmp_path / "cameras.json"),
        "--masks-npz",
        str(tmp_path / "masks.npz"),
        "--world",
        str(world),
        "--anchors",
        str(anchors),
        "--anchor-samples",
        "0,1",
        "--scale0",
        "1",
        "--size",
        str(width),
        str(height),
        "--depth-res",
        "2",
        "2",
        "--out",
        str(out),
    ]
    return tmp_path, argv, archive, cameras, sequence


def write_inputs(path, archive, cameras, sequence):
    np.savez(path / "masks.npz", **archive)
    (path / "cameras.json").write_text(json.dumps(cameras))
    if sequence is not None:
        (path / "sequence.json").write_text(json.dumps(sequence))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("same-count-grid", "indices do not match camera"),
        ("hash", "Masks source hash mismatch"),
        ("pts", "Masks source PTS mismatch"),
        ("timebase", "Masks source timebase mismatch"),
        ("padding", "nonzero padding"),
        ("incomplete", "exact source-grid shape"),
        ("camera-size", "Camera dimensions"),
        ("camera-intrinsics", "positive focal lengths"),
        ("camera-time", "Camera time does not match actual source PTS"),
        ("camera-binding", "Camera source binding requires"),
        ("camera-hash", "Camera source hash mismatch"),
        ("sequence-grid", "Camera sequence source indices"),
        ("sequence-time", "Camera sequence timestamps"),
        ("anchor-sequence", "Anchor samples or source binding"),
        ("anchor-sequence-order", "Anchor samples or source binding"),
        ("anchor-duplicate", "unique, ordered camera samples"),
        ("missing-rgb", "Decoded RGB frame count"),
    ],
)
def test_rejects_before_creating_output_or_loading_world(inputs, monkeypatch, mutation, error):
    path, argv, archive, cameras, sequence = inputs
    if mutation == "same-count-grid":
        archive.update(source_indices=[1, 3], source_pts=[200, 600])
    elif mutation == "hash":
        archive["source_sha256"] = "0" * 64
    elif mutation == "pts":
        archive["source_pts"][1] += 1
    elif mutation == "timebase":
        archive["source_time_base"] = "1/90000"
    elif mutation == "padding":
        archive["masks"][0, 0, -1] = 1
    elif mutation == "incomplete":
        archive["masks"] = archive["masks"][:1]
    elif mutation == "camera-size":
        cameras["cameras"][1]["source_image_size"] = [16, 12]
    elif mutation == "camera-intrinsics":
        cameras["cameras"][1]["source_intrinsics"][0][0] = 0
    elif mutation == "camera-time":
        cameras["cameras"][1]["time"] = 0.41
        sequence["timestamps"][1] = 0.41
    elif mutation == "camera-binding":
        sequence = None
    elif mutation == "camera-hash":
        sequence["sourceSha256"] = "0" * 64
    elif mutation == "sequence-grid":
        sequence["sourceIndices"] = [1, 3]
    elif mutation == "sequence-time":
        sequence["timestamps"][1] = 0.41
    elif mutation == "anchor-sequence":
        sequence["anchors"] = [0, 2]
    elif mutation == "anchor-sequence-order":
        sequence["anchors"] = [1, 0]
    elif mutation == "anchor-duplicate":
        argv[argv.index("--anchor-samples") + 1] = "0,0"
    elif mutation == "missing-rgb":
        decode = export.select_source_frames

        def truncated(*args, **kwargs):
            raw, provenance = decode(*args, **kwargs)
            return raw[: -17 * 12 * 3], provenance

        monkeypatch.setattr(export, "select_source_frames", truncated)

    write_inputs(path, archive, cameras, sequence)
    world = Mock(side_effect=AssertionError("world must not load during mask preflight"))
    monkeypatch.setattr(export, "read_spz", world)
    with pytest.raises(ValueError, match=error):
        export.main(argv)
    world.assert_not_called()
    assert not (path / "export").exists()


def stub_world_geometry(monkeypatch):
    monkeypatch.setattr(
        export,
        "read_spz",
        lambda _: {
            "xyz": np.zeros((1, 3)),
            "n": 1,
            "quat": np.array([[0, 0, 0, 1]]),
            "log_scale": np.zeros((1, 3)),
            "alpha": np.array([0.5]),
            "f_dc": np.zeros((1, 3)),
            "fb": 0,
        },
    )
    scene = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)

    def cloud(*args, exclusion_masks, exclusion_report):
        exclusion_report.extend(
            {
                "sample": sample,
                "candidatePoints": 2,
                "excludedPoints": int(mask.any()),
                "retainedPoints": 2 - int(mask.any()),
            }
            for sample, mask in exclusion_masks.items()
        )
        return scene

    build_cloud = Mock(side_effect=cloud)
    monkeypatch.setattr(export, "static_cloud_from_anchors", build_cloud)
    monkeypatch.setattr(export, "levelled_origin", lambda *_: (np.eye(4), None))
    monkeypatch.setattr(
        export, "zbuffer_depth", lambda *_: (np.ones((2, 2)), np.ones((2, 2), dtype=bool))
    )
    return scene, build_cloud


@pytest.mark.parametrize("direct_binding", [False, True])
def test_valid_timing_exports_exact_frames_and_pins_inputs(inputs, monkeypatch, direct_binding):
    path, argv, archive, cameras, sequence = inputs
    if direct_binding:
        cameras["sourceSha256"] = sequence["sourceSha256"]
        sequence = None
    write_inputs(path, archive, cameras, sequence)
    stale = path / "static-scene.npy"
    stale.write_bytes(b"unrelated anchor cache")
    alignment = path / "framealign.json"
    alignment.write_text("{}")
    scene, build_cloud = stub_world_geometry(monkeypatch)
    export.main(argv)
    build_cloud.assert_called_once()
    assert build_cloud.call_args.args == (path / "anchors.npz", path / "cameras.json", [0, 1])
    exclusions = build_cloud.call_args.kwargs["exclusion_masks"]
    assert not exclusions[0].any()
    assert exclusions[1][5, 9] and exclusions[1][5, 11]
    out = path / "export"
    assert stale.read_bytes() == b"unrelated anchor cache"
    np.testing.assert_array_equal(np.load(out / "static-scene.npy"), scene)
    np.testing.assert_array_equal(np.load(out / "cameras.npz")["source_index"], [0, 2])
    for index, expected in enumerate((0, 110)):
        with Image.open(out / "frames" / f"{index:04d}.jpg") as image:
            assert abs(int(np.asarray(image)[0, 0, 0]) - expected) <= 2
        with Image.open(out / "masks" / f"{index:04d}.png") as image:
            assert int(np.asarray(image)[5, 9]) == index * 255
    receipt = json.loads((out / "export-receipt.json").read_bytes())
    assert receipt["sourceSha256"] == export.sha256(path / "source.mkv")
    assert receipt["camerasSha256"] == export.sha256(path / "cameras.json")
    assert receipt["masksSha256"] == export.sha256(path / "masks.npz")
    assert receipt["anchorsSha256"] == export.sha256(path / "anchors.npz")
    assert receipt["worldSha256"] == export.sha256(path / "world.spz")
    assert receipt["framealignSha256"] == export.sha256(alignment)
    assert receipt["cameraSequenceSha256"] == (
        None if direct_binding else export.sha256(path / "sequence.json")
    )
    assert receipt["anchorSamples"] == [0, 1]
    depth_exclusions = receipt["depthExclusions"]
    assert depth_exclusions["includesOriginalSuppliedMasks"]
    assert depth_exclusions["includesFinalTrainingMasks"]
    assert [row["sourceIndex"] for row in depth_exclusions["anchors"]] == [0, 2]
    assert [row["excludedPoints"] for row in depth_exclusions["anchors"]] == [0, 1]
    for index, row in enumerate(depth_exclusions["anchors"]):
        with Image.open(out / "masks" / f"{index:04d}.png") as image:
            assert (
                row["finalTrainingMaskSha256"]
                == export.hashlib.sha256((np.asarray(image) > 127).tobytes()).hexdigest()
            )
    assert receipt["sourceIndices"] == [0, 2]
    assert receipt["sourceProvenance"]["sourceTimeBase"] == str(archive["source_time_base"])
    assert [frame["sourcePts"] for frame in receipt["sourceProvenance"]["frames"]] == archive[
        "source_pts"
    ]
    with pytest.raises(FileExistsError):
        export.main(argv)


def test_intrinsics_scale_per_frame_and_keep_origin_compatibility_matrix(inputs, monkeypatch):
    path, argv, archive, cameras, sequence = inputs
    first = [[10, 0, 6], [0, 11, 4], [0, 0, 1]]
    second = [[12, 0, 7], [0, 13, 5], [0, 0, 1]]
    cameras["cameras"][0]["source_intrinsics"] = first
    cameras["cameras"][1]["source_intrinsics"] = second
    write_inputs(path, archive, cameras, sequence)
    size_arg = argv.index("--size")
    argv[size_arg + 1 : size_arg + 3] = ["34", "36"]
    argv.extend(["--origin-frame", "1"])
    stub_world_geometry(monkeypatch)
    export.main(argv)
    with np.load(path / "export" / "cameras.npz") as exported:
        expected = np.array([first, second], dtype=np.float32)
        expected[:, 0] *= 2
        expected[:, 1] *= 3
        np.testing.assert_array_equal(exported["Ks"], expected)
        np.testing.assert_array_equal(exported["K"], expected[1])


def test_generated_mask_path_uses_exact_frames_with_legacy_cameras(inputs, monkeypatch):
    path, argv, _, cameras, _ = inputs
    for camera in cameras["cameras"]:
        del camera["time"]
    (path / "cameras.json").write_text(json.dumps(cameras))
    masks_arg = argv.index("--masks-npz")
    del argv[masks_arg : masks_arg + 2]
    _, build_cloud = stub_world_geometry(monkeypatch)

    def person_masks(images, *, dilate_px):
        np.testing.assert_array_equal(images[:, 0, 0, 0], [0, 110])
        assert dilate_px == 16
        masks = np.zeros(images.shape[:3], dtype=bool)
        masks[1, 5, 9] = True
        return masks

    monkeypatch.setitem(
        sys.modules, "wander_worker.masks", SimpleNamespace(people_masks=person_masks)
    )
    export.main(argv)
    exclusions = build_cloud.call_args.kwargs["exclusion_masks"]
    assert not exclusions[0].any()
    assert exclusions[1][5, 11] and not exclusions[1][5, 12]
    receipt = json.loads((path / "export" / "export-receipt.json").read_bytes())
    assert receipt["maskMethod"] == "person"
    assert receipt["masksSha256"] is None
    assert receipt["cameraSequenceSha256"] is None
    assert receipt["sourceIndices"] == [0, 2]


def test_original_thin_exclusion_survives_training_resize(inputs, monkeypatch):
    path, argv, archive, cameras, sequence = inputs
    original = np.zeros((2, 12, 17), dtype=bool)
    original[1, 0, 0] = True
    archive["masks"] = np.packbits(original, axis=2)
    write_inputs(path, archive, cameras, sequence)
    size_arg = argv.index("--size")
    argv[size_arg + 1 : size_arg + 3] = ["1", "1"]
    _, build_cloud = stub_world_geometry(monkeypatch)
    export.main(argv)
    with Image.open(path / "export/masks/0001.png") as image:
        assert np.asarray(image).all()
    # The single training cell contains foreground, so none of its source extent
    # may supply background geometry, even when the original person is very thin.
    assert build_cloud.call_args.kwargs["exclusion_masks"][1].all()
    assert not build_cloud.call_args.kwargs["exclusion_masks"][0].any()


def test_overlap_resize_excludes_both_cells_straddled_by_one_pixel():
    mask = np.zeros((3, 5), dtype=bool)
    mask[1, 2] = True
    # The source center cell crosses both destination grid boundaries.
    np.testing.assert_array_equal(
        export.resize_exclusion_overlap(mask, (2, 2)), np.ones((2, 2), dtype=bool)
    )
    mask = np.array([[True, False]])
    np.testing.assert_array_equal(
        export.resize_exclusion_overlap(mask, (1, 4)), [[True, True, False, False]]
    )


def brute_overlap(mask, shape):
    """Reference rectangle intersections, without rounding or summed-area tables."""
    sh, sw = mask.shape
    dh, dw = shape
    result = np.zeros(shape, dtype=bool)
    for sy, sx in np.argwhere(mask):
        for dy in range(dh):
            for dx in range(dw):
                if (
                    sy * dh < (dy + 1) * sh
                    and (sy + 1) * dh > dy * sh
                    and sx * dw < (dx + 1) * sw
                    and (sx + 1) * dw > dx * sw
                ):
                    result[dy, dx] = True
    return result


@pytest.mark.parametrize(
    ("source", "target"),
    [((7, 11), (3, 4)), ((3, 4), (7, 11)), ((7, 3), (4, 11)), ((4, 6), (2, 3))],
)
def test_overlap_resize_matches_exact_rectangle_intersections(source, target):
    # Every single-cell position checks edges, thin features and zero-area touches;
    # random multi-cell masks also check the summed-area rectangle subtraction.
    for index in range(np.prod(source)):
        mask = np.zeros(source, dtype=bool)
        mask.flat[index] = True
        np.testing.assert_array_equal(
            export.resize_exclusion_overlap(mask, target), brute_overlap(mask, target)
        )
    rng = np.random.default_rng(82)
    for _ in range(5):
        mask = rng.random(source) < 0.15
        np.testing.assert_array_equal(
            export.resize_exclusion_overlap(mask, target), brute_overlap(mask, target)
        )


@pytest.mark.parametrize("shape", [(1, 1), (3, 4), (13, 17)])
@pytest.mark.parametrize("foreground", [False, True])
def test_overlap_resize_uniform_masks(shape, foreground):
    result = export.resize_exclusion_overlap(np.full((7, 11), foreground), shape)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, np.full(shape, foreground))


def test_overlap_resize_same_grid_preserves_every_cell():
    mask = np.random.default_rng(29).random((13, 17)) < 0.3
    result = export.resize_exclusion_overlap(mask, mask.shape)
    np.testing.assert_array_equal(result, mask)
    assert not np.shares_memory(result, mask)


def test_thin_foreground_missed_by_nearest_neighbor_is_excluded(inputs, monkeypatch):
    path, argv, archive, cameras, sequence = inputs
    original = np.zeros((2, 12, 17), dtype=bool)
    original[1, :, 0] = True
    archive["masks"] = np.packbits(original, axis=2)
    write_inputs(path, archive, cameras, sequence)
    size_arg = argv.index("--size")
    argv[size_arg + 1 : size_arg + 3] = ["7", "5"]
    nearest = np.asarray(
        Image.fromarray(original[1].astype(np.uint8)).resize((7, 5), Image.Resampling.NEAREST)
    )
    assert not nearest.any()
    stub_world_geometry(monkeypatch)
    export.main(argv)
    with Image.open(path / "export/masks/0001.png") as image:
        excluded = np.asarray(image) > 0
    # Positive overlap is the left column; existing two-pixel dilation covers
    # two more columns, but must not consume the remaining clear background.
    expected = np.zeros((5, 7), dtype=bool)
    expected[:, :3] = True
    np.testing.assert_array_equal(excluded, expected)
    with Image.open(path / "export/masks/0000.png") as image:
        assert not np.asarray(image).any()
