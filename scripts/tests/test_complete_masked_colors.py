"""Color-only inference and strict offline input/provenance contracts."""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import complete_masked_colors as completion


def write_spz(path, xyz=None, colors=None):
    xyz = (
        np.array([[0, 0, -2], [0.02, 0, -2], [0.04, 0, -2], [0.06, 0, -2], [2, 0, -2]])
        if xyz is None
        else np.asarray(xyz)
    )
    n = len(xyz)
    colors = (
        np.array(
            [[210, 100, 100], [90, 140, 120], [90, 140, 120], [190, 200, 210], [0, 255, 0]],
            np.uint8,
        )
        if colors is None
        else np.asarray(colors, np.uint8)
    )
    fixed = np.rint(xyz * 4096).astype(np.int32)
    packed = np.stack([fixed & 255, (fixed >> 8) & 255, (fixed >> 16) & 255], axis=-1).astype(
        np.uint8
    )
    raw = (
        struct.pack("<IIIBBBB", 0x5053474E, 2, n, 0, 12, 0, 0)
        + packed.tobytes()
        + np.full(n, 180, np.uint8).tobytes()
        + colors.tobytes()
        + np.full((n, 3), 100, np.uint8).tobytes()
        + np.full((n, 3), 128, np.uint8).tobytes()
    )
    path.write_bytes(gzip.compress(raw, mtime=0))
    return raw


def exposure(n=5, frames=(0, 1, 2)):
    clear = np.ones(n, np.float64)
    clear[0] = 0
    masked = np.zeros(n, np.float64)
    masked[0] = 2
    views = np.full(n, 3, np.int32)
    views[0] = 0
    return dict(
        rendered_mass=clear + masked,
        masked_mass=masked,
        clear_mass=clear,
        clear_views=views,
        frame_indices=np.array(frames),
    )


@pytest.fixture
def inputs(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "masks").mkdir()
    world = tmp_path / "input.spz"
    write_spz(world)
    cameras = dict(
        c2w_gl=np.tile(np.eye(4), (4, 1, 1)),
        Ks=np.tile(np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]]), (4, 1, 1)),
        width=np.array(2),
        height=np.array(2),
        source_index=np.array([0, 2, 4, 6]),
    )
    np.savez(export / "cameras.npz", **cameras)
    # An object array must remain unread: only origin and scale are required.
    np.savez(
        export / "splats.npz",
        origin_c2w=np.eye(4),
        scale0=2.0,
        means=np.array([{"unused": True}], dtype=object),
    )
    receipt = dict(
        sourceIndices=[0, 2, 4, 6], size=[2, 2], scale0=2.0, sourceSha256="declared-only"
    )
    (export / "export-receipt.json").write_text(json.dumps(receipt))
    for index in range(3):
        Image.fromarray(np.array([[255, 0], [0, 0]], np.uint8)).save(
            export / "masks" / f"{index:04d}.png"
        )
    frames = tmp_path / "frames.json"
    frames.write_text("[0, 1, 2]")
    args = completion.parser().parse_args(
        [
            "--world",
            str(world),
            "--export",
            str(export),
            "--training-frames",
            str(frames),
            "--body-height-native",
            "1",
            "--out",
            str(tmp_path / "output"),
        ]
    )
    return args


def fake_measure(world, poses, ks, masks, *, origin_c2w, scale, frame_indices, progress):
    assert world["n"] == 5
    assert poses.shape == (4, 4, 4)
    assert ks.shape == (4, 3, 3)
    assert set(masks) == {0, 1, 2}  # frame 3 has no mask; never read or evaluated
    assert all(mask.dtype == np.bool_ for mask in masks.values())
    np.testing.assert_array_equal(origin_c2w, np.eye(4))
    assert scale == 2
    for completed, frame in enumerate(frame_indices, 1):
        progress(frame, completed, len(frame_indices))
    return exposure()


def test_cli_modifies_only_accepted_target_records_and_provenance(inputs, monkeypatch):
    before = inputs.world.read_bytes()
    raw = gzip.decompress(before)
    monkeypatch.setattr(completion, "measure_source_exposure", fake_measure)
    report = completion.run(inputs)
    result = gzip.decompress((inputs.out / "world.spz").read_bytes())
    offset = 16 + 5 * 10
    assert result[:offset] == raw[:offset]  # header, positions, alpha
    assert result[offset + 3 :] == raw[offset + 3 :]  # other colors, scales, rotations
    assert result[offset : offset + 3] == raw[offset + 3 : offset + 6]
    assert result[offset + 12 : offset + 15] == bytes([0, 255, 0])  # out-of-gamut retained bytes
    np.testing.assert_array_equal(
        np.load(inputs.out / "inferred-mask.npy"), [True, False, False, False, False]
    )
    assert inputs.world.read_bytes() == before
    assert report["inputHashesBefore"] == report["inputHashesAfter"]
    assert str(inputs.world.resolve()) in report["inputHashesBefore"]
    for name in (
        "complete_masked_colors.py",
        "splat_observations.py",
        "render_world_poses.py",
        "uv.lock",
    ):
        assert any(Path(path).name == name for path in report["inputHashesBefore"])
    assert report["status"] == "completed"
    assert report["acceptedScene"] is False
    assert report["radiusRawWorldUnits"] == 0.5
    assert report["selectionCounts"]["changedColorRecords"] == 1
    assert report["exportSourceProvenance"]["sourceSha256"] == "declared-only"
    with np.load(inputs.out / "exposure.npz") as saved:
        np.testing.assert_array_equal(saved["frame_indices"], [0, 1, 2])
    # Gzip header contains deterministic zero mtime.
    assert (inputs.out / "world.spz").read_bytes()[4:8] == b"\0" * 4
    with pytest.raises(ValueError, match="already exists"):
        completion.run(inputs)
    assert inputs.world.read_bytes() == before


def test_weighted_median_rejects_bright_outlier():
    values = np.array([[0.2, 0.4, 0.3], [0.2, 0.4, 0.3], [1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(completion.weighted_median(values, np.ones(3)), values[0])
    np.testing.assert_array_equal(
        completion.weighted_median(values, np.array([1, 1, 3])), values[2]
    )


@pytest.mark.parametrize("reason", ["no-donors", "outside-radius", "too-few"])
def test_rejected_targets_preserved(inputs, reason):
    raw, _ = completion.read_raw_spz(inputs.world)
    world = completion.render_world_poses.read_spz(inputs.world)
    exp = exposure()
    radius = 0.25
    if reason == "no-donors":
        exp["clear_views"][:] = 0
    elif reason == "outside-radius":
        radius = 0.001
    else:
        exp["clear_views"][3:] = 0
    rgb, inferred, counts = completion.infer_colors(
        world, exp, radius=radius, neighbors=16, min_neighbors=3, min_clear_views=3
    )
    assert not inferred.any()
    assert counts["rejectedInsufficientDonors"] == 1
    assert completion.patch_colors(raw, inferred, rgb) == raw


def test_observation_filter_precedes_knn_and_no_recursive_donors():
    # More than 64 nearer ineligible points must not hide 3 eligible donors.
    n = 75
    world = dict(n=n, xyz=np.zeros((n, 3)), rgb=np.full((n, 3), 0.9))
    world["xyz"][:, 0] = np.linspace(0, 0.2, n)
    world["rgb"][-3:] = 0.2
    exp = dict(rendered_mass=np.ones(n), clear_mass=np.zeros(n), clear_views=np.zeros(n, np.int32))
    exp["clear_views"][-3:] = 3
    exp["clear_mass"][-3:] = 1
    rgb, inferred, counts = completion.infer_colors(
        world, exp, radius=0.25, neighbors=3, min_neighbors=3, min_clear_views=3
    )
    assert inferred[0]
    assert counts["eligibleDonors"] == 3
    np.testing.assert_array_equal(rgb[inferred], np.full((72, 3), 0.2))
    np.testing.assert_array_equal(world["rgb"][:72], np.full((72, 3), 0.9))


def test_targets_depend_on_exposure_not_unchanged_parameters():
    world = dict(
        n=4, xyz=np.zeros((4, 3)), rgb=np.array([[0.1] * 3, [0.2] * 3, [0.3] * 3, [0.4] * 3])
    )
    exp = exposure(4)
    exp["rendered_mass"][0] = 0
    rgb, inferred, counts = completion.infer_colors(
        world, exp, radius=1, neighbors=3, min_neighbors=3, min_clear_views=3
    )
    assert not inferred.any()
    assert counts["targets"] == 0
    np.testing.assert_array_equal(rgb, world["rgb"])


@pytest.mark.parametrize("frames", [[], [True], [0.0], [0, 0], [-1], [4], {"frames": [0]}])
def test_malformed_frames_rejected(inputs, frames, monkeypatch):
    inputs.training_frames.write_text(json.dumps(frames))
    monkeypatch.setattr(
        completion, "measure_source_exposure", lambda *a, **k: pytest.fail("must not render")
    )
    with pytest.raises(ValueError):
        completion.run(inputs)
    report = json.loads((inputs.out / "report.json").read_text())
    assert report["status"] == "failed"
    assert not (inputs.out / "world.spz").exists()


def replace_archive(path, key, value):
    with np.load(path) as archive:
        values = {name: archive[name] for name in archive.files}
    values[key] = value
    np.savez(path, **values)


@pytest.mark.parametrize(
    "key,value",
    [
        ("width", np.array([2])),
        ("height", np.array(2.0)),
        ("source_index", np.array([0, 2, 2, 6])),
        ("source_index", np.array([0, 2, 4])),
        ("Ks", np.zeros((4, 3, 3))),
        ("c2w_gl", np.full((4, 4, 4), np.nan)),
        ("Ks", np.ones((3, 3))),
    ],
)
def test_malformed_camera_metadata(inputs, key, value):
    replace_archive(inputs.export / "cameras.npz", key, value)
    with pytest.raises(ValueError):
        completion.load_export(inputs.export, [0, 1, 2])


@pytest.mark.parametrize(
    "key,value",
    [
        ("size", [2, 3]),
        ("sourceIndices", [0, 2, 4, 7]),
        ("scale0", 1),
        ("scale0", True),
        ("size", [2.0, 2]),
    ],
)
def test_receipt_mismatch(inputs, key, value):
    path = inputs.export / "export-receipt.json"
    receipt = json.loads(path.read_text())
    receipt[key] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt"):
        completion.load_export(inputs.export, [0, 1, 2])


@pytest.mark.parametrize(
    "values", [np.ones((2, 2), np.uint8), np.zeros((1, 2), np.uint8), np.zeros((2, 2, 3), np.uint8)]
)
def test_invalid_masks(inputs, values):
    Image.fromarray(values).save(inputs.export / "masks/0000.png")
    with pytest.raises(ValueError, match="mask"):
        completion.load_export(inputs.export, [0, 1, 2])


def test_missing_mask_has_no_fallback(inputs):
    (inputs.export / "masks/0001.png").unlink()
    with pytest.raises(FileNotFoundError):
        completion.load_export(inputs.export, [0, 1, 2])


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--body-height-native", "nan"),
        ("--radius-body-heights", "inf"),
        ("--max-seconds", "0"),
        ("--neighbors", "0"),
        ("--min-neighbors", "1.5"),
        ("--min-clear-views", "-1"),
    ],
)
def test_invalid_cli_parameters(flag, value):
    with pytest.raises(SystemExit):
        completion.parser().parse_args(
            [
                "--world",
                "a",
                "--export",
                "b",
                "--training-frames",
                "c",
                "--body-height-native",
                "1",
                "--out",
                "d",
                flag,
                value,
            ]
        )


@pytest.mark.parametrize("field,value", [("min_neighbors", 17), ("min_clear_views", 4)])
def test_inconsistent_counts(inputs, field, value):
    setattr(inputs, field, value)
    with pytest.raises(ValueError):
        completion.run(inputs)
    assert json.loads((inputs.out / "report.json").read_text())["status"] == "failed"


def test_input_mutation_blocks_final_world(inputs, monkeypatch):
    def mutate(*args, **kwargs):
        result = fake_measure(*args, **kwargs)
        inputs.training_frames.write_text("[0,1,2]\n")
        return result

    monkeypatch.setattr(completion, "measure_source_exposure", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        completion.run(inputs)
    assert not (inputs.out / "world.spz").exists()
    report = json.loads((inputs.out / "report.json").read_text())
    assert report["inputHashesBefore"] != report["inputHashesAfter"]
    assert report["status"] == "failed"


def test_timeout_preserves_failed_receipt(inputs, monkeypatch):
    monkeypatch.setattr(completion, "measure_source_exposure", fake_measure)
    clock = iter([0, 0, 901, 902])
    monkeypatch.setattr(completion.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        completion.run(inputs)
    assert json.loads((inputs.out / "report.json").read_text())["status"] == "failed"
    assert not (inputs.out / "world.spz").exists()


@pytest.mark.parametrize("mutation", ["short", "version", "sh", "length", "flags", "gzip"])
def test_invalid_spz_rejected(inputs, mutation):
    raw = bytearray(gzip.decompress(inputs.world.read_bytes()))
    if mutation == "short":
        raw = raw[:5]
    elif mutation == "version":
        struct.pack_into("<I", raw, 4, 3)
    elif mutation == "sh":
        raw[12] = 1
    elif mutation == "length":
        raw.pop()
    elif mutation == "flags":
        raw[14] = 1
    inputs.world.write_bytes(b"bad" if mutation == "gzip" else gzip.compress(raw))
    with pytest.raises(ValueError):
        completion.read_raw_spz(inputs.world)


@pytest.mark.parametrize(
    "rotation",
    [np.diag([2, 1, 1]), np.diag([-1, 1, 1]), np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])],
)
def test_nonrigid_origin_rejected_for_radius_conversion(inputs, rotation):
    origin = np.eye(4)
    origin[:3, :3] = rotation
    np.savez(inputs.export / "splats.npz", origin_c2w=origin, scale0=2.0)
    with pytest.raises(ValueError, match="proper rigid"):
        completion.load_export(inputs.export, [0, 1, 2])


def test_rotated_rigid_origin_and_float_roundoff_allowed(inputs):
    angle = 0.4
    origin = np.eye(4, dtype=np.float32)
    origin[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    origin[:3, 3] = [1, 2, 3]
    np.savez(inputs.export / "splats.npz", origin_c2w=origin, scale0=2.0)
    loaded = completion.load_export(inputs.export, [0, 1, 2])
    np.testing.assert_array_equal(loaded[2], origin)
