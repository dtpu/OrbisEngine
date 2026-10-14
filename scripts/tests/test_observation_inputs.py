"""Source person masks exclude occluded geometry from the observation sidecar."""

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import export_observation_confidence as observation


def camera(index=7):
    return {
        "sourceIndex": index,
        "source_image_size": [8, 8],
        "camera_to_world": np.eye(4).tolist(),
        "source_intrinsics": [[4, 0, 0], [0, 4, 0], [0, 0, 1]],
    }


def source_mask():
    mask = np.zeros((8, 8), bool)
    mask[:4, :4] = True
    return mask


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def write_spz(path):
    points = [[0, 0, 1], [1, 1, 1]]
    header = struct.pack("<IIIBBBB", 0x5053474E, 2, len(points), 0, 0, 0, 0)
    positions = b"".join(
        value.to_bytes(3, "little", signed=True) for point in points for value in point
    )
    path.write_bytes(gzip.compress(header + positions))
    return path


def write_tight(path, indices=(7,)):
    masks = np.stack([source_mask() for _ in indices])
    np.savez_compressed(
        path, shape=masks.shape, masks=np.packbits(masks, axis=-1), indices=np.asarray(indices)
    )
    return path


def test_source_masks_exclude_person_pixels_and_keep_front_background(monkeypatch, tmp_path):
    monkeypatch.setattr(
        observation,
        "person_pixels",
        lambda *a, **k: pytest.fail("source masks must not use the avatar fallback"),
    )
    xyz = np.asarray([[0, 0, 1], [0, 0, 2], [1, 1, 1]], np.float32)
    seen, in_front = observation.observe(
        xyz, [camera()], 1, world_dir=tmp_path, person_masks={7: source_mask()}
    )
    np.testing.assert_array_equal(seen, [0, 0, 1])
    np.testing.assert_array_equal(in_front, [1, 1, 1])


def test_source_indices_select_correct_mask_for_each_camera():
    xyz = np.asarray([[0, 0, 1], [1, 1, 1]], np.float32)
    masks = {7: source_mask(), 42: np.zeros((8, 8), bool)}
    seen, _ = observation.observe(xyz, [camera(42), camera(7)], 1, person_masks=masks)
    np.testing.assert_array_equal(seen, [1, 2])
    with pytest.raises(ValueError, match="no coverage.*sourceIndex 43"):
        observation.observe(xyz, [camera(43)], 1, person_masks=masks)


def test_coverage_is_checked_even_when_splats_are_outside_camera():
    with pytest.raises(ValueError, match="no coverage.*sourceIndex 7"):
        observation.observe(np.asarray([[0, 0, -1]], np.float32), [camera()], 1, person_masks={})


def test_legacy_avatar_mask_and_no_mask_paths_remain_available(monkeypatch, tmp_path):
    seen_inputs = []

    def avatar_mask(world_dir, cam, width, height, ds):
        seen_inputs.append((world_dir, cam["sourceIndex"], width, height, ds))
        return np.asarray([[True, False], [False, False]])

    monkeypatch.setattr(observation, "person_pixels", avatar_mask)
    xyz = np.asarray([[0, 0, 1], [1, 1, 1]], np.float32)
    seen, _ = observation.observe(xyz, [camera()], 1, world_dir=tmp_path)
    np.testing.assert_array_equal(seen, [0, 1])
    assert seen_inputs == [(tmp_path, 7, 2, 2, 4)]
    seen, _ = observation.observe(xyz, [camera()], 1)
    np.testing.assert_array_equal(seen, [1, 1])


@pytest.fixture
def cli_inputs(tmp_path):
    world = write_spz(tmp_path / "retained.spz")
    cameras = write_json(tmp_path / "camera-export/cameras.json", {"cameras": [camera()]})
    directory = tmp_path / "packaged-world"
    directory.mkdir()
    masks = write_tight(tmp_path / "source-masks.npz")
    out = tmp_path / "sidecars/retained-obs.bin"
    return dict(world=world, cameras=cameras, world_dir=directory, masks=masks, out=out)


def arguments(inputs):
    argv = [
        "export_observation_confidence.py",
        "--scale",
        "1",
        "--saturate",
        "1",
        "--device",
        "cpu",
    ]
    for option, value in inputs.items():
        argv += ["--" + option.replace("_", "-"), str(value)]
    return argv


def test_cli_writes_expected_sidecar_from_exact_paths(cli_inputs, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(cli_inputs))
    observation.main()
    assert cli_inputs["out"].read_bytes() == bytes([0, 255])


def test_tracking_masks_map_samples_using_explicit_world_directory(cli_inputs, monkeypatch):
    write_json(cli_inputs["world_dir"] / "people.json", {"sourceIndices": [7]})
    write_json(cli_inputs["cameras"].parent / "people.json", {"sourceIndices": [999]})
    np.savez_compressed(cli_inputs["masks"], shape=[8, 8, 1], t000_s000=np.packbits(source_mask()))
    monkeypatch.setattr(sys, "argv", arguments(cli_inputs))
    observation.main()
    assert cli_inputs["out"].read_bytes() == bytes([0, 255])
    cli_inputs["out"].unlink()
    monkeypatch.setattr(
        sys, "argv", arguments({k: v for k, v in cli_inputs.items() if k != "world_dir"})
    )
    with pytest.raises(SystemExit) as error:
        observation.main()
    assert error.value.code == 2
    assert not cli_inputs["out"].exists()


def test_unmapped_supplied_masks_fail_without_writing_sidecar(cli_inputs, monkeypatch):
    write_tight(cli_inputs["masks"], indices=(99,))
    monkeypatch.setattr(sys, "argv", arguments(cli_inputs))
    with pytest.raises(SystemExit) as error:
        observation.main()
    assert error.value.code == 2
    assert not cli_inputs["out"].exists()


@pytest.mark.parametrize(
    "kind",
    [
        "missing-indices",
        "duplicate-indices",
        "wrong-shape",
        "fractional-indices",
        "invalid-archive",
    ],
)
def test_malformed_explicit_masks_fail(cli_inputs, kind):
    path = cli_inputs["masks"]
    if kind == "invalid-archive":
        path.write_bytes(b"not an archive")
    else:
        shape = (2, 8, 8)
        payload = dict(
            shape=shape, masks=np.packbits(np.ones(shape, bool), axis=-1), indices=[7, 42]
        )
        if kind == "missing-indices":
            del payload["indices"]
        elif kind == "duplicate-indices":
            payload["indices"] = [7, 7]
        elif kind == "wrong-shape":
            payload["shape"] = (3, 8, 8)
        elif kind == "fractional-indices":
            payload["indices"] = [7.1, 42.2]
        np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="invalid --masks"):
        observation.load_source_masks(path, cli_inputs["world_dir"], [camera()])


def test_tracking_archive_without_source_mapping_fails(cli_inputs):
    np.savez_compressed(cli_inputs["masks"], shape=[8, 8, 1], t000_s000=np.packbits(source_mask()))
    with pytest.raises(ValueError, match="no source-frame mapping"):
        observation.load_source_masks(cli_inputs["masks"], cli_inputs["world_dir"], [camera()])


@pytest.mark.parametrize("option", ["world", "cameras", "world_dir", "masks"])
def test_missing_explicit_paths_fail_before_any_legacy_fallback(
    cli_inputs, tmp_path, monkeypatch, option
):
    cli_inputs[option] = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", arguments(cli_inputs))
    with pytest.raises(SystemExit) as error:
        observation.main()
    assert error.value.code == 2
    assert not cli_inputs["out"].exists()


def test_no_person_mask_rejects_supplied_masks(cli_inputs, monkeypatch):
    monkeypatch.setattr(sys, "argv", arguments(cli_inputs) + ["--no-person-mask"])
    with pytest.raises(SystemExit) as error:
        observation.main()
    assert error.value.code == 2


def test_cli_no_person_mask_keeps_legacy_unmasked_measurement(cli_inputs, monkeypatch):
    inputs = {k: v for k, v in cli_inputs.items() if k != "masks"}
    monkeypatch.setattr(sys, "argv", arguments(inputs) + ["--no-person-mask"])
    observation.main()
    assert inputs["out"].read_bytes() == bytes([255, 255])


def test_partial_custom_arguments_fail_cleanly(cli_inputs, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["export_observation_confidence.py", "--world", str(cli_inputs["world"])]
    )
    with pytest.raises(SystemExit) as error:
        observation.main()
    assert error.value.code == 2
