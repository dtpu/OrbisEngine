"""Fine-tuned SPZ/PLY conversion with surviving, pruned and fully optimised splats."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import finetune_import
from bake_video_colours import read_spz
from finetune_export import quat_mul, rotmat_to_quat_xyzw


@pytest.fixture
def original(tmp_path):
    path = tmp_path / "original.spz"
    finetune_import.write_spz(
        path,
        np.asarray([[1, 2, 3], [-2, 0, 1]], np.float64),
        np.asarray([0.6, 0.8]),
        np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
        np.asarray([[-3, -4, -5], [-5, -4, -3]], np.float64),
        np.asarray([[0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]]),
        12,
        struct.pack("<IIIBBBB", 0x5053474E, 2, 2, 0, 12, 0, 0),
    )
    return path, read_spz(path)


def trained_state(tmp_path, source, indices, visible, *, omit_tracking=False):
    n = len(indices)
    # A rotated, translated, rescaled native frame exercises the actual inverse conversion.
    rotation = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float64)
    origin = np.eye(4)
    origin[:3, 3] = [3, -2, 5]
    scale = 0.5
    selected = np.asarray([max(0, i) for i in indices], dtype=int)
    xyz = source["xyz"][selected].astype(np.float64)
    native_quat = quat_mul(
        np.repeat(rotmat_to_quat_xyzw(rotation)[None], n, axis=0),
        source["quat"][selected].astype(np.float64),
    )
    alpha = source["alpha"][selected].astype(np.float64)
    payload = dict(
        means=xyz @ rotation.T / scale + origin[:3, 3],
        quats=native_quat[:, [3, 0, 1, 2]],
        log_scales=source["log_scale"][selected].astype(np.float64) - np.log(scale),
        logit_opacities=np.log(alpha / (1 - alpha)),
        f_dc=source["f_dc"][selected],
        R_raw_to_native=rotation,
        origin_c2w=origin,
        scale0=scale,
        fb=source["fb"],
    )
    if not omit_tracking:
        payload.update(visible=np.asarray(visible, bool), orig_index=np.asarray(indices, int))
    npz = tmp_path / "finetuned.npz"
    np.savez_compressed(npz, **payload)
    return npz, selected


def invoke(npz, original_path, output, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune_import.py",
            "--npz",
            str(npz),
            "--world",
            str(original_path),
            "--out",
            str(output),
        ],
    )
    finetune_import.main()


def assert_converted(output, source, selected):
    actual = read_spz(output.with_suffix(".spz"))
    assert actual["n"] == len(selected)
    assert actual["fb"] == source["fb"]
    np.testing.assert_allclose(actual["xyz"], source["xyz"][selected], atol=1 / (1 << source["fb"]))
    np.testing.assert_allclose(actual["log_scale"], source["log_scale"][selected], atol=1 / 16)
    np.testing.assert_allclose(actual["f_dc"], source["f_dc"][selected], atol=1e-6)
    np.testing.assert_allclose(actual["alpha"], source["alpha"][selected], atol=1 / 255)
    alignment = np.abs((actual["quat"] * source["quat"][selected]).sum(axis=1))
    np.testing.assert_allclose(alignment, 1, atol=1e-3)
    ply = PlyData.read(output.with_suffix(".ply"))["vertex"].data
    assert len(ply) == len(selected)
    np.testing.assert_allclose(
        np.column_stack([ply[k] for k in ("x", "y", "z")]), source["xyz"][selected], atol=1e-6
    )
    np.testing.assert_allclose(
        np.column_stack([ply[k] for k in ("scale_0", "scale_1", "scale_2")]),
        source["log_scale"][selected],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        1 / (1 + np.exp(-ply["opacity"])), source["alpha"][selected], atol=1e-6
    )


@pytest.mark.parametrize(
    "indices,visible", [([0, 1], [True, True]), ([1], [True]), ([-1], [False])]
)
def test_no_frozen_survivors_still_convert_accurately(
    original, tmp_path, monkeypatch, capsys, indices, visible
):
    path, source = original
    npz, selected = trained_state(tmp_path, source, indices, visible)
    output = tmp_path / "converted"
    invoke(npz, path, output, monkeypatch)
    assert_converted(output, source, selected)
    text = capsys.readouterr().out
    assert "no frozen original splats" in text
    assert "max |xyz - original|" not in text
    assert "re-read" in text


@pytest.mark.parametrize("indices,visible", [([0, 1], [False, True]), ([1], [False])])
def test_frozen_survivors_retain_round_trip_diagnostics(
    original, tmp_path, monkeypatch, capsys, indices, visible
):
    path, source = original
    npz, selected = trained_state(tmp_path, source, indices, visible)
    output = tmp_path / "converted"
    invoke(npz, path, output, monkeypatch)
    assert_converted(output, source, selected)
    assert "max |xyz - original|" in capsys.readouterr().out


def test_legacy_state_without_tracking_arrays_preserves_round_trip(
    original, tmp_path, monkeypatch, capsys
):
    path, source = original
    npz, selected = trained_state(tmp_path, source, [0, 1], [False, False], omit_tracking=True)
    output = tmp_path / "converted"
    invoke(npz, path, output, monkeypatch)
    assert_converted(output, source, selected)
    assert "max |xyz - original|" in capsys.readouterr().out


def test_empty_training_result_is_rejected_before_writing_outputs(original, tmp_path, monkeypatch):
    path, source = original
    npz, _ = trained_state(tmp_path, source, [], [])
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="no splats"):
        invoke(npz, path, output, monkeypatch)
    assert not output.with_suffix(".spz").exists()
    assert not output.with_suffix(".ply").exists()


def test_all_visible_refinement_preserves_changed_geometry_and_attributes(
    original, tmp_path, monkeypatch
):
    path, source = original
    npz, _ = trained_state(tmp_path, source, [0, 1], [True, True])
    with np.load(npz) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["means"][0] += [0.25, -0.5, 0.75]
    payload["log_scales"][0] += 0.125
    payload["f_dc"][0] += 0.2
    payload["logit_opacities"][0] += 0.3
    np.savez_compressed(npz, **payload)
    output = tmp_path / "refined"
    invoke(npz, path, output, monkeypatch)
    converted = read_spz(output.with_suffix(".spz"))
    expected_xyz = source["xyz"].copy()
    expected_xyz[0] += [-0.375, -0.25, 0.125]
    np.testing.assert_allclose(converted["xyz"], expected_xyz, atol=1 / (1 << source["fb"]))
    np.testing.assert_allclose(converted["log_scale"][0], source["log_scale"][0] + 0.125, atol=1e-6)
    np.testing.assert_allclose(
        converted["f_dc"][0],
        source["f_dc"][0] + 0.2,
        atol=1 / (finetune_import.SPZ_COLOR_SCALE * 255),
    )
    expected_alpha = 1 / (1 + np.exp(-payload["logit_opacities"][0]))
    assert converted["alpha"][0] == pytest.approx(expected_alpha, abs=1 / 255)
    ply = PlyData.read(output.with_suffix(".ply"))["vertex"].data
    np.testing.assert_allclose(
        np.column_stack([ply[k] for k in ("x", "y", "z")]), expected_xyz, atol=1e-6
    )
    np.testing.assert_allclose(
        [ply["f_dc_0"][0], ply["f_dc_1"][0], ply["f_dc_2"][0]], payload["f_dc"][0], atol=1e-6
    )
