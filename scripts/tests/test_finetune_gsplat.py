"""CPU checks for calibration and view routing in the standalone GPU trainer."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finetune_gsplat as trainer


@pytest.fixture
def cameras(tmp_path):
    compatibility = np.eye(3, dtype=np.float32)
    frame_Ks = np.array(
        [[[10, 0, 0], [0, 1, 1], [0, 0, 1]], [[2, 0, 0], [0, 1, 1], [0, 0, 1]]], dtype=np.float32
    )
    data = dict(
        viewmats=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        K=compatibility,
        Ks=frame_Ks,
        width=5,
        height=3,
    )
    return tmp_path / "cameras.npz", data


@pytest.mark.parametrize("legacy", [True, False])
def test_load_cameras_preserves_per_frame_and_legacy_intrinsics(cameras, legacy):
    path, data = cameras
    expected = data["Ks"]
    if legacy:
        del data["Ks"]
        expected = np.repeat(data["K"][None], 2, axis=0)
    np.savez(path, **data)
    views, Ks, K, width, height = trainer.load_cameras(path)
    np.testing.assert_array_equal(views, data["viewmats"])
    np.testing.assert_array_equal(Ks, expected)
    np.testing.assert_array_equal(K, data["K"])
    assert (width, height) == (5, 3)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("wrong-length", "Ks must contain one"),
        ("bad-focal", "Ks must contain one"),
        ("nan", "Ks must contain one"),
        ("legacy-bad-focal", "Compatibility K"),
        ("bad-size", "width and height"),
        ("bad-view-shape", "finite viewmats"),
        ("bad-figure", "must index exported cameras"),
        ("bad-offpath", "must index exported cameras"),
        ("no-training", "Holdout must be at least 2"),
        ("bad-depth-index", "map every camera"),
    ],
)
def test_invalid_camera_inputs_fail_before_cuda_and_output(cameras, monkeypatch, mutation, error):
    path, data = cameras
    out = path.parent / "output"
    poses, offpath, holdout = "0", "1", "25"
    if mutation == "wrong-length":
        data["Ks"] = data["Ks"][:1]
    elif mutation == "bad-focal":
        data["Ks"][1, 1, 1] = 0
    elif mutation == "nan":
        data["Ks"][1, 1, 2] = np.nan
    elif mutation == "legacy-bad-focal":
        data["K"][0, 0] = -1
    elif mutation == "bad-size":
        data["width"] = 0
    elif mutation == "bad-view-shape":
        data["viewmats"] = np.eye(4)
    elif mutation == "bad-figure":
        poses = "0,2"
    elif mutation == "bad-offpath":
        offpath = "-1"
    elif mutation == "no-training":
        holdout = "1"
    elif mutation == "bad-depth-index":
        np.savez(
            path.with_name("depth.npz"),
            depth=np.ones((2, 3, 5)),
            supported=np.ones((2, 3, 5), dtype=bool),
            depth_index=[0, 2],
        )
    np.savez(path, **data)

    def unexpected_device(*_):
        pytest.fail("CPU preflight must run before device selection")

    monkeypatch.setattr(trainer.torch, "device", unexpected_device)
    with pytest.raises(ValueError, match=error):
        trainer.main(
            [
                "--data",
                str(path.parent),
                "--out",
                str(out),
                "--poses",
                poses,
                "--offpath-frame",
                offpath,
                "--holdout",
                holdout,
            ]
        )
    assert not out.exists()


def test_depth_and_visibility_project_with_each_views_intrinsics(cameras):
    _, data = cameras
    means = torch.tensor([[1.0, 0.0, 1.0]])
    model = SimpleNamespace(n=len(means), means=means)
    views = torch.from_numpy(data["viewmats"])
    Ks = torch.from_numpy(data["Ks"])
    depth = torch.full((2, 3, 5), 2.0)
    support = torch.zeros_like(depth, dtype=torch.bool)
    support[1, 1, 2] = True
    valid = torch.ones_like(support)
    assert trainer.visible_mask(model, views, Ks, 5, 3, "cpu").tolist() == [True]
    observed, hits = trainer.observed_mask(means, views, Ks, 5, 3, (depth, support), valid, 0, 1)
    assert observed.tolist() == [True]
    assert hits.tolist() == [1]
    assert trainer.depth_offset(means, views, Ks, 5, 3, (depth, support), valid).tolist() == [-1.0]
    assert trainer.visible_mask(model, views, Ks[:1].repeat(2, 1, 1), 5, 3, "cpu").tolist() == [
        False
    ]


def test_heldout_depth_requires_and_counts_intersection():
    depth = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reference = torch.tensor([[2.0, 4.0, 0.0, 8.0]])
    supported = torch.tensor([[True, True, False, True]])
    valid = torch.tensor([[False, True, True, True]])
    alpha = torch.tensor([[1.0, 1.0, 1.0, 0.5]])
    error, count = trainer.heldout_depth_error(depth, reference, supported, valid, alpha, 2)
    assert error == pytest.approx(np.log(2))
    assert count == 1
    with pytest.raises(ValueError, match="frame 2 has no supported unmasked"):
        trainer.heldout_depth_error(depth, reference, supported, valid, alpha * 0, 2)
