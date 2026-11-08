"""CPU checks for calibration and view routing in the standalone GPU trainer."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finetune_gsplat as trainer


def test_topology_report_keeps_initial_count_and_row_provenance_after_prune_and_grow():
    model = trainer.Model(
        dict(
            means=np.arange(15, dtype=np.float32).reshape(5, 3),
            quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (5, 1)),
            log_scales=np.zeros((5, 3), dtype=np.float32),
            logit_opacities=np.zeros(5, dtype=np.float32),
            f_dc=np.zeros((5, 3), dtype=np.float32),
        ),
        "cpu",
    )
    orig_index = torch.arange(model.n_init, dtype=torch.int32)
    assert model.topology_stats(orig_index) == {
        "n_splats": 5,
        "n_splats_init": 5,
        "n_new": 0,
    }
    opt = torch.optim.Adam(
        [{"params": [p.requires_grad_(True)], "name": k} for k, p in model.params().items()]
    )
    # Populate real optimizer state before the same topology surgery used by training.
    sum(p.sum() for p in model.params().values()).backward()
    opt.step()
    original_means = model.means.detach().clone()
    extra = {k: p.detach()[[0, 0, 2]].clone() for k, p in model.params().items()}
    extra["means"] += 100
    (orig_index,) = trainer.surgery(
        model,
        opt,
        torch.tensor([True, False, True, False, True]),
        extra,
        side=[(orig_index, -1)],
    )
    assert orig_index.tolist() == [0, 2, 4, -1, -1, -1]
    assert model.topology_stats(orig_index) == {
        "n_splats": 6,
        "n_splats_init": 5,
        "n_new": 3,
    }
    # A later prune removes both an original and a newly created row. n_new is survivors,
    # not cumulative additions; retained original indices must not be renumbered.
    (orig_index,) = trainer.surgery(
        model,
        opt,
        torch.tensor([False, True, True, False, True, True]),
        side=[(orig_index, -1)],
    )
    assert orig_index.tolist() == [2, 4, -1, -1]
    assert model.topology_stats(orig_index) == {
        "n_splats": 4,
        "n_splats_init": 5,
        "n_new": 2,
    }
    torch.testing.assert_close(model.means[:2], original_means[[2, 4]])
    torch.testing.assert_close(model.means[2:], extra["means"][1:])
    assert all(len(p) == len(orig_index) for p in model.params().values())


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


@pytest.fixture
def initial_frame_inputs(cameras):
    camera_path, camera_data = cameras
    np.savez(camera_path, **camera_data)
    np.savez(
        camera_path.with_name("depth.npz"),
        depth=np.ones((2, 3, 5)),
        supported=np.ones((2, 3, 5), dtype=bool),
        depth_index=[0, 1],
    )
    metadata = dict(scale0=2.0, origin_c2w=np.eye(4), R_raw_to_native=np.eye(3), fb=12)
    export = camera_path.with_name("splats.npz")
    initial = camera_path.with_name("alternative.npz")
    np.savez(export, **metadata, means=np.zeros((2, 3)))
    np.savez(initial, **metadata, means=np.zeros((5, 3)))
    return export, initial, metadata


def test_initial_frame_accepts_matching_metadata_with_different_model_count(initial_frame_inputs):
    export, initial, _ = initial_frame_inputs
    trainer.validate_initial_frame(initial, export)


@pytest.mark.parametrize("field", ["scale0", "origin_c2w", "R_raw_to_native", "fb"])
@pytest.mark.parametrize("mutation", ["missing", "mismatch", "shape", "nan", "string"])
@pytest.mark.parametrize("change_export", [False, True])
def test_initial_frame_rejects_invalid_metadata_before_cuda_and_output(
    initial_frame_inputs, monkeypatch, field, mutation, change_export
):
    export, initial, metadata = initial_frame_inputs
    if mutation == "missing":
        del metadata[field]
    elif mutation == "mismatch":
        metadata[field] = np.asarray(metadata[field]) + 1
    elif mutation == "shape":
        metadata[field] = np.asarray(metadata[field])[None]
    elif mutation == "nan":
        metadata[field] = np.full(np.shape(metadata[field]), np.nan)
    else:
        metadata[field] = np.full(np.shape(metadata[field]), "invalid")
    np.savez(export if change_export else initial, **metadata)
    out = export.parent / "out"
    monkeypatch.setattr(trainer.torch, "device", lambda *_: pytest.fail("Unexpected CUDA"))
    with pytest.raises(ValueError, match=field):
        trainer.main(
            [
                "--data",
                str(export.parent),
                "--out",
                str(out),
                "--poses",
                "0",
                "--offpath-frame",
                "1",
                "--init-npz",
                str(initial),
            ]
        )
    assert not out.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("scale0", 0.0), ("scale0", -1.0), ("fb", -1), ("fb", 256), ("fb", 12.0), ("fb", True)],
)
def test_initial_frame_rejects_invalid_scale_and_fractional_bits(
    initial_frame_inputs, field, value
):
    export, initial, metadata = initial_frame_inputs
    metadata[field] = value
    np.savez(initial, **metadata)
    with pytest.raises(ValueError, match=field):
        trainer.validate_initial_frame(initial, export)


@pytest.mark.parametrize("use_override", [False, True])
def test_initial_frame_preflight_preserves_default_and_accepts_valid_override(
    initial_frame_inputs, monkeypatch, use_override
):
    export, initial, _ = initial_frame_inputs
    if not use_override:
        monkeypatch.setattr(
            trainer, "validate_initial_frame", lambda *_: pytest.fail("No alternate initialization")
        )

    def stop_at_device(*_):
        raise RuntimeError("Reached device selection")

    monkeypatch.setattr(trainer.torch, "device", stop_at_device)
    arguments = [
        "--data",
        str(export.parent),
        "--out",
        str(export.parent / "out"),
        "--poses",
        "0",
        "--offpath-frame",
        "1",
    ]
    if use_override:
        arguments += ["--init-npz", str(initial)]
    with pytest.raises(RuntimeError, match="Reached device selection"):
        trainer.main(arguments)


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


def surface_packet(depths, weights):
    logs = depths.log()
    moments = torch.stack(((logs * weights).sum(), (logs.square() * weights).sum()))[None]
    return moments, weights.sum()[None]


def surface_loss(moments, alpha, target=2.0, alpha_weight=0.1):
    return trainer.surface_depth_loss(
        moments,
        alpha,
        torch.full_like(alpha, target),
        torch.ones_like(alpha, dtype=torch.bool),
        torch.ones_like(alpha, dtype=torch.bool),
        alpha_weight,
    )


def test_surface_resolves_equal_mean_depth_ambiguity_and_moves_each_surface():
    depths = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert (depths * weights).sum().item() == 2  # Existing expected-depth loss is zero.
    moments, alpha = surface_packet(depths, weights)
    loss, metrics = surface_loss(moments, alpha)
    expected = torch.sqrt(((depths.log() - np.log(2)) ** 2 * weights).sum() + 1e-6) - 0.001
    torch.testing.assert_close(loss, expected)
    assert metrics["spread"].item() > 0
    loss.backward()
    assert depths.grad[0] < 0 < depths.grad[1]  # Gradient descent moves both toward the target.
    assert torch.isfinite(depths.grad).all()


def test_surface_exact_depth_and_common_scale_invariance():
    depths = torch.tensor([1.0, 3.0], dtype=torch.float64)
    weights = torch.tensor([0.4, 0.6], dtype=torch.float64)
    baseline, _ = surface_loss(*surface_packet(depths, weights))
    scaled, _ = surface_loss(*surface_packet(depths * 17, weights), target=34)
    torch.testing.assert_close(baseline, scaled)
    exact, _ = surface_loss(*surface_packet(torch.tensor([2.0]), torch.ones(1)))
    assert exact.item() == pytest.approx(0, abs=1e-7)


def test_surface_fixed_support_and_opacity_gradients_do_not_reward_transparency():
    depths = torch.tensor([1.0, 3.0], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    moments, alpha = surface_packet(depths, weights)
    full, full_metrics = surface_loss(moments, alpha)
    opacity_scale = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    faint, faint_metrics = surface_loss(moments * opacity_scale, alpha * opacity_scale)
    assert faint.item() - full.item() == pytest.approx(0.08)
    assert full_metrics["count"] == faint_metrics["count"] == 1
    torch.testing.assert_close(full_metrics["residual"], faint_metrics["residual"])
    faint.backward()
    assert opacity_scale.grad.item() == pytest.approx(-0.1)
    empty, empty_metrics = surface_loss(torch.zeros(1, 2), torch.zeros(1), target=1)
    assert empty.item() == pytest.approx(0.1)
    assert empty_metrics["count"] == 1
    assert torch.isfinite(empty)


def test_surface_empty_selection_ignores_garbage_and_has_zero_gradients():
    moments = torch.tensor([[float("nan"), float("nan")]], requires_grad=True)
    alpha = torch.tensor([float("nan")], requires_grad=True)
    loss, metrics = trainer.surface_depth_loss(
        moments, alpha, torch.tensor([float("nan")]), torch.tensor([False]), torch.tensor([True])
    )
    assert loss.item() == 0
    assert metrics["count"] == 0
    loss.backward()
    assert torch.equal(moments.grad, torch.zeros_like(moments))
    assert torch.equal(alpha.grad, torch.zeros_like(alpha))


def test_surface_masks_person_and_unsupported_pixels_before_log():
    moments = torch.tensor([[0.0, 0.0], [float("nan"), 0.0], [float("nan"), 0.0]])
    loss, metrics = trainer.surface_depth_loss(
        moments,
        torch.tensor([1.0, float("nan"), float("nan")]),
        torch.tensor([1.0, 0.0, float("nan")]),
        torch.tensor([True, False, True]),
        torch.tensor([True, True, False]),
    )
    assert loss.item() == 0
    assert metrics["count"] == 1


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_surface_rejects_invalid_selected_target(target):
    with pytest.raises(ValueError, match="finite and positive"):
        surface_loss(torch.zeros(1, 2), torch.ones(1), target=target)


@pytest.mark.parametrize("which", ["moments", "alpha"])
def test_surface_rejects_nonfinite_selected_render(which):
    moments, alpha = torch.zeros(1, 2), torch.ones(1)
    if which == "moments":
        moments[0, 1] = float("nan")
    else:
        alpha[0] = float("inf")
    with pytest.raises(ValueError, match="moments and alpha must be finite"):
        surface_loss(moments, alpha)


def test_quaternion_tether_sign_magnitude_angle_and_gradients():
    q = torch.tensor([[1.0, 0, 0, 0], [-2.0, 0, 0, 0], [1.0, 1.0, 0, 0]], requires_grad=True)
    reference = torch.tensor([[3.0, 0, 0, 0]]).expand_as(q)
    penalty = trainer.quaternion_tether(q, reference)
    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 0.5]))
    torch.testing.assert_close(penalty, trainer.quaternion_tether(-q, reference))
    penalty.sum().backward()
    assert torch.isfinite(q.grad).all()
    assert q.grad[2, 0] < 0 < q.grad[2, 1]


@pytest.mark.parametrize("surface", [False, True])
@pytest.mark.parametrize("depth", [False, True])
def test_render_routes_moment_channels_and_expected_depth_last(monkeypatch, surface, depth):
    captured = {}

    def rasterization(**kwargs):
        captured.update(kwargs)
        colors = kwargs["colors"]
        output = 0.25 * colors.sum(0)[None, None, None]
        if kwargs["render_mode"] == "RGB+ED":
            output = torch.cat((output, torch.full((1, 1, 1, 1), 17.0)), dim=-1)
        return output, torch.full((1, 1, 1, 1), 0.25), {}

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=rasterization))
    model = trainer.Model(
        dict(
            means=np.array([[0, 0, 1]], dtype=np.float32),
            quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
            log_scales=np.zeros((1, 3), dtype=np.float32),
            logit_opacities=np.zeros(1, dtype=np.float32),
            f_dc=np.array([[0, 1, 2]], dtype=np.float32),
        ),
        "cpu",
    )
    model.means.requires_grad_(True)
    view = torch.eye(4)
    view[2, 3] = 1  # Features must use current camera-space depth, not world z.
    K = torch.tensor([[7.0, 0, 2], [0, 9.0, 3], [0, 0, 1]])
    rgb, alpha, rendered_depth, info = model.render(
        view, K, 5, 4, depth=depth, surface_depth=surface
    )
    assert captured["colors"].shape[-1] == (5 if surface else 3)
    assert captured["render_mode"] == ("RGB+ED" if depth else "RGB")
    torch.testing.assert_close(captured["Ks"][0], K)
    torch.testing.assert_close(rgb[0, 0], 0.25 * (0.5 + trainer.SH_C0 * model.f_dc[0]))
    assert alpha.shape == (1, 1, 1)
    if depth:
        assert rendered_depth.item() == 17
    else:
        assert rendered_depth is None
    if surface:
        logs = torch.tensor([np.log(2), np.log(2) ** 2], dtype=torch.float32)
        torch.testing.assert_close(info["log_depth_moments"][0, 0], 0.25 * logs)
        info["log_depth_moments"][..., 0].sum().backward()
        assert model.means.grad[0, 2].item() == pytest.approx(0.125)
    else:
        assert "log_depth_moments" not in info


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--lambda-surface-depth", "nan"),
        ("--lambda-surface-depth", "-1"),
        ("--surface-alpha-weight", "inf"),
        ("--surface-alpha-weight", "-0.1"),
        ("--lambda-quat-reg", "-1"),
        ("--lambda-quat-reg", "nan"),
        ("--sigma-pos", "0"),
        ("--clamp-pos", "-1"),
        ("--m-per-unit", "inf"),
    ],
)
def test_invalid_objective_arguments_fail_before_files_cuda_and_output(
    tmp_path, monkeypatch, flag, value
):
    def unexpected_device(*_):
        pytest.fail("Argument preflight must run before device selection")

    monkeypatch.setattr(trainer.torch, "device", unexpected_device)
    with pytest.raises(ValueError, match="must be"):
        trainer.main(
            ["--data", str(tmp_path / "absent"), "--out", str(tmp_path / "out"), flag, value]
        )
    assert not (tmp_path / "out").exists()


def test_invalid_selected_depth_fails_before_cuda_and_output(cameras, monkeypatch):
    from PIL import Image

    path, data = cameras
    np.savez(path, **data)
    depths = np.ones((2, 3, 5), dtype=np.float32)
    depths[1, 1, 1] = float("nan")
    np.savez(
        path.with_name("depth.npz"),
        depth=depths,
        supported=np.ones_like(depths, dtype=bool),
        depth_index=[0, 1],
    )
    masks = path.parent / "masks"
    masks.mkdir()
    for i in range(2):
        Image.fromarray(np.zeros((3, 5), dtype=np.uint8)).save(masks / f"{i:04d}.png")
    monkeypatch.setattr(
        trainer.torch, "device", lambda *_: pytest.fail("Unexpected CUDA selection")
    )
    with pytest.raises(ValueError, match="finite and positive"):
        trainer.main(
            [
                "--data",
                str(path.parent),
                "--out",
                str(path.parent / "out"),
                "--poses",
                "0",
                "--offpath-frame",
                "1",
                "--lambda-surface-depth",
                "1",
            ]
        )
    assert not (path.parent / "out").exists()


def test_surface_preflight_resizes_and_ignores_masked_invalid_depth(tmp_path):
    from PIL import Image

    (tmp_path / "masks").mkdir()
    # Nearest resize from two to four columns repeats each source column twice.
    Image.fromarray(np.array([[255, 255, 0, 0]], dtype=np.uint8)).save(
        tmp_path / "masks" / "0000.png"
    )
    reference = np.array([[[float("nan"), 2.0]]], dtype=np.float32)
    supported = np.ones_like(reference, dtype=bool)
    trainer.validate_surface_targets(tmp_path, reference, supported, np.array([0]), 4, 1)
    Image.fromarray(np.zeros((1, 4), dtype=np.uint8)).save(tmp_path / "masks" / "0000.png")
    supported[0, 0, 0] = False
    trainer.validate_surface_targets(tmp_path, reference, supported, np.array([0]), 4, 1)
    supported[0, 0, 0] = True
    with pytest.raises(ValueError, match="finite and positive"):
        trainer.validate_surface_targets(tmp_path, reference, supported, np.array([0]), 4, 1)


@pytest.mark.parametrize(
    "invalid", [[0.0, 0, 0, 0], [float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0]]
)
@pytest.mark.parametrize("bad_reference", [False, True])
def test_quaternion_tether_rejects_invalid_current_and_reference(invalid, bad_reference):
    valid = torch.tensor([[1.0, 0, 0, 0]])
    invalid = torch.tensor([invalid])
    current, reference = (valid, invalid) if bad_reference else (invalid, valid)
    with pytest.raises(ValueError, match="finite and nonzero"):
        trainer.quaternion_tether(current, reference)


@pytest.mark.parametrize("magnitude", [1e-30, 1e30])
def test_quaternion_tether_normalizes_extreme_finite_magnitudes(magnitude):
    rotated = torch.tensor([[magnitude, magnitude, 0, 0]])
    reference = torch.tensor([[1.0, 0, 0, 0]])
    torch.testing.assert_close(trainer.quaternion_tether(rotated, reference), torch.tensor([0.5]))


@pytest.mark.parametrize("use_override", [False, True])
@pytest.mark.parametrize(
    "invalid", [[0.0, 0, 0, 0], [float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0]]
)
def test_invalid_initial_quaternion_fails_before_cuda_and_output(
    cameras, monkeypatch, use_override, invalid
):
    path, data = cameras
    np.savez(path, **data)
    np.savez(
        path.with_name("depth.npz"),
        depth=np.ones((2, 3, 5)),
        supported=np.ones((2, 3, 5), dtype=bool),
        depth_index=[0, 1],
    )
    default = path.with_name("splats.npz")
    np.savez(default, quats=np.array([[1.0, 0, 0, 0]] if use_override else [invalid]))
    arguments = [
        "--data",
        str(path.parent),
        "--out",
        str(path.parent / "out"),
        "--poses",
        "0",
        "--offpath-frame",
        "1",
        "--lambda-quat-reg",
        "0.1",
    ]
    if use_override:
        override = path.with_name("override.npz")
        np.savez(override, quats=np.array([invalid]))
        arguments += ["--init-npz", str(override)]
    monkeypatch.setattr(
        trainer.torch, "device", lambda *_: pytest.fail("Unexpected CUDA selection")
    )
    with pytest.raises(ValueError, match="finite and nonzero"):
        trainer.main(arguments)
    assert not (path.parent / "out").exists()
