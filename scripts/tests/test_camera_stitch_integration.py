"""Camera and point correction reaches the actual worker/runner paths without inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import ast
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from worker.stages.dense_pi3x import batch_viewer_transforms, viewer_camera
import run_clip
from stage_attempts import command_identity


def remote_body():
    # Exercise the real function without constructing Modal images or volumes.
    path = ROOT / "worker/modal_motion.py"
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reconstruct"
    )
    node.decorator_list = []
    scope = {"Path": Path, "cache": Mock(), "INFERENCE_RESOURCES": {"timeout": 600}}
    for assignment in tree.body:
        if isinstance(assignment, ast.Assign):
            for target in assignment.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_REV"):
                    scope[target.id] = ast.literal_eval(assignment.value)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["reconstruct"]


@pytest.mark.parametrize("value,experiment", [(True, "vdpm"), ("yes", "pi3x"), (1, "pi3x")])
def test_remote_rejects_incompatible_options_before_files_or_subprocess(
    monkeypatch, value, experiment
):
    import tempfile

    workspace = Mock(side_effect=AssertionError("must reject before creating workspace"))
    monkeypatch.setattr(tempfile, "mkdtemp", workspace)
    with pytest.raises(ValueError, match="anchor_lock"):
        remote_body()(b"fixture", experiment, anchor_lock=value)
    workspace.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_remote_forwards_flag_to_actual_pi3x_command(tmp_path, monkeypatch, enabled):
    import tempfile
    import subprocess

    monkeypatch.setattr(tempfile, "mkdtemp", lambda **_: str(tmp_path))
    process = Mock(return_value=SimpleNamespace(stdout=["fixture output\n"], wait=lambda: 0))
    monkeypatch.setattr(subprocess, "Popen", process)
    result = remote_body()(b"fixture", "pi3x", anchor_lock=enabled)
    command = process.call_args.args[0]
    assert command[1] == "/root/worker/stages/dense_pi3x.py"
    assert command.count("--anchor-lock") == int(enabled)
    assert result["report"]["anchorLock"] is enabled
    assert result["report"]["error"] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_corrects_exported_points_with_their_own_camera(enabled):
    times = np.array([0.2, 0.5, 1.1])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, :3] = Rotation.from_rotvec([[0, 0.1, 0], [0, 0.2, 0], [0, 0.4, 0]]).as_matrix()
    poses[:, :3, 3] = [[0, 0, 0], [0.1, 0.2, 0], [0.3, 0.2, 0.4]]
    similarity = (0.7, Rotation.from_rotvec([0, 0, 0.2]).as_matrix(), np.array([1, -2, 0.3]))
    reference = (Rotation.from_rotvec([-0.1, 0, 0]).as_matrix(), np.array([-1, 0, 0.2]), 1.3)
    original = np.array([viewer_camera(p, *similarity, *reference) for p in poses])
    canonical = original[[0, -1]].copy()
    canonical[:, :3, 3] += [[0.4, 0.2, 0], [-0.2, 0.1, 0.3]]
    cameras, maps = batch_viewer_transforms(
        poses, similarity, reference, times, times[[0, -1]], canonical, anchor_lock=enabled
    )
    if enabled:
        np.testing.assert_array_equal(cameras[[0, -1]], canonical)
    else:
        np.testing.assert_array_equal(cameras, original)
        np.testing.assert_array_equal(maps, np.broadcast_to(np.eye(4), original.shape))
    raw = np.array([[0.2, 0.1, 2], [1, 0.4, 3]])
    s, r, t = similarity
    r0, t0, scale = reference
    old_world = ((raw @ r.T * s + t) @ r0.T + t0) * scale * [1, -1, -1]
    for old, new, transform in zip(original, cameras, maps):
        changed = old_world @ transform[:3, :3].T + transform[:3, 3]
        homogeneous = np.ones((len(raw), 1))
        np.testing.assert_allclose(
            np.c_[changed, homogeneous] @ np.linalg.inv(new).T,
            np.c_[old_world, homogeneous] @ np.linalg.inv(old).T,
            atol=1e-12,
        )


@pytest.mark.parametrize("enabled", [None, False, True])
def test_runner_forwards_opt_in_once_and_keeps_pose_guard(tmp_path, enabled):
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.ctx, pipeline.clip = tmp_path, tmp_path / "source.mp4"
    pipeline.a = SimpleNamespace(**({} if enabled is None else {"pi3x_anchor_lock": enabled}))
    pipeline.paid_run, pipeline.pose_guard = Mock(), Mock()
    pipeline.pi3x()
    pipeline.paid_run.assert_called_once()
    stage, command, _ = pipeline.paid_run.call_args.args
    assert stage == "pi3x"
    assert command.count("--anchor-lock") == int(enabled is True)
    pipeline.pose_guard.assert_called_once_with()


def test_paid_identity_tracks_anchor_lock_and_helper_code():
    command = ["modal", "run", "worker/modal_motion.py", "--experiment", "pi3x"]
    legacy, legacy_code, _ = command_identity(command, ROOT, environment={})
    locked, locked_code, _ = command_identity([*command, "--anchor-lock"], ROOT, environment={})
    assert "--anchor-lock" not in legacy["options"]
    assert locked["options"]["--anchor-lock"] is True
    assert locked_code == legacy_code
    assert "worker/wander_worker/camera_stitch.py" in locked_code
