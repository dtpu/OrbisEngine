"""The opt-in colour bake stage: graph, runner command, world selection and mask input."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import bake_video_colours as bake  # noqa: E402
import run_clip  # noqa: E402
from orchestrator.api.viewer import world_suffix  # noqa: E402
from orchestrator.pipeline import multiperson_graph, single_graph  # noqa: E402
from orchestrator.steps import RunOptions, stage_graph  # noqa: E402


@pytest.mark.parametrize("graph", [single_graph, lambda *a, **k: multiperson_graph(2, *a, **k)])
def test_bake_is_opt_in_and_sits_between_scale_fit_and_placement(graph):
    stages, deps = graph("video", False, False)
    assert "bake" not in stages and deps["place_fit"] == ["scale_fit"]
    stages, deps = graph("video", False, False, bake=True)
    assert deps["bake"] == ["scale_fit"]
    assert deps["place_fit"] == ["scale_fit", "bake"]
    assert stages.index("scale_fit") < stages.index("bake") < stages.index("place_fit")


def test_bake_follows_finetune_when_both_are_requested():
    _, deps = single_graph("video", False, True, bake=True)
    assert deps["bake"] == ["scale_fit", "finetune"]
    assert deps["place_fit"] == ["scale_fit", "finetune", "bake"]


def test_a_run_without_a_world_has_no_bake():
    stages, _ = single_graph("none", False, False, bake=True)
    assert "bake" not in stages


def test_run_options_carry_bake_into_the_plan_and_runner_command():
    assert RunOptions().bake is False
    stages, _ = stage_graph({"bake": True, "finetune": False, "objects": False})
    assert "bake" in stages
    from orchestrator import cli

    run = SimpleNamespace(
        repository=ROOT,
        run_dir=Path("/tmp/run"),
        name="clip",
        source=Path("/tmp/run/source.mp4"),
        options={"bake": True, "objects": False},
        retained_delivery=None,
    )
    assert "--bake-colours" in cli.run_clip_command(run, "bake", [])
    run.options["bake"] = False
    assert "--bake-colours" not in cli.run_clip_command(run, "bake", [])


@pytest.mark.parametrize(
    "options,suffix",
    [
        ({"marble": "video"}, "clean"),
        ({"marble": "video", "finetune": True}, "finetuned"),
        ({"marble": "video", "bake": True}, "baked"),
        ({"marble": "image", "bake": True, "finetune": True}, "baked"),
    ],
)
def test_the_viewer_shows_the_requested_world_without_fallback(options, suffix):
    assert world_suffix(options) == suffix


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    p = object.__new__(run_clip.Pipeline)
    p.ctx = tmp_path / "run"
    (p.ctx / "pi3x").mkdir(parents=True)
    p.name = "fixture"
    p.clip = tmp_path / "source.mp4"
    p.state = run_clip.State(p.ctx / "state.json")
    p.a = SimpleNamespace(only=None, force=None, skip_finetune=True)
    p.stages = ["scale_fit", "bake", "place_fit"]
    p.deps = {"scale_fit": [], "bake": ["scale_fit"], "place_fit": ["scale_fit", "bake"]}
    p.marble = "video"
    public = tmp_path / "public"
    (public / "worlds" / "fixture-4d").mkdir(parents=True)
    (public / "marble-fixture-clean.spz").write_bytes(b"generated")
    monkeypatch.setattr(run_clip, "public_dir", lambda: public)
    share = tmp_path / "share"
    share.mkdir()
    monkeypatch.setattr(run_clip, "SHARE", share)
    p.cameras = Mock(return_value=p.ctx / "pi3x" / "cameras.json")
    p.person_ply = Mock(return_value=public / "worlds" / "fixture-4d" / "person" / "frame_000.ply")
    p.anchor_samples = Mock(return_value="0,1")
    p.state.record("_scale", scale0=0.75)
    return p, public


def fake_bake(monkeypatch, commands):
    def invoke(command, log, **kwargs):
        commands.append((command, kwargs))
        if "single_person_masks.py" in command[1]:
            Path(command[command.index("--out") + 1]).write_bytes(b"masks")
            return
        out = Path(command[command.index("--out") + 1])
        out.with_suffix(".spz").write_bytes(b"baked")
        n = 4
        np.savez(
            f"{out}-bake.npz",
            depthGated=np.array([True, False, True, False]),
            gatedChroma=np.array([True, False, False, False]),
            inferredShift=np.array([False, True, False, False]),
            count=np.zeros(n, np.int16),
        )

    monkeypatch.setattr(run_clip, "run", invoke)


def test_bake_stage_uses_native_cameras_masks_and_the_reviewed_recipe(pipeline, monkeypatch):
    p, public = pipeline
    commands = []
    fake_bake(monkeypatch, commands)
    p.bake()
    (masks_cmd, _), (cmd, kwargs) = commands
    assert "single_person_masks.py" in masks_cmd[1]
    assert kwargs["attempts"] == 1
    value = lambda flag: cmd[cmd.index(flag) + 1]  # noqa: E731
    assert value("--world") == str(public / "marble-fixture-clean.spz")
    assert value("--cameras") == str(p.ctx / "pi3x" / "cameras.json")
    assert value("--anchors") == str(p.ctx / "pi3x" / "anchors.npz")
    assert value("--person-masks") == str(p.ctx / "single-person-masks.npz")
    assert value("--scale0") == "0.750000"
    assert value("--depth-agreement") == str(run_clip.BAKE_DEPTH_AGREEMENT)
    assert value("--shift-radius-body-heights") == str(run_clip.BAKE_SHIFT_RADIUS_BODY_HEIGHTS)
    assert "--gated-chroma" in cmd and "--no-ply" in cmd
    assert value("--out") == str(public / "marble-fixture-baked")
    record = json.loads(p.state.path.read_text())["stages"]["_bake"]
    assert record["depthGated"] == 2 and record["gatedChroma"] == 1
    assert record["inferredShift"] == 1
    assert record["inputSha256"] == run_clip.file_sha256(public / "marble-fixture-clean.spz")
    assert record["outSha256"] == run_clip.file_sha256(public / "marble-fixture-baked.spz")


def test_tracked_runs_bake_with_the_tracked_masks(pipeline, monkeypatch):
    p, public = pipeline
    (public / "worlds" / "fixture-4d" / "people.json").write_text("{}")
    (p.ctx / "tracks").mkdir()
    (p.ctx / "tracks" / "masks.npz").write_bytes(b"tracked")
    commands = []
    fake_bake(monkeypatch, commands)
    p.bake()
    ((cmd, _),) = commands
    assert cmd[cmd.index("--person-masks") + 1] == str(p.ctx / "tracks" / "masks.npz")


def test_bake_needs_a_fitted_scale(pipeline, monkeypatch):
    p, _ = pipeline
    p.state = run_clip.State(p.ctx / "fresh-state.json")
    monkeypatch.setattr(run_clip, "run", Mock(side_effect=AssertionError("must not run")))
    with pytest.raises(RuntimeError, match="scale0"):
        p.bake()


def test_downstream_stages_wait_for_and_then_use_the_baked_world(pipeline, monkeypatch):
    p, public = pipeline
    with pytest.raises(RuntimeError, match="colour bake has not completed"):
        p.world_spz()
    commands = []
    fake_bake(monkeypatch, commands)
    p.bake()
    p.state.record("bake", status="ok")
    monkeypatch.setattr(p.state, "done", lambda stage: stage == "bake")
    assert p.world_spz() == public / "marble-fixture-baked.spz"
    assert p.unbaked_world_spz() == public / "marble-fixture-clean.spz"
    p.stages = ["scale_fit", "place_fit"]
    assert p.world_spz() == public / "marble-fixture-clean.spz"


def mask_archive(path, *, tracked=False, status=None):
    masks = np.zeros((2, 40, 16), bool)
    masks[0, 20, 8] = True
    masks[1, 5:7, 2:4] = True
    packed = np.packbits(masks, axis=-1)
    if tracked:
        np.savez(path, masks=packed, shape=np.array(masks.shape), source_indices=np.array([4, 6]))
    else:
        extra = {"status": np.array(status)} if status is not None else {}
        np.savez(
            path,
            packedMasks=packed,
            shape=np.array(masks.shape),
            sourceIndices=np.array([4, 6]),
            **extra,
        )
    return masks


@pytest.mark.parametrize("tracked", [False, True])
def test_person_masks_are_read_by_source_index_and_dilated(tmp_path, tracked):
    masks = mask_archive(tmp_path / "m.npz", tracked=tracked)
    plain = bake.PersonMasks(tmp_path / "m.npz", 0.0)
    np.testing.assert_array_equal(plain.get(4), masks[0].astype(np.float32))
    np.testing.assert_array_equal(plain.get(6), masks[1].astype(np.float32))
    grown = bake.PersonMasks(tmp_path / "m.npz", 2 / 40).get(4)
    assert grown[20, 8] == 1 and grown[22, 8] == 1 and grown[20, 10] == 1
    assert grown[23, 8] == 0 and grown[21, 10] == 0


def test_frames_without_a_known_mask_are_fully_masked(tmp_path):
    mask_archive(tmp_path / "m.npz", status=["known", "lost"])
    masks = bake.PersonMasks(tmp_path / "m.npz", 0.0)
    assert masks.get(4).sum() == 1
    assert masks.get(6).min() == 1
    assert masks.get(99).min() == 1


def test_inconsistent_mask_archives_are_refused(tmp_path):
    np.savez(
        tmp_path / "bad.npz",
        packedMasks=np.zeros((2, 4, 3), np.uint8),
        shape=np.array([2, 4, 16]),
        sourceIndices=np.array([0, 2]),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        bake.PersonMasks(tmp_path / "bad.npz")
    np.savez(tmp_path / "other.npz", frames=np.zeros(3))
    with pytest.raises(ValueError, match="not a pipeline person-mask archive"):
        bake.PersonMasks(tmp_path / "other.npz")


def test_recoloured_worlds_are_byte_reproducible(tmp_path):
    n = 3
    spz = {
        "header": b"h" * 16,
        "pos_b": np.zeros((n, 3, 3), np.uint8),
        "alpha_b": np.zeros(n, np.uint8),
        "scale_b": np.zeros((n, 3), np.uint8),
        "rot_b": np.zeros((n, 3), np.uint8),
    }
    colours = np.full((n, 3), 0.4)
    bake.write_spz_recolour(spz, colours, tmp_path / "a.spz")
    bake.write_spz_recolour(spz, colours, tmp_path / "elsewhere-b.spz")
    assert (tmp_path / "a.spz").read_bytes() == (tmp_path / "elsewhere-b.spz").read_bytes()
