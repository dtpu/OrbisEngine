"""Final-world consumers must agree, including resumes with retained outputs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
import run_clip
import world_quality_review


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    p = object.__new__(run_clip.Pipeline)
    p.name = "fixture"
    p.ctx = tmp_path / "run"
    p.ctx.mkdir()
    p.clip = tmp_path / "source.mp4"
    p.clip.write_bytes(b"source")
    p.marble = "video"
    p.multi = False
    p.stages = ["scale_fit", "finetune", "place_fit", "anchors", "verify"]
    p.deps = {
        "scale_fit": [],
        "finetune": ["scale_fit"],
        "place_fit": ["scale_fit", "finetune"],
        "anchors": ["place_fit"],
        "verify": ["anchors", "finetune"],
    }
    p.a = SimpleNamespace(
        fps=12,
        only=None,
        force=None,
        skip_finetune=False,
        reuse_world=None,
        quality_plan=None,
        quality_result=None,
        quality_evidence_root=None,
        allow_bad_size=False,
        ruler_tol=0.15,
        ruler_vlm=False,
        allow_weak_anchor=False,
        gpu_box="fixture.invalid",
        gpu_key=None,
        ft_iters=1,
        ft_mask_dilate=1,
    )
    p.state = run_clip.State(p.ctx / "state.json")
    p.state.record("_scale", scale0=1.0)
    p.info = {"duration": 1, "frames": 30, "fps": 30}
    p.video_url = "/source.mp4"
    p.placement = Mock(return_value={"floor": -1, "fittedScale0": 1})
    p.cameras = Mock(return_value=p.ctx / "cameras.json")
    p.person_ply = Mock(return_value=p.ctx / "person.ply")
    p.anchor_samples = Mock(return_value="0,1")
    p.m_per_unit = Mock(return_value=1)
    p.n_cameras = Mock(return_value=2)
    p.vacant = Mock(return_value=False)
    for suffix in ("clean", "image", "multi", "finetuned"):
        (public / f"marble-fixture-{suffix}.spz").write_bytes(suffix.encode())
    wdir = public / "worlds/fixture-4d"
    wdir.mkdir(parents=True)
    (wdir / "cameras.json").write_text('{"cameras": []}')
    (wdir / "placement.json").write_text(
        json.dumps(
            {
                "registrationScale": 1,
                "samples": 2,
                "metresPerWorldUnit": 1,
                "surface": {
                    "sizeCheck": {"pass_": True, "cameraHeightM": 1.4, "floorToCeilingM": 3}
                },
                "residualCm": {"before": {}, "after": {}},
                "reprojectionCostPx": None,
            }
        )
    )
    return p


def accept_placement(p):
    world = p.world_spz()
    report = run_clip.public_dir() / "worlds/fixture-4d/placement.json"
    p.state.record("place_fit", status="ok")
    p.state.record(
        "_placement_solve",
        world=str(world.resolve()),
        worldSha256=run_clip.file_sha256(world),
        reportSha256=run_clip.file_sha256(report),
        sizeCheck={"pass_": True},
        residualCm={"before": {}, "after": {}},
        metresPerWorldUnit=1,
    )


def accept_anchors(p):
    placement = p.state.data["stages"]["_placement_solve"]
    p.state.record("anchors", status="ok")
    p.state.record(
        "_anchors",
        ok=True,
        worldSha256=placement["worldSha256"],
        placementSha256=placement["reportSha256"],
    )


@pytest.mark.parametrize(
    "mode,suffix", [("video", "clean"), ("both", "clean"), ("image", "image"), ("multi", "multi")]
)
def test_disabled_finetune_and_other_modes_cannot_select_retained_worlds(pipeline, mode, suffix):
    p = pipeline
    p.marble = mode
    p.stages.remove("finetune")
    p.state.record("finetune", status="ok")  # retained receipt from an earlier configuration
    assert p.world_spz().name == f"marble-fixture-{suffix}.spz"
    p.world_spz().unlink()
    with pytest.raises(RuntimeError, match="no Marble world"):
        p.world_spz()


@pytest.mark.parametrize("status", ["pending", "running", "failed", "skipped"])
def test_unfinished_finetune_cannot_use_partial_or_previous_output(pipeline, status, capsys):
    pipeline.state.record("finetune", status=status)
    with pytest.raises(RuntimeError, match="has not completed"):
        pipeline.world_spz()
    pipeline.summary(1)
    assert "http://127.0.0.1" not in capsys.readouterr().out


def test_success_without_finetuned_file_cannot_fall_back_to_base(pipeline):
    pipeline.state.record("finetune", status="ok")
    pipeline.world_spz().unlink()
    with pytest.raises(RuntimeError, match="output is missing"):
        pipeline.world_spz()


def test_no_world_mode_cannot_select_leftover_outputs(pipeline):
    pipeline.marble = "none"
    pipeline.stages = []
    with pytest.raises(RuntimeError, match="does not select"):
        pipeline.world_spz()


def test_all_final_consumers_use_finetuned_world(pipeline, monkeypatch, capsys):
    p = pipeline
    p.state.record("finetune", status="ok")
    final_world = p.world_spz()
    assert final_world.name.endswith("-finetuned.spz")
    invoke = Mock()
    monkeypatch.setattr(run_clip, "run", invoke)
    p.place_fit()
    p.state.record("place_fit", status="ok")
    assert p.placement_is_current(final_world)
    place_command = invoke.call_args.args[0]
    assert Path(place_command[place_command.index("--world") + 1]) == final_world

    def anchor_checks(command, **kwargs):
        assert Path(command[command.index("--world") + 1]) == final_world
        Path(command[command.index("--json-out") + 1]).write_text('{"ok": true}')
        return SimpleNamespace(returncode=0, stdout="checked")

    monkeypatch.setattr(run_clip.subprocess, "run", anchor_checks)
    p.anchors()
    p.state.record("anchors", status="ok")
    observation = invoke.call_args.args[0]
    assert Path(observation[observation.index("--world") + 1]) == final_world
    assert Path(observation[observation.index("--out") + 1]).name.endswith("finetuned-obs.bin")
    with pytest.raises(run_clip.QualityStop):
        p.verify()
    diagnostic = invoke.call_args.args[0]
    assert Path(diagnostic[diagnostic.index("--world") + 1]) == final_world

    p.a.quality_plan = "plan.json"
    p.a.quality_result = "result.json"
    p.a.quality_evidence_root = "evidence"
    assess = Mock(return_value={"status": "passed", "errors": []})
    monkeypatch.setattr(world_quality_review, "assess", assess)
    p.verify()
    assert assess.call_args.kwargs["world"] == final_world
    p.summary(1)
    link = next(line for line in capsys.readouterr().out.splitlines() if "http://127.0.0.1" in line)
    assert "world=/marble-fixture-finetuned.spz" in link and "&place=1" in link


def test_scale_and_finetune_use_original_world_even_when_refinement_is_retained(
    pipeline, monkeypatch
):
    p = pipeline
    p.state.record("finetune", status="ok")
    p.a.force = "finetune"
    commands = []

    def invoke(command, log, **kwargs):
        commands.append(command)
        if "--diag" in command:
            log.write_text("(ratio 1.000), p10 0.98 p90 1.02")

    monkeypatch.setattr(run_clip, "run", invoke)
    p.diag(1, 0)
    p.finetune()
    consumers = [command for command in commands if "--world" in command]
    assert {command[1] for command in consumers} == {
        "scripts/bake_video_colours.py",
        "scripts/finetune_export.py",
        "scripts/finetune_import.py",
    }
    for command in consumers:
        assert Path(command[command.index("--world") + 1]) == p.base_world_spz()


@pytest.mark.parametrize("mutation", ["world", "report", "legacy"])
def test_stale_placement_cannot_be_applied_or_used_by_anchors(
    pipeline, monkeypatch, capsys, mutation
):
    p = pipeline
    p.state.record("finetune", status="ok")
    accept_placement(p)
    if mutation == "world":
        p.world_spz().write_bytes(b"changed geometry")
    elif mutation == "report":
        (run_clip.public_dir() / "worlds/fixture-4d/placement.json").write_text("{}")
    else:
        p.state.record("_placement_solve", worldSha256=None, reportSha256=None)
    invoke = Mock()
    monkeypatch.setattr(run_clip, "run", invoke)
    with pytest.raises(RuntimeError, match="rerun place_fit"):
        p.anchors()
    invoke.assert_not_called()
    p.summary(1)
    assert "&place=1" not in capsys.readouterr().out


@pytest.mark.parametrize("trigger", ["world", "report", "forced_finetune", "unchanged"])
def test_resume_rechecks_changed_world_without_repeating_paid_work(pipeline, monkeypatch, trigger):
    p = pipeline
    for stage in p.stages:
        p.state.record(stage, status="ok")
    accept_placement(p)
    accept_anchors(p)
    if trigger == "world":
        p.world_spz().write_bytes(b"new geometry")
    elif trigger == "report":
        (run_clip.public_dir() / "worlds/fixture-4d/placement.json").write_text("{}")
    elif trigger == "forced_finetune":
        p.a.force = "finetune"
    executed = []
    monkeypatch.setattr(p, "stage_fn", lambda stage: lambda: executed.append(stage))
    monkeypatch.setattr(p, "summary", Mock())
    monkeypatch.setattr(run_clip.time, "sleep", lambda _: None)
    assert p.go() == 0
    expected = ["verify"] if trigger == "unchanged" else ["place_fit", "anchors", "verify"]
    if trigger == "forced_finetune":
        expected.insert(0, "finetune")
    assert executed == expected


@pytest.mark.parametrize("only", ["verify", "place_fit,verify", "finetune,verify"])
def test_subset_review_cannot_accept_stale_dependencies(pipeline, monkeypatch, only):
    p = pipeline
    for stage in p.stages:
        p.state.record(stage, status="ok")
    accept_placement(p)
    accept_anchors(p)
    p.a.only = only
    p.a.quality_plan, p.a.quality_result, p.a.quality_evidence_root = "plan", "result", "evidence"
    world = p.world_spz()
    if "finetune" in only:
        p.a.force = "finetune"
        monkeypatch.setattr(p, "finetune", lambda: world.write_bytes(b"refined geometry"))
    else:
        world.write_bytes(b"changed geometry")
    assess = Mock(return_value={"status": "passed", "errors": []})
    monkeypatch.setattr(world_quality_review, "assess", assess)
    monkeypatch.setattr(run_clip, "run", Mock())
    monkeypatch.setattr(p, "summary", Mock())
    monkeypatch.setattr(run_clip.time, "sleep", lambda _: None)
    assert p.go() == 1
    assess.assert_not_called()
    assert not p.state.done("verify")
    assert "rerun place_fit and anchors" in p.state.data["stages"]["verify"]["error"]


def test_resume_rechecks_anchors_after_placement_only_rerun(pipeline, monkeypatch):
    p = pipeline
    for stage in p.stages:
        p.state.record(stage, status="ok")
    accept_placement(p)
    accept_anchors(p)
    p.world_spz().write_bytes(b"new geometry")
    accept_placement(p)  # A prior --only place_fit measured the new file; anchors are still old.
    executed = []
    monkeypatch.setattr(p, "stage_fn", lambda stage: lambda: executed.append(stage))
    monkeypatch.setattr(p, "summary", Mock())
    monkeypatch.setattr(run_clip.time, "sleep", lambda _: None)
    assert p.go() == 0
    assert executed == ["anchors", "verify"]


@pytest.mark.parametrize(
    "mode,suffix", [("video", "clean"), ("both", "clean"), ("image", "image"), ("multi", "multi")]
)
def test_metadata_uses_selected_provider_world_including_zero_floor(
    pipeline, monkeypatch, tmp_path, mode, suffix
):
    p = pipeline
    p.marble = mode
    monkeypatch.setattr(run_clip, "MARBLE_DIR", tmp_path)
    for index, variant in enumerate(("clean", "image", "multi"), 1):
        (tmp_path / f"fixture-{variant}-world.json").write_text(
            json.dumps(
                {
                    "world_id": variant,
                    "world_marble_url": f"https://example.invalid/{variant}",
                    "assets": {
                        "splats": {
                            "semantics_metadata": {
                                "metric_scale_factor": index,
                                "ground_plane_offset": 0,
                            }
                        }
                    },
                }
            )
        )
    result = run_clip.Pipeline.placement(p)
    assert result["world"] == suffix
    assert result["worldUrl"] == f"https://example.invalid/{suffix}"
    assert result["metricScaleFactor"] == ("clean", "image", "multi").index(suffix) + 1
    assert result["marbleFloor"] == 0
