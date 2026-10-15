"""Local packaging publication and command admission, with no provider calls."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_clip
from orchestrator.cli import run_clip_command
from scripts.tests import test_retained_scene as retained_fixtures

scene = retained_fixtures.scene


@pytest.fixture
def pipeline(scene, tmp_path):
    config, clip, _, fresh, _ = scene
    p = object.__new__(run_clip.Pipeline)
    p.name = "candidate"
    p.ctx = tmp_path / "run"
    p.ctx.mkdir()
    p.clip, p.people_limit, p.marble = clip, 2, "video"
    p.video_url = "/clips/candidate/source.mp4"
    p.a = SimpleNamespace(retained_scene=str(config), shared_scale="auto", skip_finetune=True)
    p.state = run_clip.State(p.ctx / "state.json")
    for track in range(2):
        motion = p.ctx / f"lhm-motion-{track:02d}"
        motion.mkdir()
        (motion / "sequence.json").write_text("{}")
    (p.ctx / "tracks").mkdir()
    (p.ctx / "tracks/tracks.json").write_text('{"duration": 1}')
    public = tmp_path / "public"
    public.mkdir()

    def package(command, log):
        assert command[command.index("--shared-scale") + 1] == "1"
        shutil.copytree(fresh, Path(command[command.index("--out") + 1]))

    with (
        patch.object(run_clip, "public_dir", return_value=public),
        patch.object(run_clip, "run", side_effect=package) as execute,
    ):
        yield p, public, execute


def test_delivers_package_and_selected_world_with_no_quality_acceptance(pipeline):
    p, public, execute = pipeline
    p.package_people()
    out = public / "worlds/candidate-4d"
    assert (out / "objects.json").is_file()
    assert (out / "audio/original.wav").is_file()
    assert (public / "marble-candidate-clean.spz").is_file()
    receipt = json.loads((p.ctx / "retained-delivery.json").read_text())
    assert receipt["acceptance"] == "none"
    assert set(p.state.data["stages"]) == {"_retained_delivery"}
    execute.assert_called_once()
    assert not list((public / "worlds").glob(".retained-*"))


@pytest.mark.parametrize("conflict", ["package", "world", "scale", "missing_reference", "finetune"])
def test_conflicts_preserve_existing_files_before_packaging(pipeline, conflict):
    p, public, execute = pipeline
    if conflict == "package":
        out = public / "worlds/candidate-4d"
        out.mkdir(parents=True)
        (out / "authored.txt").write_text("keep")
    elif conflict == "world":
        (public / "marble-candidate-clean.spz").write_bytes(b"another world")
    elif conflict == "scale":
        p.a.shared_scale = "0"
    elif conflict == "finetune":
        p.a.skip_finetune = False
    else:
        (p.ctx / "lhm-motion-01/sequence.json").unlink()
    before = {str(f): f.read_bytes() for f in public.rglob("*") if f.is_file()}
    with pytest.raises(ValueError):
        p.package_people()
    execute.assert_not_called()
    assert {str(f): f.read_bytes() for f in public.rglob("*") if f.is_file()} == before


def test_geometry_mismatch_cannot_publish_partial_scene(pipeline, scene):
    p, public, _ = pipeline
    fresh = scene[3]
    (fresh / "person/frame_000.ply").write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="fresh frame digest mismatch"):
        p.package_people()
    assert not (public / "worlds/candidate-4d").exists()
    assert not (public / "marble-candidate-clean.spz").exists()
    assert not list((public / "worlds").glob(".retained-*"))
    assert not p.state.data["stages"]


def test_direct_runner_link_uses_retained_presentation(pipeline, monkeypatch, capsys):
    p, public, _ = pipeline
    p.package_people()
    people_path = public / "worlds/candidate-4d/people.json"
    people = json.loads(people_path.read_text())
    for person in people["people"]:
        person["bodyHeightUnits"] = 1.0
    people_path.write_text(json.dumps(people))
    p.info, p.a.fps, p.stages = {"duration": 1, "frames": 2, "fps": 2}, 2, ["package_people"]
    monkeypatch.delenv("WANDER_RUN_DIR", raising=False)
    with (
        patch.object(p, "placement", return_value={}),
        patch.object(p, "world_spz", return_value=public / "marble-candidate-clean.spz"),
    ):
        p.summary(1)
    output = capsys.readouterr().out
    assert "&scene=%2Fworlds%2Fcandidate-4d%2Fscene.json&place=0" in output
    assert "no Marble world yet" not in output
    assert "reconstruction quality is unverified" in output


def test_failed_promotion_removes_only_new_world(pipeline, scene):
    p, public, _ = pipeline
    world = public / "marble-candidate-clean.spz"
    for existing in (False, True):
        if existing:
            shutil.copyfile(scene[4], world)
        with patch.object(Path, "rename", side_effect=OSError("promotion failed")):
            with pytest.raises(OSError, match="promotion failed"):
                p.package_people()
        assert world.exists() == existing
        assert not (public / "worlds/candidate-4d").exists()


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--people", "2"],
        ["--people", "2", "--only", "package_people"],
        ["--only", "package_people"],
        ["--people", "2", "--only", "package_people,objects"],
        ["--people", "2", "--only", "package_people", "--orchestrated"],
        ["--people", "2", "--only", "package_people", "--marble", "none"],
        ["--people", "2", "--only", "package_people", "--skip-marble"],
    ],
)
def test_unsupported_modes_stop_before_source_work(flags):
    with (
        patch.object(
            sys,
            "argv",
            [
                "run_clip.py",
                "--clip",
                "unused.mp4",
                "--name",
                "test",
                "--retained-scene",
                "config.json",
                *flags,
            ],
        ),
        patch.object(run_clip, "resolve_shots") as resolve,
        patch("sys.stderr"),
        pytest.raises(SystemExit) as exited,
    ):
        run_clip.main()
    assert exited.value.code == 2
    resolve.assert_not_called()


def test_stage_command_preserves_retained_config():
    context = SimpleNamespace(
        repository=Path(__file__).resolve().parents[2],
        source=Path("source.mp4"),
        name="candidate",
        root=Path("run"),
        options={"people": 2, "all_people": True, "marble": "video", "finetune": False},
    )
    command = run_clip_command(context, "package_people", ["--retained-scene", "config.json"])
    assert command[command.index("--only") + 1] == "package_people"
    assert "--skip-finetune" in command
    assert command[-2:] == ["--retained-scene", "config.json"]
