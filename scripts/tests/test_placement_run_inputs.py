"""Run-specific placement inputs and final, rather than preliminary, size acceptance."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import place_solve
import run_clip


def package_inputs(world, context, tracked=False):
    (world / "person").mkdir(exist_ok=True)
    (world / "person/sequence.json").write_text('{"frames": ["frame_000.ply"]}')
    (world / "person/frame_000.ply").write_bytes(b"packaged person fixture")
    (world / "cameras.json").write_text('{"cameras": []}')
    context.mkdir(parents=True, exist_ok=True)
    if tracked:
        (world / "people.json").write_text(
            '{"people": [{"id": "person", "track": 0, "sequence": "person/sequence.json"}]}'
        )
        (context / "tracks").mkdir()
        (context / "tracks/masks.npz").write_bytes(b"tracked masks fixture")


def test_masks_follow_configured_run_root_and_explicit_missing_masks_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDER_RUNS_DIR", raising=False)
    monkeypatch.setenv("WANDER_RUNS_ROOT", str(tmp_path / "active-runs"))
    assert place_solve.tracked_masks_path("clip") == tmp_path / "active-runs/clip/tracks/masks.npz"
    monkeypatch.setenv("WANDER_RUNS_DIR", str(tmp_path / "direct-runs"))
    assert place_solve.tracked_masks_path("clip") == tmp_path / "direct-runs/clip/tracks/masks.npz"
    supplied = tmp_path / "reviewed-masks.npz"
    with pytest.raises(FileNotFoundError, match="rerun the tracking stage"):
        place_solve.tracked_masks_path("clip", supplied)
    supplied.write_bytes(b"retained masks")
    assert place_solve.tracked_masks_path("clip", supplied) == supplied


def test_runner_uses_its_own_tracking_masks(tmp_path, monkeypatch):
    public = tmp_path / "external-public"
    world = public / "worlds/example-4d"
    world.mkdir(parents=True)
    (world / "people.json").write_text("{}")
    (public / "world.spz").write_bytes(b"world")
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.name = "example"
    pipeline.ctx = tmp_path / "separate-runs/example"
    package_inputs(world, pipeline.ctx, tracked=True)
    pipeline.state = SimpleNamespace(data={"stages": {"_scale": {"scale0": 1.0}}}, record=Mock())
    pipeline.a = SimpleNamespace(allow_bad_size=False)
    pipeline.placement = Mock(return_value={"floor": -1.0})
    pipeline.world_spz = Mock(return_value=public / "world.spz")
    commands = []

    def invoke(command, *args, **kwargs):
        commands.append(command)
        (world / "placement.json").write_text(
            json.dumps(
                {
                    "samples": 2,
                    "metresPerWorldUnit": 1,
                    "surface": {
                        "sizeCheck": {"cameraHeightM": 1.4, "floorToCeilingM": 3, "pass_": True}
                    },
                    "residualCm": {"before": {}, "after": {}},
                    "reprojectionCostPx": None,
                }
            )
        )

    monkeypatch.setattr(run_clip, "run", invoke)
    pipeline.place_fit()
    command = commands[0]
    assert Path(command[command.index("--masks") + 1]) == pipeline.ctx / "tracks/masks.npz"
    assert "--fail-on-size" in command


@pytest.mark.parametrize(
    "floor,size_factor,accepted", [(-1.0, 0.7, True), (-1.4, 2.0, False), (-1.4, 1.0, True)]
)
def test_cli_gates_final_fit_and_preserves_report(
    tmp_path, monkeypatch, floor, size_factor, accepted
):
    """Exercise the real CLI/geometry solve; inject only a controlled silhouette fit result."""
    world = tmp_path / "external-world"
    world.mkdir()
    cameras = world / "cameras.json"
    cameras.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "sourceIndex": i,
                        "camera_to_world": np.eye(4).tolist(),
                        "source_intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                        "source_image_size": [640, 480],
                    }
                    for i in range(8)
                ]
            }
        )
    )
    sequence = world / "person/sequence.json"
    sequence.parent.mkdir()
    sequence.write_text(json.dumps({"sourceIndices": list(range(8))}))
    people = world / "people.json"
    people.write_text(json.dumps({"people": [{"id": "person", "track": 0}]}))
    masks = tmp_path / "masks.npz"
    masks.write_bytes(b"fixture: silhouette measurements supplied by the fit stub")
    rng = np.random.default_rng(0)
    points = np.column_stack(
        [
            rng.uniform(-0.2, 0.2, 800),
            rng.uniform(floor, floor + 1.7, 800),
            rng.uniform(-2.1, -1.9, 800),
        ]
    )
    frames = [(points.copy(), points.copy()) for _ in range(8)]
    ground = np.column_stack(
        [rng.uniform(-1, 1, 2000), np.full(2000, floor), rng.uniform(-3, -1, 2000)]
    )
    ceiling = ground + [0, 2.8, 0]
    monkeypatch.setattr(place_solve, "person_frames", lambda _: iter(frames))
    monkeypatch.setattr(place_solve, "world_splats", lambda _: np.concatenate([ground, ceiling]))
    fits = []

    def silhouette(*args):
        fits.append(args)
        return {
            "person": {
                "sizeScale": size_factor,
                "heightRatioMedian": 1.0,
                "offsetUnits": [[0, 0, 0]] * 8,
                "contactResidualUnits": [0.03] * 8,
                "edgeResidualsPx": [1.0] * 8,
            }
        }

    monkeypatch.setattr(place_solve, "silhouette_fit", silhouette)
    report = tmp_path / "placement.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "place_solve.py",
            "--clip",
            "external",
            "--world",
            str(world / "room.spz"),
            "--people",
            str(people),
            "--cameras",
            str(cameras),
            "--scale",
            "1",
            "--floor",
            str(floor),
            "--masks",
            str(masks),
            "--out",
            str(report),
            "--fail-on-size",
        ],
    )
    if accepted:
        place_solve.main()
    else:
        with pytest.raises(SystemExit, match="SIZE CHECK FAILED after placement"):
            place_solve.main()
    assert len(fits) == 1
    result = json.loads(report.read_text())
    assert result["surface"]["sizeCheck"]["pass_"] is accepted
    size = result["surface"]["sizeCheck"]
    assert size["ceilingPlausible"] is (2.10 <= size["floorToCeilingM"] <= 3.60)
    assert result["perPerson"]["person"]["sizeScale"] == size_factor
    assert result["silhouetteFit"]["masks"] == str(masks)


@pytest.mark.parametrize("tracked", [False, True])
def test_runner_anchor_failure_records_current_run_inputs(tmp_path, monkeypatch, tracked):
    public = tmp_path / "external-public"
    world = public / "worlds/example-4d"
    world.mkdir(parents=True)
    (world / "placement.json").write_text('{"registrationScale": 2.5}')
    if tracked:
        (world / "people.json").write_text("{}")
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.name = "example"
    pipeline.clip = tmp_path / "upload.mp4"
    pipeline.ctx = tmp_path / "separate-runs/example"
    pipeline.ctx.mkdir(parents=True)
    package_inputs(world, pipeline.ctx, tracked=tracked)
    pipeline.clip.write_bytes(b"source video")
    pipeline.state = run_clip.State(pipeline.ctx / "state.json")
    pipeline.a = SimpleNamespace(
        ruler_tol=0.15, ruler_vlm=False, allow_weak_anchor=False, allow_bad_size=False
    )
    pipeline.placement = Mock(return_value={"floor": -1})
    pipeline.state.record("_scale", scale0=2.5)
    pipeline.world_spz = Mock(return_value=public / "retained-world.spz")
    pipeline.world_spz().write_bytes(b"retained world")
    pipeline.state.record("place_fit", status="ok")
    pipeline.state.record(
        "_placement_solve",
        world=str(pipeline.world_spz().resolve()),
        worldSha256=run_clip.file_sha256(pipeline.world_spz()),
        reportSha256=run_clip.file_sha256(world / "placement.json"),
        inputReceipt=pipeline.placement_inputs(pipeline.world_spz()),
    )
    commands = []
    exports = []

    def export_observation(command, *args, **kwargs):
        exports.append(command)
        Path(command[command.index("--out") + 1]).write_bytes(b"observation measurement")

    def invoke(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--json-out") + 1]).write_text(
            json.dumps({"ok": False, "failed": ["ruler"]})
        )
        return SimpleNamespace(returncode=3, stdout="ruler failed")

    monkeypatch.setattr(run_clip.subprocess, "run", invoke)
    monkeypatch.setattr(run_clip, "run", export_observation)
    with pytest.raises(RuntimeError, match="anchor checks failed: ruler"):
        pipeline.anchors()
    command = commands[0]
    expected = {
        "--world-dir": world,
        "--world": public / "retained-world.spz",
        "--cameras": world / "cameras.json",
        "--placement": world / "placement.json",
        "--video": pipeline.clip,
    }
    if tracked:
        expected["--masks"] = pipeline.ctx / "tracks/masks.npz"
    else:
        assert "--masks" not in command
    for flag, path in expected.items():
        assert Path(command[command.index(flag) + 1]) == path
    exported = exports[0]
    assert exported[1] == "scripts/export_observation_confidence.py"
    assert exported[exported.index("--scale") + 1] == "2.5"
    assert Path(exported[exported.index("--out") + 1]) == public / "retained-world-obs.bin"
    assert Path(command[command.index("--obs-sidecar") + 1]) == public / "retained-world-obs.bin"
    for flag in ("--world-dir", "--world", "--cameras", *(["--masks"] if tracked else [])):
        assert Path(exported[exported.index(flag) + 1]) == expected[flag]
    if not tracked:
        assert "--masks" not in exported
    assert "--no-gate" not in command
    assert pipeline.state.data["stages"]["_anchors"]["failed"] == ["ruler"]


@pytest.mark.parametrize(
    "stage_status,size_pass,placement_exists,applied",
    [
        ("ok", True, True, True),
        ("failed", True, True, False),
        ("ok", False, True, False),
        ("ok", True, False, False),
    ],
)
def test_summary_links_to_the_accepted_placement(
    tmp_path, monkeypatch, capsys, stage_status, size_pass, placement_exists, applied
):
    from urllib.parse import parse_qs, urlsplit

    public = tmp_path / "external-public"
    world = public / "worlds/example-4d"
    world.mkdir(parents=True)
    (public / "marble-example-multi.spz").write_bytes(b"world")
    if placement_exists:
        (world / "placement.json").write_text("{}")
    monkeypatch.setenv("WANDER_PUBLIC_DIR", str(public))
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.name = "example"
    pipeline.marble = "multi"
    pipeline.info = {"duration": 1, "frames": 30, "fps": 30}
    pipeline.a = SimpleNamespace(fps=12, allow_bad_size=False)
    pipeline.ctx = tmp_path / "run"
    package_inputs(world, pipeline.ctx)
    pipeline.stages = ["place_fit"]
    pipeline.video_url = "/clips/example.mp4"
    pipeline.placement = Mock(return_value={"floor": -1, "fittedScale0": 2})
    pipeline.state = run_clip.State(tmp_path / "state.json")
    pipeline.state.record("place_fit", status=stage_status)
    pipeline.state.record("_scale", scale0=2)
    pipeline.state.record(
        "_placement_solve",
        world=str((public / "marble-example-multi.spz").resolve()),
        worldSha256=run_clip.file_sha256(public / "marble-example-multi.spz"),
        reportSha256=(
            run_clip.file_sha256(world / "placement.json") if placement_exists else "missing"
        ),
        sizeCheck={"pass_": size_pass},
        residualCm={"before": {}, "after": {}},
        metresPerWorldUnit=1,
        inputReceipt=pipeline.placement_inputs(public / "marble-example-multi.spz"),
    )
    pipeline.summary(1)
    output = capsys.readouterr().out
    link = next(line.strip() for line in output.splitlines() if "http://127.0.0.1" in line)
    query = parse_qs(urlsplit(link).query)
    assert query["world"] == ["/marble-example-multi.spz"]
    assert (query.get("place") == ["1"]) is applied
    assert ("placement applied" in output) is applied
