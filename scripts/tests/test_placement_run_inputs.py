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

import camera_gravity
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
    pipeline.clip = tmp_path / "source.mp4"
    pipeline.clip.write_bytes(b"source video")
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
    assert Path(command[command.index("--video") + 1]) == pipeline.clip
    assert "--fail-on-size" in command


def stub_placement_cli(tmp_path, monkeypatch, floor, size_factor, footage, held_out=0.03):
    """The real CLI/geometry solve with only a controlled silhouette fit result and, where given,
    a controlled footage measurement (scripts/camera_gravity.py). `held_out` is the fit's contact
    residual, in world units, with each fold's contact withheld."""
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

    def silhouette(*args, footprints=None, **kwargs):
        # The real CLI hands the fit each person's drawn splats, read on demand.
        assert len(footprints) == 1 and callable(footprints[0])
        fits.append(args)
        return {
            "person": {
                "sizeScale": size_factor,
                "heightRatioMedian": 1.0,
                "offsetUnits": [[0, 0, 0]] * 8,
                "contactResidualUnits": [0.03] * 8,
                "contactHeldOutUnits": [held_out] * 8,
                "edgeResidualsPx": [1.0] * 8,
                "footageRows": footage_rows,
            }
        }

    footage_rows = [
        dict(sourceIndex=i, top=200.0, bottom=330.0, left=310.0, right=330.0) for i in range(8)
    ]
    measured = []

    def measure_footage(video, rows, cams, normal):
        measured.append((video, rows, len(cams), list(normal)))
        image = 1.4 / 1.19 * (1.0 if footage == "agrees" else 0.8)
        return dict(
            registration=dict(bodyHeightsMedian=1.4 / 1.19),
            image=dict(
                bodyHeightsMedian=image,
                inliers=80,
                framesSupporting=8,
                framesSampled=8,
                segments=200,
                tiltFromFloorNormalDeg=0.4,
            ),
            tolerance=camera_gravity.TOLERANCE,
            minFrames=camera_gravity.MIN_FRAMES,
            minInliers=camera_gravity.MIN_INLIERS,
        )

    monkeypatch.setattr(camera_gravity, "measure_footage", measure_footage)
    monkeypatch.setattr(place_solve, "silhouette_fit", silhouette)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture: frames are read only by the measurement stub")
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
            *(["--video", str(video)] if footage else []),
        ],
    )
    return SimpleNamespace(
        report=report,
        fits=fits,
        measured=measured,
        masks=masks,
        video=video,
        footage_rows=footage_rows,
    )


@pytest.mark.parametrize(
    "floor,size_factor,footage,accepted",
    [
        (-1.0, 0.7, None, True),
        (-1.4, 2.0, None, False),
        (-1.4, 1.0, None, True),
        # A camera 1.18 body-heights up: outside the handheld band unless the footage measures it.
        (-1.4, 0.7, None, False),
        (-1.4, 0.7, "agrees", True),
        (-1.4, 0.7, "disagrees", False),
        # In band: the footage is never decoded.
        (-1.4, 1.0, "agrees", True),
        # The footage's height does not describe this fitted placement (0.41 body-heights).
        (-1.4, 2.0, "agrees", False),
    ],
)
def test_cli_gates_final_fit_and_preserves_report(
    tmp_path, monkeypatch, floor, size_factor, footage, accepted
):
    cli = stub_placement_cli(tmp_path, monkeypatch, floor, size_factor, footage)
    report, fits, measured, masks = cli.report, cli.fits, cli.measured, cli.masks
    video, footage_rows = cli.video, cli.footage_rows
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
    assert "footageRows" not in result["perPerson"]["person"]
    in_band = 1.15 <= size["cameraHeightM"] <= 1.85
    if footage and not in_band:
        assert measured == [(video, footage_rows, 8, pytest.approx([0.0, 1.0, 0.0]))]
        assert size["footageGeometry"]["verdict"]["verified"] is accepted
        assert size["basis"] == ("measured from footage" if accepted else "handheld band")
    else:
        assert measured == [] and "footageGeometry" not in size
        assert size["basis"] == "handheld band"


@pytest.mark.parametrize("held_out_m,ok", [(0.05, True), (0.01, False)])
def test_noise_floor_tests_the_held_out_contact_not_the_fit_residual(
    tmp_path, monkeypatch, held_out_m, ok
):
    # The stub's in-sample residual (0.03 units, about 3 cm) is above the 2.5 cm floor in both
    # cases. The verdict follows the residual with each fold's contact withheld instead.
    cli = stub_placement_cli(tmp_path, monkeypatch, -1.0, 0.7, None, held_out=held_out_m)
    place_solve.main()
    result = json.loads(cli.report.read_text())
    floor = result["noiseFloor"]
    held = result["residualCm"]["heldOut"]["person"]["rms"]
    assert held == pytest.approx(result["residualCm"]["after"]["person"]["rms"] * held_out_m / 0.03)
    assert (held >= floor["contactCm"]) is ok
    assert floor["ok"] is ok
    assert floor["refusals"] == (
        []
        if ok
        else [
            "person: with each fifth's contact held out, the fitted feet still sit closer to the "
            "floor than the contact noise floor can measure"
        ]
    )
    assert "contactHeldOutUnits" not in result["perPerson"]["person"]


@pytest.mark.parametrize("tracked,single", [(False, False), (False, True), (True, False)])
def test_runner_anchor_failure_records_current_run_inputs(tmp_path, monkeypatch, tracked, single):
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
    if single:
        (pipeline.ctx / "single-person-masks.npz").write_bytes(b"exact source masks")
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
    elif single:
        # The checks see the same exact source masks as the placement fit and the sidecar.
        expected["--masks"] = pipeline.ctx / "single-person-masks.npz"
    else:
        assert "--masks" not in command
    for flag, path in expected.items():
        assert Path(command[command.index(flag) + 1]) == path
    exported = exports[0]
    assert exported[1] == "scripts/export_observation_confidence.py"
    assert exported[exported.index("--scale") + 1] == "2.5"
    assert Path(exported[exported.index("--out") + 1]) == public / "retained-world-obs.bin"
    assert Path(command[command.index("--obs-sidecar") + 1]) == public / "retained-world-obs.bin"
    masked = tracked or single
    for flag in ("--world-dir", "--world", "--cameras", *(["--masks"] if masked else [])):
        assert Path(exported[exported.index(flag) + 1]) == expected[flag]
    if not masked:
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
    pipeline.clip = tmp_path / "source.mp4"
    pipeline.clip.write_bytes(b"source video")
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
