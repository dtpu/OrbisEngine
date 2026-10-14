"""Run-local browser delivery; no providers or reconstruction required."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from orchestrator.api.viewer import scene_manifest
from scripts.tests import test_orchestrator_api_submission as submissions

AUTH = submissions.AUTH


@pytest.fixture
def packaged():
    case = submissions.ProductionSubmissionTests()
    case.setUp()
    try:
        assert case.upload(run_id="viewer-run", filename="preview.mp4").status_code == 201
        root = case.app.state.runs._dir("viewer-run")
        described = json.loads((root / "run.json").read_text())
        name = described["name"]
        public = root / "public"
        world = public / "worlds" / f"{name}-4d"
        person = world / "person"
        person.mkdir(parents=True)
        (person / "sequence.json").write_text(json.dumps({"frames": ["frame_000.ply"]}))
        (person / "frame_000.ply").write_bytes(b"test-frame")
        (world / "cameras.json").write_text("{}")
        (world / "audio.json").write_text('{"source":{"hasAudio":false}}')
        source = public / "clips" / name / "source.mp4"
        source.parent.mkdir(parents=True)
        shutil.copyfile(described["source"], source)
        yield case, root, world, source
    finally:
        case.doCleanups()


def test_scene_asset_range_head_and_auth(packaged):
    case, root, world, source = packaged
    prefix = "/api/pipeline/runs/viewer-run/viewer/"
    assert case.client.get(prefix + "scene.json").status_code == 401
    response = case.client.get(prefix + "scene.json", headers=AUTH)
    assert response.status_code == 200
    params = response.json()["params"]
    assert params["world"] == "none" and params["place"] == "0"
    assert params["cam"] == "0,0,0,0,0,-1"
    assert params["ruler"] == "0"
    assert case.client.head(prefix + "scene.json", headers=AUTH).content == b""
    video = params["video"]
    response = case.client.get(video, headers={**AUTH, "Range": "bytes=2-7"})
    assert response.status_code == 206
    assert response.content == source.read_bytes()[2:8]
    assert response.headers["content-range"].startswith("bytes 2-7/")
    head = case.client.head(video, headers=AUTH)
    assert head.status_code == 200 and head.content == b""
    assert int(head.headers["content-length"]) == source.stat().st_size
    person = case.client.get(params["person"], headers=AUTH).json()
    frame = params["person"].rsplit("/", 1)[0] + "/" + person["frames"][0]
    assert case.client.get(frame, headers=AUTH).content == b"test-frame"
    progress = case.client.get("/api/pipeline/runs/viewer-run/progress", headers=AUTH).json()
    assert "viewer%2Fscene.json" in progress["viewerPath"]
    assert case.client.post(prefix + "scene.json", headers=AUTH).status_code == 405


def test_missing_package_hides_link_and_stale_height_is_recomputed(packaged):
    case, root, world, source = packaged
    (root / "state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "_placement": {
                        "metricScaleFactor": 2,
                        "groundPlaneOffset": 3,
                        "height": 3.4,
                        "marbleFloor": -6,
                    }
                }
            }
        )
    )
    params = scene_manifest(root)["params"]
    assert params["height"] == "0.85" and params["floor"] == "-1.5"
    (world / "person/frame_000.ply").unlink()
    assert scene_manifest(root) is None
    assert (
        case.client.get("/api/pipeline/runs/viewer-run/viewer/scene.json", headers=AUTH).status_code
        == 404
    )


def test_root_references_people_objects_audio_and_sidecars(packaged):
    case, root, world, source = packaged
    public_relative = world.relative_to(root / "public").as_posix()
    (world / "people.json").write_text(
        json.dumps({"people": [{"sequence": f"/{public_relative}/person/sequence.json"}]})
    )
    (world / "objects").mkdir()
    (world / "objects/model.ply").write_bytes(b"object")
    (world / "person/motion.f32").write_bytes(b"motion")
    (world / "objects.json").write_text(
        json.dumps({"appearance": {"model": f"/{public_relative}/objects/model.ply"}})
    )
    (world / "audio.json").write_text(
        json.dumps({"url": "/" + source.relative_to(root / "public").as_posix()})
    )
    params = scene_manifest(root)["params"]
    people = case.client.get(params["person"], headers=AUTH).json()
    assert people["people"][0]["sequence"] == "person/sequence.json"
    objects = case.client.get(params["objects"], headers=AUTH).json()
    assert objects["appearance"]["model"] == "objects/model.ply"
    base = params["person"].rsplit("/", 1)[0]
    assert case.client.get(base + "/person/motion.f32", headers=AUTH).content == b"motion"
    audio = case.client.get(base + "/audio.json", headers=AUTH).json()
    assert audio["url"].startswith("../../clips/")


def test_private_paths_traversal_cross_run_and_symlinks_are_rejected(packaged):
    case, root, world, source = packaged
    prefix = "/api/pipeline/runs/viewer-run/viewer/assets/"
    (root / "secret.json").write_text('{"private":true}')
    (world / "escape.json").symlink_to(root / "secret.json")
    for path in [
        "run.json",
        "../run.json",
        "%2e%2e/run.json",
        "worlds/other-4d/person/frame_000.ply",
        world.relative_to(root / "public").as_posix() + "//person/frame_000.ply",
        world.relative_to(root / "public").as_posix() + "/escape.json",
    ]:
        assert case.client.get(prefix + path, headers=AUTH).status_code == 404
    assert (
        case.client.get(prefix + source.relative_to(root / "public").as_posix()).status_code == 401
    )


@pytest.mark.parametrize("motion", ["legacy-invalid", {"file": "missing.f32"}, "absolute"])
def test_optional_compact_motion_does_not_hide_playable_plys(packaged, motion):
    case, root, world, source = packaged
    if motion == "absolute":
        motion = {
            "file": "/" + world.relative_to(root / "public").as_posix() + "/person/missing.f32"
        }
    (world / "person/sequence.json").write_text(
        json.dumps({"frames": ["frame_000.ply"], "motion": motion})
    )
    scene = scene_manifest(root)
    assert scene is not None
    response = case.client.get(scene["params"]["person"], headers=AUTH)
    assert response.status_code == 200
    assert response.json()["frames"] == ["frame_000.ply"]


def test_selected_world_does_not_fall_back_when_finetune_missing(packaged):
    case, root, world, source = packaged
    described = json.loads((root / "run.json").read_text())
    described["options"].update(marble="video", finetune=True)
    (root / "run.json").write_text(json.dumps(described))
    (root / "public" / f"marble-{described['name']}-clean.spz").write_bytes(b"base")
    assert scene_manifest(root) is None
    (root / "public" / f"marble-{described['name']}-finetuned.spz").write_bytes(b"final")
    assert scene_manifest(root)["params"]["world"].endswith("-finetuned.spz")


@pytest.mark.parametrize("frames", [[], "frame_000.ply", [None], ["missing.ply"]])
def test_secondary_person_must_have_complete_frames(packaged, frames):
    case, root, world, source = packaged
    (world / "person_01").mkdir()
    (world / "person_01/sequence.json").write_text(json.dumps({"frames": frames}))
    (world / "people.json").write_text(
        json.dumps(
            {
                "people": [
                    {"sequence": "person/sequence.json"},
                    {"directory": "person_01"},
                ]
            }
        )
    )
    assert scene_manifest(root) is None


def test_provider_metadata_overrides_old_summary_height(packaged):
    case, root, world, source = packaged
    described = json.loads((root / "run.json").read_text())
    described["options"]["marble"] = "image"
    (root / "run.json").write_text(json.dumps(described))
    name = described["name"]
    (root / "public" / f"marble-{name}-image.spz").write_bytes(b"world")
    (root / "state.json").write_text(
        json.dumps({"stages": {"_placement": {"height": 100, "metricScaleFactor": 100}}})
    )
    (root / "marble").mkdir()
    (root / "marble" / f"{name}-image-world.json").write_text(
        json.dumps(
            {
                "assets": {
                    "splats": {
                        "semantics_metadata": {
                            "metric_scale_factor": 0.5,
                            "ground_plane_offset": 0.25,
                        }
                    }
                }
            }
        )
    )
    params = scene_manifest(root)["params"]
    assert float(params["height"]) == pytest.approx(3.4)
    assert float(params["floor"]) == pytest.approx(-0.5)


def test_runner_prints_scoped_scene_link(packaged, monkeypatch, capsys):
    from types import SimpleNamespace
    from unittest.mock import Mock

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import run_clip

    case, root, world, source = packaged
    pipeline = object.__new__(run_clip.Pipeline)
    pipeline.ctx = root
    pipeline.name = json.loads((root / "run.json").read_text())["name"]
    pipeline.info = {"duration": 1, "frames": 12, "fps": 12}
    pipeline.a = SimpleNamespace(fps=12)
    pipeline.stages = []
    pipeline.state = run_clip.State(root / "state.json")
    pipeline.placement = Mock(return_value={})
    monkeypatch.setenv("WANDER_RUN_DIR", str(root))
    pipeline.summary(1)
    output = capsys.readouterr().out
    assert "TOTAL WALL" in output
    assert "fourd.html?scene=%2Fapi%2Fpipeline%2Fruns%2Fviewer-run%2Fviewer%2Fscene.json" in output
    assert "WANDER_API_TOKEN" in output
    assert "?world=/" not in output
