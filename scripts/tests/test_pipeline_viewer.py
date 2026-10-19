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


def test_exported_presentation_defaults_preserve_recovered_scene_alignment(packaged):
    case, root, world, source = packaged
    (root / "state.json").write_text(
        json.dumps({"stages": {"_placement": {"metricScaleFactor": 2}}})
    )
    defaults = {
        "scale": "1.08",
        "floor": "-0.61",
        "pos": "0,0,0",
        "feetmode": "sfm",
        "feetlock": "0",
        "camdrift": "1",
        "cam": "-0.95,0.42,0.52,0.12,-0.16,-1.60",
        "box": "-1,-0.4,-3.275,1.75,0.5,0.6",
    }
    (world / "scene.json").write_text(
        json.dumps({"schema": "wander.viewer-scene/1", "params": defaults})
    )
    response = case.client.get("/api/pipeline/runs/viewer-run/viewer/scene.json", headers=AUTH)
    assert response.status_code == 200
    params = response.json()["params"]
    assert all(params[key] == value for key, value in defaults.items())
    assert "height" not in params
    assert params["place"] == "0"
    assert params["person"].startswith("/api/pipeline/runs/viewer-run/viewer/assets/")
    (world / "person/frame_000.ply").unlink()
    assert scene_manifest(root) is None


@pytest.mark.parametrize(
    "defaults",
    [
        {"world": "https://example.com/world.spz"},
        {"person": "/worlds/another-run/people.json"},
        {"demo": "elevator"},
        {"scene": "/other.json"},
        {"place": "1"},
        {"scale": 1.08},
        {"scale": "1\n"},
        {"scale": ""},
        {"scale": "garbage"},
        {"scale": "0"},
        {"height": "-1"},
        {"scale": "Infinity"},
        {"scale": "1_000"},
        {"floor": "NaN"},
        {"floor": "1e309"},
        {"pos": "0,1"},
        {"cam": "bad"},
        {"cam": "0,0,0,0,0,0"},
        {"box": "1,1,1,-1,-1,-1"},
        {"feetmode": "unknown"},
        {"feetlock": "false"},
        {"fovsrc": "180"},
        {"scale": "1", "height": "1.7"},
    ],
)
def test_exported_defaults_cannot_replace_assets_or_bypass_delivery_checks(packaged, defaults):
    case, root, world, source = packaged
    (world / "scene.json").write_text(
        json.dumps({"schema": "wander.viewer-scene/1", "params": defaults})
    )
    assert scene_manifest(root) is None


def test_exported_height_replaces_inferred_scale(packaged):
    case, root, world, source = packaged
    (world / "scene.json").write_text(
        json.dumps({"schema": "wander.viewer-scene/1", "params": {"height": "1.7"}})
    )
    params = scene_manifest(root)["params"]
    assert params["height"] == "1.7"
    assert "scale" not in params


def test_exported_defaults_symlink_cannot_escape_public_assets(packaged):
    case, root, world, source = packaged
    private = root / "private-scene.json"
    private.write_text(json.dumps({"schema": "wander.viewer-scene/1", "params": {"scale": "2"}}))
    (world / "scene.json").symlink_to(private)
    assert scene_manifest(root) is None


@pytest.mark.parametrize(
    "contents", ["broken", "{}", '{"schema":"wander.viewer-scene/1","params":[]}']
)
def test_invalid_exported_scene_is_not_silently_ignored(packaged, contents):
    case, root, world, source = packaged
    (world / "scene.json").write_text(contents)
    assert scene_manifest(root) is None


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


def declare_source_delivery(root, world):
    import hashlib

    source_sha = json.loads((root / "run.json").read_text())["sourceSha256"]
    media = world / "media"
    media.mkdir()
    receipt = {"schema": "wander.source-media/1", "sourceSha256": source_sha}
    for kind, name in (("video", "source.mp4"), ("audio", "original.wav")):
        data = f"fixture {kind}".encode()
        (media / name).write_bytes(data)
        receipt[kind] = {
            "url": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "mode": "stream-copy",
        }
    (media / "provenance.json").write_text(json.dumps(receipt))
    manifest = {
        "schema": "wander.audio/1",
        "source": {"hasAudio": False, "sha256": source_sha, "videoUrl": "media/source.mp4"},
        "original": {"url": "media/original.wav", "offsetSeconds": 0},
        "provenance": {"delivery": "media/provenance.json"},
        "timeline": {"durationSeconds": 1},
    }
    (world / "audio.json").write_text(json.dumps(manifest))
    return manifest


def test_verified_lossless_video_selected_and_range_served(packaged):
    case, root, world, source = packaged
    declare_source_delivery(root, world)
    params = scene_manifest(root)["params"]
    assert params["video"].endswith("/media/source.mp4")
    response = case.client.get(params["video"], headers={**AUTH, "Range": "bytes=0-6"})
    assert response.status_code == 206 and response.content == b"fixture"
    assert response.headers["content-type"] == "video/mp4"
    base = params["video"].rsplit("/", 1)[0]
    assert case.client.get(base + "/original.wav", headers=AUTH).content == b"fixture audio"
    # The immutable submitted/exported source is still required even with a derivative.
    source.unlink()
    assert scene_manifest(root) is None


@pytest.mark.parametrize(
    "failure",
    [
        "video",
        "audio",
        "receipt",
        "source-sha",
        "output-sha",
        "traversal",
        "cross-run",
        "symlink",
        "directory-symlink",
        "source-shape",
        "provenance-shape",
    ],
)
def test_invalid_lossless_delivery_hides_preview(packaged, failure):
    case, root, world, source = packaged
    manifest = declare_source_delivery(root, world)
    media = world / "media"
    if failure in {"video", "audio", "receipt"}:
        (
            media
            / {"video": "source.mp4", "audio": "original.wav", "receipt": "provenance.json"}[
                failure
            ]
        ).unlink()
    elif failure == "source-sha":
        manifest["source"]["sha256"] = "0" * 64
    elif failure == "output-sha":
        (media / "source.mp4").write_bytes(b"corrupt")
    elif failure in {"traversal", "cross-run"}:
        manifest["source"]["videoUrl"] = (
            "../source.mp4" if failure == "traversal" else "/worlds/other-4d/media/source.mp4"
        )
    elif failure == "symlink":
        video = media / "source.mp4"
        video.unlink()
        video.symlink_to(source)
    elif failure == "directory-symlink":
        moved = world / "other-media"
        media.rename(moved)
        media.symlink_to(moved, target_is_directory=True)
    elif failure == "source-shape":
        manifest["source"] = []
    else:
        manifest["provenance"] = []
    (world / "audio.json").write_text(json.dumps(manifest))
    assert scene_manifest(root) is None
    assert (
        case.client.get("/api/pipeline/runs/viewer-run/viewer/scene.json", headers=AUTH).status_code
        == 404
    )


def test_only_designated_package_video_is_allowlisted(packaged):
    case, root, world, source = packaged
    declare_source_delivery(root, world)
    (world / "other.mp4").write_bytes(b"unrelated")
    prefix = "/api/pipeline/runs/viewer-run/viewer/assets/"
    assert (
        case.client.get(
            prefix + (world / "other.mp4").relative_to(root / "public").as_posix(), headers=AUTH
        ).status_code
        == 404
    )
