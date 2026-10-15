"""Focused retained delivery checks using a complete two-person package."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retained_scene import RetainedScene, prepare_config


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def scene(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video-frames-with-no-embedded-sound")
    source = digest(clip.read_bytes())
    old = tmp_path / "retained"
    fresh = tmp_path / "staging" / "worlds" / "fresh-4d"
    old.mkdir()
    fresh.mkdir(parents=True)
    world = tmp_path / "retained.spz"
    world.write_bytes(b"SPZ-world")
    cameras = {"cameras": [{"sourceIndex": 0, "camera_to_world": [1, 0, 0]}]}
    write_json(old / "cameras.json", cameras)
    write_json(fresh / "cameras.json", cameras)
    shared = {"referenceTrack": 1, "applied": "height"}
    people = {
        "schema": "wander.people/1",
        "sourceSha256": source,
        "fps": 2,
        "samples": 2,
        "duration": 1.0,
        "sourceIndices": [0, 1],
        "timestamps": [0.0, 0.5],
        "primary": "person",
        "peopleCount": 2,
        "sharedScale": shared,
        "people": [],
    }
    for track, name in enumerate(("person", "person_01")):
        frames = [f"frame_{i:03d}.ply" for i in range(2)]
        frame_bytes = [f"ply-{track}-{i}".encode() for i in range(2)]
        seq = {
            "sourceSha256": source,
            "canonicalStateSha256": digest(f"canonical-{track}".encode()),
            "frames": frames,
            "frame_sha256": [digest(data) for data in frame_bytes],
            "count": 2,
            "fps": 2,
            "duration": 1.0,
            "sourceIndices": [0, 1],
            "timestamps": [0.0, 0.5],
            "sharedScaleMultiplier": 1.0 if track else 0.8,
            "sharedScaleReferenceTrack": 1,
            "camera0Reframe": {
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "translation": [0, 0, 0],
                "tiltFromYDeg": None,
                "source": "cameras.json",
            },
        }
        write_json(old / name / "sequence.json", seq)
        write_json(fresh / name / "sequence.json", seq)
        for frame, data in zip(frames, frame_bytes):
            (fresh / name / frame).write_bytes(data)
        people["people"].append(
            {
                "id": name,
                "track": track,
                "sequence": f"{name}/sequence.json",
                "sourceIndices": [0, 1],
                "timestamps": [0.0, 0.5],
                "frames": 2,
                "quality": {"fresh": False},
                "transform": {
                    "translation": [0, 0, 0],
                    "quaternionXYZW": [0, 0, 0, 1],
                    "scale": 1,
                },
            }
        )
    retained_people = copy.deepcopy(people)
    retained_people["floorFit"] = {"provenance": "measured", "offset": 0.25}
    retained_people["people"][0]["transform"]["translation"] = [0, 1, 0]
    retained_people["people"][1]["transform"]["translation"] = [0, -1, 0]
    retained_people["people"][0]["quality"] = {"stale": True}
    write_json(old / "people.json", retained_people)
    write_json(fresh / "people.json", people)
    pcm = b"RIFF" + bytes(range(64))
    (old / "audio").mkdir()
    (old / "audio" / "original.wav").write_bytes(pcm)
    retained_audio = {
        "schema": "wander.audio/1",
        "source": {"hasAudio": False},
        "timeline": {"durationSeconds": 1},
        "original": {"url": "audio/original.wav", "offsetSeconds": 0},
        "defaultMode": "original",
        "provenance": {"packagedVideoSha256": source, "audioSha256": digest(pcm)},
    }
    write_json(old / "audio.json", retained_audio)
    write_json(
        fresh / "audio.json",
        {
            "schema": "wander.audio/1",
            "source": {"hasAudio": False},
            "timeline": {"durationSeconds": 1},
        },
    )
    write_json(
        old / "scene.json",
        {"schema": "wander.viewer-scene/1", "params": {"floor": "-0.5", "cam": "0,1,2,3,4,5"}},
    )
    objects = {
        "schema": "wander.objects/2",
        "clip": "/clips/old/source.mp4",
        "people": "people.json",
        "fps": 2,
        "samples": 2,
        "sourceIndices": [0, 1],
        "timestamps": [0.0, 0.5],
        "objects": [],
    }
    for name in ("bottle", "bag"):
        model = f"objects/{name}/object.ply"
        path = old / model
        path.parent.mkdir(parents=True)
        path.write_bytes(f"model-{name}".encode())
        objects["objects"].append(
            {
                "id": name,
                "appearance": {
                    "kind": "gaussians",
                    "model": model,
                    "modelSha256": digest(path.read_bytes()),
                },
                "bakedTrack": {"positions": [[1, 2, 3]]},
            }
        )
    write_json(old / "objects.json", objects)
    write_json(
        old / "placement.json",
        {"schema": "wander.placement/1", "world": old.name, "spz": world.name, "measured": [1, 2]},
    )
    write_json(old / "framealign.json", {"schema": "wander.framealign/1", "world": old.name})
    write_json(old / "collision.json", {"schema": "wander.collision/2", "spz": world.name})
    files = {
        str(path.relative_to(old)): digest(path.read_bytes())
        for path in old.rglob("*")
        if path.is_file()
        and path.name not in {"placement.json", "framealign.json", "collision.json"}
    }
    config = tmp_path / "retained.json"
    write_json(
        config,
        {
            "schema": "wander.retained-scene/1",
            "sourceSha256": source,
            "package": "retained",
            "world": {"path": "retained.spz", "sha256": digest(world.read_bytes())},
            "files": files,
        },
    )
    return config, clip, old, fresh, world


def update_pin(config: Path, old: Path, name: str) -> None:
    doc = read_json(config)
    doc["files"][name] = digest((old / name).read_bytes())
    write_json(config, doc)


def test_complete_retained_overlay_from_compact_archive(scene):
    config, clip, old, fresh, world = scene
    assert not list(old.glob("person*/frame_*.ply"))
    overlay = RetainedScene(config, clip)
    assert overlay.shared_scale == "1"
    target = fresh.parent.parent / "new.spz"
    receipt = overlay.apply(fresh, target, "/clips/new/source.mp4")
    assert target.read_bytes() == world.read_bytes()
    assert (fresh / "audio/original.wav").read_bytes() == (old / "audio/original.wav").read_bytes()
    audio = read_json(fresh / "audio.json")
    assert audio["source"]["sha256"] == digest(clip.read_bytes())
    assert audio["original"]["url"] == "audio/original.wav"
    assert audio["provenance"] == read_json(old / "audio.json")["provenance"]
    people = read_json(fresh / "people.json")
    assert people["floorFit"] == read_json(old / "people.json")["floorFit"]
    assert [p["transform"]["translation"] for p in people["people"]] == [[0, 1, 0], [0, -1, 0]]
    assert people["people"][0]["quality"] == {"fresh": False}
    objects = read_json(fresh / "objects.json")
    assert [o["id"] for o in objects["objects"]] == ["bottle", "bag"]
    assert objects["clip"] == "/clips/new/source.mp4"
    assert all(
        (fresh / o["appearance"]["model"]).read_bytes()
        == (old / o["appearance"]["model"]).read_bytes()
        for o in objects["objects"]
    )
    assert all(
        not (fresh / name).exists()
        for name in ("placement.json", "framealign.json", "collision.json")
    )
    assert receipt["acceptance"] == "none"


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [
        ("people.json", "timestamps", [0.0, 0.6]),
        ("people.json", "sourceIndices", [0, 2]),
        ("cameras.json", "cameras", [{"sourceIndex": 0, "camera_to_world": [2, 0, 0]}]),
        ("person/sequence.json", "canonicalStateSha256", "a" * 64),
        ("person/sequence.json", "frame_sha256", ["a" * 64, "b" * 64]),
        ("person/sequence.json", "sharedScaleMultiplier", 0.7),
        ("person/sequence.json", "fps", 3),
    ],
)
def test_mismatch_preserves_staging(scene, path, field, value):
    config, clip, _, fresh, _ = scene
    target = fresh.parent.parent / "new.spz"
    overlay = RetainedScene(config, clip)
    file = fresh / path
    doc = read_json(file)
    doc[field] = value
    write_json(file, doc)
    before = {str(p.relative_to(fresh)): p.read_bytes() for p in fresh.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        overlay.apply(fresh, target, "/clips/new/source.mp4")
    assert before == {
        str(p.relative_to(fresh)): p.read_bytes() for p in fresh.rglob("*") if p.is_file()
    }
    assert not target.exists()


def test_corrupt_fresh_frame_preserves_staging(scene):
    config, clip, _, fresh, _ = scene
    overlay = RetainedScene(config, clip)
    (fresh / "person/frame_000.ply").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="frame digest"):
        overlay.apply(fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4")
    assert not (fresh / "scene.json").exists()


@pytest.mark.parametrize("changed", ["config", "asset", "world"])
def test_pinned_input_changes_rejected(scene, changed):
    config, clip, old, fresh, world = scene
    overlay = RetainedScene(config, clip)
    if changed == "config":
        config.write_text(config.read_text() + " ")
    elif changed == "asset":
        (old / "audio/original.wav").write_bytes(b"other")
    else:
        world.write_bytes(b"other")
    with pytest.raises(ValueError, match="changed"):
        overlay.apply(fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4")
    assert not (fresh / "scene.json").exists()


@pytest.mark.parametrize(
    "bad", ["../escape.wav", "https://example.com/a.wav", "/worlds/other/audio.wav"]
)
def test_bad_audio_reference_rejected(scene, bad):
    config, clip, old, _, _ = scene
    doc = read_json(old / "audio.json")
    doc["original"]["url"] = bad
    write_json(old / "audio.json", doc)
    update_pin(config, old, "audio.json")
    with pytest.raises(ValueError):
        RetainedScene(config, clip)


def test_symlink_escape_rejected(scene):
    config, clip, old, _, _ = scene
    (old / "audio/original.wav").unlink()
    (old / "audio/original.wav").symlink_to(clip)
    update_pin(config, old, "audio/original.wav")
    with pytest.raises(ValueError, match="symlink"):
        RetainedScene(config, clip)


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("audio.json", {"tracks": [{"id": "speaker"}]}),
        ("audio.json", {"source": {"hasAudio": "false"}}),
        ("scene.json", {"schema": "wander.viewer-scene/2"}),
        ("objects.json", {"schema": "wander.objects/1"}),
    ],
)
def test_malformed_manifest_rejected(scene, name, change):
    config, clip, old, _, _ = scene
    doc = read_json(old / name)
    doc.update(change)
    write_json(old / name, doc)
    update_pin(config, old, name)
    with pytest.raises(ValueError):
        RetainedScene(config, clip)


def test_prepare_config_round_trip(scene):
    config, clip, old, fresh, world = scene
    config.unlink()
    prepared = prepare_config(old, world, clip, config)
    assert set(prepared["files"]) == {
        "people.json",
        "cameras.json",
        "audio.json",
        "scene.json",
        "objects.json",
        "person/sequence.json",
        "person_01/sequence.json",
        "audio/original.wav",
        "objects/bottle/object.ply",
        "objects/bag/object.ply",
    }
    assert prepare_config(old, world, clip, config) == prepared
    assert RetainedScene(config, clip).world_sha256 == digest(world.read_bytes())
    RetainedScene(config, clip).apply(
        fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4"
    )
    assert read_json(fresh / "people.json")["retainedScene"]["acceptance"] == "none"
    altered = read_json(config)
    altered["sourceSha256"] = "a" * 64
    write_json(config, altered)
    with pytest.raises(ValueError, match="different content"):
        prepare_config(old, world, clip, config)


def test_missing_pinned_reference_rejected(scene):
    config, clip, _, _, _ = scene
    doc = read_json(config)
    del doc["files"]["objects/bag/object.ply"]
    write_json(config, doc)
    with pytest.raises(ValueError):
        RetainedScene(config, clip)


def test_root_relative_model_reference_normalized(scene):
    config, clip, old, fresh, _ = scene
    objects = read_json(old / "objects.json")
    objects["objects"][0]["appearance"]["model"] = f"/worlds/{old.name}/objects/bottle/object.ply"
    write_json(old / "objects.json", objects)
    update_pin(config, old, "objects.json")
    RetainedScene(config, clip).apply(
        fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4"
    )
    assert (
        read_json(fresh / "objects.json")["objects"][0]["appearance"]["model"]
        == "objects/bottle/object.ply"
    )


@pytest.mark.parametrize("target", ["people.json", "cameras.json"])
def test_frame_align_convention_mismatch_preserves_staging(scene, target):
    config, clip, _, fresh, _ = scene
    overlay = RetainedScene(config, clip)
    doc = read_json(fresh / target)
    doc["frameAlign"] = {"tiltFromYDeg": 7.6, "adopted": "floor", "source": "/other/source"}
    write_json(fresh / target, doc)
    with pytest.raises(ValueError, match="frame alignment mismatch"):
        overlay.apply(fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4")
    assert not (fresh / "scene.json").exists()


def test_frame_align_ignores_source_paths_but_checks_adopted(scene):
    config, clip, old, fresh, _ = scene
    for package in (old, fresh):
        for name in ("people.json", "cameras.json"):
            doc = read_json(package / name)
            doc["frameAlign"] = {
                "tiltFromYDeg": 7.6,
                "adopted": "floor",
                "source": f"{package}/provenance",
                "file": f"{package}/cameras.json",
            }
            write_json(package / name, doc)
            if package == old:
                update_pin(config, old, name)
    RetainedScene(config, clip).apply(
        fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4"
    )
    assert (fresh / "scene.json").exists()


@pytest.mark.parametrize("bad", [{"scale": "1", "height": "1.7"}, {"scale": "nan"}])
def test_scene_parameter_rejection(scene, bad):
    config, clip, old, _, _ = scene
    doc = read_json(old / "scene.json")
    doc["params"].update(bad)
    write_json(old / "scene.json", doc)
    update_pin(config, old, "scene.json")
    with pytest.raises(ValueError, match="scene"):
        RetainedScene(config, clip)


@pytest.mark.parametrize(
    "field,value",
    [
        ("translation", [0, float("nan"), 0]),
        ("quaternionXYZW", [0, 0, 0, 0]),
        ("scale", -1),
    ],
)
def test_retained_transform_rejection(scene, field, value):
    config, clip, old, _, _ = scene
    doc = read_json(old / "people.json")
    doc["people"][0]["transform"][field] = value
    write_json(old / "people.json", doc)
    update_pin(config, old, "people.json")
    with pytest.raises(ValueError, match="transform"):
        RetainedScene(config, clip)


def test_unsupported_diagnostic_pin_rejected(scene):
    config, clip, old, _, _ = scene
    update_pin(config, old, "placement.json")
    with pytest.raises(ValueError, match="unreferenced or unpinned"):
        RetainedScene(config, clip)


def test_present_retained_frame_must_match_sequence_hash(scene):
    config, clip, old, _, _ = scene
    frame = old / "person/frame_000.ply"
    frame.write_bytes(b"corrupt-retained-frame")
    with pytest.raises(ValueError, match="retained frame digest mismatch"):
        RetainedScene(config, clip)


def test_retained_frame_appearing_after_snapshot_is_rechecked(scene):
    config, clip, old, fresh, _ = scene
    overlay = RetainedScene(config, clip)
    (old / "person/frame_000.ply").write_bytes(b"corrupt-retained-frame")
    with pytest.raises(ValueError, match="retained frame digest mismatch"):
        overlay.apply(fresh, fresh.parent.parent / "new.spz", "/clips/new/source.mp4")
    assert not (fresh / "scene.json").exists()
