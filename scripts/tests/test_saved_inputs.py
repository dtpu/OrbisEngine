"""Saved-input API preparation never executes a pipeline stage."""

import json
import shutil
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestrator.api.app import ApiSettings
from orchestrator.api.service import create_service
from orchestrator.contracts import Run
from orchestrator.database import Base
from orchestrator.retained_inputs import digest, verify_delivery_inputs
from scripts.tests.test_retained_scene import scene, write_json  # noqa: F401

TOKEN = "saved-inputs-token-with-24-characters"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def prepared(scene, tmp_path):  # noqa: F811
    _, clip, old, fresh, world = scene
    root = tmp_path / "runs"
    source = root / "original"
    package = source / "public/worlds/original-4d"
    shutil.copytree(old, package)
    for path in fresh.rglob("*.ply"):
        target = package / path.relative_to(fresh)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(world, source / "public/marble-original-clean.spz")
    shutil.copyfile(clip, source / "source.mp4")
    options = {"all_people": True, "people": 2, "marble": "video", "finetune": False}
    write_json(
        source / "run.json",
        {
            "runId": "original",
            "sourceArtifactId": "source:original",
            "name": "original",
            "source": str(source / "source.mp4"),
            "sourceSha256": digest(clip),
            "options": options,
        },
    )
    write_json(source / "pi3x/cameras.json", json.loads((package / "cameras.json").read_text()))
    write_json(source / "tracks/tracks.json", {"tracks": [{"track": 0}, {"track": 1}]})
    for track, name in enumerate(("person", "person_01")):
        shutil.copytree(package / name, source / f"lhm-motion-{track:02d}")
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.sqlite'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    from orchestrator.repository import PipelineRepository

    repository = PipelineRepository(sessionmaker(engine, expire_on_commit=False))
    repository.create_run_row(
        Run(
            id="original",
            graph_version="test",
            code_revision="1234567",
            source_sha256=digest(clip),
            source_artifact_id="source:original",
            status="paused",
            created_by="test",
            configuration={"options": options},
        )
    )
    app = create_service(
        repository=repository,
        runs_root=root,
        settings=ApiSettings(bearer_token=TOKEN, code_revision="1234567"),
        artifact_root=tmp_path / "artifacts",
    )
    with TestClient(app) as client:
        yield client, app, source
    engine.dispose()


def test_prepare_paused_and_pinned(prepared):
    client, app, source = prepared
    url = "/api/pipeline/runs/original/saved-inputs"
    assert client.get(url).status_code == 401
    assert client.post(url, json={}).status_code == 401
    assert client.get(url, headers=AUTH).json()["available"]
    response = client.post(url, headers=AUTH, json={"id": "delivery"})
    assert response.status_code == 201, response.text
    assert response.json() == {"id": "delivery", "status": "paused"}
    summary = app.state.repository.run_summary("delivery")
    assert summary["status"] == "paused"
    assert summary["parentRunId"] == "original"
    assert not summary["attempts"] and not summary["reviews"]
    destination = app.state.runs._dir("delivery")
    manifest = json.loads((destination / "run.json").read_text())
    verify_delivery_inputs(destination, manifest["retainedDelivery"])
    assert (destination / "paused").is_file()
    state = json.loads((destination / "state.json").read_text())
    assert all(record["freshInference"] is False for record in state["stages"].values())
    assert client.post(url, headers=AUTH, json={"id": "delivery"}).status_code == 409
    (destination / "pi3x/cameras.json").write_text("corrupt")
    with pytest.raises(ValueError):
        verify_delivery_inputs(destination, manifest["retainedDelivery"])


@pytest.mark.parametrize("damage", ["missing", "corrupt", "symlink"])
def test_unavailable(prepared, damage):
    client, _, source = prepared
    path = source / "public/worlds/original-4d/person/frame_000.ply"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text("bad")
    else:
        target = source / "other.ply"
        path.rename(target)
        path.symlink_to(target)
    url = "/api/pipeline/runs/original/saved-inputs"
    assert client.get(url, headers=AUTH).json()["available"] is False
    assert client.post(url, headers=AUTH, json={"id": "delivery"}).status_code == 409


def test_failure_rolls_back(prepared):
    client, app, source = prepared
    with patch.object(app.state.repository, "ensure_node", side_effect=ValueError("failed seed")):
        response = client.post(
            "/api/pipeline/runs/original/saved-inputs", headers=AUTH, json={"id": "delivery"}
        )
    assert response.status_code == 409
    with pytest.raises(KeyError):
        app.state.repository.run_summary("delivery")
    assert not list((source.parent / ".submissions").iterdir())


def test_absolute_export_urls_are_normalized_in_copies(prepared):
    client, app, source = prepared
    package = source / "public/worlds/original-4d"
    audio_path = package / "audio.json"
    audio = json.loads(audio_path.read_text())
    audio["original"]["url"] = "/worlds/original-4d/" + audio["original"]["url"]
    write_json(audio_path, audio)
    objects_path = package / "objects.json"
    objects = json.loads(objects_path.read_text())
    for obj in objects["objects"]:
        obj["appearance"]["model"] = "/worlds/original-4d/" + obj["appearance"]["model"]
    write_json(objects_path, objects)
    original_hashes = [digest(path) for path in (audio_path, objects_path)]
    response = client.post(
        "/api/pipeline/runs/original/saved-inputs", headers=AUTH, json={"id": "delivery"}
    )
    assert response.status_code == 201, response.text
    destination = app.state.runs._dir("delivery")
    verify_delivery_inputs(
        destination, json.loads((destination / "run.json").read_text())["retainedDelivery"]
    )
    copied_audio = json.loads((destination / "retained/package/audio.json").read_text())
    assert not copied_audio["original"]["url"].startswith("/")
    copied_objects = json.loads((destination / "retained/package/objects.json").read_text())
    assert all(not obj["appearance"]["model"].startswith("/") for obj in copied_objects["objects"])
    assert original_hashes == [digest(path) for path in (audio_path, objects_path)]


def test_unresolved_source_is_unavailable(prepared):
    client, app, source = prepared
    from orchestrator.journal import Journal

    Journal(source).append("step.started", step="package_people", attempt="pending")
    url = "/api/pipeline/runs/original/saved-inputs"
    assert not client.get(url, headers=AUTH).json()["available"]
    assert client.post(url, headers=AUTH, json={"id": "delivery"}).status_code == 409


def test_source_changed_after_inspection_is_rejected(prepared, tmp_path):
    _, app, source = prepared
    from orchestrator.retained_inputs import copy_inputs, inspect_inputs

    snapshot = inspect_inputs(source, app.state.repository.run_summary("original"))
    (source / "source.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before copy"):
        copy_inputs(snapshot, tmp_path / "copy")


def test_removed_pin_is_rejected(prepared):
    client, app, _ = prepared
    response = client.post(
        "/api/pipeline/runs/original/saved-inputs", headers=AUTH, json={"id": "delivery"}
    )
    assert response.status_code == 201
    destination = app.state.runs._dir("delivery")
    pins_path = destination / "retained/inputs.json"
    pins = json.loads(pins_path.read_text())
    pins = [pin for pin in pins if pin["path"] != "lhm-motion-00/frame_000.ply"]
    pins_path.write_text(json.dumps(pins))
    with pytest.raises(ValueError, match="incomplete saved input pins"):
        verify_delivery_inputs(
            destination, json.loads((destination / "run.json").read_text())["retainedDelivery"]
        )


def test_symlink_ancestor_is_unavailable(prepared):
    client, _, source = prepared
    path = source / "pi3x"
    target = source / "other-cameras"
    path.rename(target)
    path.symlink_to(target, target_is_directory=True)
    response = client.get("/api/pipeline/runs/original/saved-inputs", headers=AUTH)
    assert response.json()["available"] is False


@pytest.mark.parametrize("extension", ["gltf", "glb"])
def test_external_object_dependencies_are_not_silently_dropped(prepared, extension):
    import struct

    client, _, source = prepared
    package = source / "public/worlds/original-4d"
    objects = json.loads((package / "objects.json").read_text())
    model = f"objects/bottle/model.{extension}"
    content = json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "mesh.bin"}]}).encode()
    if extension == "glb":
        content += b" " * (-len(content) % 4)
        content = (
            struct.pack("<4sIIII", b"glTF", 2, 20 + len(content), len(content), 0x4E4F534A)
            + content
        )
    (package / model).write_bytes(content)
    appearance = objects["objects"][0]["appearance"]
    appearance["model"] = model
    appearance["modelSha256"] = digest(package / model)
    write_json(package / "objects.json", objects)
    result = client.get("/api/pipeline/runs/original/saved-inputs", headers=AUTH).json()
    assert not result["available"]
    assert "self-contained" in result["reason"] or "external" in result["reason"]


def test_retry_cannot_change_prepared_run_or_unpause_it(prepared):
    client, app, _ = prepared
    assert (
        client.post(
            "/api/pipeline/runs/original/saved-inputs", headers=AUTH, json={"id": "delivery"}
        ).status_code
        == 201
    )
    before = app.state.repository.events_after("delivery", 0)
    response = client.post(
        "/api/pipeline/runs/delivery/retry",
        headers=AUTH,
        json={"node_id": "pi3x", "rationale": "Try inference"},
    )
    assert response.status_code == 409
    assert app.state.repository.events_after("delivery", 0) == before
    assert (app.state.runs._dir("delivery") / "paused").is_file()
