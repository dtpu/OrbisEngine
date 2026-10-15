"""Completion validates retained package bytes without running any model providers."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator.cli import RunContext, command_finish, plan_for
from orchestrator.journal import Journal


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDER_DATABASE_URL", raising=False)

    def create(*, people=2, all_people=True, marble="none"):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"retained source clip")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        context = RunContext(
            "completion",
            tmp_path,
            "example",
            source,
            ROOT,
            {"people": people, "all_people": all_people, "marble": marble, "objects": False},
            digest,
        )
        write_json(
            tmp_path / "run.json",
            {
                "runId": context.run_id,
                "name": context.name,
                "source": str(source),
                "sourceSha256": digest,
                "repository": str(ROOT),
                "options": context.options,
            },
        )
        delivered_source = tmp_path / "public/clips/example/source.mp4"
        delivered_source.parent.mkdir(parents=True)
        delivered_source.write_bytes(source.read_bytes())
        if marble != "none":
            (tmp_path / "public/marble-example-clean.spz").write_bytes(b"retained world")
        write_json(tmp_path / "tracks/tracks.json", {"trackCount": people})
        write_json(
            tmp_path / "state.json",
            {"stages": {step: {"status": "ok"} for step in plan_for(context)}},
        )
        world = tmp_path / "public/worlds/example-4d"
        write_json(world / "cameras.json", {"cameras": [{"sourceIndex": 0}]})
        write_json(
            world / "audio.json",
            {
                "schema": "wander.audio/1",
                "source": {"hasAudio": False, "sha256": digest},
                "timeline": {"durationSeconds": 1},
            },
        )
        entries = []
        for index in range(people):
            name = "person" if index == 0 else f"person_{index:02d}"
            frame = world / name / "frame.ply"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(f"retained frame for person {index}".encode())
            write_json(
                frame.parent / "sequence.json",
                {
                    "sourceSha256": digest,
                    "frames": [frame.name],
                    "frame_sha256": [hashlib.sha256(frame.read_bytes()).hexdigest()],
                },
            )
            entries.append({"id": name, "sequence": f"{name}/sequence.json"})
        if all_people or people > 1:
            write_json(
                world / "people.json",
                {
                    "schema": "wander.people/1",
                    "sourceSha256": digest,
                    "peopleCount": people,
                    "primary": "person",
                    "people": entries,
                },
            )
        return context, world

    return create


def finish(context):
    return command_finish(context, argparse.Namespace(status="succeeded", summary="validated"))


@pytest.mark.parametrize("people,all_people", [(2, True), (1, True), (1, False)])
def test_complete_package_is_accepted(package, people, all_people):
    context, _ = package(people=people, all_people=all_people)
    assert finish(context) == 0
    assert Journal(context.run_dir).finished() == "succeeded"


@pytest.mark.parametrize("missing", ["source", "audio", "world", "finetuned_world"])
def test_completion_requires_viewer_delivery_files(package, missing, capsys):
    context, world = package(marble="video")
    state_path = context.run_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["stages"]["_verify"] = {"passed": True}
    write_json(state_path, state)
    if missing == "source":
        (context.run_dir / "public/clips/example/source.mp4").unlink()
    elif missing == "audio":
        (world / "audio.json").unlink()
    elif missing == "world":
        (context.run_dir / "public/marble-example-clean.spz").unlink()
    else:
        context.options["finetune"] = True
        described_path = context.run_dir / "run.json"
        described = json.loads(described_path.read_text())
        described["options"] = context.options
        write_json(described_path, described)
        # A successful fine-tune record and the base world do not deliver its final output.
        state["stages"]["finetune"] = {"status": "ok"}
        write_json(state_path, state)
    assert finish(context) == 1
    assert "viewer scene" in capsys.readouterr().err
    assert Journal(context.run_dir).finished() is None


def test_exported_source_bytes_must_match_submitted_source(package, capsys):
    context, _ = package()
    (context.run_dir / "public/clips/example/source.mp4").write_bytes(b"wrong exported clip")
    assert finish(context) == 1
    assert "exported source" in capsys.readouterr().err
    assert Journal(context.run_dir).finished() is None


@pytest.mark.parametrize(
    "damage", ["truncated", "array", "schema", "has_audio", "duration", "source_hash"]
)
def test_audio_metadata_must_be_readable_and_describe_its_source(package, damage):
    context, world = package()
    audio = world / "audio.json"
    document = json.loads(audio.read_text())
    if damage == "truncated":
        audio.write_text('{"schema":')
    else:
        if damage == "array":
            document = []
        elif damage == "schema":
            document["schema"] = "unknown"
        elif damage == "has_audio":
            document["source"]["hasAudio"] = "false"
        elif damage == "duration":
            document["timeline"]["durationSeconds"] = float("nan")
        else:
            document["source"]["sha256"] = "0" * 64
        write_json(audio, document)
    assert finish(context) == 1
    assert Journal(context.run_dir).finished() is None


@pytest.mark.parametrize(
    "damage", ["missing_sequence", "missing_frame", "changed_frame", "source_hash"]
)
def test_secondary_person_must_be_intact(package, damage):
    context, world = package()
    sequence = world / "person_01/sequence.json"
    frame = world / "person_01/frame.ply"
    if damage == "missing_sequence":
        sequence.unlink()
    elif damage == "missing_frame":
        frame.unlink()
    elif damage == "changed_frame":
        frame.write_bytes(b"changed secondary frame")
    else:
        document = json.loads(sequence.read_text())
        document["sourceSha256"] = "0" * 64
        write_json(sequence, document)
    assert finish(context) == 1
    assert Journal(context.run_dir).finished() is None


@pytest.mark.parametrize(
    "damage",
    ["missing", "invalid_json", "array", "empty", "bad_entry", "count", "source_hash", "primary"],
)
def test_tracked_plan_requires_valid_people_manifest(package, damage):
    context, world = package()
    manifest = world / "people.json"
    document = json.loads(manifest.read_text())
    if damage == "missing":
        manifest.unlink()
    elif damage == "invalid_json":
        manifest.write_text("{broken")
    else:
        if damage == "array":
            document = []
        elif damage == "empty":
            document.update(people=[], peopleCount=0)
        elif damage == "bad_entry":
            document["people"][1] = {"id": "person_01", "sequence": None}
        elif damage == "count":
            document["peopleCount"] = 1
        elif damage == "source_hash":
            document["sourceSha256"] = "0" * 64
        elif damage == "primary":
            document["primary"] = "absent"
        write_json(manifest, document)
    assert finish(context) == 1
    assert Journal(context.run_dir).finished() is None


@pytest.mark.parametrize("document", [[], {}, {"frames": "frame.ply"}, {"frames": []}])
def test_malformed_secondary_sequence_is_rejected(package, document):
    context, world = package()
    if isinstance(document, dict):
        document.update(sourceSha256=context.source_sha256, frame_sha256=[])
    write_json(world / "person_01/sequence.json", document)
    assert finish(context) == 1
    assert Journal(context.run_dir).finished() is None


@pytest.mark.parametrize("target", ["sequence", "frame"])
@pytest.mark.parametrize("escape", ["relative", "absolute", "symlink"])
def test_package_paths_cannot_escape_world(package, target, escape, capsys):
    context, world = package()
    original = world / "person_01" / ("sequence.json" if target == "sequence" else "frame.ply")
    outside = context.run_dir / f"outside-{original.name}"
    outside.write_bytes(original.read_bytes())
    if escape == "symlink":
        original.unlink()
        original.symlink_to(outside)
    else:
        name = (
            str(outside)
            if escape == "absolute"
            else ("../../../" if target == "sequence" else "../../../../") + outside.name
        )
        if target == "sequence":
            manifest = world / "people.json"
            document = json.loads(manifest.read_text())
            document["people"][1]["sequence"] = name
            write_json(manifest, document)
        else:
            sequence = original.parent / "sequence.json"
            document = json.loads(sequence.read_text())
            document["frames"] = [name]
            write_json(sequence, document)
    assert finish(context) == 1
    expected = "needs a relative path" if escape == "absolute" else "escapes the world"
    assert expected in capsys.readouterr().err
    assert Journal(context.run_dir).finished() is None


@pytest.mark.parametrize("people,all_people", [(1, False), (2, True)])
def test_static_world_still_requires_passed_manual_review(package, people, all_people):
    context, _ = package(people=people, all_people=all_people, marble="video")
    assert finish(context) == 1
    assert Journal(context.run_dir).finished() is None
    state_path = context.run_dir / "state.json"
    state = json.loads(state_path.read_text())
    state["stages"]["_verify"] = {"passed": True}
    write_json(state_path, state)
    assert finish(context) == 0
    assert Journal(context.run_dir).finished() == "succeeded"
