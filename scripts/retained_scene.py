"""Bounded, hash-pinned delivery of authored scene overlays onto a fresh package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator.api.viewer import PRESENTATION_PARAMS, valid_presentation_value

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_OPTIONAL = {"objects.json"}
_REQUIRED = {"people.json", "cameras.json", "audio.json", "scene.json"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(value: str) -> str:
    _require(isinstance(value, str) and bool(_SHA.fullmatch(value)), "invalid SHA-256")
    return value


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "expected JSON object")
    return value


def _relative(value: str, package: Path, *, root_ref: bool = False) -> str:
    _require(isinstance(value, str) and value and "\\" not in value, "invalid package path")
    if root_ref and value.startswith("/worlds/"):
        prefix = f"/worlds/{package.name}/"
        _require(value.startswith(prefix), "reference belongs to another package")
        value = value[len(prefix) :]
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and not any(part in {".", ".."} for part in value.split("/"))
        and all(part for part in value.split("/"))
        and ":" not in value,
        "unconfined package path",
    )
    return value


def _confined(package: Path, value: str, *, root_ref: bool = False) -> Path:
    name = _relative(value, package, root_ref=root_ref)
    path = package / name
    _require(path.resolve().is_relative_to(package.resolve()), "escaping package symlink")
    return path


def _cast(people: dict) -> dict[tuple[str, int], dict]:
    entries = people.get("people")
    _require(
        isinstance(entries, list) and len(entries) == people.get("peopleCount") and bool(entries),
        "incomplete cast",
    )
    result = {}
    for person in entries:
        _require(isinstance(person, dict), "invalid person")
        key = (person.get("id"), person.get("track"))
        _require(
            isinstance(key[0], str) and type(key[1]) is int and key[1] >= 0, "invalid cast identity"
        )
        _require(key not in result, "duplicate cast identity")
        result[key] = person
    _require(len({key[0] for key in result}) == len(result), "duplicate person ID")
    _require(len({key[1] for key in result}) == len(result), "duplicate track")
    _require(people.get("primary") in {key[0] for key in result}, "missing primary")
    return result


def _sequence(package: Path, person: dict, *, pinned: dict[str, str] | None) -> dict:
    name = _relative(person.get("sequence"), package)
    if pinned is not None:
        _require(name in pinned, "unpinned sequence")
    sequence = _json(_confined(package, name))
    frames, hashes = sequence.get("frames"), sequence.get("frame_sha256")
    _require(
        isinstance(frames, list)
        and bool(frames)
        and len(frames) == sequence.get("count")
        and isinstance(hashes, list)
        and len(frames) == len(hashes)
        and len(sequence.get("sourceIndices", [])) == len(frames)
        and len(sequence.get("timestamps", [])) == len(frames)
        and len(set(frames)) == len(frames),
        "incomplete frame hashes",
    )
    for frame, expected in zip(frames, hashes):
        _sha(expected)
        frame_name = str(PurePosixPath(name).parent / _relative(frame, package))
        frame_path = _confined(package, frame_name)
        if pinned is None or frame_path.exists():
            _require(
                _digest(frame_path) == expected,
                "fresh frame digest mismatch"
                if pinned is None
                else "retained frame digest mismatch",
            )
    _sha(sequence.get("sourceSha256"))
    _sha(sequence.get("canonicalStateSha256"))
    return sequence


def _reframe(sequence: dict) -> dict:
    value = sequence.get("camera0Reframe")
    _require(isinstance(value, dict), "missing camera reframe")
    return {key: value.get(key) for key in ("rotation", "translation", "tiltFromYDeg")}


def _frame_align(value: object) -> dict | None:
    if value is None:
        return None
    _require(isinstance(value, dict), "invalid frame alignment")
    keys = ("tiltFromYDeg", "adopted", "rotationRowMajor", "quaternionXYZW")
    return {key: value[key] for key in keys if key in value}


def _transform(value: object) -> None:
    _require(isinstance(value, dict), "missing retained transform")
    translation = value.get("translation")
    quaternion = value.get("quaternionXYZW")
    scale = value.get("scale")
    for vector, length in ((translation, 3), (quaternion, 4)):
        _require(
            isinstance(vector, list)
            and len(vector) == length
            and all(type(item) in (int, float) and math.isfinite(item) for item in vector),
            "invalid retained transform vector",
        )
    _require(
        type(scale) in (int, float) and math.isfinite(scale) and scale > 0,
        "invalid retained transform scale",
    )
    norm = math.sqrt(sum(item * item for item in quaternion))
    _require(0.99 <= norm <= 1.01, "invalid retained transform rotation")


class RetainedScene:
    """Validate a retained package, then overlay authored data into staging."""

    def __init__(self, config_path: Path, clip: Path):
        self.config_path = Path(config_path)
        self.clip = Path(clip)
        self.config_hash = _digest(self.config_path)
        config = _json(self.config_path)
        _require(config.get("schema") == "wander.retained-scene/1", "unsupported retention schema")
        self.source_hash = _sha(config.get("sourceSha256"))
        _require(_digest(self.clip) == self.source_hash, "source clip digest mismatch")
        package_value = config.get("package")
        _require(isinstance(package_value, str) and bool(package_value), "missing package")
        self.package = (self.config_path.parent / package_value).resolve()
        _require(self.package.is_dir(), "missing retained package")
        world = config.get("world")
        _require(isinstance(world, dict), "missing retained world")
        world_value = world.get("path")
        _require(isinstance(world_value, str) and world_value.endswith(".spz"), "world must be SPZ")
        self.world = (self.config_path.parent / world_value).resolve()
        self.world_hash = _sha(world.get("sha256"))
        self.world_sha256 = self.world_hash
        _require(_digest(self.world) == self.world_hash, "retained world digest mismatch")
        files = config.get("files")
        _require(isinstance(files, dict), "missing pinned files")
        self.files = {_relative(name, self.package): _sha(digest) for name, digest in files.items()}
        _require(_REQUIRED <= self.files.keys(), "missing required retained manifest")
        self._validate_pins()
        self.people = _json(self.package / "people.json")
        _require(self.people.get("schema") == "wander.people/1", "unsupported people schema")
        _require(self.people.get("sourceSha256") == self.source_hash, "retained source mismatch")
        self.cast = _cast(self.people)
        _require(isinstance(self.people.get("floorFit"), dict), "missing retained floor fit")
        self.sequences = {}
        needed = set(_REQUIRED)
        for person in self.cast.values():
            _transform(person.get("transform"))
            name = _relative(person["sequence"], self.package)
            _require(name not in self.sequences, "shared sequence across tracks")
            needed.add(name)
            sequence = _sequence(self.package, person, pinned=self.files)
            _require(sequence["sourceSha256"] == self.source_hash, "sequence source mismatch")
            for field, expected in (("personId", person["id"]), ("trackIndex", person["track"])):
                if field in sequence:
                    _require(sequence[field] == expected, "sequence cast mismatch")
            self.sequences[name] = sequence
        self.cameras = _json(self.package / "cameras.json")
        _require(
            isinstance(self.cameras.get("cameras"), list) and self.cameras["cameras"],
            "missing cameras",
        )
        self.scene = _json(self.package / "scene.json")
        _require(self.scene.get("schema") == "wander.viewer-scene/1", "unsupported scene schema")
        params = self.scene.get("params")
        _require(isinstance(params, dict), "invalid scene parameters")
        _require(not {"scale", "height"} <= params.keys(), "conflicting scene scale and height")
        for key, value in params.items():
            _require(
                key in PRESENTATION_PARAMS
                and isinstance(value, str)
                and bool(value)
                and len(value) <= 1024
                and not re.search(r"[\x00-\x1f\x7f]", value)
                and valid_presentation_value(key, value),
                "invalid scene parameter",
            )
        self.audio = _json(self.package / "audio.json")
        _require(self.audio.get("schema") == "wander.audio/1", "unsupported audio schema")
        _require(
            not self.audio.get("tracks") and not self.audio.get("spatialMixComplete"),
            "spatial stems unsupported",
        )
        _require(self.audio.get("defaultMode", "original") == "original", "unsupported audio mode")
        source = self.audio.get("source")
        _require(
            isinstance(source, dict) and type(source.get("hasAudio")) is bool,
            "invalid audio source",
        )
        _require(isinstance(self.audio.get("provenance", {}), dict), "invalid audio provenance")
        for value in (
            source.get("sha256"),
            self.audio.get("provenance", {}).get("videoClipSha256"),
            self.audio.get("provenance", {}).get("packagedVideoSha256"),
        ):
            if value is not None:
                _require(_sha(value) == self.source_hash, "audio source mismatch")
        _require(
            isinstance(self.audio.get("timeline"), dict)
            and abs(self.audio["timeline"].get("durationSeconds", -1) - self.people["duration"])
            <= 0.1,
            "audio duration mismatch",
        )
        original = self.audio.get("original")
        if original is not None:
            _require(
                isinstance(original, dict)
                and isinstance(original.get("offsetSeconds"), (int, float)),
                "invalid original audio",
            )
            audio_name = _relative(original.get("url"), self.package, root_ref=True)
            needed.add(audio_name)
            _require(original["offsetSeconds"] >= 0, "invalid audio offset")
            provenance_hash = self.audio.get("provenance", {}).get("audioSha256")
            if provenance_hash is not None:
                _require(
                    _sha(provenance_hash) == self.files.get(audio_name), "audio provenance mismatch"
                )
        self.objects = None
        if "objects.json" in self.files:
            needed.add("objects.json")
            self.objects = _json(self.package / "objects.json")
            _require(self.objects.get("schema") == "wander.objects/2", "unsupported objects schema")
            _require(isinstance(self.objects.get("objects"), list), "invalid objects")
            for obj in self.objects["objects"]:
                appearance = obj.get("appearance", {})
                if "model" in appearance:
                    model = _relative(appearance["model"], self.package, root_ref=True)
                    needed.add(model)
                    if "modelSha256" in appearance:
                        _require(
                            _sha(appearance["modelSha256"]) == self.files.get(model),
                            "object model digest mismatch",
                        )
        _require(set(self.files) == needed, "unreferenced or unpinned retained file")
        self._validate_pins()
        shared = self.people.get("sharedScale", {})
        reference = shared.get("referenceTrack")
        _require(
            reference is None or type(reference) is int and reference >= 0,
            "invalid shared scale reference",
        )
        _require(
            reference is None or reference in {key[1] for key in self.cast},
            "unknown shared scale reference",
        )
        self.shared_scale = "none" if reference is None else str(reference)

    def _validate_pins(self) -> None:
        _require(_digest(self.config_path) == self.config_hash, "retention config changed")
        _require(_digest(self.clip) == self.source_hash, "source clip changed")
        _require(_digest(self.world) == self.world_hash, "retained world changed")
        for name, expected in self.files.items():
            _require(_digest(_confined(self.package, name)) == expected, "retained file changed")

    def apply(self, fresh_world: Path, world_target: Path, video_url: str) -> dict:
        fresh_world, world_target = Path(fresh_world), Path(world_target)
        _require(
            fresh_world.resolve() != self.package and fresh_world.is_dir(), "fresh package required"
        )
        _require(
            world_target.suffix == ".spz"
            and not world_target.exists()
            and not world_target.is_symlink(),
            "staging SPZ target must be absent",
        )
        _require(
            world_target.parent.resolve()
            in {fresh_world.parent.resolve(), fresh_world.parent.parent.resolve()},
            "SPZ target outside staging",
        )
        _require(
            isinstance(video_url, str)
            and video_url.startswith("/")
            and not video_url.startswith("//"),
            "invalid video URL",
        )
        self._validate_pins()
        fresh = _json(_confined(fresh_world, "people.json"))
        _require(fresh.get("schema") == "wander.people/1", "unsupported fresh people schema")
        for key in (
            "sourceSha256",
            "fps",
            "samples",
            "duration",
            "sourceIndices",
            "timestamps",
            "primary",
        ):
            _require(fresh.get(key) == self.people.get(key), f"people {key} mismatch")
        _require(fresh["sourceSha256"] == self.source_hash, "fresh source mismatch")
        fresh_cast = _cast(fresh)
        _require(fresh_cast.keys() == self.cast.keys(), "cast mismatch")
        fresh_cameras = _json(_confined(fresh_world, "cameras.json"))
        _require(fresh_cameras.get("cameras") == self.cameras["cameras"], "camera mismatch")
        _require(
            _frame_align(fresh.get("frameAlign")) == _frame_align(self.people.get("frameAlign")),
            "people frame alignment mismatch",
        )
        _require(
            _frame_align(fresh_cameras.get("frameAlign"))
            == _frame_align(self.cameras.get("frameAlign")),
            "camera frame alignment mismatch",
        )
        _require(
            fresh.get("sharedScale", {}).get("referenceTrack")
            == self.people.get("sharedScale", {}).get("referenceTrack"),
            "shared scale mismatch",
        )
        for key, old_person in self.cast.items():
            new_person = fresh_cast[key]
            for field in ("sequence", "sourceIndices", "timestamps", "frames"):
                _require(new_person.get(field) == old_person.get(field), f"person {field} mismatch")
            old_seq = self.sequences[_relative(old_person["sequence"], self.package)]
            _sequence(self.package, old_person, pinned=self.files)
            new_seq = _sequence(fresh_world, new_person, pinned=None)
            for field, expected in (
                ("personId", new_person["id"]),
                ("trackIndex", new_person["track"]),
            ):
                if field in new_seq:
                    _require(new_seq[field] == expected, "fresh sequence cast mismatch")
            for field in (
                "sourceSha256",
                "canonicalStateSha256",
                "sourceIndices",
                "timestamps",
                "count",
                "fps",
                "duration",
                "frame_sha256",
                "sharedScaleMultiplier",
                "sharedScaleReferenceTrack",
                "frames",
            ):
                _require(new_seq.get(field) == old_seq.get(field), f"sequence {field} mismatch")
            _require(_reframe(new_seq) == _reframe(old_seq), "camera reframe mismatch")
        fresh_audio = _json(_confined(fresh_world, "audio.json"))
        _require(
            fresh_audio.get("schema") == "wander.audio/1"
            and fresh_audio.get("source", {}).get("hasAudio") == self.audio["source"]["hasAudio"],
            "audio stream mismatch",
        )
        if fresh_audio["source"].get("sha256") is not None:
            _require(
                fresh_audio["source"]["sha256"] == self.source_hash, "fresh audio source mismatch"
            )
        _require(
            abs(
                fresh_audio.get("timeline", {}).get("durationSeconds", -1) - self.people["duration"]
            )
            <= 0.1,
            "fresh audio duration mismatch",
        )
        if self.objects is not None:
            for field in ("fps", "samples", "sourceIndices", "timestamps"):
                _require(self.objects.get(field) == fresh.get(field), f"object {field} mismatch")
        # Every check above, including all fresh frame bytes, precedes writes.
        merged = copy.deepcopy(fresh)
        for key, old_person in self.cast.items():
            current = fresh_cast[key]
            current["transform"] = copy.deepcopy(old_person.get("transform"))
        if "floorFit" in self.people:
            merged["floorFit"] = copy.deepcopy(self.people["floorFit"])
        merged["retainedScene"] = {
            "schema": "wander.retained-scene/1",
            "configSha256": self.config_hash,
            "scope": ["person transforms", "floorFit"],
            "acceptance": "none",
        }
        merged["people"] = [fresh_cast[(p["id"], p["track"])] for p in fresh["people"]]
        audio = copy.deepcopy(self.audio)
        audio["source"]["sha256"] = self.source_hash
        if audio.get("original") is not None:
            audio["original"]["url"] = _relative(
                audio["original"]["url"], self.package, root_ref=True
            )
        objects = copy.deepcopy(self.objects)
        if objects is not None:
            objects["clip"] = video_url
            objects["people"] = "people.json"
            for obj in objects["objects"]:
                appearance = obj.get("appearance", {})
                if "model" in appearance:
                    appearance["model"] = _relative(
                        appearance["model"], self.package, root_ref=True
                    )
        files_out = {"people.json": merged, "audio.json": audio, "scene.json": self.scene}
        if objects is not None:
            files_out["objects.json"] = objects
        copied = {}
        for name in self.files:
            if name in files_out or name in _REQUIRED or name.endswith("/sequence.json"):
                continue
            target = _confined(fresh_world, name)
            _require(not target.exists(), "retained asset would overwrite staging asset")
        for name, doc in files_out.items():
            target = _confined(fresh_world, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(doc, indent=2) + "\n")
            copied[name] = _digest(target)
        for name, expected in self.files.items():
            if name in files_out or name in _REQUIRED or name.endswith("/sequence.json"):
                continue
            target = _confined(fresh_world, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_confined(self.package, name), target)
            _require(_digest(target) == expected, "copied asset digest mismatch")
            copied[name] = expected
        shutil.copyfile(self.world, world_target)
        _require(_digest(world_target) == self.world_hash, "copied world digest mismatch")
        return {
            "configSha256": self.config_hash,
            "sourceSha256": self.source_hash,
            "worldSha256": self.world_hash,
            "compatibility": "exact source, cast, cameras, timeline, frame geometry and scale",
            "copiedSha256": copied,
            "retainedOnly": [
                "presentation",
                "person transforms",
                "floorFit",
                "original audio",
                "objects",
            ],
            "acceptance": "none",
        }


def prepare_config(package: Path, world: Path, clip: Path, out: Path) -> dict:
    """Pin only supported manifests and their directly referenced assets."""
    package, world, clip, out = map(Path, (package, world, clip, out))
    _require(package.is_dir(), "missing retained package")
    _require(world.suffix == ".spz" and world.is_file(), "world must be SPZ")
    source = _digest(clip)
    people = _json(_confined(package, "people.json"))
    _require(people.get("sourceSha256") == source, "retained source mismatch")
    needed = set(_REQUIRED)
    for person in _cast(people).values():
        needed.add(_relative(person.get("sequence"), package))
    needed |= {name for name in _OPTIONAL if (package / name).exists()}
    audio = _json(_confined(package, "audio.json"))
    if audio.get("original") is not None:
        needed.add(_relative(audio["original"].get("url"), package, root_ref=True))
    if "objects.json" in needed:
        for obj in _json(_confined(package, "objects.json")).get("objects", []):
            model = obj.get("appearance", {}).get("model")
            if model is not None:
                needed.add(_relative(model, package, root_ref=True))
    files = {name: _digest(_confined(package, name)) for name in sorted(needed)}
    config = {
        "schema": "wander.retained-scene/1",
        "sourceSha256": source,
        "package": os.path.relpath(package.resolve(), out.parent.resolve()),
        "world": {
            "path": os.path.relpath(world.resolve(), out.parent.resolve()),
            "sha256": _digest(world),
        },
        "files": files,
    }
    if out.exists():
        _require(_json(out) == config, "retention config already exists with different content")
        RetainedScene(out, clip)
        return config
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=".retained-", dir=out.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(config, stream, indent=2)
        stream.write("\n")
    try:
        RetainedScene(temporary, clip)
        temporary.rename(out)
    finally:
        temporary.unlink(missing_ok=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pin a bounded retained scene without copying assets"
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--world", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepared = prepare_config(args.package, args.world, args.clip, args.out)
    print(
        json.dumps(
            {
                "config": str(args.out),
                "sourceSha256": prepared["sourceSha256"],
                "worldSha256": prepared["world"]["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
