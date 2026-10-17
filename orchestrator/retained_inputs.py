"""Confined, immutable snapshots for a single local retained-scene delivery."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import struct
import tempfile
from pathlib import Path

from orchestrator.api.viewer import WORLD_SUFFIXES, run_name
from orchestrator.journal import Journal
from scripts.retained_scene import RetainedScene, prepare_config

SCHEMA = "wander.saved-inputs/1"
GOAL = """Perform exactly one detached local package_people stage using the pinned retained scene. No provider calls, inference, retries, other stages, edits to inputs, or quality acceptance. Start package_people once and end the turn. On the next turn inspect its result and ask for human viewer review. If blocked or failed, ask for help; never retry or finish the run automatically."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("invalid saved input path")
    parts = relative.split("/")
    if any(not part or part.startswith(".") for part in parts):
        raise ValueError("unconfined saved input path")
    path = root
    if root.is_symlink():
        raise ValueError("symlink input root")
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink saved input")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("nonregular saved input")
    return path


def read(root: Path, relative: str):
    return json.loads(safe_file(root, relative).read_text())


def require_self_contained_model(path: Path) -> None:
    """Only copy object formats whose viewer dependencies fit in the pinned file."""
    if path.suffix.lower() == ".ply":
        return
    if path.suffix.lower() != ".glb":
        raise ValueError("saved object models must be self-contained PLY or GLB files")
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError("invalid saved GLB header")
        magic, version, total, length, chunk_type = struct.unpack("<4sIIII", header)
        if (
            magic != b"glTF"
            or version != 2
            or total != path.stat().st_size
            or chunk_type != 0x4E4F534A
            or not 0 < length <= min(total - 20, 16_000_000)
        ):
            raise ValueError("invalid saved GLB manifest")
        manifest = json.loads(stream.read(length))
    for item in [*manifest.get("buffers", []), *manifest.get("images", [])]:
        uri = item.get("uri")
        if uri is not None and (not isinstance(uri, str) or not uri.startswith("data:")):
            raise ValueError("saved GLB references external buffers or textures")


def verify_delivery_inputs(run_dir: Path, delivery: dict) -> None:
    if (
        not isinstance(delivery, dict)
        or delivery
        != {
            "schema": SCHEMA,
            "sourceRunId": delivery.get("sourceRunId"),
            "config": "retained/scene.json",
            "inputs": "retained/inputs.json",
            "stage": "package_people",
        }
        or not isinstance(delivery.get("sourceRunId"), str)
    ):
        raise ValueError("invalid retained delivery manifest")
    entries = read(run_dir, delivery["inputs"])
    if not isinstance(entries, list) or not entries:
        raise ValueError("missing saved input pins")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise ValueError("invalid saved input pin")
        path = safe_file(run_dir, entry["path"])
        if (
            entry["path"] in seen
            or path.stat().st_size != entry["size"]
            or digest(path) != entry["sha256"]
        ):
            raise ValueError("saved input digest mismatch")
        seen.add(entry["path"])
    source = Path(read(run_dir, "run.json")["source"])
    if "retained/scene.json" not in seen or source.name not in seen:
        raise ValueError("missing source or config pin")
    config = read(run_dir, "retained/scene.json")
    if config.get("package") != "package" or config.get("world", {}).get("path") != "world.spz":
        raise ValueError("retained config must reference copied assets")
    scene = RetainedScene(run_dir / "retained/scene.json", safe_file(run_dir, source.name))
    needed = {
        source.name,
        "retained/scene.json",
        "retained/world.spz",
        "pi3x/cameras.json",
        "tracks/tracks.json",
    }
    needed.update(f"retained/package/{name}" for name in scene.files)
    for name, sequence in scene.sequences.items():
        needed.update(
            f"retained/package/{Path(name).parent}/{frame}" for frame in sequence["frames"]
        )
    tracks = read(run_dir, "tracks/tracks.json")["tracks"]
    for track in tracks:
        folder = f"lhm-motion-{track['track']:02d}"
        needed.add(f"{folder}/sequence.json")
        sequence = read(run_dir, f"{folder}/sequence.json")
        needed.update(f"{folder}/{frame}" for frame in sequence["frames"])
    # Optional supported inputs must also be pinned if present; removal of a receipt entry
    # cannot silently remove a packager dependency from verification.
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        if relative.startswith(("retained/package/", "lhm-motion-", "pi3x/", "tracks/")) and (
            path.is_file() or path.is_symlink()
        ):
            needed.add(relative)
    if not needed <= seen:
        raise ValueError("incomplete saved input pins")


def inspect_inputs(root: Path, summary: dict) -> dict:
    manifest = read(root, "run.json")
    options = manifest.get("options", {})
    if (
        not (options.get("all_people") or options.get("people", 1) > 1)
        or options.get("finetune")
        or options.get("marble", "video") not in WORLD_SUFFIXES
    ):
        raise ValueError("requires tracked people, an enabled world, and skip-finetune")
    if (
        summary.get("recordedStatus", summary["status"])
        not in {"paused", "succeeded", "failed", "canceled"}
        and not (root / "paused").is_file()
    ):
        raise ValueError("source run must be paused or finished")
    if Journal(root).active_attempts() or any(
        a["status"] in {"pending", "running", "unknown"} for a in summary.get("attempts", [])
    ):
        raise ValueError("source has unresolved stage attempts")
    source = Path(manifest["source"])
    if source.parent.resolve() != root.resolve():
        raise ValueError("source must be inside the run")
    source = safe_file(root, source.name)
    if digest(source) != manifest.get("sourceSha256") or digest(source) != summary["sourceSha256"]:
        raise ValueError("source clip digest mismatch")
    name = run_name(root)
    package = root / "public" / "worlds" / f"{name}-4d"
    world_rel = f"public/marble-{name}-{WORLD_SUFFIXES[options.get('marble', 'video')]}.spz"
    world = safe_file(root, world_rel)
    files = {"people.json", "cameras.json", "audio.json", "scene.json"}
    people = read(package, "people.json")
    for person in people["people"]:
        sequence = person["sequence"]
        files.add(sequence)
        seq = read(package, sequence)
        for frame, expected in zip(seq["frames"], seq["frame_sha256"], strict=True):
            rel = str(Path(sequence).parent / frame)
            if digest(safe_file(package, rel)) != expected:
                raise ValueError("exported frame digest mismatch")
            files.add(rel)

    def asset(value):
        prefix = f"/worlds/{package.name}/"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        if Path(value).suffix.lower() not in {
            ".ply",
            ".glb",
            ".gltf",
            ".wav",
            ".mp3",
            ".m4a",
            ".ogg",
        }:
            raise ValueError("unsupported retained asset")
        safe_file(package, value)
        return value

    audio = read(package, "audio.json")
    if audio.get("original"):
        files.add(asset(audio["original"]["url"]))
    if (package / "objects.json").exists() or (package / "objects.json").is_symlink():
        files.add("objects.json")
        for obj in read(package, "objects.json")["objects"]:
            model = obj.get("appearance", {}).get("model")
            if model:
                relative = asset(model)
                require_self_contained_model(safe_file(package, relative))
                files.add(relative)
    copies = {source.name: source, "retained/world.spz": world}
    for rel in files:
        copies[f"retained/package/{rel}"] = safe_file(root, str((package / rel).relative_to(root)))
    for rel in ("pi3x/cameras.json", "tracks/tracks.json"):
        copies[rel] = safe_file(root, rel)
    if not read(root, "pi3x/cameras.json").get("cameras"):
        raise ValueError("missing saved cameras")
    tracks = read(root, "tracks/tracks.json")
    ids = [p["track"] for p in people["people"]]
    if set(ids) != {t["track"] for t in tracks["tracks"]}:
        raise ValueError("retained cast differs from tracked people")
    if sorted(ids) != list(range(len(ids))) or len(ids) > options.get("people", 1):
        raise ValueError("unsupported retained track slots")
    for track in ids:
        if type(track) is not int or track < 0:
            raise ValueError("invalid track")
        folder = f"lhm-motion-{track:02d}"
        seq = read(root, f"{folder}/sequence.json")
        if any(
            not re.fullmatch(r"frame_[A-Za-z0-9_-]+\.ply", frame) for frame in seq.get("frames", [])
        ):
            raise ValueError("unsupported motion frame path")
        if not seq.get("frames"):
            raise ValueError("empty saved motion")
        for rel in [f"{folder}/sequence.json", *[f"{folder}/{frame}" for frame in seq["frames"]]]:
            copies[rel] = safe_file(root, rel)
        if "frame_sha256" in seq:
            for frame, expected in zip(seq["frames"], seq["frame_sha256"], strict=True):
                if digest(copies[f"{folder}/{frame}"]) != expected:
                    raise ValueError("saved motion frame digest mismatch")
        for extra in ("registration.json", "motion.json", "missing-poses.json"):
            rel = f"{folder}/{extra}"
            if (root / rel).exists() or (root / rel).is_symlink():
                copies[rel] = safe_file(root, rel)
    if (root / "pi3x/framealign.json").exists() or (root / "pi3x/framealign.json").is_symlink():
        copies["pi3x/framealign.json"] = safe_file(root, "pi3x/framealign.json")
    with tempfile.TemporaryDirectory() as tmp:
        prepare_config(package, world, source, Path(tmp) / "scene.json")
    return {
        "manifest": manifest,
        "copies": copies,
        "tracks": ids,
        "root": root,
        "hashes": {name: digest(path) for name, path in copies.items()},
    }


def copy_inputs(snapshot: dict, destination: Path) -> dict:
    pins = []
    for relative, source in sorted(snapshot["copies"].items()):
        source = safe_file(snapshot["root"], str(source.relative_to(snapshot["root"])))
        expected = snapshot["hashes"][relative]
        if digest(source) != expected:
            raise ValueError("saved input changed before copy")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected:
            raise ValueError("saved input changed during copy")
        pins.append({"path": relative, "sha256": expected, "size": target.stat().st_size})
    retained = destination / "retained"
    # Exported viewer URLs name the old package. Normalize only supported asset
    # references in our copies so the snapshot remains valid after relocation.
    prefix = f"/worlds/{snapshot['manifest']['name']}-4d/"
    audio_path = retained / "package/audio.json"
    audio = json.loads(audio_path.read_text())
    if audio.get("original") and audio["original"]["url"].startswith(prefix):
        audio["original"]["url"] = audio["original"]["url"][len(prefix) :]
        audio_path.write_text(json.dumps(audio, indent=2) + "\n")
    objects_path = retained / "package/objects.json"
    if objects_path.exists():
        objects = json.loads(objects_path.read_text())
        changed = False
        for obj in objects["objects"]:
            appearance = obj.get("appearance", {})
            model = appearance.get("model")
            if model and model.startswith(prefix):
                appearance["model"] = model[len(prefix) :]
                changed = True
        if changed:
            objects_path.write_text(json.dumps(objects, indent=2) + "\n")
    for pin in pins:
        copied = destination / pin["path"]
        pin.update(sha256=digest(copied), size=copied.stat().st_size)
    config = retained / "scene.json"
    prepare_config(
        retained / "package",
        retained / "world.spz",
        destination / Path(snapshot["manifest"]["source"]).name,
        config,
    )
    pins.append(
        {"path": "retained/scene.json", "sha256": digest(config), "size": config.stat().st_size}
    )
    (retained / "inputs.json").write_text(json.dumps(pins, indent=2) + "\n")
    return {
        "schema": SCHEMA,
        "sourceRunId": snapshot["manifest"]["runId"],
        "config": "retained/scene.json",
        "inputs": "retained/inputs.json",
        "stage": "package_people",
    }
